# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/vsb-power-line-fault-detection"))

print(os.listdir("../input/dataprocessing"))

# Any results you write to the current directory are nsaved as output.
import pandas as pd

import numpy as np

from scipy import stats





def get_outliers(df, outlier_z_score_abs_threshold=5, outlier_feature_fraction=0.3):

    # features is level 1. ts is level 0.

    f1 = df.groupby(df.columns.get_level_values(1), axis=1).mean().T

    f2 = df.abs().groupby(df.columns.get_level_values(1), axis=1).mean().T

    f3 = df.groupby(df.columns.get_level_values(1), axis=1).std().T

    f4 = df.groupby(df.columns.get_level_values(1), axis=1).quantile(0.5).T

    f5 = df.groupby(df.columns.get_level_values(1), axis=1).quantile(0.1).T

    f6 = df.groupby(df.columns.get_level_values(1), axis=1).quantile(0.9).T



    print('Features', f1.shape, f1.columns[:3])

    if 'diff_smoothend_by_1 Quant-0.0' in f1.columns:

        f1 = f1.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)

        f2 = f2.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)

        f3 = f3.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)

        f4 = f4.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)

        f5 = f5.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)

        f6 = f6.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)



    outlier_feature_count = int(f1.shape[0] * outlier_feature_fraction)

    fs = [f1, f2, f3, f4, f5, f6]

    zscores = list(map(lambda f: stats.zscore(f, axis=1), fs))

    outliers = list(map(lambda zscore: np.abs(zscore) > outlier_z_score_abs_threshold, zscores))

    examplewise_outliers = list(map(lambda outlier: np.sum(outlier, axis=0), outliers))

    # print('Shape of example_outliers', examplewise_outliers[0].shape)

    outlier_filters = []

    for i, ex_out in enumerate(examplewise_outliers):

        outlier_filter = ex_out > outlier_feature_count

        outlier_filters.append(outlier_filter)

        # zero_percent = round(ex_out[ex_out == 0].shape[0] / ex_out.shape[0] * 100, 2)

        # outlier_count = ex_out[outlier_filter].shape[0]

        # print('FeatureIndex', i, 'zero percent', zero_percent)

        # print('FeatureIndex', i, 'outlier count', outlier_count)



    outlier_filter = np.sum(outlier_filters, axis=0)

    outlier_filter = outlier_filter >= len(outlier_filters) // 2

    print('Outlier percent:', round(outlier_filter.sum() / outlier_filter.shape[0] * 100, 2))

    print('Outlier count:', outlier_filter.sum())



    output_df = pd.Series(outlier_filter, index=f1.columns).to_frame('outliers')

    # output_df.index = output_df.index.astype(int)

    return output_df





def target_class_outlier_distribution_grid_search(df, meta_fname):

    meta_df = pd.read_csv(meta_fname).set_index('signal_id')[['target']]

    thresholds = [4, 5, 6]

    fractions = [0.3, 0.5, 0.7]

    vc = round(meta_df['target'].value_counts().loc[1] / meta_df.shape[0] * 100, 2)

    print('In original data, target class 1 is {}%'.format(vc))



    target_one_percent = pd.DataFrame([], columns=thresholds, index=fractions)

    target_one_percent.index.name = 'outlier_feature_fraction'

    target_one_percent.columns.name = 'outlier_z_score_abs_threshold'

    for thresh in thresholds:

        for frac in fractions:

            outliers_df = get_outliers(df, thresh, frac)

            outliers_df = outliers_df.join(meta_df, how='left')

            vc = outliers_df[outliers_df['outliers'] == True]['target'].value_counts()

            if 1 not in vc.index:

                target_one_percent.loc[frac, thresh] = 0

            else:

                percnt = round(vc.loc[1] / vc.sum() * 100, 2)

                target_one_percent.loc[frac, thresh] = percnt

                print('In outliers with thresh:{}, frac:{}, target class 1 is {}%'.format(thresh, frac, percnt))



    return target_one_percent

# Taken from https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/

import threading





class ThreadSafeIter:

    def __init__(self, iter):

        self._itr = iter

        self.lock = threading.Lock()



    def __iter__(self):

        return self



    def __next__(self):

        with self.lock:

            return self._itr.__next__()





def threadsafe(f):

    """A decorator that takes a generator function and makes it thread-safe.

    """



    def g(*a, **kw):

        return ThreadSafeIter(f(*a, **kw))



    return g

from typing import Tuple

import numpy as np

import pandas as pd

from keras import backend as K

from keras.callbacks import ModelCheckpoint

from keras.layers import (Bidirectional, Dense, Input, CuDNNLSTM, Activation, BatchNormalization, LeakyReLU, Dropout)

from keras.models import Model

from sklearn.metrics import matthews_corrcoef

from sklearn.model_selection import StratifiedKFold

from keras import regularizers



# from threadsafe_iterator import threadsafe





# It is the official metric used in this competition

# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)

def matthews_correlation(y_true, y_pred):

    '''Calculates the Matthews correlation coefficient measure for quality

    of binary classification problems.

    '''

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos



    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos



    tp = K.sum(y_pos * y_pred_pos)

    tn = K.sum(y_neg * y_pred_neg)



    fp = K.sum(y_neg * y_pred_pos)

    fn = K.sum(y_pos * y_pred_neg)



    numerator = (tp * tn - fp * fn)

    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))



    return numerator / (denominator + K.epsilon())





class LSTModel:

    def __init__(

            self,

            units: int,

            dense_count: int,

            train_fname='/home/ashesh/Documents/initiatives/kaggle_competitions/vsb_powerline/data/transformed_train.csv',

            meta_train_fname='/home/ashesh/Documents/initiatives/kaggle_competitions/vsb_powerline/data/metadata_train.csv',

            skip_fraction: float = 0,

            data_aug_num_shifts=1,

            data_aug_flip=False,

            dropout_fraction=0.3,

            remove_outliers_from_training=False,

            outlier_removal_kwargs={},

            plot_stats=True):

        """

        Args:

            skip_fraction: initial fraction of timestamps can be ignored.

        """

        self._units = units

        self._dense_c = dense_count

        self._data_fname = train_fname

        self._meta_fname = meta_train_fname

        self._skip_fraction = skip_fraction

        self._data_aug_num_shifts = data_aug_num_shifts

        self._data_aug_flip = data_aug_flip

        self._plot_stats = plot_stats

        self._dropout_fraction = dropout_fraction

        self._remove_outliers_from_training = remove_outliers_from_training

        self._outlier_removal_kwargs = outlier_removal_kwargs



        self._skip_features = [

            'diff_smoothend_by_1 Quant-0.25', 'diff_smoothend_by_1 Quant-0.75', 'diff_smoothend_by_1 abs_mean',

            'diff_smoothend_by_1 mean', 'diff_smoothend_by_16 Quant-0.25', 'diff_smoothend_by_16 Quant-0.75',

            'diff_smoothend_by_16 abs_mean', 'diff_smoothend_by_16 mean', 'diff_smoothend_by_2 Quant-0.25',

            'diff_smoothend_by_2 Quant-0.75', 'diff_smoothend_by_4 Quant-0.25', 'diff_smoothend_by_4 Quant-0.75',

            'diff_smoothend_by_8 Quant-0.25', 'diff_smoothend_by_8 Quant-0.5', 'signal_Quant-0.25', 'signal_Quant-0.75'

        ]



        self._n_splits = 3

        self._feature_c = None

        self._ts_c = None

        self._train_batch_size = None

        # validation score is saved here.

        self._val_score = -1

        # normalization is done using this.

        self._n_split_scales = []

        # a value between 0 and 1. a prediction greater than this value is considered as 1.

        self.threshold = None



    def get_model(self):

        inp = Input(shape=(

            self._ts_c,

            self._feature_c,

        ))

#         x = Dropout(self._dropout_fraction, noise_shape=(self._train_batch_size, 1, self._feature_c))(inp)

        x = Bidirectional(

            CuDNNLSTM(

                self._units,

                return_sequences=False,

                #                 kernel_regularizer=regularizers.l1(0.001),

                # activity_regularizer=regularizers.l1(0.01),

                # bias_regularizer=regularizers.l1(0.01)

            ))(inp)



        #         x = Bidirectional(CuDNNLSTM(self._units, return_sequences=False))(inp)

        #         x = Bidirectional(CuDNNLSTM(self._units // 2, return_sequences=False,

        #                                    kernel_regularizer=regularizers.l1(0.001),))(x)

        x = Dropout(self._dropout_fraction)(x)

        x = Dense(self._dense_c)(x)

        x = BatchNormalization()(x)

        x = LeakyReLU()(x)

        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inp, outputs=x)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])

        return model



    @staticmethod

    def skip_quantile_features(cols, quantiles):

        filt_cols1 = LSTModel._skip_quantiles(cols, quantiles, '_')

        filt_cols2 = LSTModel._skip_quantiles(cols, quantiles, '-')

        cols3 = list(set(filt_cols1) & set(filt_cols2))

        cols3.sort()

        return cols3



    @staticmethod

    def _skip_quantiles(cols, quantiles, delimiter):

        filtered_cols = []

        for col in cols:

            try:

                val = float(col.split(delimiter)[-1])

                if val in quantiles:

                    continue

            except:

                pass

            filtered_cols.append(col)

        return filtered_cols



    def get_processed_data_df(self, fname: str):

        processed_data_df = pd.read_csv(fname, compression='gzip', index_col=[0, 1])

        processed_data_df = processed_data_df.T

        processed_data_df = processed_data_df.swaplevel(axis=1).sort_index(axis=1)

        if 'Unnamed: 0' in processed_data_df.index:

            processed_data_df = processed_data_df.drop('Unnamed: 0', axis=0)



        processed_data_df.index = list(map(int, processed_data_df.index))



        # skip unnecessary columns

        # feature_cols = LSTModel.skip_quantile_features(processed_data_df.columns.levels[1], [0.25, 0.75])

        feature_cols = list(set(processed_data_df.columns.levels[1]) - set(self._skip_features))

        processed_data_df = processed_data_df.iloc[:, processed_data_df.columns.get_level_values(1).isin(feature_cols)]



        # skip first few timestamps. (from paper.)

        ts_units = len(processed_data_df.columns.levels[0])

        skip_end_ts_index = int(ts_units * self._skip_fraction) - 1

        if skip_end_ts_index > 0:

            print('Skipping first ', skip_end_ts_index + 1, 'timestamp units out of total ', ts_units, ' units')

            col_filter = processed_data_df.columns.get_level_values(0) > skip_end_ts_index

            processed_data_df = processed_data_df.iloc[:, col_filter]



        return processed_data_df.sort_index(axis=0)



    def get_y_df(self):

        fname = self._meta_fname

        df = pd.read_csv(fname)

        return df.set_index('signal_id')



    def add_phase_data(self, processed_data_df, meta_fname):

        return processed_data_df



        print('Phase data is about to be added')

        metadata_df = pd.read_csv(meta_fname).set_index('signal_id')

        processed_data_df = processed_data_df.join(metadata_df[['id_measurement']], how='left')

        assert not processed_data_df.isna().any().any()



        # pandas does not have a cyclic shift facility. therefore copying it.

        temp_df = pd.concat([processed_data_df, processed_data_df])

        grp = temp_df.groupby('id_measurement')



        data_1 = grp.shift(0)

        data_1 = data_1[~data_1.index.duplicated(keep='first')]



        data_2 = grp.shift(-1)

        data_2 = data_2[~data_2.index.duplicated(keep='first')]



        data_3 = grp.shift(-2)

        data_3 = data_3[~data_3.index.duplicated(keep='first')]

        del grp

        del temp_df



        assert set(data_1.index.tolist()) == (set(data_2.index.tolist()))

        assert set(data_1.index.tolist()) == (set(data_3.index.tolist()))



        # change indicators name to ensure uniqueness of columns

        feat_names = ['Phase1-' + e for e in data_1.columns.levels[1].tolist()]

        data_1.columns.set_levels(feat_names, level=1, inplace=True)



        feat_names = ['Phase2-' + e for e in data_2.columns.levels[1].tolist()]

        data_2.columns.set_levels(feat_names, level=1, inplace=True)



        feat_names = ['Phase3-' + e for e in data_3.columns.levels[1].tolist()]

        data_3.columns.set_levels(feat_names, level=1, inplace=True)



        processed_data_df = pd.concat([data_1, data_2, data_3], axis=1)

        print(processed_data_df.shape)

        print('Phase data added')

        return processed_data_df



    @staticmethod

    def add_ts_segment_feature(processed_data_df):

        # add segment feature. from paper. (it is said that the 2nd and 4th component of sine wave has information and

        # first and 3rd are non-informative )

        segment_size = len(processed_data_df.columns.levels[0]) // 4

        print('Adding segment data in one-hot encoding form')

        for ts_index in processed_data_df.columns.levels[0]:

            segment = ts_index // segment_size

            segment_0 = 'segment_0'

            segment_1 = 'segment_1'

            segment_2 = 'segment_2'

            segment_3 = 'segment_3'



            processed_data_df[ts_index, segment_0] = int(segment == 0)

            processed_data_df[ts_index, segment_1] = int(segment == 1)

            processed_data_df[ts_index, segment_2] = int(segment == 2)

            processed_data_df[ts_index, segment_3] = int(segment >= 3)



            assert processed_data_df.iloc[0][ts_index][[segment_0, segment_1, segment_2, segment_3]].sum() == 1



        return processed_data_df.sort_index(axis=1)



    def get_X_df(self, fname, meta_fname):

        processed_data_df = self.get_processed_data_df(fname)



        # NOTE: there are 8 columns which are being zero. one needs to fix it.

        assert processed_data_df.isna().any(axis=0).sum() <= 9, processed_data_df.isna().any(axis=0).sum()

        assert processed_data_df.isna().all(axis=0).sum() <= 9, processed_data_df.isna().all(axis=0).sum()



        processed_data_df = processed_data_df.fillna(0)

        processed_data_df = self.add_phase_data(processed_data_df, meta_fname)

        assert not processed_data_df.isna().any().any(), 'Training data has nan'



        processed_data_df.columns = processed_data_df.columns.remove_unused_levels()

        return LSTModel.add_ts_segment_feature(processed_data_df)



    def get_X_in_parts_df(self, fname, meta_fname):

        processed_data_df = self.get_processed_data_df(fname)



        # NOTE: there are 8 columns which are being zero. one needs to fix it.

        assert processed_data_df.isna().any(axis=0).sum() <= 9

        assert processed_data_df.isna().all(axis=0).sum() <= 9



        processed_data_df = processed_data_df.fillna(0)

        meta_df = pd.read_csv(meta_fname)

        chunksize = 2 * 999

        s_index = 0

        e_index = chunksize

        sz = processed_data_df.shape[0]

        while e_index < sz:

            last_accesible_id = meta_df.iloc[e_index - 1]['id_measurement']

            first_inaccesible_id = meta_df.iloc[e_index]['id_measurement']

            while e_index < sz and last_accesible_id == first_inaccesible_id:

                e_index += 1

                last_accesible_id = meta_df.iloc[e_index - 1]['id_measurement']

                first_inaccesible_id = meta_df.iloc[e_index]['id_measurement']



            # making all three phases data available.

            data_df = self.add_phase_data(processed_data_df.iloc[s_index:e_index], meta_fname)

            assert not data_df.isna().any().any(), 'Training data has nan'

            s_index = e_index

            e_index = s_index + chunksize

            print('Completed Test data preprocessing', round(e_index / sz * 100), '%')

            yield LSTModel.add_ts_segment_feature(data_df)



        data_df = self.add_phase_data(processed_data_df.iloc[s_index:], meta_fname)

        assert not data_df.isna().any().any(), 'Training data has nan'

        yield LSTModel.add_ts_segment_feature(data_df)



    def get_X_y(self):

        """

        Returns:

            Tuple(X,y):

                X.shape should be: (#examples,#ts,#features)

                y.shape should be: (#examples,)

        """

        processed_train_df = self.get_X_df(self._data_fname, self._meta_fname)



        y_df = self.get_y_df()

        y_df = y_df.loc[processed_train_df.index]



        if self._remove_outliers_from_training:

            outlier_filter = get_outliers(processed_train_df, **self._outlier_removal_kwargs)['outliers']

            processed_train_df = processed_train_df.loc[~outlier_filter]

            y_df = y_df.loc[processed_train_df.index]

            print('Removed', outlier_filter.sum(), 'many outlier entries from training data ')



        examples_c = processed_train_df.shape[0]

        self._ts_c = len(processed_train_df.columns.levels[0])

        self._feature_c = len(processed_train_df.columns.levels[1])



        print('#examples', examples_c)

        print('#ts', self._ts_c)

        print('#features', self._feature_c)

        print('data shape', processed_train_df.shape)



        X = processed_train_df.values.reshape(examples_c, self._ts_c, self._feature_c)

        y = y_df.target.values



        assert X.shape == (examples_c, self._ts_c, self._feature_c)

        assert y.shape == (examples_c, )

        return X, y



    def predict(self, fname: str, meta_fname: str):

        ser = self._predict(fname, meta_fname)

        ser.index.name = 'signal_id'



        ser = ser.to_frame('prediction')

        meta_df = pd.read_csv(meta_fname).set_index('signal_id')

        df = ser.join(meta_df[['id_measurement']], how='left')

        ser = df.groupby('id_measurement').transform(np.mean)['prediction']

        ser[ser >= 0.5] = 1

        ser[ser < 0.5] = 0

        ser = ser.astype(int)



        return ser



    def _predict(self, fname: str, meta_fname):

        """

        Using the self._n_splits(5) models, it returns a pandas.Series with values belonging to {0,1}

        """

        output = []

        output_index = []

        for df in self.get_X_in_parts_df(fname, meta_fname):

            examples_c = df.shape[0]

            X = df.values.reshape(examples_c, self._ts_c, self._feature_c)



            pred_array = []

            for split_index in range(self._n_splits):

                weight_fname = 'weights_{}.h5'.format(split_index)

                model = self.get_model()

                model.load_weights(weight_fname)



                scale = self._n_split_scales[split_index]

                pred_array.append(model.predict(X / scale, batch_size=128))



            # Take average value over different models.

            pred_array = np.array(pred_array).reshape(len(pred_array), -1)

            pred_array = (pred_array > self.threshold).astype(int)

            pred = np.mean(np.array(pred_array), axis=0)

            # majority prediction

            pred = (pred > 0.5).astype(int)

            assert pred.shape[0] == X.shape[0]



            output.append(pred)

            output_index.append(df.index.tolist())

        return pd.Series(np.squeeze(np.concatenate(output)), index=np.concatenate(output_index))



    def fit_threshold(self, prediction, actual, start=0.08, end=0.98, n_count=20, center_alignment_offset=0.01):

        best_score = -1

        self.threshold = 0

        scores = []

        thresholds = np.linspace(start, end, n_count)

        for threshold in thresholds:

            score = matthews_corrcoef(actual, (prediction > threshold).astype(np.float64))

            scores.append(score)

            center_alignment = 1 if threshold > (1 - self.threshold) else -1

            if score > best_score + center_alignment * center_alignment_offset:

                best_score = score

                self.threshold = threshold



        if self._plot_stats:

            import matplotlib.pyplot as plt



            plt.plot(thresholds, scores)

            plt.title('Fitting threshold')

            plt.ylabel('mathews correlation coef')

            plt.xlabel('threshold')

            plt.show()



        print('Matthews correlation on train set is ', best_score, ' with threshold:', self.threshold)



    @staticmethod

    def get_generator(train_X: np.array, train_y: np.array, batch_size: int, flip: bool, num_shifts: int = 2):



        shifts = list(map(int, np.linspace(0, train_X.shape[1] * 0.2, num_shifts + 1)[1:-1]))

        shifts = [0] + shifts

        flip_ts = [1, -1] if flip else [1]



        @threadsafe

        def augument_by_timestamp_shifts() -> Tuple[np.array, np.array]:

            """

            num_shifts: factor by which the training data is to be increased.

            We shift the timestamps to get more data to train. It assumes timestamp is in 2nd dimension of

            train_X

            """

            num_times = len(flip_ts) * num_shifts

            print('After data augumentation, training data has become ', num_times, ' times its original size.')

            # generator = DataGenerator('training_data_augumented.csv', batch_size, train_X.shape[1], train_X.shape[2],)

            # generator.add(train_X, train_y)

            # 1 time is the original data itself.

            while True:

                for shift_amount in shifts:

                    for flip_direction in flip_ts:

                        flipped_X = train_X[:, ::flip_direction, :]

                        train_X_shifted = np.roll(flipped_X, shift_amount, axis=1)

                        for index in range(0, train_X_shifted.shape[0], batch_size):

                            X = train_X_shifted[index:(index + batch_size), :, :]

                            y = train_y[index:(index + batch_size)]

                            if X.shape[0] < batch_size:

                                continue

                            assert X.shape[0] == batch_size

                            yield (X, y)



        steps_per_epoch = len(flip_ts) * len(shifts) * train_X.shape[0] // batch_size

        return augument_by_timestamp_shifts, steps_per_epoch



    def _plot_acc_loss(self, history):

        import matplotlib.pyplot as plt



        plt.plot(history.history['matthews_correlation'])

        plt.plot(history.history['val_matthews_correlation'])

        plt.title('model accuracy')

        plt.ylabel('accuracy')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

        # summarize history for loss

        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

        plt.title('model loss')

        plt.ylabel('loss')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()



    def train(self, batch_size=128, epoch=50):



        # to be used in get_model()

        self._train_batch_size = batch_size



        X, y = self.get_X_y()

        print('X shape', X.shape)

        print('Y shape', y.shape)



        splits = list(StratifiedKFold(n_splits=self._n_splits, shuffle=True).split(X, y))



        preds_array = []

        y_array = []

        # Then, iteract with each fold

        # If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]

        for idx, (train_idx, val_idx) in enumerate(splits):

            K.clear_session()  # I dont know what it do, but I imagine that it "clear session" :)

            print("Beginning fold {}".format(idx + 1))

            # use the indexes to extract the folds in the train and validation data

            train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]



            # We should get scale using normalization on train data only.

            # axis 0 is #examples, 1 is #timestamps, 2 is features.

            scale = np.abs(np.max(train_X, axis=(0, 1)))

            scale[scale == 0] = 1



            train_X = train_X / scale

            val_X = val_X / scale

            self._n_split_scales.append(scale)



            # data augumentation

            generator, steps_per_epoch = LSTModel.get_generator(

                train_X,

                train_y,

                batch_size,

                self._data_aug_flip,

                num_shifts=self._data_aug_num_shifts,

            )

            # print('Train X shape', train_X.shape)

            # print('Val X shape', val_X.shape)

            # print('Train Y shape', train_y.shape)

            # print('Val y shape', val_y.shape)



            model = self.get_model()

            print(model.summary())

            # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an

            # validation matthews_correlation greater than the last one.

            ckpt = ModelCheckpoint(

                'weights_{}.h5'.format(idx),

                save_best_only=True,

                save_weights_only=True,

                verbose=0,

                monitor='val_matthews_correlation',

                mode='max',

            )



            # Train

            history = model.fit_generator(

                generator(),

                epochs=epoch,

                validation_data=[val_X, val_y],

                callbacks=[ckpt],

                steps_per_epoch=steps_per_epoch,

                # workers=2,

                # use_multiprocessing=True,

                verbose=0,

            )



            if self._plot_stats:

                self._plot_acc_loss(history)



            # loads the best weights saved by the checkpoint

            model.load_weights('weights_{}.h5'.format(idx))

            # Add the predictions of the validation to the list preds_val

            preds_array.append(model.predict(val_X, batch_size=20))

            y_array.append(val_y)



        prediction = np.concatenate(preds_array)

        actual = np.concatenate(y_array)



        self.fit_threshold(model.predict(train_X), train_y)

        self._val_score = matthews_corrcoef(actual, (prediction > self.threshold).astype(np.float64))

        print('On validation data, score is:', self._val_score)

import pandas as pd
pd.read_csv('../input/vsb-power-line-fault-detection/sample_submission.csv').head()
# fname = '../input/dataprocessing/test_data.csv'

# processed_data_df2 = pd.read_csv(fname, compression='gzip', index_col=[0, 1])

# fname = '../input/dataprocessing/train_data_0.95_10_200.csv'

# processed_data_df1 = pd.read_csv(fname, compression='gzip', index_col=[0, 1])

# a = processed_data_df1.abs().groupby('features').quantile(0.9).mean(axis=1)

# b = processed_data_df2.abs().groupby('features').quantile(0.9).mean(axis=1)

# percentage_deviation = ((b-a)/a * 100 ).round(2)

# percentage_deviation[percentage_deviation.abs() > 10].abs().sort_values()
# model = LSTModel(128, 64, train_fname='../input/dataprocessing/train_data_3_0.95_15.csv', 

#                  meta_train_fname='../input/vsb-power-line-fault-detection/metadata_train.csv',

#                 skip_fraction=0)

# # df = model.get_X_df('../input/dataprocessing/train_data.csv',

# #                    '../input/vsb-power-line-fault-detection/metadata_train.csv')

# model.train(epoch=40)

fnames = ['train_data_0.25_10_150_100.csv',

'train_data_0.75_10_150_50.csv',

'train_data_0.75_10_150_100.csv',

'train_data_0.25_10_150_50.csv']

for data_aug_num_times in [3]:

    for dropout_fraction in [0.1]:

        for fname in fnames:

            fname = '../input/dataprocessing/' + fname

            model = LSTModel(128, 64, train_fname=fname, 

                         meta_train_fname='../input/vsb-power-line-fault-detection/metadata_train.csv',

                        skip_fraction=0,

                        data_aug_num_shifts=data_aug_num_times,

                         dropout_fraction=dropout_fraction,

                            data_aug_flip=False,

                            remove_outliers_from_training=True)

            model.train(epoch=25)

            print(fname, ' ', data_aug_num_times, ' ', model._val_score)

            del model
# norm_train_df = model.get_X_df(model._data_fname, model._meta_fname)



# y_df = model.get_y_df()

# y_df = y_df.loc[norm_train_df.index]



# feat1 = norm_train_df.groupby(axis=1, level=1).mean()

# feat2 = norm_train_df.abs().groupby(axis=1, level=1).mean()

# feat3 = norm_train_df.groupby(axis=1, level=1).std()

# feat4 = norm_train_df.groupby(axis=1, level=1).max()



# from sklearn.feature_selection import mutual_info_classif

# dfs= []

# for index, feat in enumerate([feat1, feat2, feat3, feat4]):

#     mut_df = pd.Series(mutual_info_classif(feat, y_df['target']), index=feat.columns)

#     dfs.append(mut_df.to_frame(index))

# mutual_information_df = pd.concat(dfs, axis=1).max(axis=1).sort_values()

# columns_to_skip = mutual_information_df.head(40).index.tolist()

# columns_to_skip.sort()



# features_to_skip = list(set(map(lambda x: '-'.join(x.split('-')[1:]), columns_to_skip)))

# features_to_skip.sort()
# mutual_information_df = pd.concat(dfs, axis=1).max(axis=1)
# cols = mutual_information_df.index.tolist()
# df = model.predict('../input/dataprocessing/test_data.csv', '../input/vsb-power-line-fault-detection/metadata_test.csv')

# df.index.name = 'signal_id'

# df = df.to_frame('target').reset_index()

# df.to_csv('submission.csv', index=False)