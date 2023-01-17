# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pyarrow.parquet as pq

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
meta_train_df = pd.read_csv('../input/metadata_train.csv')

meta_test_df = pd.read_csv('../input/metadata_test.csv')
signal_ids_for_one = meta_train_df[meta_train_df.target ==1].sample(300).signal_id.tolist()

if '1577' not in signal_ids_for_one:

    signal_ids_for_one.append('1577')

signal_ids_for_zero = meta_train_df[meta_train_df.target ==0].sample(300).signal_id.tolist()
cols_one = list(map(str, signal_ids_for_one))

df_one = pq.read_pandas('../input/train.parquet', columns=cols_one).to_pandas()
cols_zero = list(map(str, signal_ids_for_zero))

df_zero = pq.read_pandas('../input/train.parquet', columns=cols_zero).to_pandas()
df_one['1577'].plot()
peak_threshold = 40

corona_max_distance=10

corona_max_height_ratio=0.1

corona_cleanup_distance=20

clean_df = DataProcessor.remove_corona_discharge(df_one['1577'], 

                                      peak_prominence=peak_threshold,

                                     corona_max_distance=corona_max_distance,

                                     corona_max_height_ratio=corona_max_height_ratio,

                                     corona_cleanup_distance=corona_cleanup_distance)

clean_df.plot()
(df_one['1577'] - clean_df).plot()
df_one['1577'].iloc[:216400].plot()
corona_max_distance
clean_df = DataProcessor.remove_corona_discharge(df_one['1577'].iloc[:216400], 

                                      peak_prominence=peak_threshold,

                                     corona_max_distance=corona_max_distance,

                                     corona_max_height_ratio=corona_max_height_ratio,

                                     corona_cleanup_distance=corona_cleanup_distance)

clean_df.plot()
#(118142, 118143), (124161, 124163), (127994, 127996)
df_one['1577'].iloc[124150:124250].plot()
peaks, peak_data =find_peaks(df_one['1577'].iloc[124150:124250],prominence=20)

peak_data
peaks, peak_data =find_peaks(-1*df_one['1577'].iloc[124150:124250],prominence=20)

peak_data
peaks
clean_df.iloc[124150:124250].plot()
removed_data = (df_one['1577'].iloc[:216400] - clean_df)
(df_one['1577'].iloc[:216300] - clean_df).iloc[124150:124250].plot()
clean_df.iloc[216200:216250].plot()
from scipy.signal import find_peaks, peak_prominences

peaks, dict_ = find_peaks(clean_df.iloc[216200:216250], prominence=peak_threshold)

print(peak_prominences(clean_df.iloc[216200:216250],peaks)[0])



peaks, _ = find_peaks(-1*clean_df.iloc[216200:216250], prominence=peak_threshold)

print(peak_prominences(-1* clean_df.iloc[216200:216250],peaks)[0])

df_one.astype(float).diff().abs().quantile([0.1, 0.5, 0.9, 0.95,0.98, 0.99, 1]).max(axis=1)
df_zero.astype(float).diff().abs().quantile([0.1, 0.5, 0.9, 0.95, 0.98, 0.99, 1]).max(axis=1)
ax = df['6'].plot()

df['6'].rolling(4).mean().plot(ax=ax)



df['7'].rolling(4).mean().plot(ax=ax)

df['7'].plot(ax=ax)



df['8'].rolling(4).mean().plot(ax=ax)

df['8'].plot(ax=ax)
ax = df['0'].plot()

df['0'].rolling(3).mean().plot(ax=ax)
from typing import Dict, List, Tuple

from multiprocessing import Pool

from scipy.signal import find_peaks

import numpy as np

import pandas as pd

import pyarrow.parquet as pq



import numpy as np





class DataPipeline:

    """

    Top level class which takes in the parquet file and converts it to a low dimensional dataframe.

    """



    def __init__(

            self,

            parquet_fname: str,

            data_processor_args,

            start_row_num: int,

            num_rows: int,

            concurrency: int = 100,

    ):

        self._fname = parquet_fname

        self._concurrency = concurrency

        self._nrows = num_rows

        self._start_row_num = start_row_num

        self._processor_args = data_processor_args

        self._process_count = 4



    @staticmethod

    def run_one_chunk(arg_tuple):

        fname, processor_args, start_row, end_row, num_rows = arg_tuple

        cols = [str(i) for i in range(start_row, end_row)]

        data = pq.read_pandas(fname, columns=cols)

        data_df = data.to_pandas()

        processor = DataProcessor(**processor_args)

        output_df = processor.transform(data_df)

        print('Another ', round((end_row - start_row) / num_rows * 100, 2), '% Complete')

        del processor

        del data_df

        del data

        return output_df



    def run(self):

        outputs = []



        args_list = []

        for s_index in range(self._start_row_num, self._start_row_num + self._nrows, self._concurrency):

            e_index = s_index + self._concurrency

            args_this_chunk = [self._fname, self._processor_args, s_index, e_index, self._nrows]

            args_list.append(args_this_chunk)



        pool = Pool(self._process_count)

        outputs = pool.map(DataPipeline.run_one_chunk, args_list)

        pool.close()

        pool.join()



        final_output_df = pd.concat(outputs, axis=1)

        return final_output_df





class DataProcessor:

    def __init__(

            self,

            intended_time_steps: int,

            original_time_steps: int,

            peak_prominence: int,

            smoothing_window: int = 3,

            remove_corona=False,

            corona_max_distance=5,

            corona_max_height_ratio=0.3,

            corona_cleanup_distance=100,

            num_processes: int = 7,

    ):

        self._o_steps = original_time_steps

        self._steps = intended_time_steps

        # 50 is a confident smoothed version

        # resulting signal after subtracting the smoothened version has some noise along with signal.

        # with 10, smoothened version seems to have some signals as well.

        self._smoothing_window = smoothing_window

        self._num_processes = num_processes

        self._peak_prominence = peak_prominence

        self._remove_corona = remove_corona

        self._corona_max_distance = corona_max_distance

        self._corona_max_height_ratio = corona_max_height_ratio

        self._corona_cleanup_distance = corona_cleanup_distance



    def get_noise(self, X_df: pd.DataFrame):

        """

        TODO: we need to keep the noise. However, we don't want very high freq jitter.

        band pass filter is what is needed.

        """

        msg = 'Expected len:{}, found:{}'.format(self._o_steps, len(X_df))

        assert self._o_steps == X_df.shape[0], msg

        smoothe_df = X_df.rolling(self._smoothing_window, min_periods=1).mean()

        noise_df = X_df - smoothe_df

        del smoothe_df

        return noise_df



    @staticmethod

    def peak_data(

            ser: pd.Series,

            prominence: float,

            quantiles=[0, 0.1, 0.5, 0.95, 0.99, 1],

    ) -> Dict[str, np.array]:

        maxima_peak_indices, maxima_data_dict = find_peaks(ser, prominence=prominence, width=0)

        maxima_width = maxima_data_dict['widths']

        maxima_height = maxima_data_dict['prominences']



        minima_peak_indices, minima_data_dict = find_peaks(-1 * ser, prominence=prominence, width=0)

        minima_width = minima_data_dict['widths']

        minima_height = minima_data_dict['prominences']



        peak_indices = np.concatenate([maxima_peak_indices, minima_peak_indices])

        peak_width = np.concatenate([maxima_width, minima_width])

        peak_height = np.concatenate([maxima_height, minima_height])

        maxima_minima = np.concatenate([np.array([1] * len(maxima_height)), np.array([-1] * len(minima_height))])



        index_ordering = np.argsort(peak_indices)

        peak_width = peak_width[index_ordering]

        peak_height = peak_height[index_ordering]

        peak_indices = peak_indices[index_ordering]

        maxima_minima = maxima_minima[index_ordering]

        return {

            'width': peak_width,

            'height': peak_height,

            'maxima_minima': maxima_minima,

            'indices': peak_indices,

        }



    @staticmethod

    def corona_discharge_index_pairs(

            ser: pd.Series,

            peak_prominence: float,

            corona_max_distance: int,

            corona_max_height_ratio: float,

    ) -> List[Tuple[int, int]]:

        """

        Args:

            ser: time series data.

            peak_prominence: for detecting peaks, if elevation is more than this value, then consider it a peak.

            corona_max_distance: maximum distance between consequitive alternative peaks for it to be a corona discharge.

            corona_max_height_ratio: the alternate peaks should have similar peak heights.

        Returns:

            List of (peak1, peak2) indices. Note that these peaks are consequitive and have opposite sign

        """

        data = DataProcessor.peak_data(ser, peak_prominence)

        corona_indices = []

        for index, data_index in enumerate(data['indices']):

            if index < len(data['indices']) - 1:

                opposite_peaks = data['maxima_minima'][index] * data['maxima_minima'][index + 1] == -1

                nearby_peaks = data['indices'][index + 1] - data['indices'][index] < corona_max_distance



                # for height ratio, divide smaller by larger height

                h1 = data['height'][index]

                h2 = data['height'][index + 1]

                height_ratio = (h1 / h2 if h1 < h2 else h2 / h1)

                similar_height = height_ratio > corona_max_height_ratio

                if opposite_peaks and nearby_peaks and similar_height:

                    corona_indices.append((data_index, data['indices'][index + 1]))

        return corona_indices



    @staticmethod

    def remove_corona_discharge(

            ser: pd.Series,

            peak_prominence: float,

            corona_max_distance: int,

            corona_max_height_ratio: float,

            corona_cleanup_distance: int,

    ) -> pd.Series:

        """

        Args:

            ser: time series data.

            peak_prominence: for detecting peaks, if elevation is more than this value, then consider it a peak.

            corona_max_distance: maximum distance between consequitive alternative peaks for it to be a corona discharge.

            corona_max_height_ratio: the alternate peaks should have similar peak heights.

            corona_cleanup_distance: how many indices after the corona discharge should the data be removed.

        Returns:

            ser: cleaned up time series data.

        """

        pairs = DataProcessor.corona_discharge_index_pairs(

            ser,

            peak_prominence,

            corona_max_distance,

            corona_max_height_ratio,

        )

        print('[Corona discharge peaks removal]', len(pairs), 'many peaks removed')

        ser = ser.copy()

        for start_index, end_index in pairs:

            smoothing_start_index = max(0, start_index - 1)

            smoothing_end_index = min(end_index + corona_cleanup_distance, ser.index[-1])

            start_val = ser.iloc[smoothing_start_index]

            end_val = ser.iloc[smoothing_end_index]

            count = smoothing_end_index - smoothing_start_index



            ser.iloc[smoothing_start_index:smoothing_end_index] = np.linspace(start_val, end_val, count)



        return ser



    @staticmethod

    def peak_stats(ser: pd.Series, prominence, quantiles=[0, 0.1, 0.5, 0.95, 0.99, 1]):

        """

        Returns quantiles of peak width, height, distances from next peak.

        """

        data = DataProcessor.peak_data(ser, prominence, quantiles=quantiles)

        peak_indices = data['indices']

        peak_width = data['width']

        peak_height = data['height']



        if len(peak_indices) == 0:

            # no peaks

            width_stats = [0] * len(quantiles)

            height_stats = [0] * len(quantiles)

            distance_stats = [0] * len(quantiles)

        else:

            peak_distances = np.diff(peak_indices)



            peak_width[peak_width > 100] = 100



            width_stats = np.quantile(peak_width, quantiles)

            height_stats = np.quantile(peak_height, quantiles)



            # for just one peak, distance will be empty array.

            if len(peak_distances) == 0:

                assert len(peak_indices) == 1

                distance_stats = [ser.shape[0]] * len(quantiles)

            else:

                distance_stats = np.quantile(peak_distances, quantiles)



        width_names = ['peak_width_' + str(i) for i in quantiles]

        height_names = ['peak_height_' + str(i) for i in quantiles]

        distance_names = ['peak_distances_' + str(i) for i in quantiles]



        index = width_names + height_names + distance_names + ['peak_count']

        data = np.concatenate([width_stats, height_stats, distance_stats, [len(peak_indices)]])



        return pd.Series(data, index=index)



    @staticmethod

    def get_peak_stats_df(df, peak_prominence):

        """

        Args:

            df:

                columns are different examples.

                axis is time series.

        """

        return df.apply(lambda x: DataProcessor.peak_stats(x, peak_prominence), axis=0)



    @staticmethod

    def pandas_describe(df, quantiles=[0, 0.1, 0.5, 0.95, 0.99, 1]):

        output_df = df.quantile(quantiles, axis=0)

        output_df.index = list(map(lambda x: 'Quant-' + str(x), output_df.index.tolist()))

        abs_mean_df = df.abs().mean().to_frame('abs_mean')

        mean_df = df.mean().to_frame('mean')

        std_df = df.std().to_frame('std')

        return pd.concat([output_df, abs_mean_df.T, mean_df.T, std_df.T])



    @staticmethod

    def transform_chunk(signal_time_series_df: pd.DataFrame, peak_prominence: float) -> pd.DataFrame:

        """

        It sqashes the time series to a single point multi featured vector.

        """

        df = signal_time_series_df

        # mean, var, percentile.

        # NOTE pandas.describe() is the costliest computation with 95% time of the function.

        metrics_df = DataProcessor.pandas_describe(df)

        peak_metrics_df = DataProcessor.get_peak_stats_df(df, peak_prominence)



        metrics_df.index = list(map(lambda x: 'signal_' + x, metrics_df.index))

        temp_metrics = [metrics_df, peak_metrics_df]



        for smoothener in [1, 2, 4, 8, 16, 32]:

            diff_df = df.rolling(smoothener).mean()[::smoothener].diff().abs()

            temp_df = DataProcessor.pandas_describe(diff_df)



            temp_df.index = list(map(lambda x: 'diff_smoothend_by_' + str(smoothener) + ' ' + x, temp_df.index))

            temp_metrics.append(temp_df)



        df = pd.concat(temp_metrics)

        df.index.name = 'features'

        return df



    def transform(self, X_df: pd.DataFrame):

        """

        Args:

            X_df: dataframe with each column being one data point. Rows are timestamps.

        """



        def cleanup_corona(x: pd.Series):

            return DataProcessor.remove_corona_discharge(

                x,

                self._peak_prominence,

                self._corona_max_distance,

                self._corona_max_height_ratio,

                self._corona_cleanup_distance,

            )



        # Corona removed.

        if self._remove_corona:

            X_df = X_df.apply(cleanup_corona)



        # Remove the smoothened version of the data so as to work with noise.

        if self._smoothing_window > 0:

            X_df = self.get_noise(X_df)



        # stepsize many consequitive timestamps are compressed to form one timestamp.

        # this will ensure we are left with self._steps many timestamps.

        stepsize = self._o_steps // self._steps

        transformed_data = []

        for s_tm_index in range(0, self._o_steps, stepsize):

            e_tm_index = s_tm_index + stepsize

            # NOTE: dask was leading to memory leak.

            #one_data_point = delayed(DataProcessor.transform_chunk)(X_df.iloc[s_tm_index:e_tm_index, :])

            one_data_point = DataProcessor.transform_chunk(X_df.iloc[s_tm_index:e_tm_index, :], self._peak_prominence)

            transformed_data.append(one_data_point)



        # transformed_data = dd.compute(*transformed_data, scheduler='processes', num_workers=self._num_processes)

        # Add timestamp

        for ts in range(0, len(transformed_data)):

            transformed_data[ts]['ts'] = ts



        df = pd.concat(transformed_data, axis=0).set_index(['ts'], append=True)

        df.columns.name = 'Examples'

        return df



corona_cleanup_distance = 10

intended_time_steps = 150

peak_prominence_arr = [50, 100]

for peak_prominence in peak_prominence_arr:

    for max_height_ratio in [0.25, 0.75]:

    

        dp_dict = {'intended_time_steps':intended_time_steps,'original_time_steps':800000,

                   'smoothing_window':3,

                  'peak_prominence':peak_prominence,'remove_corona':True,

                   'corona_max_height_ratio':max_height_ratio,

                  'corona_cleanup_distance':corona_cleanup_distance}

        start_row_num = meta_train_df.signal_id[0]

        num_rows = meta_train_df.shape[0]

        pipeline = DataPipeline('../input/train.parquet', dp_dict,start_row_num, num_rows)

        df = pipeline.run()

        fname = 'train_data_{}_{}_{}_{}.csv'.format(max_height_ratio, corona_cleanup_distance, intended_time_steps,

                                                peak_prominence)

        df.to_csv(fname, compression='gzip')

        del df

        del pipeline
# dp_dict = {'intended_time_steps':150,'original_time_steps':800000,

#            'smoothing_window':3,

#           'peak_threshold':15,'remove_corona':True,

#            'corona_max_height_ratio':0.95,

#           'corona_cleanup_distance':50}

# start_row_num = meta_train_df.signal_id[0]

# num_rows = meta_train_df.shape[0]

# pipeline = DataPipeline('../input/train.parquet', dp_dict,start_row_num, num_rows)

# df = pipeline.run()

# fname = 'train_data.csv'

# df.to_csv(fname, compression='gzip')

ax = df['3'].plot(figsize=(20,10))

df['4'].plot(ax=ax)

df['5'].plot(ax=ax)
ax = df['0'].plot(figsize=(20,10))

df['1'].plot(ax=ax)

df['2'].plot(ax=ax)


# start_row_num = meta_test_df.signal_id[0]

# num_rows = meta_test_df.shape[0]

# pipeline = DataPipeline('../input/test.parquet', dp_dict,start_row_num, num_rows)

# df = pipeline.run()

# df.to_csv('test_data.csv', compression='gzip')