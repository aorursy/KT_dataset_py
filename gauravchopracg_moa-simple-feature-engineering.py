# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Load data

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
# copy the original dataframe for feature generation

test_features1 = test_features.copy()

train_features1 = train_features.copy()

train_targets_nonscored1 = train_targets_nonscored.copy()

train_targets_scored1 = train_targets_scored.copy()

sample_submission1 = sample_submission.copy()
# take a look at train_features.csv before digging deeper

train_features.head(5)
# extract columns containing numerical values

float_cols = train_features.select_dtypes(include=[np.float]).columns

# for each column create a new column emthusiasing floating point value

for col in float_cols:

    train_features[col] = train_features[col].apply(np.exp)

    

train_features.head()
# how much days does it take for samples to show (MoA) response(s)

train_features['days'] = (train_features['cp_time']/24).astype(np.int)

test_features['days'] = (test_features['cp_time']/24).astype(np.int)

train_features.head()
train_features['minutes'] = train_features['cp_time']*60

test_features['minutes'] = test_features['cp_time']*60

train_features['seconds'] = train_features['cp_time']*3600

test_features['seconds'] = test_features['cp_time']*3600

train_features.head()
# extract columns containing numerical values

float_cols = train_features.select_dtypes(include=[np.float]).columns

# for each column create a new column emthusiasing floating point value

for col in float_cols:

    train_features['frac_'+col] = train_features[col].apply(lambda x: x%1)

    test_features['frac_'+col] = test_features[col].apply(lambda x: x%1)

    

train_features.head()
# feature interaction between days and cp_dose

train_features['cp_time_dose'] = train_features['cp_time'].astype(str)+train_features['cp_dose']

test_features['cp_time_dose'] = test_features['cp_time'].astype(str)+test_features['cp_dose']



train_features.head()
GENES = [col for col in train_features1.columns if col.startswith('g-')]

CELLS = [col for col in train_features1.columns if col.startswith('c-')]
# study gene expression data

g_train_features = train_features1[GENES]

g_test_features = test_features1[GENES]

g_train_features.head()
# study cell viability data

c_train_features = train_features1[CELLS]

c_test_features = test_features1[CELLS]

c_train_features.head()
plt.plot(g_train_features.iloc[0].sort_values(), '.');
import numpy as np



g_exp_train = g_train_features.copy()

g_exp_test = g_test_features.copy()

cols = g_exp_train.columns



for col in cols:

    g_exp_train[col] = np.exp(g_exp_train[col])

    g_exp_test[col] = np.exp(g_exp_test[col])
plt.plot(g_exp_train.iloc[0].sort_values(), '.');
# copy of dataframes to apply label encoding

train_features_fe = train_features1.copy()

test_features_fe = test_features1.copy()



temp = train_features_fe['cp_dose'].value_counts().to_dict()

train_features_fe['cp_dose_counts'] = train_features_fe['cp_dose'].map(temp)

train_features_fe.head()
temp = train_features_fe.groupby('cp_dose')['g-0'].agg(['mean']).rename({'mean':'g-0_cp_dose_mean'},axis=1)

train_features_fe = pd.merge(train_features_fe,temp,on='cp_dose',how='left')

train_features_fe.head()
# Source: https://www.kaggle.com/simakov/keras-multilabel-neural-network-v1-2

# add seed and change create_model



'''

from typing import Tuple, List, Callable, Any



from sklearn.utils import check_random_state  # type: ignore



### from eli5

def iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=False,

                  random_state=None):

    rng = check_random_state(random_state)



    if columns_to_shuffle is None:

        columns_to_shuffle = range(X.shape[1])



    if pre_shuffle:

        X_shuffled = X.copy()

        rng.shuffle(X_shuffled)



    X_res = X.copy()

    for columns in tqdm(columns_to_shuffle):

        if pre_shuffle:

            X_res[:, columns] = X_shuffled[:, columns]

        else:

            rng.shuffle(X_res[:, columns])

        yield X_res

        X_res[:, columns] = X[:, columns]







def get_score_importances(

        score_func,  # type: Callable[[Any, Any], float]

        X,

        y,

        n_iter=5,  # type: int

        columns_to_shuffle=None,

        random_state=None

    ):

    rng = check_random_state(random_state)

    base_score = score_func(X, y)

    scores_decreases = []

    for i in range(n_iter):

        scores_shuffled = _get_scores_shufled(

            score_func, X, y, columns_to_shuffle=columns_to_shuffle,

            random_state=rng, base_score=base_score

        )

        scores_decreases.append(scores_shuffled)



    return base_score, scores_decreases







def _get_scores_shufled(score_func, X, y, base_score, columns_to_shuffle=None,

                        random_state=None):

    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state)

    res = []

    for X_shuffled in Xs:

        res.append(-score_func(X_shuffled, y) + base_score)

    return res



def metric(y_true, y_pred):

    metrics = []

    for i in range(y_pred.shape[1]):

        if y_true[:, i].sum() > 1:

            metrics.append(log_loss(y_true[:, i], y_pred[:, i].astype(float)))

    return np.mean(metrics)   



perm_imp = np.zeros(train.shape[1])

all_res = []

for n, (tr, te) in enumerate(KFold(n_splits=7, random_state=0, shuffle=True).split(train_targets)):

    print(f'Fold {n}')



    model = create_model(len(train.columns))

    checkpoint_path = f'repeat:{seed}_Fold:{n}.hdf5'

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

    cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,

                                     save_weights_only = True, mode = 'min')

    model.fit(train.values[tr],

                  train_targets.values[tr],

                  validation_data=(train.values[te], train_targets.values[te]),

                  epochs=35, batch_size=128,

                  callbacks=[reduce_lr_loss, cb_checkpt], verbose=2

                 )

        

    model.load_weights(checkpoint_path)

        

    def _score(X, y):

        pred = model.predict(X)

        return metric(y, pred)



    base_score, local_imp = get_score_importances(_score, train.values[te], train_targets.values[te], n_iter=1, random_state=0)

    all_res.append(local_imp)

    perm_imp += np.mean(local_imp, axis=0)

    print('')

    

top_feats = np.argwhere(perm_imp < 0).flatten()

top_feats

'''



# print(top_feats)