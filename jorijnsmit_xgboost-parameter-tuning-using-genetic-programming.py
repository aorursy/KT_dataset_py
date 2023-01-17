# make sure all packages are up to date

!pip install pip --upgrade

!pip install pandas --upgrade

!pip install xgboost --upgrade

!pip install git+https://github.com/EpistasisLab/tpot@development
import os

from datetime import datetime



import numpy as np

import pandas as pd

from scipy import stats

from sklearn.metrics import make_scorer

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

from tpot import TPOTRegressor

from tpot.export_utils import expr_to_tree, generate_export_pipeline_code
def convert_to_float16(df):

    dtypes = {}

    for col_name in df.columns.tolist():

        if col_name.startswith(('feature', 'target')):

            dtypes[col_name] = np.float16

    return df.set_index('id').astype(dtypes)





def rank_correlation(y_true, y_pred):

    return stats.spearmanr(y_true, y_pred, axis=1)[0]





class TimeSeriesSplitGroups(_BaseKFold):

    def __init__(self, n_splits=5):

        super().__init__(n_splits, shuffle=False, random_state=None)





    def split(self, X, y=None, groups=None):

        X, y, groups = indexable(X, y, groups)

        n_samples = _num_samples(X)

        n_splits = self.n_splits

        n_folds = n_splits + 1

        group_list = np.unique(groups)

        n_groups = len(group_list)

        if n_folds > n_groups:

            raise ValueError(

                ("Cannot have number of folds ={0} greater"

                 " than the number of samples: {1}.").format(n_folds, n_groups))

        indices = np.arange(n_samples)

        test_size = (n_groups // n_folds)

        test_starts = range(test_size + n_groups % n_folds, n_groups, test_size)

        test_starts = list(test_starts)[::-1]

        for test_start in test_starts:

            yield (indices[groups.isin(group_list[:test_start])],

                   indices[groups.isin(group_list[test_start:test_start + test_size])])
# define ranges for all of xgboost's parameters

config_dict = {

    'xgboost.XGBRegressor': {

        'max_depth': range(1, 11),

        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],

        'objective': [

            'reg:squarederror',

            'reg:squaredlogerror',

            'reg:logistic',

            'binary:logistic',

            'binary:logitraw',

            'binary:hinge',

            'count:poisson',

            'survival:cox',

            'survival:aft',

            'rank:pairwise',

            'rank:ndcg',

            'rank:map',

            'reg:gamma',

            'reg:tweedie',

        ],

        'gamma': [0, 0.2, 0.5, 2, 5, 10],

        'min_child_weight': range(1, 21),

        'max_delta_step': range(0, 11),

        'colsample_bytree': np.linspace(0.05, 1, 20),

        'colsample_bylevel': np.linspace(0.05, 1, 20),

        'colsample_bynode': np.linspace(0.05, 1, 20),

    },

}
# configure xgboost to use gpu

config_dict['xgboost.XGBRegressor']['tree_method'] = ['gpu_hist']
# load dataset

training_data = convert_to_float16(pd.read_csv(f'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'))

eras = pd.Series([int(era[3:]) for era in training_data['era']])

feature_names = [f for f in training_data.columns if f.startswith('feature')]



x_train = training_data[feature_names].to_numpy()

y_train = training_data[f'target_kazutsugi']
# initialise TPOT model

model = TPOTRegressor(

    generations=10,

    population_size=10,

    scoring=make_scorer(rank_correlation),

    cv=TimeSeriesSplitGroups(5),

    n_jobs=-1,

    max_eval_time_mins=60,

    config_dict=config_dict,

    template='Regressor',

    memory='auto',

    verbosity=0

)
model.fit(x_train, y_train, groups=eras)
print('Rank correlation (train):', model.score(x_train, y_train))



pipeline_tree = expr_to_tree(model._optimized_pipeline, model._pset)

print(generate_export_pipeline_code(pipeline_tree, model.operators))