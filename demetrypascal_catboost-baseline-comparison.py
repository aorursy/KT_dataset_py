import numpy as np

import pandas as pd



from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt







from pylab import rcParams



rcParams['figure.figsize'] = 18, 8









files = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



files = sorted(files)



files
train = pd.read_csv(files[2])



test = pd.read_csv(files[1])



submission = pd.read_csv(files[0])



targets = pd.read_csv(files[4])



train.head()
import catboost

print(catboost.__version__)
from catboost import CatBoostClassifier, Pool

from catboost.eval.catboost_evaluation import *
from catboost.utils import create_cd

import os



train_df = train



train_df['target'] = targets.iloc[:,1]



train_df = train_df.drop(['sig_id'], axis = 1)



train_file = 'train.csv'

description_file = 'train.cd'



train_df.to_csv(train_file, header=False, index=False)



feature_names = dict()

for column, name in enumerate(train_df):

    if column != 0:

        feature_names[column - 1] = name

    

create_cd(

    label = train_df.columns.shape[0] - 1, 

    cat_features = [0,1,2], #list(range(0, train_df.columns.shape[0] - 1)),

    feature_names=feature_names,

    output_path= 'train.cd'

)

!cat $description_file
fold_size = 3000

fold_offset = 0

folds_count = 25

random_seed = 0
learn_params = {'iterations': 250, 

                'random_seed': 0, 

                'logging_level': 'Silent',

                'loss_function': 'Logloss',

                # You could set learning process to GPU

                # 'devices': '1',  

                # 'task_type': 'GPU',

                'loss_function' : 'Logloss',

                'boosting_type': 'Plain', 

                # For feature evaluation learning time is important and we need just the relative quality

                'max_ctr_complexity' : 4}
features_to_evaluate = list(range(10))   #list(range(train_df.columns.shape[0] - 1)) #[0, 1, 2, 3, 4]
evaluator = CatboostEvaluation(train_file,

                               fold_size,

                               folds_count,

                               delimiter=',',

                               column_description=description_file,

                               partition_random_seed=random_seed)
result = evaluator.eval_features(learn_config = learn_params,

                                 eval_metrics = ["Logloss"],

                                 features_to_eval = features_to_evaluate)
logloss_result = result.get_metric_results("Logloss")



logloss_result.get_baseline_comparison()