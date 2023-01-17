# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import h2o

h2o.init()
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



import math



from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.grid.grid_search import H2OGridSearch





import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')



plt.style.use('seaborn')

sns.set(font_scale=1)
random_state = 82

np.random.seed(random_state)

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
X_train, X_val = train_test_split(df_train, test_size=0.2, random_state=random_state)
X_train_h2o = h2o.H2OFrame(X_train.drop(["ID_code"], axis=1))

X_train_h2o["target"]=X_train_h2o["target"].asfactor()    ## Converting Target Variable as Factor
X_val_h2o = h2o.H2OFrame(X_val.drop(["ID_code"], axis=1))

X_val_h2o["target"]=X_val_h2o["target"].asfactor()    ## Converting Target Variable as Factor
X_test = h2o.H2OFrame(df_test.drop(["ID_code"], axis=1))
hyper_params = {'max_depth' : list(range(11,20,1)),

#                 'sample_rate': [x/100. for x in range(20,101)],

#                 'col_sample_rate' : [x/100. for x in range(20,101)],

#                 'col_sample_rate_per_tree': [x/100. for x in range(20,101)],

#                 'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],

                'min_rows': [2**x for x in range(0,int(math.log(X_train_h2o.nrow,2)-1)+1)],

                'nbins': [2**x for x in range(4,11)],

                'nbins_cats': [2**x for x in range(4,13)],

                'min_split_improvement': [0,1e-8,1e-6,1e-4],

                'histogram_type': ["UniformAdaptive","QuantilesGlobal","RoundRobin"]}

gbm = H2OGradientBoostingEstimator(

    ntrees = 1000, 

    

    distribution = "bernoulli",



    learn_rate = 0.02012,

    

    learn_rate_annealing=0.99,



    stopping_rounds = 7, 

    

    stopping_tolerance = 1e-4, 

    

    stopping_metric = "AUC",



#     sample_rate = 0.8,                                        



#     col_sample_rate = 0.8,

    

#     col_sample_rate_change_per_level = 0.999,



    seed = random_state,

    

#     calibrate_model = True,

    

#     calibration_frame = X_train_h2o,

    

    score_tree_interval = 10,

    

    nfolds = 3)
grid = H2OGridSearch(gbm,hyper_params,

                         grid_id = 'depth_grid',

                         search_criteria = {'strategy': "Cartesian"})
grid.train(x=X_train_h2o.names[1:],y=X_train_h2o.names[0],

           training_frame = X_train_h2o)
sorted_grid = grid.get_grid(sort_by='auc',decreasing=True)

print(sorted_grid)
model = h2o.get_model(sorted_grid.sorted_metric_table()['model_ids'][0])

performance_model = model.model_performance(X_val_h2o)

print(performance_model.auc())
model.cross_validation_metrics_summary()
model.varimp_plot()
pred = model.predict(test_data=X_test)
pred = pred.as_data_frame() 
sub = pd.read_csv("../input/sample_submission.csv")
sub["target"] = pred["predict"]
sub.head()
sub.to_csv("h2o_submission.csv", index=False)