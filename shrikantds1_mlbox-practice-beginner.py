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
!pip install mlbox
from mlbox.preprocessing import *

from mlbox.optimisation import *

from mlbox.prediction import *
paths = ["../input/train.csv","../input/test.csv"]

target_name = "SalePrice"
rd = Reader(sep = ",")

df = rd.train_test_split(paths, target_name)   #reading and preprocessing (dates, ...)
dft = Drift_thresholder()

df = dft.fit_transform(df)   #removing non-stable features (like ID,...)
rmse = make_scorer(lambda y_true, y_pred: np.sqrt(np.sum((y_true - y_pred)**2)/len(y_true)), greater_is_better=False, needs_proba=False)

opt = Optimiser(scoring = rmse, n_folds = 3)
space = {

    

        'est__strategy':{"search":"choice",

                                  "space":["LightGBM"]},    

        'est__n_estimators':{"search":"choice",

                                  "space":[150]},    

        'est__colsample_bytree':{"search":"uniform",

                                  "space":[0.8,0.95]},

        'est__subsample':{"search":"uniform",

                                  "space":[0.8,0.95]},

        'est__max_depth':{"search":"choice",

                                  "space":[5,6,7,8,9]},

        'est__learning_rate':{"search":"choice",

                                  "space":[0.07]} 

    

        }



params = opt.optimise(space, df,15)
prd = Predictor()

prd.fit_predict(params, df)
submit = pd.read_csv("../input/sample_submission.csv",sep=',')

preds = pd.read_csv("save/"+target_name+"_predictions.csv")



submit[target_name] =  preds[target_name+"_predicted"].values



submit.to_csv("mlbox.csv", index=False)
submit = pd.read_csv("../input/sample_submission.csv",sep=',')