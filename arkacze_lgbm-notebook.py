



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

import optuna

from sklearn.model_selection import train_test_split



import sklearn.metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def returnlabels(label):

    global labels

    l = labels

    return l.index(label)
import pandas as pd

testDf = pd.read_csv( "../input/automodele/iris_test_data.csv" )

trainDF = pd.read_csv( "../input/automodele/iris_train_data.csv" )


labels = list(trainDF['label'].unique())







trainDF['goal'] = trainDF.apply(lambda x: returnlabels(x['label']),axis=1)







def objective(trial):

    data = trainDF[['a1', 'a2', 'a3', 'a4']]

    target = trainDF['goal']

    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25, random_state=1)

    dtrain = lgb.Dataset(train_x, label=train_y)

    dval = lgb.Dataset(val_x, label=val_y)



    param = {

        "objective": "multiclass",

        "num_class":3,

        "metric" : "multi_error",

        "verbosity": -1,

        "boosting_type": "gbdt",

        "max_depth": trial.suggest_int('max_depth', 2, 10),

        "num_leaves": trial.suggest_int("num_leaves", 2, 5),

    }



    

    gbm = lgb.train(param, dtrain)

    preds = gbm.predict(val_x)

    accuracy = sklearn.metrics.accuracy_score(val_y, [np.argmax(x) for x in preds])

    return accuracy





study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50,callbacks=None)







print("Number of finished trials: {}".format(len(study.trials)))



print("Best trial:")

trial = study.best_trial



print("  Value: {}".format(trial.value))



print("  Params: ")

for key, value in trial.params.items():

    print("    {}: {}".format(key, value))

    





param = {

        "objective": "multiclass",

        "num_class":3,

        "metric" : "multi_error",

        "verbosity": -1,

        "boosting_type": "gbdt",

    }



for key, value in trial.params.items():

    param[key] = value
    data = trainDF[['a1', 'a2', 'a3', 'a4']]

    target = trainDF['goal']

    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.14, random_state=1)

    dtrain = lgb.Dataset(train_x, label=train_y)

    dval = lgb.Dataset(val_x, label=val_y)

    gbm = lgb.train(param, dtrain)
submission = pd.read_csv("/kaggle/input/automodele/sample_submission.csv")
def predct(vals):

    global labels

    test = np.array(vals)

    B = np.reshape(test, (1, -1))

    preds = gbm.predict(B)

    maxind = np.argmax(preds)

    return labels[maxind]

    
testDf['label'] = testDf.apply(lambda row: predct(list([row['a1'],row['a2'],row['a3'],row['a4']])),axis=1)
submissionDf = testDf.filter(['id','label'], axis=1)
testDf.head()
submissionDf.to_csv("submission.csv")
!ls