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
import pandas as pd
testDf = pd.read_csv( "../input/irisdataset/iris_test_data.csv" )

trainDF = pd.read_csv( "../input/irisdataset/iris_train_data.csv" )
labels = list(trainDF['label'].unique())
def returnlabels(label):

    global labels

    l = labels

    return l.index(label)
trainDF['goal'] = trainDF.apply(lambda x: returnlabels(x['label']),axis=1)
def objective(trial):

    data = trainDF[['a1', 'a2', 'a3', 'a4']]

    target = trainDF['goal']

    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.5, random_state=1)

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





if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=100)



    print("Number of finished trials: {}".format(len(study.trials)))



    print("Best trial:")

    trial = study.best_trial



    print("  Value: {}".format(trial.value))



    print("  Params: ")

    for key, value in trial.params.items():

        print("    {}: {}".format(key, value))
returnlabels('Iris-setosa')
testDf.head()

trainDF.head()

trainData = []

objective1 = []
for index, row in trainDF.iterrows():

    temp = list([row['a1'],row['a2'],row['a3'],row['a4']])

    objective1.append(row.goal)

    trainData.append(temp)

    

    

trainData
submission = pd.read_csv("/kaggle/input/irisdataset/sample_submission.csv")
submission.count
submission['label'] = submission['label'].replace({'Iris-versicolor' : 0, 'Iris-setosa' : 1, 'Iris-virginica' : 2})
df1 = submission['label']

testDf = testDf.join(df1)

testDf.head()