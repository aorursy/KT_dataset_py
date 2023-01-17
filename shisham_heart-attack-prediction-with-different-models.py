# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

import os

import pandas as pd

import numpy as np
df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

df.head()
import seaborn as sn



sn.kdeplot(df.age, shade = True)



sn.kdeplot(df[df['target'] == 1].age, shade = True, label = 'diseased')

sn.kdeplot(df[df['target'] == 0].age, shade = True, label = 'not diseased')
sn.kdeplot(df[(df['target'] == 1) & (df['sex'] == 1)].age, shade = True, label = 'male')

sn.kdeplot(df[(df['target'] == 1) & (df['sex'] == 0)].age, shade = True, label = 'female')
import matplotlib.pyplot as plt



plt.figure(figsize = (8,8))

sn.heatmap(df.corr(), annot = True)
#CORRELATION WITH RESPONSE

dummy = df.copy().drop(columns = ['target'])

dummy



dummy.corrwith(df.target).plot.bar(figsize = (8,8),

                                      title = 'Correlation with response variable',

                                      fontsize = 15, rot = 45,

                                      grid = True)
df = df.drop(columns = ['fbs', 'chol', 'trestbps'])
df.head()
df.isna().any()
from sklearn.preprocessing import StandardScaler



y = df.target

x = df.drop(columns = ['target'])
from sklearn.model_selection import train_test_split



xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb





def models(xtrain, xtest, ytrain, ytest):

    

    #logistic regression

    lrmodel = LogisticRegression(random_state = 0)

    lrmodel.fit(xtrain, ytrain)

    lrypred = lrmodel.predict(xtest)

    

    

    #decision tree

    dtmodel = tree.DecisionTreeClassifier()

    dtmodel.fit(xtrain, ytrain)

    dtypred = dtmodel.predict(xtest)

    tree.plot_tree(dtmodel)

    

    

    #random forest

    rfmodel = RandomForestClassifier(max_depth = 8, random_state = 42)

    rfmodel.fit(xtrain, ytrain)

    rfypred = rfmodel.predict(xtest)

    

    

    #SVM

    svmodel = SVC(random_state = 0, kernel = 'linear')

    svmodel.fit(xtrain, ytrain)

    svypred = svmodel.predict(xtest)

    

    

    #Gradient boosting classifier

    gbmodel = GradientBoostingClassifier(random_state = 0)

    gbmodel.fit(xtrain, ytrain)

    gbypred = gbmodel.predict(xtest)

    

    

    #XGBoost

    xgbmodel = xgb.XGBRegressor(objective = 'reg:logistic', random_state = 42)

    xgbmodel.fit(xtrain, ytrain)

    xgbypred = xgbmodel.predict(xtest)

    for i in range(len(xgbypred)):

        if xgbypred[i] >= 0.5:

            xgbypred[i]=1

        else:

            xgbypred[i]=0

    

    return lrypred, dtypred, rfypred, svypred, gbypred, xgbypred
lr, dt, rf, svm, gb, xgb = models (xtrain, xtest, ytrain, ytest)
xgb
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
# results for Logistic regression





ac = accuracy_score(ytest, lr)

f1 = f1_score(ytest, lr)

ps = precision_score(ytest, lr)

rs = recall_score(ytest, lr)



results = pd.DataFrame([['Logistic regression', ac, f1, ps, rs ]], columns = ['Model', 'Accuracy', 'F1 score', 'Precision', 'Recall score'])

models = [dt, rf, svm, gb, xgb]

model_names = ['Decision tree', 'Random forest', 'SVM', 'Gradient boost', 'XG boost']

index = 0





for model in models:

    

    ac = accuracy_score(ytest, model)

    f1 = f1_score(ytest, model)

    ps = precision_score(ytest, model)

    rs = recall_score(ytest, model)



    temp_results = pd.DataFrame([[model_names[index], ac, f1, ps, rs ]], columns = ['Model', 'Accuracy', 'F1 score', 'Precision', 'Recall score'])

    results = results.append(temp_results, ignore_index = True)

    index = index + 1

    print(results)

    print("\n")

results
import numpy as np



lrcm = confusion_matrix(ytest, lr)

dtcm = confusion_matrix(ytest, dt)

rfcm = confusion_matrix(ytest, rf)

svmcm = confusion_matrix(ytest, svm)

gbcm = confusion_matrix(ytest, gb)

xgbcm = confusion_matrix(ytest, xgb)





models_list = [lrcm,  dtcm, rfcm, svmcm, gbcm, xgbcm]

model_names = ['Logistic regression', 'Decision tree', 'Random forest', 'SVM', 'Gradient boost', 'XGBoost']

row = 0



for cl in models_list:

    df_cm = pd.DataFrame(cl, index = (0,1), columns = (0,1))

    plt.figure(figsize = (10,7))

    sn.set(font_scale = 1.4)

    sn.heatmap(df_cm, annot = True, fmt = 'g')

    plt.title(model_names[row])

    row = row+1

 