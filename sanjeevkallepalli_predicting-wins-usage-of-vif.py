# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
LOL = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
LOL.isnull().sum()
def levels(df):

    return (pd.DataFrame({'dtype':df.dtypes, 

                         'levels':df.nunique(), 

                         'levels':[df[x].unique() for x in df.columns],

                         'null_values':df.isna().sum(),

                         'unique':df.nunique()}))

levels(LOL)
cols = ['gameId','blueWins','blueFirstBlood','blueEliteMonsters','blueDragons','blueHeralds','blueTowersDestroyed','redFirstBlood',

       'redEliteMonsters','redDragons','redHeralds','redTowersDestroyed']



for col in cols:

    LOL[col] = LOL[col].astype('category')
LOL.drop(['gameId'],axis = 1, inplace = True)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(18,18))

sns.heatmap(LOL.corr(),annot=True)
x = LOL.copy().drop("blueWins",axis=1)

y = LOL["blueWins"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 200)
num_col = [i for i in LOL.columns if i not in cols]



from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scale.fit(x_train[num_col])

x_train[num_col] = scale.transform(x_train[num_col])

x_test[num_col] = scale.transform(x_test[num_col])
from statsmodels.stats.outliers_influence import variance_inflation_factor

Vif = pd.DataFrame()

Vif["VIF Factor"] = [variance_inflation_factor(x_train[num_col].values,i) for i in range(x_train[num_col].shape[1])]

Vif["features"] = x_train[num_col].columns

Vif
vif_cols = ['blueWardsPlaced','blueWardsDestroyed','blueAssists','blueAvgLevel','blueTotalJungleMinionsKilled',

            'redWardsPlaced','redWardsDestroyed','redAssists','redAvgLevel','redTotalJungleMinionsKilled']

cols = ['blueFirstBlood','blueEliteMonsters','blueDragons','blueHeralds','blueTowersDestroyed','redFirstBlood',

       'redEliteMonsters','redDragons','redHeralds','redTowersDestroyed']

x_train = pd.concat([x_train[cols].reset_index(drop=True),x_train[vif_cols].reset_index(drop=True)],axis=1)

x_test = pd.concat([x_test[cols].reset_index(drop=True),x_test[vif_cols].reset_index(drop=True)],axis=1)
x_train.shape,x_test.shape
from sklearn.linear_model import LogisticRegression

FN1 = LogisticRegression(random_state=200)

FN1.fit(x_train,y_train)

train_pred_lr = FN1.predict(x_train)

test_pred_lr = FN1.predict(x_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



cm_train = confusion_matrix(y_pred=train_pred_lr,y_true=y_train)

print("Accuracy_train:", sum(np.diag(cm_train))/np.sum(cm_train))

print("Error_train:", np.round(1-sum(np.diag(cm_train))/np.sum(cm_train),2))



cm_test = confusion_matrix(y_pred=test_pred_lr,y_true=y_test)

print("Accuracy_test:", sum(np.diag(cm_test))/np.sum(cm_test))

print("Error_test:", np.round(1-sum(np.diag(cm_test))/np.sum(cm_test),2))
from sklearn.tree import DecisionTreeClassifier

regressor_1=DecisionTreeClassifier(max_depth=30,max_features=8,min_samples_split=2,min_samples_leaf=1)

regressor_1.fit(x_train,y_train)

train_pred_dtr = regressor_1.predict(x_train)

test_pred_dtr = regressor_1.predict(x_test)
cm_train = confusion_matrix(y_pred=train_pred_dtr,y_true=y_train)

print("Accuracy_train_dtr:", sum(np.diag(cm_train))/np.sum(cm_train))

print("Error_train_dtr:", np.round(1-sum(np.diag(cm_train))/np.sum(cm_train),2))



cm_test = confusion_matrix(y_pred=test_pred_dtr,y_true=y_test)

print("Accuracy_test_dtr:", sum(np.diag(cm_test))/np.sum(cm_test))

print("Error_test_dtr:", np.round(1-sum(np.diag(cm_test))/np.sum(cm_test),2))