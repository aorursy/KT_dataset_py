# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
train.describe()
# we have 177 missing age value

ax = train['Age'].hist()

ax.set(xlabel = 'Age', ylabel = 'Count')

plt.show()
train["Age"].median(skipna=True)
sns.countplot(x='Embarked', data=train,)

plt.show()
train_data = train #i dont want to make change in the original dataframe

train_data['Age'].fillna(28, inplace=True)

train_data['Embarked'].fillna("S", inplace=True)

train_data.drop('Cabin', axis=1, inplace=True)
train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"] #

train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1) #if the person is traveling alone 1 else 0





train_data.drop('SibSp', axis=1, inplace=True)

train_data.drop('Parch', axis=1, inplace=True)

train_data.drop('TravelBuds', axis=1, inplace=True)



#one code encoding with pd.get_dummies



train2 = pd.get_dummies(train_data, columns=["Pclass"])



train3 = pd.get_dummies(train2, columns=["Embarked"])



train4=pd.get_dummies(train3, columns=["Sex"])

train4.drop('Sex_female', axis=1, inplace=True)



#Drop Unwanted

train4.drop('PassengerId', axis=1, inplace=True)

train4.drop('Name', axis=1, inplace=True)

train4.drop('Ticket', axis=1, inplace=True)

train4.head(5)

df_final = train4
test["Age"].fillna(28, inplace=True)

test["Fare"].fillna(14.45, inplace=True)

test.drop('Cabin', axis=1, inplace=True)
test['TravelBuds']=test["SibSp"]+test["Parch"]

test['TravelAlone']=np.where(test['TravelBuds']>0, 0, 1)



test.drop('SibSp', axis=1, inplace=True)

test.drop('Parch', axis=1, inplace=True)

test.drop('TravelBuds', axis=1, inplace=True)



test2 = pd.get_dummies(test, columns=["Pclass"])

test3 = pd.get_dummies(test2, columns=["Embarked"])



test4=pd.get_dummies(test3, columns=["Sex"])

test4.drop('Sex_female', axis=1, inplace=True)



test4.drop('PassengerId', axis=1, inplace=True)

test4.drop('Name', axis=1, inplace=True)

test4.drop('Ticket', axis=1, inplace=True)

final_test = test4
final_test.head(5)
#lest check our dataframe in case we have no missing value :)

df_final.info()
df_final.corr()
plt.figure(figsize=(15,8))

sns.kdeplot(train["Age"][df_final.Survived == 1], color="blue")

sns.kdeplot(train["Age"][df_final.Survived == 0], color="red")

plt.legend(['Survived', 'Died'])

plt.show()
plt.figure(figsize=(35, 10))

avg_survival_byage = df_final[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")
df_final['IsChild']=np.where(train_data['Age']<=16, 1, 0)

final_test['IsChild']=np.where(final_test['Age']<=16, 1, 0)
plt.figure(figsize=(15,8))

sns.kdeplot(df_final["Fare"][train.Survived == 1], color="blue", shade=True)

sns.kdeplot(df_final["Fare"][train.Survived == 0], color="red", shade=True)

plt.legend(['Survived', 'Died'])

sns.barplot('Pclass', 'Survived', data=train, color="blue")

plt.show()
sns.barplot('Embarked', 'Survived', data=train)

plt.show()
sns.barplot('TravelAlone', 'Survived', data=df_final )

plt.show()
col=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsChild"] 

X=df_final[col]

Y=df_final['Survived']
import statsmodels.api as sm

from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit_model=sm.Logit(Y,X)

result=logit_model.fit()

print(result.summary())
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X, Y)



print("Model Accuracy : {:.2f}%".format(logreg.score(X, Y)*100))
from sklearn.model_selection import train_test_split

train1, test1 = train_test_split(df_final, test_size=0.25)
X1=train1[col]

Y1=train1['Survived']

logit_model3=sm.Logit(Y1,X1)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



logreg = LogisticRegression()

logreg.fit(X1, Y1)

print("Model Accuracy : {:.2f}%".format(logreg.score(X1, Y1)*100))
from sklearn import metrics

logreg.fit(X1, Y1)



X1_test = test1[col]

Y1_test = test1['Survived']



Y1test_pred = logreg.predict(X1_test)



print('Accuracy of logistic regression: {:.2f}'.format(logreg.score(X1_test, Y1_test)*100))
from sklearn.metrics import roc_auc_score

logreg.fit(X1, Y1)

Y1_pred = logreg.predict(X1)



y_true = Y1

y_scores = Y1_pred

print("Model ROC_AUC : {:.2f}%".format(roc_auc_score(y_true, y_scores)))
final_test.info()
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()