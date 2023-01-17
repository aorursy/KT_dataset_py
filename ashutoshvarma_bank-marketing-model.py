# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bank = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')
bank.head()
bank.info()
sns.distplot(bank['balance'])



sns.FacetGrid(data=bank,hue='deposit',size=6).map(sns.distplot,'age').add_legend()
sns.set_style('whitegrid')

sns.boxplot(x='age',y='education',data=bank)

num1 = bank[['age','balance','day','duration','campaign','pdays','previous']]
cat = bank[['job','marital','education','housing','contact','month','poutcome','default','loan','deposit']]
from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

cat1 = cat.apply(lab.fit_transform)
cat1
data = num1.join(cat1)
bank.head()
data.head()
data.corr()
X = data.drop('deposit',axis=1)
y = data['deposit']
sns.heatmap(data.isnull())
data.isnull().sum()
X['pdays'].describe()
X = X.drop('pdays',axis=1)
X
plt.figure(figsize=(15,15))

sns.heatmap(data.corr())
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import f_regression

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

logit = LogisticRegression()

pred = logit.fit(X,y)
pred.score(X,y)
coef = pred.coef_
coef.round(3)
f_regression(X,y)
p_values = f_regression(X,y)[1]
p_values
regressor_OLS = sm.OLS(y,X)

result = regressor_OLS.fit()

result.summary()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=101)
logit.fit(X_train,y_train)
y_pred = logit.predict(X_test)
print("Accuracy score is",accuracy_score(y_test,y_pred))

print("Confusion matrix is\n",confusion_matrix(y_test,y_pred))

print("Classification report is\n",classification_report(y_test,y_pred))

print("F1 Score is",f1_score(y_test,y_pred))
print("cross value score is",cross_val_score(logit,X,y,cv=5))
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 1000],'max_iter' :[100, 500 , 1000]}]

grid_search = GridSearchCV(estimator = logit,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 5,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_
final = pd.DataFrame({'method' : ['Logit Reg Score','GridSearchCV Score'],'Result' :[accuracy_score(y_test,y_pred), grid_search.best_score_]})
final