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

df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()
df.describe()
df.info()
(df['Sex']=='female').sum()


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt



sns.distplot(a=df[df['Sex']=='male']['Fare'], label='Men', bins=10, kde=False, norm_hist=True);
sns.distplot(a=df[df['Sex']=='female']['Fare'], label='women', bins=10, kde=False, norm_hist=True);
plt.legend();

df.replace(['male','female'],['M','F'],inplace=True)
df['Sex'].head()
df.info()
df[['Sex', 'Age', 'SibSp', 'Parch', 'Embarked']] =df[['Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].astype('category')
df.info()
pd.pivot_table(df,values='Survived',index=['Pclass'],columns=['Sex'])
import seaborn as sns; sns.set()
ax=pd.pivot_table(df,values='Survived',index=['Pclass'],columns=['Sex']).plot(kind='bar')
ax.set_ylabel('Survival_Rate');

pivot=pd.pivot_table(df,index='SibSp',values='Survived')
ax=pivot.plot(kind='bar')
ax.set_ylabel('Survival_Rate')
pivot = pd.pivot_table(df,index='Embarked', values='Survived')
ax=pivot.plot(kind='bar')
ax.set_ylabel('Survival_Rate');
pivot = pd.pivot_table(df, index='Parch',values='Survived')
ax=pivot.plot(kind='bar')
ax.set_ylabel('Survival_Rate')
df['cut']=pd.cut(df['Age'],6)
pivot = pd.pivot_table(df,index='cut',values='Survived')
pivot.plot(kind='bar')
srs=df['cut'].value_counts(normalize=False)
srs.plot(kind='bar')
#filling null values with random values with each category have a probability proportional to the number of it in the column.
srs = df['cut'].value_counts(normalize=True)
filling = pd.Series(np.random.choice(srs.index,p=srs,size=len(df)))
df['cut'].fillna(filling,inplace=True)
df['cut'].isna().any()
titanic_df = pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_df.info()
titanic_df.dropna(axis=0,inplace=True,subset=['Embarked'])
titanic_df.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)
titanic_df.set_index('PassengerId',inplace=True)
titanic_df.info()
titanic_df=pd.get_dummies(titanic_df, columns=['Sex','Embarked'])
titanic_df.info()
titanic_df['Age'].fillna(titanic_df['Age'].mean(),axis=0,inplace=True)
titanic_df.info()
titanic_df=pd.get_dummies(titanic_df,columns=['Pclass'])
titanic_df.info()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class AttributeAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['FareAge'] = X['Fare']/X['Age']
        return X

    

X = titanic_df.drop('Survived',axis=1)
y = titanic_df['Survived']
X.head()
#Addind a pipe for scaling
procpipe = Pipeline([
    
    ('attradd', AttributeAdder()),
    ('minmaxscale', MinMaxScaler()),
    ]
)
#Support Vector Classifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
svcpipe = Pipeline([
    ('procpipe',procpipe),
    ('svc',SVC())
])

y_predict = cross_val_predict(svcpipe, X, y, cv=4,n_jobs=-1)

print('The accuracy for the Support Vector Classifier using an rbf kernel is:',accuracy_score(y_predict, y))


from sklearn.ensemble import RandomForestClassifier

rfcpipe = Pipeline([
    ('procpipe', procpipe),
    ('randomforestregressor',RandomForestClassifier())
])
y_predict = cross_val_predict(rfcpipe, X, y, cv=4,n_jobs=-1)

print('The accuracy for the Random Forest Classifier is:',accuracy_score(y_predict, y))


from sklearn.linear_model import LogisticRegression

lrpipe = Pipeline([
    ('procpipe', procpipe),
    ('logisticregressor',LogisticRegression())
])
y_predict = cross_val_predict(lrpipe, X, y, cv=4,n_jobs=-1)

print('The accuracy for the Logistic Regression is:',accuracy_score(y_predict, y))
#Models to use. They are classification algorithims
#SVC, DecisionTreeClassifier, RandomForestRegressor, SGDClassifier

from sklearn.linear_model import SGDClassifier

SGDpipe = Pipeline([
    ('procpipe', procpipe),
    ('sgdclassifier', SGDClassifier())
])


y_predict = cross_val_predict(SGDpipe, X, y, cv=4,n_jobs=-1)

print('The accuracy for the Stochastic Gradient Descent Classifier is:',accuracy_score(y_predict, y))


from sklearn.naive_bayes import GaussianNB

GNBpipe = Pipeline([
    ('procpipe', procpipe),
    ('sgdclassifier', GaussianNB())
])


y_predict = cross_val_predict(GNBpipe, X, y, cv=4,n_jobs=-1)

print('The accuracy for the Naive Gaussian Bayes  is:',accuracy_score(y_predict, y))


from sklearn.naive_bayes import MultinomialNB

MTNpipe = Pipeline([
    ('procpipe', procpipe),
    ('sgdclassifier', MultinomialNB())
])


y_predict = cross_val_predict(MTNpipe, X, y, cv=4,n_jobs=-1)

print('The accuracy for the Naive Multinomial Bayes  is:',accuracy_score(y_predict, y))



from sklearn.naive_bayes import BernoulliNB

BNLpipe = Pipeline([
    ('procpipe', procpipe),
    ('sgdclassifier', BernoulliNB())
])


y_predict = cross_val_predict(BNLpipe, X, y, cv=4,n_jobs=-1)

print('The accuracy for the Naive Bernoulli Bayes  is:',accuracy_score(y_predict, y))

from sklearn.ensemble import AdaBoostClassifier

ABCpipe = Pipeline([
    ('procpipe', procpipe),
    ('sgdclassifier', AdaBoostClassifier())
])


y_predict = cross_val_predict(ABCpipe, X, y, cv=4,n_jobs=-1)

print('The accuracy for the AdaBoostClassifier  is:',accuracy_score(y_predict, y))

import xgboost as xgb

xgbmodel = xgb.XGBClassifier(objective='binary:logistic')

XGBpipe = Pipeline([
    ('procpipe', procpipe),
    ('xgbclassifier', xgbmodel)
])

y_pred = cross_val_predict(XGBpipe,X,y,cv=6);
print('The accuracy for the XGBClassifier is:',accuracy_score(y_predict, y))
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svc__C':[0.5, 1, 10, 100, 300, 350],
    'svc__gamma':['scale', 'auto', 1,10,100,300],
    
}

grid = GridSearchCV(svcpipe,param_grid, cv=6, n_jobs=-1).fit(X,y)





print('The best params for SVC are:\n',grid.best_params_)
print('The best score was',grid.best_score_)
m=titanic_df.corr()
plt.figure(figsize = (16,16))
sns.heatmap(m,annot=True)
df = pd.read_csv("/kaggle/input/titanic/test.csv")
df.head()

df.info()
titanic_df = df

titanic_df.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)
titanic_df.set_index('PassengerId',inplace=True)
titanic_df=pd.get_dummies(titanic_df, columns=['Sex','Embarked'])

titanic_df.head()


titanic_df['Age'].fillna(titanic_df['Age'].mean(),axis=0,inplace=True)
titanic_df.info()
titanic_df.loc[titanic_df['Fare'].isna(),'Fare']=titanic_df['Fare'].mean()
titanic_df.info()
titanic_df=pd.get_dummies(titanic_df,columns=['Pclass'])
bestModel = grid.best_estimator_

y_predict = bestModel.predict(titanic_df)
y_predict
data = {titanic_df.index.name: titanic_df.index, 'Survived': y_predict}
theframe = pd.DataFrame(data)
theframe.head()
theframe.to_csv("./submission.csv")
