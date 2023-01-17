import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/titanic/train.csv') 

test = pd.read_csv('../input/titanic/test.csv')

df=pd.concat([train,test]).reset_index()
df.info()
df[df['Embarked'].isnull()]
df['Embarked']=df['Embarked'].fillna('S')
df[df['Fare'].isnull()]
t=df[(df['Embarked']=='S')&(df['Pclass']==3)]['Fare'].median()

df['Fare']=df['Fare'].fillna(t)
sns.heatmap(df.corr(),annot=True)
from patsy import dmatrices

Y, X = dmatrices('Survived ~ Pclass+Sex+Fare', df, return_type='dataframe')

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif
from sklearn.ensemble import RandomForestClassifier



y = train["Survived"]



#Filling NaNs of Fare in train dataset

h=train[(train['Embarked']=='S')&(train['Pclass']==3)]['Fare'].median()

train['Fare']=train['Fare'].fillna(h)



#Filling NaNs of Fare in test dataset

l=test[(test['Embarked']=='S')&(test['Pclass']==3)]['Fare'].median()

test['Fare']=test['Fare'].fillna(l)



features = ["Pclass", "Sex", "Fare"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('final_submission.csv', index=False)