import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/titanic/train.csv') 
test = pd.read_csv('../input/titanic/test.csv')
df=pd.concat([train,test]).reset_index()
df.describe()
df.head()
cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)
pd.set_option('display.max_rows', 5000)
train_test_null_info=pd.DataFrame(df.isnull().sum(),columns=['Count of NaN'])
train_test_dtype_info=pd.DataFrame(df.dtypes,columns=['DataTypes'])
train_tes_info=pd.concat([train_test_null_info,train_test_dtype_info],axis=1)
train_tes_info
g = {str(k): list(v) for k,v in df.groupby(df.dtypes, axis=1)}
g
df['Embarked']=df['Embarked'].fillna('S')
df[df.Fare.isnull()]
g=df[(df['Embarked']=='S')&(df['Pclass']==3)]['Fare'].median()
df['Fare']=df['Fare'].fillna(g)
train_test_null_info=pd.DataFrame(df.isnull().sum(),columns=['Count of NaN'])
train_test_dtype_info=pd.DataFrame(df.dtypes,columns=['DataTypes'])
train_tes_info=pd.concat([train_test_null_info,train_test_dtype_info],axis=1)
train_tes_info
dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
df = pd.concat((df,titanic_dummies), axis=1)
df = df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
test=df[df.Age.isnull()]
train=df[df.Age.notnull()]
sns.heatmap(train.corr(),annot=True)
top_corr_features = corrmat.index[abs(corrmat["Age"])>0.0]
top_corr_features
train_y = train.Age
predictor_cols = ['SibSp',1,2,3]

# Create training predictors data
train_X = train[predictor_cols]

my_model = linear_model.LinearRegression()
my_model.fit(train_X, train_y)
test_X = test[predictor_cols]
test['Age'] = my_model.predict(test_X)
test
test['Age']=test['Age'].round(decimals=0)
test['Age']=test['Age'].abs()
test
df=pd.concat([train,test]).reset_index()
df
train_test_null_info=pd.DataFrame(df.isnull().sum(),columns=['Count of NaN'])
train_test_dtype_info=pd.DataFrame(df.dtypes,columns=['DataTypes'])
train_tes_info=pd.concat([train_test_null_info,train_test_dtype_info],axis=1)
train_tes_info
sns.heatmap(df.corr(),annot=True)
corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat["Survived"])>0.2]
plt.figure(figsize=(10,10))
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
test=df[df.Survived.isnull()]
train=df[df.Survived.notnull()]
y = train["Survived"]
features = [2, 1, 3, 'male', 'female', "Fare"]
X = train[features]
X_test = test[features]
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
predictions
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('final_submission.csv', index=False)