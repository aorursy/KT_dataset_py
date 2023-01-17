#importing useful libararies
import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#read the Sales Records excel file
df = pd.read_csv(r'../input/income-classification/income_evaluation.csv')
df.head(5)
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship','race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.isna().sum()
df.dtypes
num = [i for i in df.columns if df[i].dtype!='O']
cat = [i for i in df.columns if df[i].dtype=='O']
df[cat].nunique().plot(kind='bar')
for i in df[cat]:
    print(df[i].value_counts())
df.replace(' ?', np.nan, inplace= True)
df.isna().sum()
df.fillna(method = 'bfill', inplace=True)
df.isna().sum().sum()
df.income.value_counts()
sea.countplot(x= 'income' ,data =df)
df.income.value_counts().plot(kind='pie')
sea.countplot(x="income", hue="sex", data=df)
plt.subplots(figsize=(12, 8))
sea.countplot(x="income", hue="workclass", data=df)
plt.subplots(figsize=(12, 8))
sea.countplot(hue="sex", x="workclass", data=df)
for i in cat:
    
    print(i, ' contains ', len(df[i].unique()), ' labels')
y = df.income
y = pd.get_dummies(y,drop_first=True)
df.drop(['income'],axis =1, inplace=True)
x = pd.get_dummies(df[cat])
df.drop(df[cat],axis = 1,inplace = True)
df[num].head()
sea.distplot(df.age, bins=10)
sea.boxplot(df.age)
df.corr()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df = scaler.fit_transform(df)
df = pd.DataFrame(df)
x = pd.concat([x,df],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
X_train.head(2)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy_score(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0,n_estimators=10)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0,n_estimators=100)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
import xgboost as xgb
model=xgb.XGBClassifier(base_estimator = rfc,random_state=1,learning_rate=0.1)
model.fit(X_train, y_train)
model.score(X_test,y_test)