import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv("../input/hr-analytics/HR_comma_sep.csv")
df.head()
plt.figure(figsize=(10,7))
sns.countplot(x="Department",hue="left",data=df)
plt.figure(figsize=(10,7))
sns.countplot(x="Department",hue="salary",data=df)
plt.figure(figsize=(10,7))
sns.countplot(x="salary",hue="left",data=df)
sal = pd.get_dummies(df['salary'],prefix='sal')
sal
df = pd.concat([df,sal],axis=1)
df.head()
df.columns
X = df[['satisfaction_level','average_montly_hours', 'promotion_last_5years','sal_high', 'sal_low',
       'sal_medium']]
y = df['left']
df.drop(['last_evaluation', 'number_project', 'time_spend_company','salary','Department','left'],axis=1,inplace=True)
df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)
# Accuracy of the model
model.score(X_train,y_train)
