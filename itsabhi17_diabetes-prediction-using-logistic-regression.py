import pandas as pd

import numpy as np
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.describe().T
df1 = df.copy(deep=True)

df1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
import seaborn as sns

sns.heatmap(df1.isnull())

print(df1.isnull().sum())
p = df1.hist(figsize = (20,20),rwidth=0.9)
df1['Glucose'].fillna(df1['Glucose'].mean(), inplace = True)

df1['BloodPressure'].fillna(df1['BloodPressure'].mean(), inplace = True)

df1['SkinThickness'].fillna(df1['SkinThickness'].median(), inplace = True)

df1['Insulin'].fillna(df1['Insulin'].median(), inplace = True)

df1['BMI'].fillna(df1['BMI'].mean(), inplace = True)
p = df1.hist(figsize = (20,20), rwidth=0.9)
sns.heatmap(df1.isnull())

print(df1.isnull().sum())
x = df1[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction', 'Age']].values

y = df1[['Outcome']].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()

lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))