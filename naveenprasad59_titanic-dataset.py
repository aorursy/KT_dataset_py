import numpy as np

import pandas as pd

import seaborn as sb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from  sklearn.metrics import accuracy_score,classification_report

from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('../input/titanic/train_and_test2.csv')

df.head()
df.shape
df.isnull().sum()
df.dropna(inplace=True)
scaler = MinMaxScaler()

scaler.fit_transform(df)
df.rename(columns={'2urvived':'Survived'},inplace=True)
sb.countplot('Survived',data=df)
df.groupby(['Sex', 'Survived'])['Survived'].count()
Y = df['Survived'].to_numpy()

df = df.drop(columns=['Survived'])

X = df.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=42)

clf = LogisticRegression(max_iter=10000)

clf.fit(X_train,y_train)

predict = clf.predict(X_test)

print('Accuracy score:',accuracy_score(predict,y_test))
print(classification_report(y_test,predict))