import numpy as np

import pandas as pd
df=pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent.csv')

df2=pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
df.head()
df2.head()
df.drop('Unnamed: 0',axis=1,inplace=True)
df
df['floor'].replace(to_replace='-',value=0,inplace=True)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['animal']=le.fit_transform(df['animal'])
df['furniture']=le.fit_transform(df['furniture'])
df
df.columns
for col in ['hoa', 'rent amount', 'property tax','fire insurance', 'total']:

    df[col].replace(to_replace='R\$',value='',regex=True,inplace=True)

    df[col].replace(to_replace=',',value='',regex=True,inplace=True)
df=df.astype(dtype=np.int64)
df.isin(['Sem info']).any()
df['hoa'].replace('Sem info',value='0',inplace=True)
df.isin(['Incluso']).any()
df['hoa'].replace('Incluso',value='0',inplace=True)

df['property tax'].replace('Incluso',value='0',inplace=True)
df
df=df.astype(dtype=np.int64)
df
df=df.sample(frac=1).reset_index(drop=True)
df
y=df['city']

X=df.drop('city',axis=1)
X
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)

print(lr.score(X_test,y_test))
from sklearn.svm import SVC

svm=SVC(kernel='rbf')

svm.fit(X_train,y_train)

print(svm.score(X_test,y_test))
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(hidden_layer_sizes=(16,16),activation='relu',solver='adam',max_iter=300)

nn.fit(X_train, y_train)

print(nn.score(X_test,y_test))