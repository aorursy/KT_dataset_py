import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.DataFrame()
df['Weather'] = ['Sunny', 'Sunny', 'Overcast', 'Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy' ]
df['Temperature'] = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
df['Play'] = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
df.head()
print('Number of logical states: ',dict(zip(df.columns,[len(df[i].unique())for i in df.columns])))
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df)
arr = enc.transform(df.values).toarray()
X,y = arr[:,:-1],arr[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
pd.DataFrame(metrics.classification_report(nb.predict(X_test), y_test, output_dict=True))