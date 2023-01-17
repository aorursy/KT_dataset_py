import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df= pd.read_csv('/kaggle/input/hepatitis.csv')

data = pd.DataFrame(df)
data.describe()
data.shape
data.info()
data.isnull().sum()
data.hist('age')
data.keys()

X = df[['class', 'age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise',

       'anorexia', 'liver_big', 'liver_firm', 'spleen_palable', 'spiders',

       'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin',

       'protime']]

y = df['histology']
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(X_train, y_train)

y_predict = modelo.predict(X_test)
score = modelo.score(X_test, y_test)

score