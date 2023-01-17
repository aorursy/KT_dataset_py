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
file = r'/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv'
import pandas as pd

import seaborn as sns

import keras

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt
data = pd.read_csv(file)
data.head()
data.isna().sum()
data.describe()
data.corr()
data['oldpeak'] = data['oldpeak'].astype(int)

data.tail()
sns.heatmap(data.corr(), cmap="YlGnBu")
sns.countplot(data['sex'])
ax = sns.distplot(data['age'])
sns.catplot(x='age', y='oldpeak', hue='sex', data=data, kind='point')
X = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]

y = data['target']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))



# Adding the second hidden layer

classifier.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 13, epochs = 1000)
classifier.summary()
results = X_test.copy()

results["predicted"] = classifier.predict(X_test)

results["actual"]= y_test.copy()

results = results[['predicted', 'actual']]

results['predicted'] = results['predicted'].round(1)

results
from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(results['actual'].round(), results['predicted'].round())

sns.heatmap(cf_matrix)