import pandas as pd

import matplotlib.pyplot as pt

import warnings
from sklearn.model_selection import train_test_split , cross_val_score

from sklearn.metrics import classification_report , confusion_matrix, mean_squared_error , r2_score 

from sklearn.neighbors import KNeighborsClassifier

from math import sqrt
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
df = pd.read_csv("../input/datacsv/datasets_9066_12635_Indian automoble buying behavour study 1.0.csv")
df.shape
df.columns
df.describe()
df.hist()
df['Price'].max()
df['Price'].min()
df.isnull().sum()
df['Make'].unique()
df['Make'].value_counts()
df.head(3)
# Convert string into numeric

df['Profession'] = df['Profession'].replace ('Salaried',1)

df['Profession'] = df['Profession'].replace ('Business',2)
df['Marrital Status'] = df['Marrital Status'].replace ('Single',1)

df['Marrital Status'] = df['Marrital Status'].replace ('Married',2)
df['Education'] = df['Education'].replace ('Graduate',1)

df['Education'] = df['Education'].replace ('Post Graduate',2)
df.replace('Yes', 2, inplace = True)

df.replace('No', 1, inplace = True)

df.replace('m', 0, inplace = True)
df.columns
cars_model = df['Make'].unique()

cars_model
cars_model_mapping = [1,2,3,4,5,6,7,8,9]
for car, maper in zip(cars_model, cars_model_mapping):

    df['Make'] = df['Make'].replace(car, maper)
df['Make'].unique()
y = df['Make']

X = df.drop(['Make'], axis= 1)
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.3)
model =  KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
rms = sqrt(mean_squared_error (y_test , y_pred)) 

rms
r2_score (y_test, y_pred)
# Classification report shows our model accuracy for each class , which is not good.

# rms (difference between actual and predicted) should be less (in number 0.2 to 0.5 is good) for better results, it's too high.

# R2 must be high (0.7 is a good number), we have negative r2score.