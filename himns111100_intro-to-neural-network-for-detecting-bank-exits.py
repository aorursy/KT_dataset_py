

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import zscore

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras import models

from tensorflow.keras.layers import Dense


### Removes warnings that occassionally show up

import warnings

warnings.filterwarnings('ignore')
data_bank=pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')

data_bank.head()
data_bank.shape
data_bank.info()
data_bank.isna().sum()
# 5 point summary

data_bank.describe().T
# CORRELATION - HEAT MAP

colormap = plt.cm.plasma

plt.figure(figsize=(17,10))

plt.title('Correlation of Cutomer Exiting Bank', y=1.05, size=15)

sns.heatmap(data_bank.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 

            linecolor='white', annot=True)
sns.countplot(x="Geography", data=data_bank,hue="Exited")
gendermap = sns.FacetGrid(data_bank,hue = 'Exited')

(gendermap.map(plt.hist,'Age',edgecolor="w").add_legend())
bank_data_new = data_bank.drop(['RowNumber', 'CustomerId', 'Surname'], axis =1)


bank_data_new.head()
numerical_distribution = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

for i in numerical_distribution:

    plt.hist(data_bank[i])

    plt.title(i)

    plt.show()
## Label Encoding of all the columns

# instantiate labelencoder object

le = LabelEncoder()



# Categorical boolean mask

categorical_feature_mask = bank_data_new.dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = bank_data_new.columns[categorical_feature_mask].tolist()

bank_data_new[categorical_cols] = bank_data_new[categorical_cols].apply(lambda col: le.fit_transform(col))

print(bank_data_new.info())


df_scaled = bank_data_new.apply(zscore)

X_columns =  df_scaled.columns.tolist()[1:10]

Y_Columns = bank_data_new.columns.tolist()[-1:]



X = df_scaled[X_columns].values

y = np.array(bank_data_new['Exited']) # Exited



print(y)

print(X)
#splitting the dataset into training and test set

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 8)
X_train.shape
#Encoding the output class label (One-Hot Encoding)

y_train=to_categorical(y_train,2)

y_test=to_categorical(y_test,2)
from sklearn.preprocessing import Normalizer

normalize=Normalizer(norm="l2")

X_train=normalize.transform(X_train)



print(X_train)


X_test=normalize.transform(X_test)

print(X_test)
#Initialize Sequential Graph (model)

model = tf.keras.Sequential()
model.add(Dense(units=6, activation='relu', input_shape=(9,)))

model.add(Dense(20, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
history=model.fit(X_train, y_train, batch_size=45, epochs=200, validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test,verbose=1)



print(score)


score = model.evaluate(X_train, y_train,verbose=1)



print(score)


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix



confmatrx= confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
confmatrx
plt.plot(np.array(history.history['accuracy']) * 100)

plt.plot(np.array(history.history['val_accuracy']) * 100)

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.legend(['train', 'validation'])

plt.title('Accuracy over epochs')

plt.show()
print (((confmatrx[0][0]+confmatrx[1][1])*100)/(len(y_test)), '% of testing data was classified correctly')
# Checking the accuracy using accuracy_score as well to check if the above calculation is correct or not.

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)