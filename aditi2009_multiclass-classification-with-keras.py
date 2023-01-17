#Multiclass Classification using Iris Dataset
#This example is to show how we actually deal with data set where we have more than two possible classes in our traget
#Load the packages
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#load the dataset
df = pd.read_csv('../input/Iris.csv')
df = df.drop("Id",axis=1)
#Use seaborn pairplot to show how the dataset looks
sns.pairplot(df,hue="Species")
#Checking the dataset for numerical columns
df.head()
X = df.drop("Species",axis=1)
X.head()
#add the species in the target names
target_names = df['Species'].unique()
target_names
#Build a target dictionarywhere we enumerate the target names where it gives Setosa=0, Versicolor = 1, Virginica=2
target_dict = {n:i for i,n in enumerate(target_names)}
target_dict
#Map the species column to the target data
y= df['Species'].map(target_dict)
y.head()
#In Keras we have to_categorical which does exactly the same function as performed above
from keras.utils.np_utils import to_categorical
y_cat = to_categorical(y)
y_cat[:10]
#Split the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X.values,y_cat,test_size=0.2)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
#We use categorical_crossentropy as it goes in tandem with the soft max function

model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='softmax'))
model.compile(Adam(lr=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, validation_split=0.1)
y_pred = model.predict(X_test)
y_pred[:5]
#So we take the maximum probability for each row
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
from sklearn.metrics import classification_report
#Compare the test class with the predicted calss 
print(classification_report(y_test_class, y_pred_class))
