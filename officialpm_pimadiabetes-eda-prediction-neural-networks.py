from mlxtend.plotting import plot_decision_regions

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Loading the dataset

diabetesData = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')



#Print the first 5 rows of the dataframe.

diabetesData.head()
diabetesDatacopy = diabetesData.copy(deep = True) # creating the copy of the dataset

# replacing the 0 values with Nan

diabetesDatacopy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetesDatacopy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)



## showing the count of Nans

print(diabetesDatacopy.isnull().sum())
p = diabetesDatacopy.hist(figsize = (20,20))
diabetesDatacopy['BloodPressure'].fillna(diabetesDatacopy['BloodPressure'].mean(), inplace = True)

diabetesDatacopy['SkinThickness'].fillna(diabetesDatacopy['SkinThickness'].median(), inplace = True)

diabetesDatacopy['Insulin'].fillna(diabetesDatacopy['Insulin'].median(), inplace = True)

diabetesDatacopy['BMI'].fillna(diabetesDatacopy['BMI'].median(), inplace = True)

diabetesDatacopy.isna().sum()
p = diabetesDatacopy.hist(figsize = (20,20))
from keras.models import Sequential

from keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split

import numpy

import pandas as pd 

diabetes_data = diabetesDatacopy.copy()
X = diabetes_data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values

Y = diabetes_data[['Outcome']].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = Sequential()



model.add(Dense(12, input_dim=8, activation='relu')) # Input layer requires input_dim param



model.add(Dense(10, activation='relu'))



model.add(Dense(8, activation='relu'))



model.add(Dropout(.2))



model.add(Dense(1, activation='sigmoid')) # Sigmoid instead of relu for final probability between 0 and 1



# Compile the model, adam gradient descent (optimized)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
# training the network

history  = model.fit(x_train, y_train, epochs = 1000, batch_size=1, validation_data=(x_test, y_test),verbose=0)
import matplotlib.pyplot as plt

n = len(history.history["loss"])

fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(range(n), history.history["loss"],'r', marker='.', label="Train Loss")

ax.legend()
results = model.evaluate(x_test, y_test)

print(f"Accuracy: {results[1]*100}%")