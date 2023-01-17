import numpy as np

import pandas as pd
#Importing the dataset

dataset = pd.read_csv('../input/car-dataset/car_data.csv')

dataset.head()
dataset.info()
dataset["fuel"].value_counts()
dataset["transmission"].value_counts()
X = dataset.iloc[:,[1,3,4,6]].values

y = dataset.iloc[:,2].values
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

X[:,2]=lb.fit_transform(X[:,2])

lb1 = LabelEncoder()

X[:,3]=lb1.fit_transform(X[:,3])
print(X.shape)
#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)



print(X_train[0, :])
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)

regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test, y_test)

print(accuracy*100, '%')
new_data = [2017, 7000, "Petrol", "Manual"]

new_data[2] = lb.transform([new_data[2]])[0]

new_data[3] = lb1.transform([new_data[3]])[0]
print(new_data)

regressor.predict([new_data])
import pickle

pickle.dump(regressor, open('regressor.pkl', 'wb'))

pickle.dump(lb, open('lb', 'wb'))

pickle.dump(lb1, open('lb1', 'wb'))