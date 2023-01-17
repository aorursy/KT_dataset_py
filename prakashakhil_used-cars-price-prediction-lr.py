from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.qda import QDA

from sklearn.cross_validation import cross_val_score

from sklearn import preprocessing

from sklearn.cross_validation import train_test_split





import pandas as pd

import numpy as np





all_data= pd.read_csv("../input/autos.csv" , encoding='latin1')



cars_updated = all_data.dropna()

cars_updated = cars_updated.apply(preprocessing.LabelEncoder().fit_transform)



car_price_column = cars_updated.loc[:,["price"]]

car_attributes = cars_updated.loc[:,["vehicleType","gearbox","powerPS","brand","model","fuelType","brand"]]



lr=LinearRegression()



scores = cross_val_score(lr, cars_updated.values,car_price_column, cv=5)

print("Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))

#fit the model



#Split into train and validation

x_train, x_test, y_train, y_test = train_test_split(car_attributes, 

                                                    car_price_column,

                                                    test_size=0.2)



lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)



#Find error : RMSE

error_factor=np.sqrt(np.mean((y_test - y_pred)**2))







print("RMS error %0.10f "%(error_factor))