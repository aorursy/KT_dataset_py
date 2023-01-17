import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler





from sklearn.metrics import mean_squared_error

import math



data = pd.read_csv('../input/train.csv')

data=data.drop('Date_Time', axis=1)

data=data.drop('NMHC(GT)', axis=1)



correlation = data.corr(method='pearson')

columns = correlation.nlargest(5, 'T').index



#print(columns)





X_train = data[columns]

Y_train= X_train['T'].values

X_train = X_train.drop('T', axis = 1).values





#X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, random_state=0)



#header = ['AH', 'C6H6(GT)', 'RH', 'PT08.S4(NO2)', 'PT08.S1(CO)','PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S5(O3)', 'NO2(GT)']



header = ['AH','C6H6(GT)', 'RH', 'PT08.S4(NO2)']



df_test = pd.read_csv('../input/test.csv')



XX = []

date = []



for q in range(len(df_test)):

	date = date + [df_test['Date_Time'][q]]

	x1 = []

	for i in header:

		x1.append(df_test[i][q])

	XX.append(x1)



X_test = np.asarray(XX)

















#from sklearn.ensemble import GradientBoostingRegressor

#from sklearn.tree import DecisionTreeRegressor

#from sklearn.linear_model import LinearRegression

#from sklearn import linear_model

#from sklearn.linear_model import ElasticNet

#from sklearn.neighbors import KNeighborsRegressor







####  2. Polynomial Regression  ####

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures













#print("WITHOUT SCALING\n")



'''

model1_ = GradientBoostingRegressor(random_state=42, n_estimators=8000)

model1_.fit(X_train, Y_train)

Y_pred1_ = model1_.predict(X_test)



model2_ = DecisionTreeRegressor(random_state=42)

model2_.fit(X_train, Y_train)

Y_pred2_ = model2_.predict(X_test)







poly_reg = PolynomialFeatures(degree = 3)

X_poly = poly_reg.fit_transform(X_train)

poly_reg_model = LinearRegression()

poly_reg_model.fit(X_poly, Y_train)

X_test_poly = poly_reg.fit_transform(X_test)

Y_pred3_ = poly_reg_model.predict(X_test_poly)







#print(math.sqrt(mean_squared_error(Y_test, Y_pred1_)))

#print()

#print(math.sqrt(mean_squared_error(Y_test, Y_pred2_)))

#print()

print(math.sqrt(mean_squared_error(Y_test, Y_pred3_)))

print()

'''











#print("WITH SCALING\n")



scaler = StandardScaler().fit(X_train)

rescaled_X_train = scaler.transform(X_train)

rescaled_X_test = scaler.transform(X_test)





'''

model1 = GradientBoostingRegressor(random_state=42, n_estimators=8000)

model1.fit(rescaled_X_train, Y_train)

Y_pred1 = model1.predict(rescaled_X_test)



model2 = DecisionTreeRegressor(random_state=42)

model2.fit(rescaled_X_train, Y_train)

Y_pred2 = model2.predict(rescaled_X_test)



'''

poly_reg = PolynomialFeatures(degree = 3)

X_poly = poly_reg.fit_transform(rescaled_X_train)

poly_reg_model = LinearRegression()

poly_reg_model.fit(X_poly, Y_train)

X_test_poly = poly_reg.fit_transform(rescaled_X_test)

Y_pred3 = poly_reg_model.predict(X_test_poly)







'''

model3 = KNeighborsRegressor(n_neighbors=300)

model3.fit(rescaled_X_train, Y_train) 

Y_pred3 = model3.predict(rescaled_X_test)



'''









#print(math.sqrt(mean_squared_error(Y_test, Y_pred1)))

#print()

#print(math.sqrt(mean_squared_error(Y_test, Y_pred2)))

#print()







#print(math.sqrt(mean_squared_error(Y_test, Y_pred3)))

#print()



classifier_output = pd.DataFrame(data={"Date_Time":date , "T": Y_pred3})

classifier_output.to_csv(('pred_output_2.csv'), index=False, quoting=3, escapechar='\\')







print("\nTHANK YOU !!\n")





































#from sklearn.linear_model import LinearRegression

#from sklearn.linear_model import Lasso

#from sklearn.linear_model import ElasticNet

#from sklearn.tree import DecisionTreeRegressor

#from sklearn.neighbors import KNeighborsRegressor