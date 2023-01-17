import cmath

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import linear_model

import warnings

warnings.filterwarnings("ignore")
olympics = pd.read_csv('../input/athlete_events.csv')
olympics["Medal"] = olympics["Medal"].fillna(0)

olympics["Medal"] = olympics["Medal"].replace(["Bronze","Silver","Gold"],1)
us_train_data = olympics[olympics.Team =="United States"][olympics.Sport=="Athletics"][olympics.Year<=1988].groupby("Year", as_index=False)[["Medal"]].sum().values
us_test_data = olympics[olympics.Team =="United States"][olympics.Sport=="Athletics"][olympics.Year>1988].groupby("Year", as_index=False)[["Medal"]].sum().values
us_data = olympics[olympics.Team =="United States"][olympics.Sport=="Athletics"].groupby("Year", as_index=False)[["Medal"]].sum().values
x_us = us_data[:,0] 

y_us = us_data[:,1] 
x_us_train = us_train_data[:,0] 

y_us_train = us_train_data[:,1] 
x_us_test = us_test_data[:,0] 

y_us_test = us_test_data[:,1]
ex_us = sum(x_us_train)/x_us_train.size

ex2_us = sum(x_us_train**2)/x_us_train.size

var_x_us = ex2_us - ex_us**2

sd_x_us = cmath.sqrt(var_x_us)
ey_us = sum(y_us_train)/y_us_train.size

ey2_us = sum(y_us_train**2)/y_us_train.size

var_y_us = ey2_us - ey_us**2

sd_y_us = cmath.sqrt(var_y_us)
exy_us = sum(x_us_train*y_us_train)/x_us_train.size

cov_us = exy_us - ex_us*ey_us
corr_us = (cov_us/(sd_x_us*sd_y_us)).real
beta_us = ((corr_us*sd_y_us)/sd_x_us).real

print("Coefficient: \n", beta_us) 
alpha_us = ey_us - beta_us*ex_us

print("Intercept: \n", alpha_us)
response_us = alpha_us + beta_us*x_us_test

response_us = np.round(response_us)

print("Predicted response: \n", response_us)
error_us = y_us_test - response_us

print("Error: \n", np.round(error_us))
std_error_us = cmath.sqrt(sum(error_us**2)/error_us.size)

print("Standard error: \n", np.round(std_error_us.real))
x_us_skt = us_train_data[:,0].reshape(-1,1)

y_us_skt = us_train_data[:,1].reshape(-1,1) 

x_us_sktest = us_test_data[:,0].reshape(-1,1)
# Create linear regression object



linear_us = linear_model.LinearRegression()



# Train the model using the training sets and check score



linear_us.fit(x_us_skt, y_us_skt)

linear_us.score(x_us_skt, y_us_skt)



#Equation coefficient and Intercept



print('Coefficient: \n', linear_us.coef_)

print('Intercept: \n', linear_us.intercept_)



#Predict Output



predicted_us = linear_us.predict(x_us_sktest)

predicted_us = np.round(predicted_us)
print("Verification of predicted response: \n", predicted_us)
plt.title("Medals Scored by US in the event of Athletics at  Olympics")

plt.scatter(x_us,y_us,c="dodgerblue")

plt.ylabel("Number of Medals")

plt.xlabel("Year")

x_reg_us = np.array([min(x_us),max(x_us)])

f = lambda x: alpha_us + beta_us*x 

plt.plot(x_reg_us,f(x_reg_us),c="lightcoral")

plt.yticks(np.arange(0,80,10).tolist())

plt.xticks(np.arange(1896,2016,16).tolist())

plt.grid(True)

plt.show()
aus_train_data = olympics[olympics.Team =="Australia"][olympics.Sport=="Athletics"][olympics.Year<=1988].groupby("Year", as_index=False)[["Medal"]].sum().values
aus_test_data = olympics[olympics.Team =="Australia"][olympics.Sport=="Athletics"][olympics.Year>1988].groupby("Year", as_index=False)[["Medal"]].sum().values
aus_data = olympics[olympics.Team =="Australia"][olympics.Sport=="Athletics"].groupby("Year", as_index=False)[["Medal"]].sum().values
x_aus = aus_data[:,0]

y_aus = aus_data[:,1]
x_aus_train = aus_train_data[:,0]

y_aus_train = aus_train_data[:,1]
x_aus_test = aus_test_data[:,0]

y_aus_test = aus_test_data[:,1]
ex_aus = sum(x_aus_train)/x_aus_train.size

ex2_aus = sum(x_aus_train**2)/x_aus_train.size

var_x_aus = ex2_aus - ex_aus**2

sd_x_aus = cmath.sqrt(var_x_aus)
ey_aus = sum(y_aus_train)/y_aus_train.size

ey2_aus = sum(y_aus_train**2)/y_aus_train.size

var_y_aus = ey2_aus - ey_aus**2

sd_y_aus = cmath.sqrt(var_y_aus)
exy_aus = sum(x_aus_train*y_aus_train)/x_aus_train.size

cov_aus = exy_aus - ex_aus*ey_aus
corr_aus = (cov_aus/(sd_x_aus*sd_y_aus)).real
beta_aus = ((corr_aus*sd_y_aus)/sd_x_aus).real

print("Coefficient: \n", beta_aus)
alpha_aus = ey_aus - beta_aus*ex_aus

print("Intercept: \n", alpha_aus)
response_aus = alpha_aus + beta_aus*x_aus_test

print("Predicted response: \n", np.round(response_aus))
error_aus = y_aus_test - response_aus

print("Error: \n", np.round(error_aus))
std_error_aus = cmath.sqrt(sum(error_aus**2)/error_aus.size)

print("Standard error: \n", np.round(std_error_aus.real))
x_aus_skt = aus_train_data[:,0].reshape(-1,1)

y_aus_skt = aus_train_data[:,1].reshape(-1,1) 

x_aus_sktest = aus_test_data[:,0].reshape(-1,1)
# Create linear regression object



linear_aus = linear_model.LinearRegression()



# Train the model using the training sets and check score



linear_aus.fit(x_aus_skt, y_aus_skt)

linear_aus.score(x_aus_skt, y_aus_skt)



#Equation coefficient and Intercept



print('Coefficient: \n', linear_aus.coef_)

print('Intercept: \n', linear_aus.intercept_)



#Predict Output



predicted_aus = linear_aus.predict(x_aus_sktest)

predicted_aus = np.round(predicted_aus)
predicted_aus
plt.title("Medals Scored by Australia in the event of Athletics at Olympics")

plt.scatter(x_aus,y_aus,c="mediumaquamarine")

plt.ylabel("Number of Medals")

plt.xlabel("Year")

x_reg_aus = np.array([min(x_aus),max(x_aus)])

f = lambda x: alpha_aus + beta_aus*x 

plt.plot(x_reg_aus,f(x_reg_aus),c="lightcoral")

plt.yticks(np.arange(0,24,4).tolist())

plt.xticks(np.arange(1896,2016,16).tolist())

plt.grid(True)

plt.show()
plt.title("US vs Australia")

us = plt.scatter(x_us,y_us,c="dodgerblue")

plt.ylabel("Number of Medals")

plt.xlabel("Year")

aus = plt.scatter(x_aus,y_aus,c="mediumaquamarine")

plt.ylabel("Number of Medals")

plt.xlabel("Year")

plt.yticks(np.arange(0,90,10).tolist())

plt.xticks(np.arange(1896,2016,16).tolist())

plt.legend([us, aus], ["US", "Australia"])

plt.grid(True)

plt.show()