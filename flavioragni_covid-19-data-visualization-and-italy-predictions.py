#Import libraries

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
#Load covid-19 dataset

covid = pd.read_csv('../input/covid_19_data.csv')

covid.head()
#Check for missing data

covid.isnull().sum().to_frame('nulls')
#Province/state has a lot of missing values, so I can discard it

covid.drop(['Province/State', 'SNo'], axis = 1, inplace=True)

covid.head()
#Convert Observation Date column to date type to be able to operate on it

covid['ObservationDate'] = pd.to_datetime(covid['ObservationDate'], format='%m/%d/%Y')

covid.head()
#Plot summary of all countries

world_df = covid[covid['ObservationDate'] == '2020-03-18'].groupby(['Country/Region']).sum().sort_values(by=['Confirmed'], ascending=False)

world_df.head(60).style.background_gradient(cmap='Reds')
#Create dataset for China

china_df = covid[covid['Country/Region'] == 'Mainland China']

#Create a variable that express time as days since outbreak

outbreak_china = pd.Timestamp('2020-01-22')

china_df['timeSince'] = (china_df['ObservationDate'] - outbreak_china).dt.days

#china_df.head()

china_plot = china_df.groupby(['timeSince']).sum()

china_plot.head()

#Create dataset for Italy

italy_df = covid[covid['Country/Region'] == 'Italy']

italy_df.head()

#Create a variable that express time as days since outbreak

outbreak_italy = pd.Timestamp('2020-01-31')

italy_df['timeSince'] = (italy_df['ObservationDate'] - outbreak_italy).dt.days

italy_plot = italy_df.groupby(['timeSince']).sum()

italy_plot.head()
#Plot a comparison of two countries - Confirmed cases

ax = plt.figure(figsize=(15,10))

ax = sns.lineplot(data=china_plot['Confirmed'], label = 'China')

ax = sns.lineplot(data=italy_plot['Confirmed'], label = 'Italy')

ax.set_xlim([0, 60])

ax.set(xlabel='Days since 1st case', ylabel='Number of confirmed cases')

plt.legend()
#Plot a comparison of two countries - Deaths

ax = plt.figure(figsize=(15,10))

ax = sns.lineplot(data=china_plot['Deaths'], label = 'China')

ax = sns.lineplot(data=italy_plot['Deaths'], label = 'Italy')

ax.set_xlim([0, 60])

ax.set(xlabel='Days since 1st case', ylabel='Number of deaths')

plt.legend()
#Plot a comparison of two countries - Recovered

ax = plt.figure(figsize=(15,10))

ax = sns.lineplot(data=china_plot['Recovered'], label = 'China')

ax = sns.lineplot(data=italy_plot['Recovered'], label = 'Italy')

ax.set_xlim([0, 60])

ax.set(xlabel='Days since 1st case', ylabel='Number of recovered patients')

plt.legend()
#Fit a linear regression model to predict spreading of COVID-19 in italy

#Define X (indep var) = days since outbreak and Y (dep var) = confirmed cases

x_data = np.array(italy_df['timeSince']).reshape(-1, 1) 

y_data = np.array(italy_df['Confirmed']).reshape(-1, 1) 

#Divide in training and test set

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.4,random_state=0)

x_cv, x_test, y_cv, y_test = train_test_split(x_test, y_test, test_size=0.5,random_state=0)
#Find best degree of polynomial for the model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

J_poly_train=[]

J_poly_cv=[]

m_train = len(x_train)

m_cv = len(x_cv)

for i in range(1,11):

    poly = PolynomialFeatures(degree=i)

    x_train_poly = poly.fit_transform(x_train)

    #Train the model with polynomial features

    linReg_italy_poly = Ridge(alpha=0)

    linReg_italy_poly.fit(x_train_poly, y_train)

    #Predict on train set

    y_train_pred_poly = linReg_italy_poly.predict(x_train_poly)

    sq_err_train_poly = (y_train_pred_poly-y_train)**2

    J_poly_train.append(np.sum(sq_err_train_poly)/(2*m_train))

    #Before prediction you need to transform test_data to same polynomial degree

    x_cv_poly = poly.fit_transform(x_cv)

    #Now predict

    y_cv_pred_poly = linReg_italy_poly.predict(x_cv_poly)

    #Compute cost

    sq_err_cv_poly = (y_cv_pred_poly-y_cv)**2

    J_poly_cv.append(np.sum(sq_err_cv_poly)/(2*m_cv))
#Plot the cost of training and cv set to find best poly degree

plt.plot(range(1,11), J_poly_train, color='red')

plt.plot(range(1,11), J_poly_cv, color='blue')

plt.title("Cost of the model as a function of polynomial degree")
#Now find best lambda for the model

J_poly_train=[]

J_poly_cv=[]

for i in range(1,20001,1):

    poly = PolynomialFeatures(degree=4) 

    x_train_poly = poly.fit_transform(x_train)

    #Train the model with polynomial features

    linReg_italy_poly = Ridge(alpha=i)

    linReg_italy_poly.fit(x_train_poly, y_train)

    #Predict on train set

    y_train_pred_poly = linReg_italy_poly.predict(x_train_poly)

    sq_err_train_poly = (y_train_pred_poly-y_train)**2

    J_poly_train.append(np.sum(sq_err_train_poly)/(2*m_train))

    #Before prediction you need to transform test_data to same polynomial degree

    x_cv_poly = poly.fit_transform(x_cv)

    #Now predict

    y_cv_pred_poly = linReg_italy_poly.predict(x_cv_poly)

    #Compute cost

    sq_err_cv_poly = (y_cv_pred_poly-y_cv)**2

    J_poly_cv.append(np.sum(sq_err_cv_poly)/(2*m_cv))
#Plot the cost of training and cv set to find best poly degree

#Find index of lowest cost for J_poly

best_alpha=J_poly_cv.index(min(J_poly_cv))

plt.plot(range(1,20001,1), J_poly_cv, color='blue')

plt.plot([best_alpha,best_alpha],[0,180000],color='red')

plt.title("Best regularization alpha value for Ridge regression")

print("Best alpha value: %d" % best_alpha)
#Chosen parameters: poly=4, lambda=30

#Compute J on test set

poly = PolynomialFeatures(degree=4) 

x_train_poly = poly.fit_transform(x_train)

#Train the model with polynomial features

linReg_italy_poly = Ridge(alpha=best_alpha)

linReg_italy_poly.fit(x_train_poly, y_train)

#Predict on test set

x_test_poly = poly.fit_transform(x_test)

y_test_pred_poly = linReg_italy_poly.predict(x_test_poly)

sq_err_test_poly = (y_test_pred_poly-y_test)**2

J_poly_test=np.sum(sq_err_test_poly)/(2*m_cv)

#Compute r2 score

from sklearn.metrics import r2_score

r2score=r2_score(y_test, y_test_pred_poly)

print('Cost for linear regression model on test set: %d \nR2 score: %.4f' % (J_poly_test, r2score))
#Plot the model

plt.scatter(x_data,y_data,color='red')

#Plot predictions for all dataset

y_data_pred = linReg_italy_poly.predict(poly.fit_transform(x_data))

x_data_ord, y_data_pred_ord = zip(*sorted(zip(x_data, y_data_pred)))

plt.plot(x_data_ord, y_data_pred_ord, color='blue')

plt.title("Polynomial Regression Model ")
#Predict number of confirmed cases at day 60

to_predict = np.array(60)

print('Number of cases on March 22nd: 41035\nPrediction linear regression model for March 22nd: %d' % linReg_italy_poly.predict(poly.fit_transform(to_predict.reshape(1, -1))))
#Luckyly quarantine measure seems to have worked. 

#The diffusion of the disease deviated from the linear increasing trend predicted by the model.

#Hopefully the spread will reach a plateau soon!