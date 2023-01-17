import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn import preprocessing 

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression , LogisticRegression



data=pd.read_csv("../input/energydata_complete.csv",index_col='date', parse_dates=True)# data in index

#data['date'] =pd.to_datetime(data.date)# convert in datetime



#Copy of the data

data_copy=data.copy()



#Supression useless data



data_copy.head()



#plt.plot(data_copy.index,data_copy['Appliances'])





data_copy=data_copy.drop(columns=['rv1','rv2'])#feature



#Cleaning data -> count the missing values

    #missing_data = data_copy.isnull()

#for column in missing_data.columns.values.tolist():

#    print(column)

#    print (missing_data[column].value_counts())

#    print("") 



## Correlation

print('Correlation'.center(50))

corr=data_copy.corr()

print(corr.Appliances)

f, ax = plt.subplots(figsize=(7, 7))

sb.heatmap(corr, square=False)

plt.show()



#Scalling data

scaler=preprocessing.StandardScaler()



X=data_copy.drop(columns=['Appliances'])#features

Y=data_copy[['Appliances']]#target



X=X.astype(float)#transform int in float



colnames=list(X)

idxnames=X.index

X=scaler.fit_transform(X) # apply the standardization

X=pd.DataFrame(X, columns=colnames, index=idxnames)



#Creation TestSet and TrainSet

    #divide into 2 datasets: 3 months = training / 1.5 months = testing

#X_train=X[0:2161]

#Y_train=Y[0:2161]

#X_test=X[2161:len(data_copy)]

#Y_test=Y[2161:len(data_copy)]







X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)



############################# Multiple Linear Regression ##################################

print(' Multiple Linear Regression '.center(50))



lin_reg = LinearRegression()

lin_reg.fit(X_train, Y_train['Appliances'])# X=predictor variables / Y_train response variable 



print('Intercept: \n', lin_reg.intercept_)



print('Coefficients: \n', lin_reg.coef_) # Regression coefficient 



#accuracy of the model

Y_predic = lin_reg.predict(X_train)

print('Prediction:',Y_predic[0:5])

print('Y_train:',list (Y_train['Appliances'])[0:5])





#train

#Rsq_train=lin_reg.score(X_train,Y_train['Appliances'])

Rsq_train= 1- np.sum((Y_train['Appliances']-Y_predic)**2)/np.sum((Y_train['Appliances']-Y_train['Appliances'].mean())**2)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list(Y_train['Appliances'])-Y_predic)**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train['Appliances'], Y_predic)) 



#Prediction model and Y_train

plt.figure(figsize=(20,8))

plt.title('Predictive model and Y_train')

plt.xlabel('Dates')

plt.ylabel('Appliances energy consumption (Wh)')

plt.plot(X_train.index,Y_train['Appliances'],label='Y_train')

plt.plot(X_train.index, Y_predic, c='r', label='Prediction')

plt.legend()

plt.show()



#Residuals

residual= Y_train['Appliances']-Y_predic

plt.scatter(Y_train['Appliances'],residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances energy consumption (Wh)")

plt.ylabel("Residuals")

plt.show()





def normalEq(X,Y):

    theta_norm = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    return theta_norm



def gradDescent(X,Y,alpha,nb_it):

    theta_grad = np.random.randn(len(X.T),1)

    for i in range(nb_it):

        pred=np.dot(X,theta_grad)

        loss=pred-Y

        theta_grad=np.dot(X.T,loss)/len(X)*alpha

    return theta_grad



########################## The Normal equation #####################################

print('The Normal equation'.center(50))



Y_predic = X_train.dot(normalEq(X_train,Y_train))



print('Prediction:',list(Y_predic[0])[0:5])

print('Y_train:',list (Y_train['Appliances'])[0:5])



Rsq_train= 1- np.sum((Y_train['Appliances']-Y_predic[0])**2)/np.sum((Y_train['Appliances']-Y_train['Appliances'].mean())**2)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list(Y_train['Appliances'])-Y_predic[0])**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train['Appliances'], Y_predic)) 



#Residuals

residual= Y_train['Appliances']-Y_predic[0]

plt.scatter(Y_train['Appliances'],residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances energy consumption (Wh)")

plt.ylabel("Residuals")

plt.show()



########################## The Gradient Descent #####################################

print('The Gradient Descent'.center(50))

alpha=0.1



Y_predic = X_train.dot(gradDescent(X_train,Y_train,alpha,5))



print('Prediction:',list(Y_predic[0])[0:5])

print('Y_train:',list (Y_train['Appliances'])[0:5])



Rsq_train= 1- np.sum((Y_train['Appliances']-Y_predic[0])**2)/np.sum((Y_train['Appliances']-Y_train['Appliances'].mean())**2)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list(Y_train['Appliances'])-Y_predic[0])**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train['Appliances'], Y_predic)) 



#Residuals

residual= Y_train['Appliances']-Y_predic[0]

plt.scatter(Y_train['Appliances'],residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances energy consumption (Wh)")

plt.ylabel("Residuals")

plt.show()



############################# Polynomial Regression ##################################

print(' Multiple Polynomial regression'.center(50))



poly = preprocessing.PolynomialFeatures(degree=4)

X_poly = poly.fit_transform(X_train)

poly_reg= LinearRegression()

poly_reg.fit(X_poly, Y_train)



print('Intercept: \n', poly_reg.intercept_) 



print('Coefficients: \n', poly_reg.coef_) # Regression coefficient 



#accuracy of the model

Y_predic_poly = poly_reg.predict(X_poly)

print('Prediction:',Y_predic_poly[0:5])

print('Y_train:',list (Y_train['Appliances'])[0:5])



#train

Rsq_train=metrics.r2_score(Y_train,Y_predic_poly)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list(Y_train['Appliances'])-Y_predic_poly)**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train, Y_predic_poly)) 



#Prediction model and Y_train

plt.figure(figsize=(20,8))

plt.title('Predictive model and Y_train')

plt.xlabel('Dates')

plt.ylabel('Appliances energy consumption (Wh)')

plt.plot(X_train.index,Y_train['Appliances'],label='Y_train')

plt.plot(X_train.index, Y_predic_poly, c='r', label='Prediction')

plt.legend()

plt.show()



#Residuals

residual= Y_train-Y_predic_poly

plt.scatter(Y_train,residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances Energy consumption")

plt.ylabel("Residuals")

plt.show()









############################# Polynomial Regression ##################################

print(' Evolution of Squared'.center(50))

plt.figure(figsize=(10,8))

plt.title('Evolution RSquared in function of degree')

plt.xlabel('Degree')

plt.ylabel('RSquared')

for x in range(1,5):

    poly = preprocessing.PolynomialFeatures(degree=x)

    X_poly = poly.fit_transform(X_train)

    poly_reg= LinearRegression()

    poly_reg.fit(X_poly, Y_train)

    Y_predic_poly = poly_reg.predict(X_poly)

    Rsq_train=metrics.r2_score(Y_train,Y_predic_poly)

    plt.scatter(x,Rsq_train, s=70)

    

plt.show()
print(' Multiple Polynomial regression with TestSet'.center(50))



poly = preprocessing.PolynomialFeatures(degree=4)

X_poly = poly.fit_transform(X_train)

poly_reg= LinearRegression()

poly_reg.fit(X_poly, Y_train)



print('Intercept: \n', poly_reg.intercept_) 



print('Coefficients: \n', poly_reg.coef_) # Regression coefficient 





X_poly_test = poly.fit_transform(X_test)# transform X_test to higher degree features



#accuracy of the model

Y_predic_poly_test = poly_reg.predict(X_poly_test)

print('Prediction:',Y_predic_poly_test[0:5])

print('Y_test:',list (Y_test['Appliances'])[0:5])





#Model performance

Rsq_test=metrics.r2_score(Y_test['Appliances'],Y_predic_poly_test)

print('Rsquared test:',Rsq_test)



rmse_test = np.sqrt(np.mean((list(Y_test['Appliances'])-Y_predic_poly_test)**2))

print('RMSE test:',rmse_test)



print('MAE test:', metrics.mean_absolute_error(Y_test['Appliances'], Y_predic_poly_test)) 



#Prediction model and Y_test

plt.figure(figsize=(20,8))

plt.title('Predictive model and Y_test')

plt.xlabel('Dates')

plt.ylabel('Appliances energy consumption (Wh)')

plt.plot(X_test.index,Y_test['Appliances'],label='Y_test')

plt.plot(X_test.index, Y_predic_poly_test, c='r', label='Prediction')

plt.legend()

plt.show()







#Residuals

residual= Y_test-Y_predic_poly_test

plt.scatter(Y_test,residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances Energy consumption")

plt.ylabel("Residuals")

plt.show()
############################# Simple Linear Regression ##################################

print(' Simple Linear Regression '.center(50))

lin_reg = LinearRegression()



     

 #####Lights 

   

print('***Lights***'.center(20))

lin_reg .fit(X_train[['lights']],Y_train)



print('Intercept: \n', lin_reg .intercept_) # Expected mean value of Y when all X=0





print('Coefficient: \n', lin_reg .coef_) # Regression coefficient 



#accuracy of the model

Y_predic = lin_reg .predict(X_train[['lights']])

print('Prediction:',Y_predic[0:5])

print('Y_train:', list(Y_train['Appliances'])[0:5])





Rsq_train=lin_reg .score(X_train[['lights']],Y_train)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list (Y_train['Appliances'])-Y_predic)**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train, Y_predic)) 



plt.plot(X_train['lights'], Y_predic, label='Y_predic')

plt.scatter(X_train['lights'], Y_train, color='red', s=1, label='Y_train')

plt.title('Appliances in function of lights')

plt.xlabel('Lights (Wh)')

plt.ylabel('Appliances(Wh)')

plt.legend()

plt.show()

 

#Residuals

residual= Y_train-Y_predic



plt.scatter(Y_train,residual, facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances Energy consumption")

plt.ylabel("Residuals")

plt.show()



 ##### RH_out 

   

print('*** RH_out ***'.center(20))

lin_reg .fit(X_train[['RH_out']],Y_train)



print('Intercept: \n', lin_reg .intercept_) 





print('Coefficient: \n', lin_reg .coef_) # Regression coefficient 



#accuracy of the model

Y_predic = lin_reg.predict(X_train[['RH_out']])

print('Prediction:',Y_predic[0:5])

print('Y_train:', list(Y_train['Appliances'])[0:5])





Rsq_train=lin_reg.score(X_train[['RH_out']],Y_train)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list (Y_train['Appliances'])-Y_predic)**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train, Y_predic)) 



plt.plot(X_train['RH_out'], Y_predic, label='Y_predic')

plt.scatter(X_train['RH_out'], Y_train, color='red', s=1, label='Y_train')

plt.title('Appliances in function of RH_out ')

plt.xlabel('RH_out  (%)')

plt.ylabel('Appliances(Wh)')

plt.legend()

plt.show()

            

#Residuals

residual= Y_train-Y_predic



plt.scatter(Y_train,residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances Energy consumption")

plt.ylabel("Residuals")







############################# Simple Linear Regression ##################################

print(' Simple Polynomial Regression '.center(50))

poly = preprocessing.PolynomialFeatures(degree=2)

poly_reg= LinearRegression()



#####Lights



X_poly = poly.fit_transform(X_train[['lights']])

poly_reg= LinearRegression()



print('***Lights***'.center(20))

poly_reg.fit(X_poly,Y_train)



print('Intercept: \n', poly_reg.intercept_) 



print('Coefficient: \n', poly_reg.coef_) # Regression coefficient 



#accuracy of the model

Y_predic_poly = poly_reg.predict(X_poly)

print('Prediction:',Y_predic_poly[0:5])

print('Y_train:', list(Y_train['Appliances'])[0:5])





Rsq_train=metrics.r2_score(Y_train,Y_predic_poly)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list (Y_train['Appliances'])-Y_predic_poly)**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train, Y_predic_poly)) 





plt.plot(X_train['lights'], Y_predic_poly, label='Y_predic')

plt.scatter(X_train['lights'], Y_train, color='red', s=1, label='Y_train')

plt.title('Appliances in function of Lights ')

plt.xlabel('Lights  (Wh)')

plt.ylabel('Appliances(Wh)')

plt.legend()

plt.show()



#Residuals

residual= Y_train-Y_predic_poly



plt.scatter(Y_train,residual, facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances Energy consumption")

plt.ylabel("Residuals")

plt.show()



#####RH_out



X_poly = poly.fit_transform(X_train[['RH_out']])

poly_reg= LinearRegression()



print('***RH_out***'.center(20))

poly_reg.fit(X_poly,Y_train)



print('Intercept: \n', poly_reg.intercept_) 



print('Coefficient: \n', poly_reg.coef_) # Regression coefficient 



#accuracy of the model

Y_predic_poly = poly_reg.predict(X_poly)

print('Prediction:',Y_predic_poly[0:5])

print('Y_train:', list(Y_train['Appliances'])[0:5])





Rsq_train=metrics.r2_score(Y_train,Y_predic_poly)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list (Y_train['Appliances'])-Y_predic_poly)**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train, Y_predic_poly)) 



plt.plot(X_train['RH_out'], Y_predic_poly, label='Y_predic')

plt.scatter(X_train['RH_out'], Y_train, color='red', s=1, label='Y_train')

plt.title('Appliances in function of RH_out ')

plt.xlabel('RH_out(%)')

plt.ylabel('Appliances(Wh)')

plt.legend()

plt.show()



#Residuals

residual= Y_train-Y_predic_poly



plt.scatter(Y_train,residual, facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances Energy consumption")

plt.ylabel("Residuals")

plt.show()


