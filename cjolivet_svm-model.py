import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn import preprocessing 

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import metrics

from sklearn.linear_model import LinearRegression , LogisticRegression , RidgeCV, Ridge

from sklearn import svm



data=pd.read_csv("../input/energydata_complete.csv",index_col='date', parse_dates=True)# data in index



#Copy of the data

data_copy=data.copy()



#plt.plot(data_copy.index,data_copy['Appliances'])



#Supression useless data

data_copy=data_copy.drop(columns=['rv1','rv2'])#feature



#Cleaning data -> count the missing values

    #missing_data = data_copy.isnull()

#for column in missing_data.columns.values.tolist():

#    print(column)

#    print (missing_data[column].value_counts())

#    print("") 



## Correlation

print('Correlation with T9'.center(50))

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

########################## SVM #####################################

print('SVM'.center(50))



svm_reg = svm.SVR(gamma=0.4 , kernel= 'rbf', C= 10000,  epsilon =10)

svm_reg.fit(X_train, Y_train['Appliances'])



Y_predic=svm_reg.predict(X_train)

print('Prediction:',Y_predic[0:5])

print('Y_train:',list (Y_train['Appliances'])[0:5])



Rsq_train=svm_reg.score(X_train, Y_train)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list(Y_train['Appliances'])-Y_predic)**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train['Appliances'], Y_predic)) 



#Residuals

residual= Y_train['Appliances']-Y_predic[0]

plt.scatter(Y_train['Appliances'],residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances energy consumption (Wh)")

plt.ylabel("Residuals")

plt.show()
########################## SVM #####################################

print('SVM + gridSearchCV'.center(50))



grid = GridSearchCV(

        estimator=svm.SVR(kernel='rbf'),

        param_grid={

            'C': [0.1, 1, 100],

            'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],

           'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]}, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)



grid_result = grid.fit(X_train, Y_train)

best_hyp = grid_result.best_params_





svm_reg = svm.SVR(kernel='rbf', C=best_hyp["C"], epsilon=best_hyp["epsilon"], gamma=best_hyp["gamma"])



svm_reg.fit(X_train, Y_train['Appliances'])



Y_predic=svm_reg.predict(X_train)

print('Prediction:',Y_predic[0:5])

print('Y_train:',list (Y_train['Appliances'])[0:5])



Rsq_train=svm_reg.score(X_train, Y_train)

print('Rsquared train:',Rsq_train)



rmse_train = np.sqrt(np.mean((list(Y_train['Appliances'])-Y_predic)**2))

print('RMSE train:',rmse_train)



print('MAE train:', metrics.mean_absolute_error(Y_train['Appliances'], Y_predic)) 



#Residuals

residual= Y_train['Appliances']-Y_predic[0]

plt.scatter(Y_train['Appliances'],residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances energy consumption (Wh)")

plt.ylabel("Residuals")

plt.show()
########################## SVM TestSet #####################################

print('SVM TestSet'.center(50))



svm_reg = svm.SVR(kernel= 'rbf', C= 1000, epsilon=10, gamma=0.4)

svm_reg.fit(X_train, Y_train['Appliances'])



Y_predic=svm_reg.predict(X_test)

print('Prediction:',Y_predic[0:5])

print('Y_train:',list (Y_test['Appliances'])[0:5])



Rsq_test=svm_reg.score(X_test, Y_test)

print('Rsquared test:',Rsq_test)



rmse_test = np.sqrt(np.mean((list(Y_test['Appliances'])-Y_predic)**2))

print('RMSE test:',rmse_test)



print('MAE test:', metrics.mean_absolute_error(Y_test['Appliances'], Y_predic)) 



#Residuals

residual= Y_test['Appliances']-Y_predic[0]

plt.scatter(Y_test['Appliances'],residual,facecolors='none', edgecolors='r')

plt.title('Residuals')

plt.xlabel("Appliances energy consumption (Wh)")

plt.ylabel("Residuals")

plt.show()