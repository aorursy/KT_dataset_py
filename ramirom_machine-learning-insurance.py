#carga e importe de librerías
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as pl

#carga del dataset "Insurance"
insurance = pd.read_csv("../input/insurance.csv")
insurance.tail(10)
#Procedemos a verificar los valores nulos (missing)
insurance.isnull().sum()
#Transformación variables
smoker = np.unique(insurance['smoker'])
smoker
def map_smoker(smoker):
    if smoker == 'yes':
        return 1
    else:
        return 0
insurance['smoker'] = insurance['smoker'].apply(map_smoker)
insurance.head()
#Transformación variables
sex = np.unique(insurance['sex'])
sex
def map_sex(sex):
    if sex == 'male':
        return 1
    else:
        return 0
insurance['sex'] = insurance['sex'].apply(map_sex)
insurance.head()
#Transformación variables
region = np.unique(insurance['region'])
region
def map_region(region):
    if region == 'northeast':
        return 1
    elif region == 'northwest':
        return 2
    elif region == 'southeast':
        return 3
    elif region == 'southwest':
        return 4
    else:
        return 0
insurance ['region'] = insurance ['region'].apply(map_region)
insurance.head()
#Definición de variables
x = insurance.drop(['charges', 'sex', 'region'], axis = 1)
y = insurance.charges
#Partición del dataset en train y test
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.30, 
                                                    train_size=0.70, random_state =0)

#Definición Random Forest Regressor
forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))
pl.figure(figsize=(10,6))

pl.scatter(forest_train_pred,forest_train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
pl.scatter(forest_test_pred,forest_test_pred - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
pl.xlabel('Predicted values')
pl.ylabel('Tailings')
pl.legend(loc = 'upper left')
pl.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
pl.show()
#Definición de Learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
def learningCurve(model, X_train, y_train, k=10):
    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=model,
                                   X=X_train,
                                   y=y_train,
                                   train_sizes=np.linspace(0.1, 1.0, 10),
                                   cv=k,
                                   n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.rcParams["figure.figsize"] = [6,6]
    fsize=14
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples', fontsize=fsize)
    plt.ylabel('Accuracy', fontsize=fsize)
    plt.legend(loc='lower right')
    plt.ylim([0.4, 1.03])
    plt.tight_layout()
    plt.show()

learningCurve(forest, x_train, y_train)