import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # Train-test split
from sklearn.preprocessing import MinMaxScaler # Scaling data
from sklearn.feature_selection import SelectKBest, f_regression # N° variables
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore') # Ignoring sklearn warnings

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sns.set()
dataset = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/audi.csv')
dataset.head()
dataset.describe().T
sns.distplot(dataset['year'], bins = 10, color = 'orange', label = 'KDE')
plt.legend()
plt.gcf().set_size_inches(12, 5)
print(dataset.model.unique())
print('--'* 50)

print(dataset.transmission.unique())
print('--'* 50)

print(dataset.fuelType.unique())
print('--'* 50)
fig, ax =plt.subplots(1,2, sharey = True)
plt.gcf().set_size_inches(12, 5)
sns.countplot(dataset['fuelType'], ax = ax[0])
sns.countplot(dataset['transmission'], ax = ax[1])
plt.show()
Model = pd.DataFrame(dataset['model'].value_counts())
sns.barplot(x = Model.index, y = Model['model'])

labels = Model.index.tolist()
plt.gcf().set_size_inches(15, 7)

plt.title('Models vs quantity', fontsize = 20)
plt.xlabel('Model', fontsize = 15)

plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] , labels = labels, rotation = 'vertical')
plt.show()
sns.heatmap(dataset.corr(), annot = True, linewidths=.5, cmap='cubehelix')
plt.title('Correlation', fontsize = 20)
plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

plt.gcf().set_size_inches(15, 7)
ax1.plot(dataset.mileage, dataset.price, c = 'green')
ax1.set_title('Mileage vs. Price', c = 'green', fontsize = 25)
ax2.scatter(dataset.engineSize, dataset.price, c='red')
ax2.set_title('Engine size vs. Price', c ='red', fontsize = 25)

plt.ylabel('Price', fontsize = 25)

plt.show()
dataset2 = dataset.copy()
dataset2 = dataset2[['model','year','transmission','mileage','fuelType','tax','mpg','engineSize','price']]
dataset2.head(3)
data_audi_D = pd.get_dummies(dataset2)
data_audi_D.head(3)
data_audi_D = data_audi_D.drop(['model_ A1', 'transmission_Automatic', 'fuelType_Diesel'], axis=1)
MinMaxScaler = MinMaxScaler() 
data_audi_D_Scaled = MinMaxScaler.fit_transform(data_audi_D)
data_audi_D_Scaled = pd.DataFrame(data_audi_D_Scaled, columns = data_audi_D.columns)
data_audi_D_Scaled.head(3)
X_train, X_test, y_train, y_test = train_test_split(data_audi_D_Scaled.drop(columns = ['price']),
                                                    data_audi_D_Scaled[['price']],
                                                    test_size = 0.2, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
column_names = data_audi_D_Scaled.drop(columns = ['price']).columns

no_of_features = []
r_squared_train = []
r_squared_test = []

for k in range(3, 35, 2): # From 3 to 35 variables (every single one)
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train, y_train)
    X_test_transformed = selector.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared_train.append(regressor.score(X_train_transformed, y_train))
    r_squared_test.append(regressor.score(X_test_transformed, y_test))
    
sns.lineplot(x = no_of_features, y = r_squared_train, legend = 'full')
sns.lineplot(x = no_of_features, y = r_squared_test, legend = 'full')
plt.show()
# Doing the same as above but only with k = 27

selector = SelectKBest(f_regression, k = 27)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)
column_names[selector.get_support()]
def regression_model(model):
    """
    Will fit the regression model passed and will return the regressor object and the score
    """
    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_test_transformed, y_test) # R2
    return regressor, score
model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])

models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index=True)

model_performance
poly = PolynomialFeatures()
X_train_transformed_poly = poly.fit_transform(X_train)
X_test_transformed_poly = poly.transform(X_test)

print(X_train_transformed_poly.shape)
no_of_features = []
r_squared = []

for k in range(10, 400, 5): # Seeing what happens up to 400 variables
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train_transformed_poly, y_train)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared.append(regressor.score(X_train_transformed, y_train))
    
sns.lineplot(x = no_of_features, y = r_squared)
plt.show()
selector = SelectKBest(f_regression, k = 250)

X_train_transformed = selector.fit_transform(X_train_transformed_poly, y_train)
X_test_transformed = selector.transform(X_test_transformed_poly)

models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Polynomial","Model": model, "Score": score}, ignore_index=True)

model_performance
regressor_final = RandomForestRegressor(n_estimators = 1000, random_state = 42)
regressor.fit(X_train_transformed_poly, y_train)

y_pred = regressor.predict(X_test_transformed_poly)
y_test = y_test.reset_index() # To join the Dataframes


y_pred_df = pd.DataFrame({'Price_prediction': y_pred.flatten()})
Comparison = y_test.join(y_pred_df) 
Comparison = Comparison.drop(['index'], axis=1)
Comparison.head()
Fifty_comparison = Comparison.head(50)
Fifty_comparison.plot(kind = 'bar', figsize=(20,15))
plt.grid(which = 'both', linestyle = '-', linewidth = '0.5', color = 'green')
plt.show()