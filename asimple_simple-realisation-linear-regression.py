import pandas as pd
import numpy as np
from sympy import *
import seaborn as sns
import matplotlib.pyplot as plt
path = {'train': '/kaggle/input/house-prices-advanced-regression-techniques/train.csv', 'test':'/kaggle/input/house-prices-advanced-regression-techniques/train.csv'}
data = pd.read_csv(path['train'])
test_data = pd.read_csv(path['test'])
data.SalePrice.describe()
sns.boxplot(data.SalePrice.values)
plt.title("Распределение целевой переменной - SalePrice\n", fontsize=15)
plt.xlabel('Цена')
plt.ylabel("Кол-ство")
plt.show();
gr_liv_area_data = pd.concat([data['SalePrice'], data['GrLivArea']], axis=1)
gr_liv_area_data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800_000))
total_bsmtSF = pd.concat([data.SalePrice, data.TotalBsmtSF], axis=1)
total_bsmtSF.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800_000))
overallQual = pd.concat([data.SalePrice, data.OverallQual], axis=1)
f, ax = plt.subplots(figsize=(14,8))
fig = sns.boxplot(x="OverallQual", y='SalePrice', data=overallQual)
fig.axis(ymin=0, ymax=800_000)
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
k = 9 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
f, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(data[cols].corr(), vmax=.8, square=True);
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
sns.pairplot(data[cols], size = 4);
total = data.isnull().sum()
total
percent = (total/ data.isnull().count()).sort_values(ascending=False)
percent
missing_data = pd.concat([total.sort_values(ascending=False), percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
x = data['GrLivArea']

y = data['SalePrice']
def feature_scaling(data):
    data_ = (data - data.mean()) / data.std()
    return np.c_[np.ones(data_.shape[0]), data_]
x = feature_scaling(x)
x.shape 
class LinearRegression:
    
    """Реализация линейной регрессии"""
    
    def predict(self, X):
        return np.dot(X, self._W)

    
    def loss(self, y_prediction, y):
        return sum((y_prediction - y)**2) / len(y)
   

    def _step_gradient_descent(self, X, y, prediction, step_learn):
        error = prediction - y
        self._W -= step_learn * 2 * np.dot(X.T, error) / len(X)

        
    def fit(self, X, y, iterations = 100_000, step_learn = .01):
        self._W = np.zeros(X.shape[1])
        self._cost_history = []
        self._w_history = [self._W]
        
        for i in range(iterations):
            
            prediction = self.predict(X)
            cost = self.loss(prediction, y)
            self._cost_history.append(cost)
            self._step_gradient_descent(X, y, prediction, step_learn)
            self._w_history.append(self._W.copy())
        return "=> Обучение прошло успешно <="    
simple_regression = LinearRegression()
simple_regression.fit(x, y , iterations=2000, step_learn=0.01)
simple_regression._W
plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(simple_regression._cost_history)
plt.show()
simple_regression._cost_history[0]
simple_regression.predict([x[0]]) - data.SalePrice[0]
x_m = data[['OverallQual', 'GrLivArea', 'GarageCars']]
x_m = feature_scaling(x_m)
multivariable_regression = LinearRegression()
multivariable_regression.fit(x_m, y, iterations=2000, step_learn=0.01)
multivariable_regression._W
plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(multivariable_regression._cost_history)
plt.show()
multivariable_regression._cost_history[-1]
simple_regression._cost_history[-1]
test_simple = test_data['GrLivArea']
test_multi =  test_data[['OverallQual', 'GrLivArea', 'GarageCars']]
test_simple = feature_scaling(test_simple)
test_multi = feature_scaling(test_multi)
predictions_simple = simple_regression.predict(test_simple)
predictions_simple
predictions_multi = multivariable_regression.predict(test_multi)
predictions_multi
def save(feature, result:pd.DataFrame, name:np.ndarray) -> None:
    output_multi = pd.DataFrame({'Id': feature.Id, 'SalePrice': result})
    output_multi.to_csv(f'{name}_submission.csv', index=False)
    print("=> Your submission was successfully saved! <=")
save(test_data, predictions_simple, 'Simple_regression')
save(test_data, predictions_multi, 'Multi_regression')