import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importar datos de archivo train.csv
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(train['SalePrice'].describe())

figsize_rect = (25, 10)  # dimensiones para graficos en formato rectangular

mpl.rc('figure', max_open_warning=0)
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.color'] = '#1372B2'
colors = ['#1372B2', "#F19917", '#F76413', '#2B6B85', '#359CAE']

correlations = train.corr()['SalePrice'].nlargest(11)
correlations.pop('SalePrice')
print(correlations)


def preditSalePrice(features):
    print(features)
    y = train.SalePrice.values
    X = train[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    print(regressor.intercept_)
    print(regressor.coef_)
    y_pred = regressor.predict(X_test)

    X_test = X_test.assign(SalePricePredicted=y_pred)
    X_test = X_test.assign(SalePrice=y_test)
    X_test = X_test.assign(Difference=np.abs(X_test['SalePrice'] - X_test['SalePricePredicted']))
    X_test = X_test.assign(
        DifferencePerct=np.abs(X_test['SalePrice'] - X_test['SalePricePredicted']) * 100 / X_test['SalePrice'])
    X_test.index.name = 'Id'
    X_test = X_test.sort_values('Id')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Coefficient of determination:', regressor.score(X_train, y_train))

    fig, ax = plt.subplots()
    X_test.plot(ax=ax,y=['SalePricePredicted', 'SalePrice'], figsize=figsize_rect)
    X_test.plot(ax=ax, y='Difference', figsize=figsize_rect)

    plt.xlabel("Id")
    plt.ylabel("SalePrice")
    plt.draw()

print("\n\n")
preditSalePrice(
    ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
     'YearBuilt', 'YearRemodAdd'])

plt.show()
