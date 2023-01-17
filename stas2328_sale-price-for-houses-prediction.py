import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import seaborn as sns
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
def displayCorrMatrix():
    sns.set(font_scale=0.6)
    correlation_train = train.corr()
    correlation_train.drop('Id', axis=1, inplace=True)
    correlation_train.drop(['Id'], inplace=True)
    plt.figure(figsize=(17, 17))
    sns.heatmap(correlation_train,
                annot=True,
                fmt='.1f',
                cmap='coolwarm',
                linewidths=0.2,
                cbar=True,
                cbar_kws={"shrink": 1.0, "aspect": 40})
    plt.draw()
    sns.reset_orig()

displayCorrMatrix()
variablesForPrediction = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath','TotRmsAbvGrd','YearBuilt', 'YearRemodAdd', 'Fireplaces', 'BsmtFinSF1']
train[variablesForPrediction].isnull().any()
def predictSalePrice(features, dataset):
    y = dataset.SalePrice.values
    my_imputer = SimpleImputer()
    X = dataset[features]
    # Dividimos datos en 80% para entrenamiento 20% para pruebas y realizamos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    model1 = LinearRegression()
    model1.fit(X_train, y_train)

    y_pred = model1.predict(X_test)

    X_test = X_test.assign(SalePricePredicted=y_pred)
    X_test = X_test.assign(SalePrice=y_test)
    X_test.index.name = 'Id'
    X_test = X_test.sort_values('Id')
    print('Error Cuadratico Medio:', metrics.mean_squared_error(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(19,7))
    X_test.plot(ax=ax, y=['SalePricePredicted', 'SalePrice'])
    plt.xlabel("Id")
    plt.ylabel("SalePrice")
    plt.draw()
predictSalePrice(variablesForPrediction, train)
train.FullBath = np.power(train.FullBath, 10)  # fullbath = fullbath^10
predictSalePrice(variablesForPrediction, train)
plt.show()