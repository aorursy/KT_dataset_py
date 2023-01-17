# enhances numpy by allowing the organization of data in a tabular form with descriptive column and row labels.

import pandas as pd

# built on NumPy and SciPy and integrated with Pandas. Provides summaries to allow greater understanding of sklearn.

import statsmodels.api as sm

# two dimensional plotting library designed for visualization of NumPy computations.

class Linear_Regression():
    
    file_name = ''
    file_price = ''
    file_factors = ''
    regression_result = ''
    
    def get_result(self):
        file_factors_constant = sm.add_constant(self.file_factors)
        self.regression_result = sm.OLS(self.file_price, file_factors_constant).fit()
        print(self.regression_result.summary())
        print("Price = " + str(self.regression_result.params[1]) +  " * Size + " + str(self.regression_result.params[2]) +  " * Year + " + str(self.regression_result.params[0]))
    
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_price = file_data['price']
        self.file_factors = file_data[['size','year']]

    def __init__(self):
        
        pass


def perform_regression():    
    
    linear_regression = Linear_Regression()
    linear_regression.file_name = '../input/real-estate-price-size-year/real_estate_price_size_year.csv'
    linear_regression.load_file()    
    linear_regression.get_result()
    
perform_regression()    


class Linear_Regression():
    
    file_name = ''
    file_price = ''
    file_factors = ''
    regression_result = ''
    
    def get_result(self):
        file_factors_constant = sm.add_constant(self.file_factors)
        self.regression_result = sm.OLS(self.file_price, file_factors_constant).fit()
        print(self.regression_result.summary())
        print("Price = " + str(self.regression_result.params[1]) +  " * Size + " + str(self.regression_result.params[0]))
    
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_price = file_data['price']
        self.file_factors = file_data[['size']]

    def __init__(self):
        
        pass


def perform_regression():    
    
    linear_regression = Linear_Regression()
    linear_regression.file_name = '../input/real-estate-price-size-year/real_estate_price_size_year.csv'
    linear_regression.load_file()    
    linear_regression.get_result()
    
perform_regression()    

class Linear_Regression():
    
    file_name = ''
    file_price = ''
    file_factors = ''
    regression_result = ''
    
    def get_result(self):
        file_factors_constant = sm.add_constant(self.file_factors)
        self.regression_result = sm.OLS(self.file_price, file_factors_constant).fit()
        print(self.regression_result.summary())
        print("Price = " + str(self.regression_result.params[1]) +  " * Year + " + str(self.regression_result.params[0]))
    
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_price = file_data['price']
        self.file_factors = file_data[['year']]

    def __init__(self):
        
        pass


def perform_regression():    
    
    linear_regression = Linear_Regression()
    linear_regression.file_name = '../input/real-estate-price-size-year/real_estate_price_size_year.csv'
    linear_regression.load_file()    
    linear_regression.get_result()
    
perform_regression()    

from sklearn.linear_model import LinearRegression

class Linear_Regression():
    
    file_name = ''
    file_price = ''
    file_factors = ''
    regression_result = ''
    
    def get_result(self):
        self.regression_result = LinearRegression()
        self.regression_result.fit(self.file_factors, self.file_price)
        print("Price = " + str(self.regression_result.coef_[0]) + " * Size + " + str(self.regression_result.coef_[1]) + " * Year + " + str(self.regression_result.intercept_)) 
        
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_price = file_data['price']
        self.file_factors = file_data[['size','year']]

    def __init__(self):
        
        pass


def perform_regression():    
    
    linear_regression = Linear_Regression()
    linear_regression.file_name = '../input/real-estate-price-size-year/real_estate_price_size_year.csv'
    linear_regression.load_file()    
    linear_regression.get_result()
    
perform_regression()    

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class Linear_Regression():
    
    file_name = ''
    file_price = ''
    file_factors = ''
    file_factors_scaled = ''
    regression_result = ''
    
    def get_result(self):
        self.regression_result = LinearRegression()
        self.regression_result.fit(self.file_factors_scaled, self.file_price)
        print("Price = " + str(self.regression_result.coef_[0]) + " * Size + " + str(self.regression_result.coef_[1]) + " * Year + " + str(self.regression_result.intercept_)) 
        
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_price = file_data['price']
        self.file_factors = file_data[['size','year']]
        
        scaler = StandardScaler()
        scaler.fit(self.file_factors)        
        self.file_factors_scaled = scaler.transform(self.file_factors)

    def __init__(self):
        
        pass


def perform_regression():    
    
    linear_regression = Linear_Regression()
    linear_regression.file_name = '../input/real-estate-price-size-year/real_estate_price_size_year.csv'
    linear_regression.load_file()    
    linear_regression.get_result()
    
perform_regression()    

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class Linear_Regression():
    
    file_name = ''
    file_price = ''
    file_price_train = ''
    file_price_test = ''
    file_price_predict = ''
    file_factors = ''
    file_factors_scaled = ''
    file_factors_train = ''
    file_factors_test = ''
    regression_result = ''
    
    def get_result(self):
        self.regression_result = LinearRegression()
        self.regression_result.fit(self.file_factors_train, self.file_price_train)
        self.file_price_predict = self.regression_result.predict(self.file_factors_test)
        print("Price = " + str(self.regression_result.coef_[0]) + " * Size + " + str(self.regression_result.coef_[1]) + " * Year + " + str(self.regression_result.intercept_)) 
        print("Correlation: " + str(r2_score(self.file_price_test.values, self.file_price_predict)))
        
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_price = file_data['price']
        self.file_factors = file_data[['size','year']]
        
        scaler = StandardScaler()
        scaler.fit(self.file_factors)        
        self.file_factors_scaled = scaler.transform(self.file_factors)
        
        self.file_price_train, self.file_price_test, self.file_factors_train, self.file_factors_test = train_test_split(self.file_price, self.file_factors_scaled, test_size = 0.1)
        
    def __init__(self):
        
        pass


def perform_regression():    
    
    linear_regression = Linear_Regression()
    linear_regression.file_name = '../input/real-estate-price-size-year/real_estate_price_size_year.csv'
    linear_regression.load_file()    
    linear_regression.get_result()
    
perform_regression()    
