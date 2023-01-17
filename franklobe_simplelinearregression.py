# enhances numpy by allowing the organization of data in a tabular form with descriptive column and row labels.

import pandas as pd

# built on NumPy and SciPy and integrated with Pandas. Provides summaries to allow greater understanding of sklearn.

import statsmodels.api as sm

# two dimensional plotting library designed for visualization of NumPy computations.

import matplotlib as plt

# visualization library based on matplotlib providing high-level interface for statistical graphics

import seaborn as sns
sns.set()

class Linear_Regression():
    
    file_gpa = ''
    file_name = ''
    file_sat = ''
    regression_result = ''
    
    def get_result(self):
        file_sat_constant = sm.add_constant(self.file_sat)
        self.regression_result = sm.OLS(self.file_gpa, file_sat_constant).fit()
    
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_gpa = file_data['GPA']
        self.file_sat = file_data['SAT']

        
    def plot_data(self):
        
        print(self.regression_result.summary())
                
        plt.pyplot.scatter(self.file_sat, self.file_gpa)
        regression_function = self.regression_result.params[1] * self.file_sat + self.regression_result.params[0]
        regression_line = plt.pyplot.plot(self.file_sat, regression_function, lw = 5, c = 'red', label = 'Regression Line')
        plt.pyplot.xlabel('SAT', fontsize = 20)
        plt.pyplot.ylabel('GPA', fontsize = 20)
        plt.pyplot.show()
        
        print("GPA = " + str(self.regression_result.params[1]) +  " * SAT + " + str(self.regression_result.params[0]))
    
    def __init__(self):
        
        pass


def perform_regression():    
    
    linear_regression = Linear_Regression()
    linear_regression.file_name = '../input/simple-linear-regression/simple_linear_regression.csv'
    linear_regression.load_file()    
    linear_regression.get_result()
    linear_regression.plot_data()

    
perform_regression()    


from sklearn.linear_model import LinearRegression

class Linear_Regression():
    
    file_gpa = ''
    file_name = ''
    file_sat = ''
    file_sat_matrix = ''
    regression_result = ''
    
    def get_result(self):
        self.regression_result = LinearRegression()
        self.file_sat_matrix = self.file_sat.values.reshape(-1,1)
        self.regression_result.fit(self.file_sat_matrix, self.file_gpa)
    
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_gpa = file_data['GPA']
        self.file_sat = file_data['SAT']
        
    def plot_data(self):
        
        plt.pyplot.scatter(self.file_sat, self.file_gpa)
        regression_function = self.regression_result.coef_[0] * self.file_sat + self.regression_result.intercept_
        regression_line = plt.pyplot.plot(self.file_sat, regression_function, lw = 5, c = 'red', label = 'Regression Line')
        plt.pyplot.xlabel('SAT', fontsize = 20)
        plt.pyplot.ylabel('GPA', fontsize = 20)
        plt.pyplot.show()
        
        print("GPA = " + str(self.regression_result.coef_[0]) +  " * SAT + " + str(self.regression_result.intercept_))
    
    def __init__(self):
        
        pass

def perform_regression():    
    
    linear_regression = Linear_Regression()
    linear_regression.file_name = '../input/simple-linear-regression/simple_linear_regression.csv'
    linear_regression.load_file()    
    linear_regression.get_result()
    linear_regression.plot_data()
    
perform_regression()    
