import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
input_data = pd.read_csv("../input/US_graduate_schools_admission_parameters_dataset.csv")
input_data.head()
X = input_data[['GRE Score', 'TOEFL Score']]
y= input_data['Change of Admit']
X=sm.add_constant(X)
multiple_linear_regression_model= sm.OLS(y,x)
multiple_linear_regression_model_fit=multiple_linear_regression_model
