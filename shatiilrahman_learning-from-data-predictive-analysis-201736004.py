import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
input_data = pd.read_csv("../input/US_graduate_schools_admission_parameters_dataset.csv")
input_data.head()
x = input_data[['GRE Score','TOEFL Score']]
y = input_data['Chance of Admit ']
x = sm.add_constant(x)
multiple_linear_regression_model = sm.OLS(y,x) # OLS =ordinary least square
multiple_linear_regression_model_fit = multiple_linear_regression_model.fit()
print(multiple_linear_regression_model_fit.params)
x1 = input_data['GRE Score']
y = input_data['Chance of Admit ']
plt.scatter(x1, y)
plt.show()
multiple_linear_regression_model_fit.summary()