import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
input_data = pd.read_csv("../input/US_graduate_schools_admission_parameters_dataset.csv")
input_data.head()

X1 = input_data['GRE Score']
y = input_data['Chance of Admit ']
plt.scatter(X1, y)
plt.show()

