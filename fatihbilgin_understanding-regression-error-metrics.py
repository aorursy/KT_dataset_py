import numpy as np 

import pandas as pd 

import os



from warnings import filterwarnings

filterwarnings('ignore')
df = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv",sep = ",")



#First column is "Serial No" so i'll pass it. I'll take the first 10 rows...

df = df.iloc[:10,1:]

df



y = df["Chance of Admit "].values

X = df.drop(["Chance of Admit "],axis=1)
#the model and predict

from sklearn.linear_model import LinearRegression



linear_ref = LinearRegression()

model = linear_ref.fit(X, y)



y_head = model.predict(X)
#The metric_table is consisted of real y values and predicts of y values.

metric_table = pd.DataFrame({"y_values": y, "y_heads": y_head})

metric_table
metric_table["errors"] = metric_table["y_values"] - metric_table["y_heads"]

metric_table["errors_sq"] = metric_table["errors"]**2

metric_table
my_MSE = np.mean(metric_table["errors_sq"])

print(my_MSE)
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y, y_head)

print(MSE)
metric_table["errors_abs"] = np.abs(metric_table["y_values"] - metric_table["y_heads"])

metric_table.loc[:,["y_values","y_heads","errors_abs"]]
my_MAE = np.mean(metric_table["errors_abs"])

print(my_MAE)
from sklearn.metrics import mean_absolute_error



MAE = mean_absolute_error(y, y_head)

print(MAE)
avg_y = np.mean(metric_table["y_values"])

metric_table["y_differ_mean"] = metric_table["y_values"] - avg_y

metric_table["y_differ_mean_sq"] = metric_table["y_differ_mean"]**2

metric_table.loc[:,["y_values","y_heads","y_differ_mean","y_differ_mean_sq"]]
my_MSE_Baseline = np.mean(metric_table["y_differ_mean_sq"])
my_R2 = 1 - (my_MSE/my_MSE_Baseline)

print(my_R2)
from sklearn.metrics import r2_score

R2 = r2_score(y, y_head)

print(R2)
import statsmodels.api as sm





X = sm.add_constant(X)



lm = sm.OLS(y, X)

model_lm = lm.fit()

model_lm.summary().tables[0]
print(model_lm.rsquared)