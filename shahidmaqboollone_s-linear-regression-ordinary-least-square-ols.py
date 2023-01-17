import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import math

data = pd.read_csv('../input/predictingese/AttendanceMarksSA.csv')
data.head()
data.describe()
corr = data.corr()
corr.style.background_gradient(cmap = 'coolwarm')
X = data["MSE"]
y = data["ESE"]

sns.scatterplot(X,y)
endog = data['ESE']

exog = sm.add_constant(data[['MSE']])

print(exog.head())
print(endog.head())
model = sm.OLS(endog , exog)
results = model.fit()
print(results.summary())
def RSE(y_true , y_predicted):
    
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    
    RSS = np.sum(np.square(y_true-y_predicted))
    
    rse = math.sqrt( RSS / (len(y_true) - 2))
    
    return rse
rse = RSE(data['ESE'],results.predict())
print(rse)
marks = 17
end_marks = results.predict([1,marks])
print(end_marks)
X1 = data["Attendance"]
y1 = data["ESE"]

sns.scatterplot(X1 ,y1)

endog1 = data['ESE']
exog1 = sm.add_constant(data[['Attendance']])
print(exog1.head(),end="\n\n")
print(endog1.head())
model1 = sm.OLS(endog1, exog1)
results1 = model1.fit()
rse = RSE(data['Attendance'],results1.predict())
print(rse)
print (results1.summary())