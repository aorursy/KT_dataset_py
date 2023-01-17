import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# read in data
kaggle = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding="ISO-8859-1")
stackOverflow = pd.read_csv("../input/so-survey-2017/survey_results_public.csv")
display(kaggle.head())
display(stackOverflow.head())
CompensationAmount = kaggle['CompensationAmount']
CompensationAmount = CompensationAmount.str.replace(',','')
CompensationAmount = CompensationAmount.str.replace('-','')
CompensationAmount.fillna(0,inplace=True)
CompensationAmount = pd.to_numeric(CompensationAmount)

CompensationAmount.head()
Age = kaggle['Age']
Age.fillna(0,inplace=True)
Age = pd.to_numeric(Age)
Age = Age[CompensationAmount>0]
CompensationAmount = CompensationAmount[CompensationAmount>0]
Age.head()
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

reg = LinearRegression()

x = Age[:,np.newaxis]
y = CompensationAmount
reg.fit(x,y)
y_pred= reg.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred,color='blue',linewidth=3)
residual = y - y_pred
plt.scatter(x,residual)
Age = Age[CompensationAmount<200000]
CompensationAmount = CompensationAmount[CompensationAmount<200000]
x = Age[:,np.newaxis]
y = CompensationAmount
reg = LinearRegression()
reg.fit(x,y)
y_pred = reg.predict(x)
residual = y - y_pred
plt.axes=(1,2)
plt.plot(x,y_pred,color='red',linewidth=3)
plt.scatter(x,y,color='yellow')
plt.xlabel('age')
plt.ylabel('compensation')
Age = Age[CompensationAmount<200000]
CompensationAmount = CompensationAmount[CompensationAmount<200000]
x = Age[:, np.newaxis]
y = CompensationAmount
reg = LinearRegression()
reg.fit(x,y)
y_pred = reg.predict(x)
residual = y - y_pred
plt.scatter(x,residual)
plt.xlabel('age')
plt.ylabel('compensation')
plt.scatter(x, y)
plt.plot(x, y_pred, color='blue', linewidth=3)
plt.xlabel('age')
plt.ylabel('compensation')
stackOverflow.head()
import seaborn as sns

plt.figure(figsize=( 10,10))
plt.hist(stackOverflow['Professional'])
CompanySize = stackOverflow['CompanySize']
CompanySize.fillna('0',inplace=True)
options = CompanySize.unique();
options = options[1:-2]
print(options)
size_options = [55,10000,15,5,7500,255,2500,750]
CompanySize_number = []

for i in range(len(CompanySize)):
    to_append = 0
    for j in range(len(options)):
        if CompanySize[i]==options[j]:
            to_append = size_options[j]
            break
    CompanySize_number.append(to_append)
x = pd.to_numeric(CompanySize_number)
print(len(x))
def getValue(x):
    output = stackOverflow[x]
    output.fillna(0,inplace=True)
    output = pd.to_numeric(output)
    return output

ExpectedSalary = getValue('ExpectedSalary')
Salary = getValue('Salary')

salary_combined = []
for i in range(len(ExpectedSalary)):
    if(ExpectedSalary[i]>Salary[i]):
        salary_combined.append(ExpectedSalary[i])
    else:
        salary_combined.append(Salary[i])

y = pd.Series(salary_combined)
y = y[x>0]
x = x[x>0]
x = x[y>0]
y = y[y>0]
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x_fitting = x[:,np.newaxis]
reg = LinearRegression()
reg.fit(x_fitting,y)
y_pred = reg.predict(x_fitting)
plt.scatter(x,y)
plt.plot(x,y_pred,color='blue',linewidth=3)
plt.xlabel('Company Size')
plt.ylabel('Salary')