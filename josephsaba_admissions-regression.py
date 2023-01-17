import numpy as np 
import pandas as pd 
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")

#print(df.head())
#print(df.shape)

y_var = df['Chance of Admit ']
x_var = df[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
df2 = df[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']].copy()
lm = linear_model.LinearRegression()
model = lm.fit(x_var,y_var)

print('Intercept: \n', lm.intercept_)
print('Coefficients: \n', lm.coef_)

gremean = df2.mean()

averages = []

#running model using averages
for columnmean in gremean:
    print (columnmean)
    averages.append(columnmean)

#with research component 
averages.pop(6) #remove last element since research can only be 1 or 0
averages.append(1) #with research component
predict = lm.predict([[b for b in averages]])
predict_pct_research = round(predict[0]*100,1)

#without research component
averages.pop(6) #remove last element since research can only be 1 or 0
averages.append(0) #without research component
predict = lm.predict([[b for b in averages]])
predict_pct_noresearch = round(predict[0]*100,1)

print ("""A remarkably average student has a """,predict_pct_research, """% chance of getting into this university with a previous research component, 
or a """,predict_pct_noresearch,"""% chance without a previous research component.""")
