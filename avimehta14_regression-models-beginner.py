import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model
%matplotlib inline
df= pd.read_csv("../input/canadas-per-capita-income/Canada-per-capita-income.csv")
df.head()
df.rename(columns={'year':"Years","per capita income (US$)":"P.C.I. $"} ,inplace=True)
df.head(5)
plt.scatter(df['Years'], df['P.C.I. $'], color="red",marker='+')

plt.xlabel("YEARS")

plt.ylabel("Per-Capita-Income ($)")

reg = linear_model.LinearRegression()
reg.fit(df[['Years']],df['P.C.I. $'])
reg.predict([[2050]])
reg.coef_
reg.intercept_
plt.scatter(df['Years'], df['P.C.I. $'], color="red",marker='+')

plt.xlabel("YEARS")

plt.ylabel("Per-Capita-Income ($)")

plt.plot(df['Years'],reg.predict(df[['Years']]),color="blue")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 9.0)





data = pd.read_csv("../input/linearreg-data/data.csv")

X = data.iloc[:, 0]

Y = data.iloc[:, 1]

plt.scatter(X, Y)

plt.show()
# creating the model



m = 0 

c = 0



L = 0.0001

iters = 1000



n= float(len(X))



for i in range(len(X)):

    Y_pred = m*X+c

    deri_m = (-2/n) * sum(X*(Y-Y_pred))

    deri_c = (-2/n)* sum(Y-Y_pred)

    

    m = m - L*deri_m

    c = c - L*deri_c

    

print(m,c)    



Y_pred = m*X+c

plt.scatter(X, Y)

plt.plot(X, Y_pred,color='red')

plt.show()
Y_pred
df1= pd.read_csv("../input/linearreg-data/hiring.csv")
df1
df1.rename(columns={'test_score(out of 10)':"testscore_10","interview_score(out of 10)":"inter_score","salary($)":"salary"},

           inplace =True)
df1
import math



med=df1.testscore_10.median()

df1.testscore_10 = df1.testscore_10.fillna(med)

from word2number import w2n



df1.experience = df1.experience.fillna("Zero")
for i in range(len(df1.experience)):

    s=df1.iloc[i,0]

    df1.iloc[i,0]=w2n.word_to_num(s)

df1
for j in range(len(df1.salary)):

    s=df1.iloc[j,3]

    if "S" in s:

        y=s.replace("S"," ")

        df1.iloc[j,3]=int(y)

        

    else:

        df1.iloc[j,3]=int(df1.iloc[j,3])

        

 
df1
# creating our model



reg1 = linear_model.LinearRegression()

reg1.fit(df1[['experience','testscore_10','inter_score']],df1['salary'])
reg1.coef_
reg1.intercept_
reg1.predict([[2,9,6]])
reg1.predict([[12,10,10]])