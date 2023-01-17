import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
df=pd.read_csv('/kaggle/input/loanapproval/loan.csv')
df.keys()  #shows the column names
df #will display the whole dataset
df=df.dropna()  #to drop all null values
df

x=df[["FICO.Score","Loan.Amount"]].values

y=df["Interest.Rate"].values
model=LinearRegression()

model.fit(x,y)
y_pred=model.predict(x)
exp=np.exp(-y_pred)+1

log=1/exp
y_con = y<17    #store the condition where interest rate is less than 17
df["TF"]=df["Interest.Rate"] <17   #store boolean values in TF where interest rate is less than 17

df
df.TF.value_counts()     # no of types of values, 
log_reg=LogisticRegression()    #to use logistic regression
log_reg.fit(x,y_con)   #y_con is where interest rate i.e. y is < 17
log_reg.predict([[10,20000]])
dat=log_reg.predict_proba([[1000,20000]])   #this prints the probability :(false,true)

dat
dat[0][1]>0.8
y_pred = log_reg.predict(x)
df["Predict"] = y_pred

df.Predict
df.Predict.value_counts()  #predicted data

#real data



df.TF.value_counts()
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(df.TF.values, df.Predict.values)
cm
df.shape
accuracy_score(df.TF.values, df.Predict.values)