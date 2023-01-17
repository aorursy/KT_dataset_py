import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import numpy as np

from sklearn import metrics
pima=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
pima.shape
pima.head(20)
sum(pima.BloodPressure==0)
pima.Pregnancies.nunique()
pima.describe().T
pima.groupby(["Outcome"]).count()
sns.pairplot(pima)
X= pima.iloc[:,0:8] # sele ting all rows and first 8 columns

Y = pima.iloc[:,8]  # selecting all rows of last column
X.head()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)
model = LogisticRegression()
model.fit(X_train,Y_train)  # model is getting trained



Y_predict = model.predict(X_test) # prediction of label and run on the test data.This is y hat



t= list(X_train.columns)



coef_df = pd.DataFrame(model.coef_,columns=t) #extracting coefficient of the model 

coef_df["intercept"] = model.intercept_

print(coef_df)
model_score = model.score(X_test,Y_test)

print(model_score )

print(metrics.confusion_matrix(Y_test,Y_predict))

# Confusion matrix shows that this model has correctly identified 135 as non diabetic and 48 as diabatic which is true in both cases which are diagonally, there in the confusion matrix. 

pima.head(10)
pima[pima.BloodPressure != 0].mean()