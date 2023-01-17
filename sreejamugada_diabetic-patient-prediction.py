import pandas as pd
dia=pd.read_csv("/kaggle/input/pima-indians-diabetes-dataset/diabetes.csv")
dia
X=dia.drop(columns=["Outcome"],axis=1).values
Y=dia[["Outcome"]].values
X=X/X.max()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.4,random_state=58)
(ytest==1).sum()
(ytest==0).sum()
from sklearn.naive_bayes import GaussianNB
nmodel=GaussianNB()
modelpre=nmodel.fit(xtrain,ytrain)
#print(nmodel.score(xtrain,ytrain))
print(nmodel.score(xtest,ytest))
ytrain_prod=modelpre.predict(xtrain)
ytest_prod=modelpre.predict(xtest)
modelpre.predict([[0,137,40.0,35,168.0,43.1,2.288,33]])
modelpre.predict([[10,101,76.0,48,180.0,32.9,0.171,63]])
d=dia[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]]
d.corr()
