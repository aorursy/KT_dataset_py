import pandas as pd
import numpy as np
name=["num_preg","glucose","diastolic","thickness","insulin","bmi","diab_pred","age","diabetes"]
df=pd.read_csv("../input/pima-indians-diabetes.csv",names=name)
df.head(10)
df.isnull().values.any()
corr=df.corr()
corr
num_true=len(df[df["diabetes"]==True])
num_false=len(df[df["diabetes"]==False])
print(num_true)
print(num_false)
from sklearn.model_selection import train_test_split

feature_names=["num_preg","glucose","diastolic","thickness","insulin","bmi","diab_pred","age"]
label_names=["diabetes"]

X=df[feature_names].values
y=df[label_names].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
#traning naive bayes
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()

model.fit(X_train,y_train.ravel())


#test on training

train_predict=model.predict(X_train)

from sklearn import metrics
a=metrics.accuracy_score(y_train,train_predict)
print(a)
#test on test data

test_predict=model.predict(X_test)

from sklearn import metrics
a=metrics.accuracy_score(y_test,test_predict)
print(a)
#confusion matrix

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test,test_predict)))
print("")


print("Classification Report")
print(metrics.classification_report(y_test,test_predict))
print("")

#training random forest
from sklearn.ensemble import RandomForestClassifier
rfmodel=RandomForestClassifier(min_samples_leaf=8)

rfmodel.fit(X_train,y_train.ravel())

#predift training data
rf_train_predict=rfmodel.predict(X_train)

from sklearn import metrics
a=metrics.accuracy_score(y_train,rf_train_predict)
print(a)
#predift test data
rf_test_predict=rfmodel.predict(X_test)

from sklearn import metrics
a=metrics.accuracy_score(y_test,rf_test_predict)
print(a)
#confusion matrix

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test,rf_test_predict)))
print("")


print("Classification Report")
print(metrics.classification_report(y_test,rf_test_predict))
print("")
#logistic regression
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(C=0.7922077922077922,max_iter=100,random_state=0)

lr_model.fit(X_train,y_train.ravel())
#predift training data
lr_train_predict=lr_model.predict(X_train)

from sklearn import metrics
a=metrics.accuracy_score(y_train,lr_train_predict)
print(a)
#predift test data
lr_test_predict=lr_model.predict(X_test)

from sklearn import metrics
a=metrics.accuracy_score(y_test,lr_test_predict)
print(a)
from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test,lr_test_predict)))
print("")


print("Classification Report")
print(metrics.classification_report(y_test,lr_test_predict))
print("")
l=[]
values=[]
#to obatin c value
for i in range(1,10):
    lr_model=LogisticRegression(C=i,random_state=0)
    values.append(i)
    lr_model.fit(X_train,y_train.ravel())
    #predift test data
    lr_test_predict=lr_model.predict(X_test)
    from sklearn import metrics
    a=metrics.accuracy_score(y_test,lr_test_predict)
    print(a)
    l.append(a)

plt.plot(values,l)
plt.show()
from sklearn.linear_model import LogisticRegressionCV
cv_model=LogisticRegressionCV(Cs=3,cv=10,refit=False,class_weight="balanced",max_iter=1000)
cv_model.fit(X_train,y_train.ravel())
#predift training data
cv_train_predict=cv_model.predict(X_train)

from sklearn import metrics
a=metrics.accuracy_score(y_train,cv_train_predict)
print(a)
#predift training data
cv_test_predict=cv_model.predict(X_test)

from sklearn import metrics
a=metrics.accuracy_score(y_test,cv_test_predict)
print(a)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test,cv_test_predict)))
print("")


print("Classification Report")
print(metrics.classification_report(y_test,cv_test_predict))
print("")
from sklearn.preprocessing import MinMaxScaler as Scaler

scaler = Scaler()
scaler.fit(y)

def get_value(new_df,model):

    new_df_scaled = scaler.transform(new_df)
    prediction = model.predict(new_df_scaled)
    print("model used is :", [model][0])
    print("Predicted Value is :" ,prediction)
    if prediction==[1]:
        print("The patient has DB2")
        print(" ")
    else:
        print("The patient does not have DB2")
        print(" ")
new_df = pd.DataFrame([[2, 120, 62, 60, 0.5, 46, 0.47, 56]])   
get_value(new_df,model)
get_value(new_df,rfmodel)
get_value(new_df,lr_model)