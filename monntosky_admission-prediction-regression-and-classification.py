import pandas as pd
df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")

df
df = df.drop("Serial No.",axis=1)

df
import seaborn as sns 

import matplotlib.pyplot as plt

_,fig = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(),annot=True)

plt.show()
# taking all the coloumns expect the last one i.e 'chance_of_admission'

x = df.iloc[:,:-1]

x
# taking our target coloumn which chance_of_admit

y = df.iloc[:,-1]

y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size = 0.2,random_state = 42)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier

from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier

from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier

from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier

from sklearn.svm import SVR,SVC

from sklearn.naive_bayes import GaussianNB



from sklearn.metrics import accuracy_score,mean_squared_error
import numpy as np

regressors=[['Linear Regression :',LinearRegression()],

       ['Decision Tree Regression :',DecisionTreeRegressor()],

       ['Random Forest Regression :',RandomForestRegressor()],

       ['Gradient Boosting Regression :', GradientBoostingRegressor()],

       ['Ada Boosting Regression :',AdaBoostRegressor()],

       ['Extra Tree Regression :', ExtraTreesRegressor()],

       ['K-Neighbors Regression :',KNeighborsRegressor()],

       ['Support Vector Regression :',SVR()]]

reg_pred=[]

print('Results time for regression ...\n')

print("Excited !!!!!! :)")
for name,model in regressors:

    model=model

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    rms=np.sqrt(mean_squared_error(y_test, predictions))

    reg_pred.append(rms)

    print(name,rms)
y_ax=['Linear Regression' ,'Decision Tree Regression', 'Random Forest Regression','Gradient Boosting Regression', 'Ada Boosting Regression','Extra Tree Regression' ,'K-Neighbors Regression', 'Support Vector Regression' ]

x_ax=reg_pred

sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.1")
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)

#If Chance of Admit greater than 80% we classify it as 1

y_train_c = [1 if each > 0.8 else 0 for each in y_train]

y_test_c  = [1 if each > 0.8 else 0 for each in y_test]
classifiers=[['Logistic Regression :',LogisticRegression()],

       ['Decision Tree Classification :',DecisionTreeClassifier()],

       ['Random Forest Classification :',RandomForestClassifier()],

       ['Gradient Boosting Classification :', GradientBoostingClassifier()],

       ['Ada Boosting Classification :',AdaBoostClassifier()],

       ['Extra Tree Classification :', ExtraTreesClassifier()],

       ['K-Neighbors Classification :',KNeighborsClassifier()],

       ['Support Vector Classification :',SVC()],

       ['Gausian Naive Bayes :',GaussianNB()]]

cla_pred=[]

for name,model in classifiers:

    model=model

    model.fit(X_train,y_train_c)

    predictions = model.predict(X_test)

    cla_pred.append(accuracy_score(y_test_c,predictions))

    print(name,accuracy_score(y_test_c,predictions))
y_ax=['Logistic Regression' ,

      'Decision Tree Classifier',

      'Random Forest Classifier',

      'Gradient Boosting Classifier',

      'Ada Boosting Classifier',

      'Extra Tree Classifier' ,

      'K-Neighbors Classifier',

      'Support Vector Classifier',

       'Gaussian Naive Bayes']

x_ax=cla_pred

sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.8")

plt.xlabel('Accuracy')
