# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
cancer = pd.read_csv("../input/cervical-cancer/cervical-cancer.csv")
print(cancer)

X= cancer.iloc[:,0:34].values
y= cancer.iloc[:,35].values
print(X)
print(y)
cancer.isnull().sum()
cancer.columns
#data preprocessing
for i in cancer.columns:
    #filling the null values with median
    cancer[i]=cancer[i].fillna(cancer[i].median())
    cancer[i].isnull().any()
print(cancer)
X= cancer.iloc[:,0:34].values
y= cancer.iloc[:,35].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=123)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.feature_selection import VarianceThreshold
sel_variance_threshold = VarianceThreshold() 
X_train1 = sel_variance_threshold.fit_transform(X_train)
print(X_train1.shape)
#print(X_train1.scores)

#RANDOMFOREST 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#creating model
model= RandomForestClassifier()

#feeding training data
model.fit(X_train,y_train)

#predict the test data
y_pred=model.predict(X_test)

#accuracy
print("training accuracy:",model.score(X_train,y_train))
print("testing accuracy:",model.score(X_test,y_test))

#classification report
print(classification_report(y_test,y_pred))
rf_acc = accuracy_score(y_test,y_pred)
print("accuracy:",accuracy_score(y_test,y_pred))


#confusion matrix
print(confusion_matrix(y_test,y_pred))

#DECISIONTREE
from sklearn.tree import DecisionTreeClassifier

# creating the model
model = DecisionTreeClassifier()

# feeding the training data into the model
model.fit(X_train, y_train)

# predicting the test set results
y_pred = model.predict(X_test)

# Calculating the accuracies
print("Training accuracy :", model.score(X_train, y_train))
print("Testing accuracy :", model.score(X_test, y_test))

# classification report
print(classification_report(y_test, y_pred))
 
#accuracy
df_acc = accuracy_score(y_test,y_pred)
print("accuracy:",accuracy_score(y_test,y_pred))

# confusion matrix 
print(confusion_matrix(y_test, y_pred))
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# creating the model
model = LogisticRegression()

# feeding the training data into the model
model.fit(X_train, y_train)

# predicting the test set results
y_pred = model.predict(X_test)

# Calculating the accuracies
print("Training accuracy :", model.score(X_train, y_train))
print("Testing accuracy :", model.score(X_test, y_test))

# classification report
print(classification_report(y_test, y_pred))
log_acc = accuracy_score(y_test,y_pred)
print("accuracy:",accuracy_score(y_test,y_pred))


# confusion matrix 
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dt = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)

# training the model
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Calculating the accuracies
print("Training accuracy :", clf.score(X_train, y_train))
print("Testing accuracy :", clf.score(X_test, y_test))

# classification report
print(classification_report(y_test, y_pred))
ada_acc=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy_score(y_test,y_pred))
# confusion matrix 
print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import VotingClassifier
lreg=LogisticRegression()
dt=DecisionTreeClassifier()
rt=RandomForestClassifier()
vc=VotingClassifier(estimators=[('lreg',lreg),('dt',dt),('rt',rt)],voting="hard")
vc.fit(X_train,y_train)
y_pred=vc.predict(X_test)
print("accuracy:",accuracy_score(y_test,y_pred))

scores = [rf_acc, df_acc, log_acc, ada_acc]
algoritham = ["randomforest", "decisiontree", "logisticregression", "adaboostclassifier"]
m = len(algoritham)
for i in range(m):
    print("the accuracy score acheived using", algoritham[i], "is:",str(scores[i]*100),"%")
    



from matplotlib import pyplot as plt
a = algoritham
b = scores
plt.plot(a,b)
plt.show()
def fun(a,b,c,d,e,f,g):
    a,b,c,d,e,f,g=int(a),int(b),int(c),int(d),float(e),int(f),float(g)
    new=(a,b,c,d,e,f,g)
    myText=StringVar()
    g=np.resize(new,(167.7))
    x=max(model.predict(g))
    myText.set(x)
    Label(first,text="",textvariable=myText).grid(row=10,column=1)
    

from tkinter import*
from tkinter import messagebox
from PIL import ImageTk,Image
m = Tk()
global first
first = frame(master, wdith=500, height=500)
master.title("PREDICTIVE ANAYLSIS OF BIOPSY IN CERVICAL CANCER")
first.place(x=0,y=0,width=500,height=500)
Label=tkinter.Label(first,text="Age").grid(row=0,column=0)
e1 = Entry(first)
e1.grid(row=0,column=1)
Label=tkinter.Label(first,text="first sexual intercourse").grid(row=1,column=0).pack()
e1 = Entry(first)
e1.grid(row=1,column=1)
Label=tkinter.Label(first,text="no of sexual intercourse").grid(row=2,column=0).pack()
e1 = Entry(first)
e1.grid(row=2,column=1)
Label=tkinter.Label(first,text="no of pregnencies").grid(row=3,column=0).pack()
e1 = Entry(first)
e1.grid(row=3,column=1)
Label=tkinter.Label(first,text="Smokes").grid(row=4,column=0).pack()
e1 = Entry(first)
e1.grid(row=4,column=1)
Label=tkinter.Label(first,text="STDs").grid(row=5,column=0).pack()
e1 = Entry(first)
e1.grid(row=5,column=1)
master.mainloop()





