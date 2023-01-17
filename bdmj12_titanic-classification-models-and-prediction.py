import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import math
%matplotlib inline
folder ="../input/"
data = pd.read_csv(folder + "train.csv")
data.head()
clean_data = data.drop(["PassengerId","Name", "Ticket","Cabin"], axis=1)
clean_data["Sex"] = clean_data["Sex"].map({"female": 1, "male": 0})
clean_data.head()
clean_data.info()
clean_data["Embarked"].value_counts()
for i in range(len(clean_data["Embarked"])):

    if clean_data["Embarked"][i]!="S":
          if clean_data["Embarked"][i]!="Q":
            if clean_data["Embarked"][i]!="C":
                #change the null entries:
                clean_data["Embarked"][i]= "S"
    
em_encoded,em_cats= clean_data["Embarked"].factorize()
clean_data[em_cats]=pd.get_dummies(em_encoded)
clean_data = clean_data.drop("Embarked",axis=1)
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean')
imp.fit(clean_data)
imp_data =pd.DataFrame(imp.transform(clean_data.values))
imp_data.columns = clean_data.columns
clean_data = imp_data
clean_data.head()
clean_data.info()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_data=clean_data

scaled_data[["Fare","Age"]] = scaler.fit_transform(scaled_data[["Fare","Age"]])
clean_data = scaled_data
def cleanup_data(data):
    data = data.drop(["PassengerId","Name", "Ticket","Cabin"], axis=1)
    data["Sex"] = data["Sex"].map({"female": 1, "male": 0})
    
    for i in range(len(data["Embarked"])):

        if data["Embarked"][i]!="S":
              if data["Embarked"][i]!="Q":
                if data["Embarked"][i]!="C":
                    #change the null entries:
                    data["Embarked"][i]= "S"
                
    em_encoded,em_cats= data["Embarked"].factorize()
    data[em_cats]=pd.get_dummies(em_encoded)
    data = data.drop("Embarked",axis=1)
    
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='mean')
    imp.fit(data)
    imp_data =pd.DataFrame(imp.transform(data.values))
    imp_data.columns = data.columns
    data = imp_data
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data=data
    scaled_data[["Fare","Age"]] = scaler.fit_transform(scaled_data[["Fare","Age"]])
    data = scaled_data
    return data
cleanup_data(data).head()
randpred = []
n_correct=0
for i in range(len(clean_data)):
    randpred.append(np.random.randint(low=0,high=2))
    if randpred[i]== clean_data["Survived"][i]:
        n_correct = n_correct+1
print("Accuracy: ", n_correct/len(clean_data))
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(clean_data.drop("Survived",axis=1),clean_data["Survived"] , test_size=0.33, random_state=42)
X_train =(clean_data.drop("Survived",axis=1))
y_train = clean_data["Survived"]
from sklearn import metrics, model_selection

def eval_model_withC(model, n, init, increment):
    values = dict()
    for x in range(n):
        c_val = init+x*increment
        predicted = model_selection.cross_val_predict(model(C=c_val), X_train, y_train, cv=10)
        values["C="  + str(round(c_val,4))] = (metrics.accuracy_score(y_train, predicted))
#for k, v in values.items():
#   print(f'{k:<4} {v}')

    maximum = max(values, key=values.get) 
    print(maximum, "  ", "Accuracy is", values[maximum])
    
def eval_model(model):
    predicted = model_selection.cross_val_predict(model(), X_train, y_train, cv=10)
    print(metrics.accuracy_score(y_train, predicted))
    

    
from sklearn.linear_model import LogisticRegression

eval_model(LogisticRegression)
eval_model_withC(LogisticRegression, 100,0.001, 0.001)


#pred = model_selection.cross_val_predict(LogisticRegression(C=0.01+4*0.02), X_train, y_train, cv=10)
#print(metrics.classification_report(y_train, pred) )
from sklearn import svm

eval_model_withC(svm.SVC,50,0.01, 0.01)
from sklearn.naive_bayes import GaussianNB

eval_model(GaussianNB)
from sklearn.neighbors import KNeighborsClassifier

eval_model(KNeighborsClassifier)
from sklearn.tree import DecisionTreeClassifier

def eval_tree(model, n, init, increment):
    values = dict()
    for x in range(n):
        val = init+x*increment
        predicted = model_selection.cross_val_predict(model(min_impurity_decrease=val), X_train, y_train, cv=10)
        values["min_impurity_decrease="  + str(round(val,4))] = (metrics.accuracy_score(y_train, predicted))
#for k, v in values.items():
#   print(f'{k:<4} {v}')

    maximum = max(values, key=values.get) 
    print(maximum, "  ", "Accuracy is", values[maximum])
scaled_data = pd.concat([scaled_data, clean_data[["Survived","Sex","S","C","Q"]]],sort=True)
eval_tree(DecisionTreeClassifier,100,0,0.00005)
from sklearn.ensemble import RandomForestClassifier

eval_model(RandomForestClassifier)
eval_tree(RandomForestClassifier,100,0.001,0.0001)


test_data = pd.read_csv(folder + "test.csv")
passID = test_data["PassengerId"]
clean_test_data = cleanup_data(test_data)
clean_test_data.head()
X_train.head()
X_train.columns
clean_test_data = clean_test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'S', 'C', 'Q']]
clean_test_data.head()
RFClassifier = RandomForestClassifier(min_impurity_decrease=0.0013)
                                         
RFClassifier.fit(X_train,y_train)

pred = pd.DataFrame(RFClassifier.predict(clean_test_data))
        
pred["PassengerId"] = passID

pred.columns = ["Survived","PassengerId"]
    
pred = pred[["PassengerId","Survived"]]

#needs to be int or wont submit properly!
pred = pred.astype(int)

pred.head()
                                         
pred.to_csv("predictions.csv", index=False)
