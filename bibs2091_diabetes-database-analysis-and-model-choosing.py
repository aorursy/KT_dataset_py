import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# disable warnings
import warnings
warnings.filterwarnings('ignore')
filename = '../input/diabetes.csv'
data=pd.read_csv(filename)
print(data.columns) # to know all the features(variables) we got in our data
print(data.head()) #the first 5 rows
data['Outcome'].describe()

data['Outcome'].hist(figsize=(7,7))

#borrowed from my friend Ayoub Benaissa.
def plot_diabetic_per_feature(data, feature):
    grouped_by_Outcome = data[feature].groupby(data["Outcome"])
    diabetic_per_feature = pd.DataFrame({"Sick": grouped_by_Outcome.get_group(1),
                                        "Not Sick": grouped_by_Outcome.get_group(0),
                                        })
    hist = diabetic_per_feature.plot.hist(bins=60, alpha=0.6)
    hist.set_xlabel(feature)
    plt.show()
    


plot_diabetic_per_feature(data, "Age")

plot_diabetic_per_feature(data, "Glucose")

print(data["Glucose"].min())
plot_diabetic_per_feature(data, "BMI")

import seaborn as sns #the librery we'll use for the job xD

corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, annot=True, square=True, vmax=.8);
sns.set()
cols = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
sns.pairplot(data[cols], size = 2.5)
plt.show();
data.min()

data = data.drop(data[data['Glucose'] == 0].index)
data = data.drop(data[data['SkinThickness'] == 0].index) # even it will be deleted xD
data = data.drop(data[data['BloodPressure'] == 0].index) #same
data = data.drop(data[data['BMI'] == 0].index)
data = data.drop(data[data['Insulin'] == 0].index)

print(data.min()) # let's check



#just the libreries we need
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import accuracy_score, make_scorer,f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


#preparing the data
cols = ['Pregnancies','Glucose','DiabetesPedigreeFunction','Insulin','BMI','Age']

Y=data['Outcome']
#rescaledX = StandardScaler().fit_transform(data[cols])
#X=pd.DataFrame(data = rescaledX, columns= cols)
X=data[cols]

# I deleted BloodPressure and Skinthikness
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state = 25, test_size = 0.2)




svm1 = svm.SVC(kernel='linear')
svm2 = svm.SVC(kernel='rbf') 
lr = LogisticRegression()
rf = RandomForestClassifier()
knn=KNeighborsClassifier()
models = {"Logistic Regression": lr,"Random Forest": rf, "svm linear": svm1 , "svm rbf": svm2,"KNeighborsClassifier": knn }
l=[]
for model in models:
    l.append(make_pipeline(Imputer(),  models[model]))
#Finally get the cross-validation scores
i=0

for Classifier in l:    
    accuracy = cross_val_score(Classifier,X_train,Y_train,scoring='accuracy',cv=10)
    print("===", [*models][i] , "===")
    print("accuracy = ",accuracy)
    print("accuracy.mean = ", accuracy.mean())
    print("accuracy.variance = ", accuracy.var())
    i=i+1
    print("")
    



lr = LogisticRegression()
lr.fit(X_train,Y_train)
predictions = lr.predict(X_test)
sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, cmap="YlGn")
plt.title(' LogisticRegression ')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print("the f1 score for Logistic Regression is :",(f1_score(Y_test, predictions, average="macro")))
print("the precision score is :",(precision_score(Y_test, predictions, average="macro")))
print("the recall score is :",(recall_score(Y_test, predictions, average="macro")))   


rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
predictions = rf.predict(X_test)
sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, cmap="YlGn")
plt.title(' Random Forest Classifier ')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print("the f1 score for Random Forest Classifier is :",(f1_score(Y_test, predictions, average="macro")))
print("the precision score is :",(precision_score(Y_test, predictions, average="macro")))
print("the recall score is :",(recall_score(Y_test, predictions, average="macro")))   
svm = svm.SVC(kernel='linear')
svm.fit(X_train,Y_train)
predictions = svm.predict(X_test)
sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, cmap="YlGn")
plt.title(' SVM kernel(linear) ')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print("the f1 score for SVM linear is :",(f1_score(Y_test, predictions, average="macro")))
print("the precision score is :",(precision_score(Y_test, predictions, average="macro")))
print("the recall score is :",(recall_score(Y_test, predictions, average="macro")))   


    
