# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.linear_model import LogisticRegression
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from IPython.display import Image  
import graphviz
from eli5.sklearn import PermutationImportance

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#data.shape
#data.columns
# Any results you write to the current directory are saved as output.
columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       ]




one=OneHotEncoder(n_values=[2,2,2,2,2,3,3,3,3,3,3,3,3,3,2])
#transform=one.fit_transform(data[columns])
data[columns].values[0]

one_hot_encoded_training=pd.get_dummies(data[columns])

label_encode=LabelEncoder()
encode=label_encode.fit_transform(data['Churn'])
data['Churn']=encode

Target=data['Churn']

clf=RandomForestClassifier(n_estimators=600)
Logistic=LogisticRegression()

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
grid_Random=GridSearchCV(clf,parameters)
Decision_tree=DecisionTreeClassifier(criterion='entropy',min_samples_split=50)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(one_hot_encoded_training,Target,test_size=0.2)

clf.fit(Xtrain,Ytrain)
Logistic.fit(Xtrain,Ytrain)
Decision_tree.fit(Xtrain,Ytrain)

prediction_Random=clf.predict(Xtest)
logistic_pred=clf.predict(Xtest)
decision_pred=Decision_tree.predict(Xtest)

logistic_accuracy=accuracy_score(Ytest,logistic_pred)
accuracy=accuracy_score(Ytest,prediction_Random)
Decision_accuracy=accuracy_score(Ytest,decision_pred)
#Accuracy formula : TF+TN/TP+FP+TN+FN
print(accuracy)

print("RandomForest Accuracy is  {}".format(accuracy))
print("Logistic regression_Accuracy is {}".format(logistic_accuracy))
print("Decision_tree {}".format(Decision_accuracy))

#binarizer = preprocessing.Binarizer().fit(data['PhoneService'])
#one_hot_encoded_training.head(12)


#Confusion Matrix:
random_pred=clf.predict(Xtrain)
conf=confusion_matrix(y_true=Ytrain,y_pred=random_pred)

#perm=PermutationImportance(random_pred,random_state=1).fit(Xtrain,Ytrain)


sns.heatmap(conf,annot=True,annot_kws={"size":16})
plt.show()
print(conf)





from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision=precision_score(y_true=Ytrain,y_pred=random_pred)
print("Precision score is {}".format(precision*100))
recall=recall_score(y_true=Ytrain,y_pred=random_pred)
print("Recall score is {}".format(recall*100))




#Visualization of Decision Tree Classifier


Features=one_hot_encoded_training.columns
Target=['1','0']
data1=tree.export_graphviz(Decision_tree,class_names=Target,feature_names=Features,filled=True,rounded=True)
graphviz.Source(data1)

data.head(10)
data['PaymentMethod'].value_counts().plot.bar()
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import sparse_encode


Feature_Importance=pd.DataFrame({'feature':one_hot_encoded_training.columns,'Importance':clf.feature_importances_})
Feature_Importance

Feature_Importance.min()
Feature_Importance=Feature_Importance.sort_values(by='Importance',ascending=False)
plt.rcParams['figure.figsize']=(20,10)
Feature_Importance.plot.bar(color='green')
plt.xlabel("Features")
plt.ylabel("Probabilities")
plt.show()


data.columns
data.StreamingMovies.value_counts()
data.dtypes

for i,v in enumerate(zip(data.iloc[488])):
    if i==19:
        data.iloc[488][19]=0
        
        
data['TotalCharges']=data.TotalCharges.str.strip()
data['TotalCharges']=pd.to_numeric(data.TotalCharges)
data.dtypes
Total_Charges_By_customer=data.groupby(by='customerID')['TotalCharges'].sum()

Total_Charges_By_customer.drop([data.iloc[488][0]])
sns.boxplot(x='StreamingTV',y='TotalCharges',data=data)
data.columns
sns.boxplot(x='OnlineSecurity',y='TotalCharges',data=data)
