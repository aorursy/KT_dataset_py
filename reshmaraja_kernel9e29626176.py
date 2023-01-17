import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
#importing dataset

titan=pd.read_csv('Titanic.csv')
titan
titan=titan.drop(['Cabin'],axis=1)
# A way to think about textual preprocessing is: Given my character column, what are some regularities that occur often. In our case we see titles (miss,mr etc).

#Then extract second word from every row and assign it to a new column. Not only that let us make it categorical (so that we can one-hot encode it) where we observe the most frequent ones.

def title_parser(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # Check whether title exists, then return it, if not ""

    if title_search:

        return title_search.group(1)

    return ""
titan['Title'] = titan['Name'].apply(title_parser)
titan['Title'] =titan['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 

                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'irrelevant')

# Let us make sure they are categorical, where we replace similiar names

titan['Title'] = titan['Title'].replace('Mlle', 'Miss')

titan['Title'] = titan['Title'].replace('Ms', 'Miss')

titan['Title'] = titan['Title'].replace('Mme', 'Mrs')
#Replacing the missing values with mean of Age

titan['Age'].fillna(titan['Age'].mean(),inplace=True)
#Replacing the missing values with the most frequent values

titan['Embarked'].fillna(titan['Embarked'].mode()[0], inplace = True)
#Checking the data frame has any null values

titan.isnull().sum()
titan
#Visualize the gender wise Survived people 

sns.set_style('dark')

sns.countplot(data=titan, x='Survived', hue='Sex')
#Visualizing the Pclass Survived people

sns.set_style('dark')

sns.countplot(data=titan, x='Survived', hue='Pclass')
fig = plt.figure(figsize=(18,6))

plt.subplots_adjust(hspace=.5)



plt.subplot2grid((2,3), (0,0))

titan['Survived'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Survival Distribution (Normalized)')



plt.subplot2grid((2,3), (0,1))

titan['Sex'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Gender Distribution (Normalized)')



plt.subplot2grid((2,3), (0,2))

titan['Pclass'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Pclass (Normalized)')



plt.subplot2grid((2,3), (1,0))

titan['SibSp'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('# of Siblings/Spouses (Normalized)')



plt.subplot2grid((2,3), (1,1))

titan['Parch'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('# of Parents/Children (Normalized)')



plt.subplot2grid((2,3), (1,2))

titan['Embarked'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Embarked')
#Correlation values respective with each other

sns.heatmap(titan.corr(),annot=True)

plt.show()
g = sns.pairplot(data=titan, hue='Survived', palette = 'seismic',

                 size=2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
#pair plotting 

sns.set(style="ticks")

sns.pairplot(titan, palette="Set1")

plt.show()
#Strip plotting

sns.stripplot(x='Survived',y='Age',hue='Pclass',jitter=True,dodge=True,palette="Set2",data=titan)

plt.show()
#Factor plotting 

sns.factorplot(x="Survived", y="Fare", hue="Embarked", kind='violin',data=titan)

plt.show()
#Creating Dummy values for Title

Title=pd.get_dummies(titan['Title'],prefix='Title')
#Concatenating the dummy vales

titan=pd.concat([titan,Title],axis=1)
titan
titan['Sex']=LabelEncoder().fit_transform(titan['Sex'])
Pclass=pd.get_dummies(titan['Pclass'],prefix='Pclass')
titan=pd.concat([titan,Pclass],axis=1)
bins=[0,12,19,25,50,100]

range=['kid','teens','youth','couples','old']

titan['Age_New']=pd.cut(titan['Age'],bins,labels=range)
Age_New=pd.get_dummies(titan['Age_New'],prefix='Age_New')

    
titan=pd.concat([titan,Age_New],axis=1)
Embarked=pd.get_dummies(titan['Embarked'],prefix="Embarked")
titan=pd.concat([titan,Embarked],axis=1)
#Binning is ranging the values for allocating some values to it 

bins1=[0,7.91,14.45,31,120]

range1=['Low_fare','median_fare','Average_fare','high_fare']

titan['Fare']=pd.cut(titan['Fare'],bins1,labels=range1)
#Creating dummt values for Fare 

Fare=pd.get_dummies(titan['Fare'],prefix='Fare')
#Concatenating the dummy values to the data frame

titan=pd.concat([titan,Fare],axis=1)
#Collecting the family size

titan['FamilySize']=titan['SibSp']+titan['Parch']+1
titan
#Dropping the unwanted columns from the data frame

titan=titan.drop(['Pclass','Age_New','Embarked','Title','PassengerId','Name','Fare','Ticket'],axis=1)
titan
#Visualzing the correlation values for each columns respective to each other

sns.heatmap(titan.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
#Scaling the age value 

sc=MinMaxScaler()

titan[['Age']]=sc.fit_transform(titan[['Age']])
titan
#Splitting the depepndent and independent variables

X=titan.drop(['Survived'],axis=1)

y=titan['Survived']
from sklearn.model_selection import train_test_split #split the dat in test and train sets

from sklearn.model_selection import cross_val_score #score evaluation with cross validation

from sklearn.model_selection import cross_val_predict #prediction with cross validation

from sklearn.metrics import confusion_matrix #for confusion matrix (metric of succes)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import  classification_report,roc_curve,roc_auc_score,accuracy_score
#Splitting train test values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
lr=LogisticRegression()
lr.fit(X_train,y_train)
#predicting the values with X_test

prediction=lr.predict(X_test)
classification_report(y_test,prediction)
conf=confusion_matrix(y_test,prediction)

conf
fpr, tpr,_=roc_curve(lr.predict(X),y,drop_intermediate=False)

plt.figure()

##Adding the ROC

plt.plot(fpr, tpr, color='red',

 lw=2, label='ROC curve')

##Random FPR and TPR

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

##Title and label

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve')

plt.show()
accuracy_score(y_test,prediction)
f, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(conf, annot=True, fmt="d", linewidths=.5, ax=ax)

plt.title("Confusion Matrix", fontsize=20)

plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)

ax.set_yticks(np.arange(conf.shape[0]) + 0.5, minor=False)

ax.set_xticklabels("")

ax.set_yticklabels(['Refused T. Deposits', 'Accepted T. Deposits'], fontsize=16, rotation=360)

plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
print('Accuracy',round(accuracy_score(pred_knn,y_test)*100,2))

result_knn=cross_val_score(knn,X,y,cv=10,scoring='accuracy')

print('The cross validated score',round(result_knn.mean()*100,2))

y_pred = cross_val_predict(knn,X,y,cv=10)

conf1=confusion_matrix(y,y_pred)

#sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="cool")

#plt.title('Confusion matrix', y=1.05, size=15)
f, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(conf1, annot=True, fmt="d", linewidths=.5, ax=ax)

plt.title("Confusion Matrix", fontsize=20)

plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)

ax.set_yticks(np.arange(conf1.shape[0]) + 0.5, minor=False)

ax.set_xticklabels("")

ax.set_yticklabels(['Refused T. Deposits', 'Accepted T. Deposits'], fontsize=16, rotation=360)

plt.show()
from sklearn.tree import DecisionTreeClassifier 
def train_using_gini(X_train, X_test, y_train): 

  

    # Creating the classifier object 

    clf_gini = DecisionTreeClassifier(criterion = "gini", 

            random_state = 100,max_depth=3, min_samples_leaf=5) 

  

    # Performing training 

    clf_gini.fit(X_train, y_train) 

    return clf_gini 

      
def tarin_using_entropy(X_train, X_test, y_train): 

  # Decision tree with entropy 

    clf_entropy = DecisionTreeClassifier( 

            criterion = "entropy", random_state = 100, 

            max_depth = 3, min_samples_leaf = 5) 

  # Performing training 

    clf_entropy.fit(X_train, y_train) 

    return clf_entropy 
def prediction(X_test, clf_object):   

    # Predicton on test with giniIndex 

    y_pred = clf_object.predict(X_test) 

    print("Predicted values:") 

    print(y_pred) 

    return y_pred 

      
# Function to calculate accuracy 

def cal_accuracy(y_test, y_pred):       

    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred)) 

    print ("Accuracy : ",accuracy_score(y_test,y_pred)*100) 

    print("Report : ",classification_report(y_test, y_pred)) 

  

# Driver code 

def main(): 

    # Building Phase 

    data = titan 

    clf_gini = train_using_gini(X_train, X_test, y_train) 

    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 

    # Operational Phase 

    print("Results Using Gini Index:") 

    # Prediction using gini 

    y_pred_gini = prediction(X_test, clf_gini) 

    cal_accuracy(y_test, y_pred_gini) 

    print("Results Using Entropy:") 

    # Prediction using entropy 

    y_pred_entropy = prediction(X_test, clf_entropy) 

    cal_accuracy(y_test, y_pred_entropy) 

      

      

#Calling main function 

if __name__=="__main__": 

    main() 

