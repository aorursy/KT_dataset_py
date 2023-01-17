import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(color_codes=True)

%matplotlib inline
%pylab inline
df = pd.read_csv('../input/Iris.csv')
df.head()
#ANY OF THE BELOW THREE CAN BE USED
#del df['Id']
#df.__delitem__('Id')
#df.drop('Id',axis=1)

if 'Id' in df.columns:
  df.__delitem__('Id')

df.head()
#SUMMARY OF THE DATA SET
df.shape
df.info()
df['Species'].unique()
# COMPARING THE DIFFERENT NUMERICAL COLUMNS IN THE GIVEN DATASET 

df.describe()

listOfColumns = df.columns
listOfNumericalColumns = []

for column in listOfColumns:
    if df[column].dtype == float64:
        listOfNumericalColumns.append(column)

print('listOfNumericalColumns :',listOfNumericalColumns)
spices = df['Species'].unique()
print('spices :',spices)

fig, axs = plt.subplots(nrows=len(listOfNumericalColumns),ncols=len(spices),figsize=(15,15))

for i in range(len(listOfNumericalColumns)):
    for j in range(len(spices)):  
        print(listOfNumericalColumns[i]," : ",spices[j])
        axs[i,j].boxplot(df[listOfNumericalColumns[i]][df['Species']==spices[j]])
        axs[i,j].set_title(listOfNumericalColumns[i]+""+spices[j])  
#descriptions
df.describe()
df.groupby('Species').size()
#box and whisker plots for different numerical columns
df.plot(kind='box')
#HIST PLOT OF ALL NUMERICAL COLUMNS

df.hist(figsize=(10,5))
plt.show()
print("HIST PLOT OF INDIVIDUAL Species")
print(spices)

for spice in spices:
        df[df['Species']==spice].hist(figsize=(10,5))  
df.boxplot(by='Species',figsize=(15,15))
sns.violinplot(data=df,x='Species',y='PetalLengthCm')
pd.scatter_matrix(df,figsize=(15,10))
sns.pairplot(df,hue="Species")
sns.pairplot(df,diag_kind='kde',hue='Species')
#Importing Metrics for Evaluation

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# SEPARATING THE DEPENDENT AND INDEPENDENT VARIABLES ( X, Y )
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.metrics import accuracy_score

def generateClassificationReport(y_test,y_pred):
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))    
    print('accuracy is ',accuracy_score(y_test,y_pred))
#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)
#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)
#SUPPORT VECTOR MACHINE'S
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)
#K-NEAREST NEIGHBOUR
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

generateClassificationReport(y_test,y_pred)
#DECISION TREE 
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

generateClassificationReport(y_test,y_pred)