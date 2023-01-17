# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Gaussian naive Bayes classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#reading the iris dataset 
iris = pd.read_csv('../input/Iris.csv')
print(iris.shape)
iris=iris.drop('Id',axis=1)
display(iris.head())
#plotting the confusion matrix 
def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
       
    )
plot_correlation_map(iris)
plt.show()
iris.describe()
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=(12,10))
sns.stripplot(x="Species", y="SepalLengthCm", data=iris,ax=ax1, jitter=True);
sns.swarmplot(x="Species", y="SepalWidthCm", data=iris,ax=ax2);
fig, (ax3, ax4) = plt.subplots(ncols=2, sharey=True,figsize=(12,10))
sns.stripplot(x="Species", y="PetalLengthCm", data=iris,ax=ax3);
sns.stripplot(x="Species", y="PetalWidthCm", data=iris,ax=ax4);
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=(15,5))
sns.swarmplot(x="SepalWidthCm", y="SepalLengthCm", hue="Species", data=iris,ax=ax1);
sns.swarmplot(x="PetalWidthCm", y="PetalLengthCm", hue="Species", data=iris,ax=ax2);
fig, (ax3, ax4) = plt.subplots(ncols=2, sharey=True,figsize=(15,5))
sns.swarmplot(x="PetalWidthCm", y="SepalWidthCm", hue="Species", data=iris,ax=ax3);
sns.swarmplot(x="PetalLengthCm", y="SepalWidthCm", hue="Species", data=iris,ax=ax4);
sns.lmplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", data=iris);
iris=iris.drop('SepalWidthCm',axis=1)
trainSet, testSet = train_test_split(iris, test_size = 0.33)
print(trainSet.shape)
print(testSet.shape)
# Format the data and expected values for SKLearn
trainData = pd.DataFrame(trainSet[['SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']]).values
trainTarget = pd.DataFrame(trainSet[['Species']]).values.ravel()
testData = pd.DataFrame(testSet[['SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']]).values
testTarget = pd.DataFrame(testSet[['Species']]).values.ravel()

#using Niave Bayes algorithm
classifier = GaussianNB()
classifier.fit(trainData, trainTarget)
predicted_value = classifier.predict(testData)

predictions = dict()
accuracy = accuracy_score(testTarget,predicted_value) 
predictions['Naive-Bayes']=accuracy*100
print("The accuracy of the model is {}".format(accuracy))
confusionmatrix = confusion_matrix(testTarget, predicted_value)
cm=pd.DataFrame(confusion_matrix(testTarget, predicted_value))
print("The confusion matrix of the model is \n{}".format(cm))
skplt.metrics.plot_confusion_matrix(testTarget, predicted_value, normalize=True)
plt.show()
#Using Random forest
clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
clf.fit(trainData, trainTarget)
predicted_value = clf.predict(testData)
accur = accuracy_score(testTarget,predicted_value) 
predictions['Random-Forest']=accur*100
cm = (confusion_matrix(testTarget, predicted_value))
print("The accuracy score of the model is {}".format(accur))
print("The confusion matrix of the model is \n{}".format(cm))
#using decision Tree
clf1=DecisionTreeClassifier()
clf1.fit(trainData, trainTarget)
predicted_value = clf1.predict(testData)
accur = accuracy_score(testTarget,predicted_value) 
predictions['Decision Tree']=accur*100
cm = (confusion_matrix(testTarget, predicted_value))
print("The accuracy score of the model is {}".format(accur))
print("The confusion matrix of the model is \n{}".format(cm))
#using KNN algorithm
clf2=KNeighborsClassifier()
clf2.fit(trainData, trainTarget)
predicted_value = clf2.predict(testData)
accur = accuracy_score(testTarget,predicted_value)
predictions['KNN']=accur*100
cm = (confusion_matrix(testTarget, predicted_value))
print("The accuracy score of the model is {}".format(accur))
print("The confusion matrix of the model is \n{}".format(cm))
fig, (ax1) = plt.subplots(ncols=1, sharey=True,figsize=(15,5))
df=pd.DataFrame(list(predictions.items()),columns=['Algorithms','Percentage'])
display(df)
sns.barplot(x="Algorithms", y="Percentage", data=df,ax=ax1);
#submission file 
submission= pd.DataFrame()
columns=['SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']
submission[columns] = testSet[columns]
submission['Species']=predicted_value
submission.to_csv('Submission.csv',index=True)
print("Submission file Created")