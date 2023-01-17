import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
data=pd.read_csv('../input/diabetes.csv')
data.head()
data.describe()
data.isnull().sum()
plt.figure(figsize=(5,5))
plt.title('How many have diabetes(0=No,1=Yes)')
locs, labels = plt.xticks()
sns.countplot(data['Outcome'])
data['Outcome'].value_counts()
from plotnine import *
ggplot(data,aes(x='Age',y='Glucose',colour='Outcome'))+geom_point()+stat_smooth()
ggplot(data,aes(x='Age',y='Glucose',colour='BloodPressure'))+geom_point()+stat_smooth()+facet_wrap('~Outcome')
ggplot(data,aes(x='Age',y='Pregnancies'))+geom_point(aes(color='BMI'))+facet_wrap('~Outcome')+stat_smooth()
m=data.loc[:,data.columns!='Outcome'].corr()
plt.figure(figsize=(10,10))
sns.heatmap(m,annot=True,cmap="Reds")
sns.lmplot(x='Age',y='Pregnancies',hue='Outcome',data=data)
sns.lmplot(x='Insulin',y='SkinThickness',hue='Outcome',data=data)
sns.lmplot(x='BMI',y='SkinThickness',hue='Outcome',data=data)
sns.lmplot(x='Insulin',y='Glucose',hue='Outcome',data=data)
sns.pairplot(data[['Age','Pregnancies','Insulin','BMI','SkinThickness','Glucose']])
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
x,y = data.loc[:,data.columns != 'Outcome'], data.loc[:,'Outcome']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=5) accuracy is: ',knn.score(x_test,y_test)) # accuracy

# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
rf = RandomForestClassifier(random_state = 4)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))
predictors=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': rf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()