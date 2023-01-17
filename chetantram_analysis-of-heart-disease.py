import numpy as np 

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier



from sklearn import metrics

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
data=pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head()
data.describe().T
data.info()
data.isnull().sum()
corr=data.corr()
plt.figure(figsize=[10,7])

sns.heatmap(corr,annot=True)
data.target.value_counts()
sns.pairplot(data)
sns.countplot(data["target"])
nodisease = len(data[data.target == 0])

hasdisease = len(data[data.target == 1])

print("% of Patients dont have Heart Disease: {:.2f}%".format((nodisease / (len(data.target))*100)))

print("% of Patients Have Heart Disease: {:.2f}%".format((hasdisease / (len(data.target))*100)))
sns.countplot(x='sex',data=data)#seems like male patients are prone to heart disease
data.age.value_counts()[:10]
sns.barplot(x=data.age.value_counts()[:10].index,y=data.age.value_counts()[:10].values)

plt.title("Age Analysis")
pd.crosstab(data.cp,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency wrt  Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.show()
#dropping few cols which are not needed 

data = data.drop(columns = ['cp', 'thal', 'slope'])
#initialising 'x' and 'y' and split at 80%

x = data.drop(['target'], axis = 1)

y = data["target"]
#Normalizing the data

x_data = (x - np.min(x)) / (np.max(x) - np.min(x)).values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
#Logistic Regression

accuracies = {}



lr = LogisticRegression()

lr.fit(x_train,y_train)

acc = lr.score(x_test,y_test)*100



accuracies['Logistic Regression'] = acc

print("Test Accuracy {:.2f}%".format(acc))
#Naive Bayes

nb = GaussianNB()

nb.fit(x_train, y_train)

acc = nb.score(x_test,y_test)*100



accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
#K-Nearest Neigbors

knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)



accuracies['KNN'] = acc

print("Maximum KNN Score is {:.2f}%".format(acc))
#Decision Trees

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

acc = dt.score(x_test, y_test)*100



accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
ada = AdaBoostClassifier(n_estimators=100)

ada.fit(x_train, y_train)

y_pred = ada.predict(x_test)



accuracies['AdaBoost'] = acc

print("Maximum AdaBoost Score is {:.2f}%".format(acc))
#Random Forest

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train, y_train)

acc = rf.score(x_test,y_test)*100



accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
xax=list(accuracies.keys())
yax=list(accuracies.values())
#Comparing the Models

x_pos = [i for i, _ in enumerate(xax)]

fig, ax = plt.subplots(figsize=(15,6))



rects1 = ax.bar(x_pos, yax,color=['violet','red','blue','green','orange','cyan'])

plt.xlabel("Models")

plt.ylabel("Accuracy Scores %")

plt.title("Models Comparision")

plt.xticks(x_pos, xax)



def autolabel(rects):

    

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%f' % float(height),

        ha='center', va='bottom')

autolabel(rects1)

plt.show()