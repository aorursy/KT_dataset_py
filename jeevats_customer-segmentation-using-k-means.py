!pip install math
!pip install mpl_toolkits
!pip install info_gain
!pip install sklearn
!pip install info_gain
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation
import warnings
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D 
from info_gain import info_gain
from sklearn.linear_model import LogisticRegression

train=pd.read_csv("../input/german_credit_train.csv")
train.head()
train.info()
train.nunique()
ax = train['Risk'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Risk Values")
ax.set_xlabel("Risk Good or Bad")
ax.set_ylabel("Frequency")
count_no_sub = len(train[train['Risk']=='good'])
count_sub = len(train[train['Risk']=='bad'])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of good is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of bad", pct_of_sub*100)
def scatters(data, h=None, pal=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.scatterplot(x="Credit amount",y="Duration", hue=h, palette=pal, data=data, ax=ax1)
    sns.scatterplot(x="Age",y="Credit amount", hue=h, palette=pal, data=data, ax=ax2)
    sns.scatterplot(x="Age",y="Duration", hue=h, palette=pal, data=data, ax=ax3)
    plt.tight_layout()
scatters(train, h="Sex")
ax = train['Housing'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Housing")
ax.set_xlabel("Housing own or not")
ax.set_ylabel("Frequency")
def boxes(x,y,h,r=45):
    fig, ax = plt.subplots(figsize=(10,6))
    box = sns.boxplot(x=x,y=y, hue=h, data=train)
    box.set_xticklabels(box.get_xticklabels(), rotation=r)
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    
boxes("Purpose","Credit amount","Sex")
boxes("Purpose","Credit amount","Job")
boxes("Purpose","Duration","Sex")
boxes("Job","Duration","Sex")
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train["Credit amount"], train["Duration"], train["Job"])
ax.set_xlabel("Credit amount")
ax.set_ylabel("Duration")
ax.set_zlabel("Job")
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train["Credit amount"], train["Duration"], train["Age"])
ax.set_xlabel("Credit amount")
ax.set_ylabel("Duration")
ax.set_zlabel("Age")
columns = ["Age","Credit amount", "Duration"]
clust_train = train.loc[:,columns]
def distributions(data):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,12))
    sns.distplot(data["Age"], ax=ax1)
    sns.distplot(data["Credit amount"], ax=ax2)
    sns.distplot(data["Duration"], ax=ax3)
    plt.tight_layout()
distributions(clust_train)
cluster_log = np.log(clust_train)
distributions(cluster_log)
cluster_log.head()
sns.heatmap(train.corr())
plt.show()
train.drop(["Unnamed: 0"], axis = 1, inplace = True) 
  
train = train.fillna(0)
train.head()

categories = ['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose',]


X, Y = train[categories], train.Risk
#from info_gain import info_gain
#ig  = info_gain.info_gain(X, Y)
#iv  = info_gain.intrinsic_value(fruit, colour)
#igr = info_gain.info_gain_ratio(fruit, colour)

ig  = info_gain.info_gain(X['Age'], Y)
ig
ig  = info_gain.info_gain(X['Sex'], Y)
ig
ig  = info_gain.info_gain(X['Job'], Y)
ig
ig  = info_gain.info_gain(X['Housing'], Y)
ig
ig  = info_gain.info_gain(X['Saving accounts'], Y)
ig
ig  = info_gain.info_gain(X['Checking account'], Y)
ig
ig  = info_gain.info_gain(X['Credit amount'], Y)
ig
ig  = info_gain.info_gain(X['Duration'], Y)
ig
ig  = info_gain.info_gain(X['Purpose'], Y)
ig
train['Job'].value_counts()
train['Housing'].value_counts()
train['Saving accounts'].value_counts()
train['Checking account'].value_counts()
train['Purpose'].value_counts()
tmp=train
matplotlib.pyplot.boxplot(tmp['Credit amount'])
matplotlib.pyplot.boxplot(tmp['Duration'])
matplotlib.pyplot.boxplot(tmp['Age'])
train['Credit amount']=cluster_log['Credit amount']
train['Duration']=cluster_log['Duration']
train['Age']=cluster_log['Age']
train1=train
matplotlib.pyplot.boxplot(train['Credit amount'])
matplotlib.pyplot.boxplot(cluster_log['Duration'])
matplotlib.pyplot.boxplot(cluster_log['Age'])
#import numpy as np
#import pandas as pd
"""outliers=[]
def detect_outlier(data_1):
    
    threshold=1.1
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
outliers=detect_outlier(train1['Credit amount'])
outliers
minval=min(outliers)
median = train1['Credit amount'].median()
median
train1.loc[train1['Credit amount'] >=minval, 'Credit amount'] = median

"""
train['Saving accounts'] = le.fit_transform(train['Saving accounts'].astype(str))
train['Checking account'] = le.fit_transform(train['Checking account'].astype(str))
train['Sex'] = le.fit_transform(train['Sex'].astype(str))
train['Housing'] = le.fit_transform(train['Housing'].astype(str))
train['Purpose'] = le.fit_transform(train['Purpose'].astype(str))
train['Risk'] = le.fit_transform(train['Risk'].astype(str))
train.head()
clms = ['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose',]


from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(train[clms], train['Risk'], test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=2)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)  
y_pred = classifier.predict(X_test)
print('Accuracy of Knn classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=200)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#importance based selection
clms = ['Age','Saving accounts','Checking account','Credit amount','Duration']
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(train[clms], train['Risk'], test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=2)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
print('Accuracy of Knn classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
