import pandas as pd  # Load data

import numpy as np # Scientific Computing

import matplotlib.pyplot as plt  # Data Visualization

import seaborn as sns  # Data Visualization

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve,auc

from sklearn.metrics import confusion_matrix, classification_report

import missingno as msno  # showing null values by bar graph

import warnings  # Ignore Warnings

warnings.filterwarnings("ignore")

sns.set()   # Set Graphs Background
data = pd.read_csv('../input/heartdata1/heart.csv')

data.head()
# shape showing how many data rows & columns have

data.shape
# info() showing rows & columns number column name data type Non-null count

data.info()
# isnull() check null value

data.isnull()
# any() check null values by columns

data.isnull().any()
# innull().sum() show total null values 

data.isnull().sum()
# missingno() showing null values by bar graph

msno.bar(data, figsize=(12,6))

plt.show()
# heatmap() showing null values

sns.heatmap(data.isnull(), yticklabels=False,cbar=False, cmap='viridis')

plt.show()
# describe() statistical information

data.describe()
# hist() histogram 

data.hist(figsize = (15,12))

plt.show()
# value_counts() total unique value count

print(data.sex.value_counts())

sns.countplot(x='sex', data=data)

plt.show()
# value_counts() total unique value count

print(data.ca.value_counts())

sns.countplot(x='ca', data=data)

plt.show()
# value_counts() total unique value count

print(data.fbs.value_counts())

sns.countplot(x='fbs', data=data)

plt.show()
# value_counts() total unique value count

print(data.cp.value_counts())

sns.countplot(x='cp', data=data)

plt.show()
# value_counts() total unique value count

print(data.exang.value_counts())

sns.countplot(x='exang', data=data)

plt.show()
# value_counts() total unique value count

print(data.restecg.value_counts())

sns.countplot(x='restecg', data=data)

plt.show()
# value_counts() total unique value count

print(data.slope.value_counts())

sns.countplot(x='slope', data=data)

plt.show()
# value_counts() total unique value count

print(data.thal.value_counts())

sns.countplot(x='thal', data=data)

plt.show()
# value_counts() total unique value count

print(data.target.value_counts())

sns.countplot(x='target', data=data)

plt.show()
# scatter() relation between two columns

plt.figure(figsize=(10,8))

plt.scatter(data['age'],data['chol'])

plt.title('Age VS Chol', fontsize=20)

plt.xlabel('Age', fontsize=20)

plt.ylabel('Chol', fontsize=20)

plt.show()
# scatter() relation between two columns

plt.figure(figsize=(10,8))

plt.scatter(data['age'],data['trestbps'])

plt.title('Age VS Trestbps', fontsize=20)

plt.xlabel('Age', fontsize=20)

plt.ylabel('Trestbps', fontsize=20)

plt.show()
# scatter() relation between two columns

plt.figure(figsize=(10,8))

plt.scatter(data['age'],data['thalach'])

plt.title('Age VS Thalach', fontsize=20)

plt.xlabel('Age', fontsize=20)

plt.ylabel('Thalach', fontsize=20)

plt.show()
# scatter() relation between two columns

plt.figure(figsize=(10,8))

plt.scatter(data['trestbps'],data['chol'])

plt.title('Trestbps VS Chol', fontsize=20)

plt.xlabel('Trestbps', fontsize=20)

plt.ylabel('Chol', fontsize=20)

plt.show()
# scatter() relation between two columns

plt.figure(figsize=(10,8))

plt.scatter(data['trestbps'],data['thalach'])

plt.title('Trestbps VS Thalach', fontsize=20)

plt.xlabel('Trestbps', fontsize=20)

plt.ylabel('Thalach', fontsize=20)

plt.show()
# scatter() relation between two columns

plt.figure(figsize=(10,8))

plt.scatter(data['chol'],data['thalach'])

plt.title('Chol VS Thalach', fontsize=20)

plt.xlabel('Chol', fontsize=20)

plt.ylabel('Thalach', fontsize=20)

plt.show()
# boxplot() showing outlier

box = data[['age','trestbps','chol','thalach']]

plt.figure(figsize=(12,8))

sns.boxplot(data=box)

plt.show()
data_iqr = box

Q1 = data_iqr.quantile(0.25)

Q3 = data_iqr.quantile(0.75)

iqr = Q3 - Q1



data_iqr_clean = data_iqr[~((data_iqr < (Q1 - 1.5*iqr)) | (data_iqr > (Q3 + 1.5*iqr))).any(axis=1)]
# boxplot() showing outlier

box = data_iqr_clean[['age','trestbps','chol','thalach']]

plt.figure(figsize=(12,8))

sns.boxplot(data=box)

plt.show()
# distplot() same as histogram

fig, ax = plt.subplots(2,2, figsize=(10,8))

sns.distplot(data_iqr_clean.age, bins = 20, ax=ax[0,0]) 

sns.distplot(data_iqr_clean.trestbps, bins = 20, ax=ax[0,1]) 

sns.distplot(data_iqr_clean.chol, bins = 20, ax=ax[1,0])

sns.distplot(data_iqr_clean.thalach, bins = 20, ax=ax[1,1])

plt.show()
# corr() relation with data

corr=data.corr()



plt.figure(figsize=(14,8))



sns.heatmap(corr, vmax=.8, linewidths=0.01,annot=True,cmap='summer',linecolor="black")

plt.title('Correlation between features')

plt.show()
x = data.iloc[:,0:-1].values # All rows & columns present except Target column

y = data.iloc[:,-1].values # Only target column present
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=4)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
svm = SVC(kernel='rbf',random_state=0)

svm.fit(x_train,y_train)
svm.score(x_test,y_test)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
svm = SVC(kernel='rbf',random_state=0,probability=True) #probability for predict_proba

svm.fit(x_train,y_train)
svm.score(x_test,y_test)
param_grid = {'C':[1,10,100,200],

              'kernel':['rbf','poly','linear','sigmoid'],

              'degree':[1,2,4,6],

              'gamma':[0.01,0.1,0.5,1]}



grid=GridSearchCV(SVC(), param_grid=param_grid, cv=4)

grid.fit(x_train,y_train)



y_pred = grid.predict(x_test)



print("Accuracy: {}".format(grid.score(x_test, y_test)))

print("Tuned Model Parameters: {}".format(grid.best_params_))
svm = SVC(C=1,kernel='poly',degree=1,gamma=0.5,probability=True)

svm.fit(x_train,y_train)
svm.score(x_test,y_test)
y_pred = svm.predict(x_test)

cm = confusion_matrix(y_pred,y_test)

print('Confusion Matrix \n',cm)
plt.figure(figsize=(6,4))

sns.heatmap(cm,annot=True,fmt="d") 

plt.show()
cr = classification_report(y_pred,y_test)

print('Classification Report\n',cr)
y_prob = svm.predict_proba(x_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test,y_prob)

Auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr,linestyle='-',label='(auc=%0.3f)' %Auc)

plt.plot([0,1],[0,1])

plt.title('ROC CURVE')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc=4)

plt.show()
num = np.arange(1, 30)

train_accuracy = []

test_accuracy = []

for i, k in enumerate(num):

    svm = SVC(C=k)

    svm.fit(x_train,y_train)

    train_accuracy.append(svm.score(x_train, y_train))

    test_accuracy.append(svm.score(x_test, y_test))



# Plot

plt.figure(figsize=(10,6))

plt.plot(num, test_accuracy, label = 'Testing Accuracy')

plt.plot(num, train_accuracy, label = 'Training Accuracy')

plt.legend(loc=10)

plt.title('value VS Accuracy')

plt.xlabel('Number of C')

plt.ylabel('Accuracy')

plt.xticks(num)

plt.show()

print("Best accuracy is {} with C = {}".format(np.max(test_accuracy),

                                               1+test_accuracy.index(np.max(test_accuracy))))
data1 = data.copy() # copy data

data1.head()
data1_thal = pd.get_dummies(data['thal'],prefix='thal')

data1_thal.head()
data1_slope = pd.get_dummies(data['slope'],prefix='slope')

data1_slope.head()
data1_restecg = pd.get_dummies(data['restecg'],prefix='restecg')

data1_restecg.head()
data1_cp = pd.get_dummies(data['cp'],prefix='cp')

data1_cp.head()
data1_ca = pd.get_dummies(data['ca'],prefix='ca')

data1_ca.head()
data2 = pd.concat([data1_cp,data1_restecg,data1_slope,data1_ca,data1_thal],axis='columns')
data2.head()
data3 = pd.concat([data1,data2],axis='columns')

data3.head()
data3 = data3.drop(['cp','restecg','slope','thal','ca','target'], axis=1)

data3.head()
data3 = pd.concat([data3,data.target],axis=1)

data3.head()
x = data3.iloc[:,:-1].values # All rows & columns present except Target column

y = data3.iloc[:,-1].values # Only target column present
xx_train,xx_test,yy_train,yy_test = train_test_split(x,y, test_size=0.2, random_state=4)
svm = SVC(C=1,kernel='poly',degree=1,gamma=0.5,probability=True)

svm.fit(xx_train,yy_train)
param_grid = {'ccp_alpha':[0.0,0.1,0.2,0.3,0.4,1],

              'criterion':['gini','entropy'],

              'max_depth':[5,10,50,100,200],

              'max_leaf_nodes':[5,10,50,100,200],

              'random_state':[2,5,10,20,42]}



grid=GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=4)

grid.fit(xx_train,yy_train)



print("Tuned Model Parameters: {}".format(grid.best_params_))
dtc = DecisionTreeClassifier(criterion='gini',max_depth=10,max_leaf_nodes=10,

                            ccp_alpha=0.0,random_state=2)

dtc.fit(xx_train,yy_train)
param_grid = {'C':[1.0,2.0,5.0,10.0,20.0],

              'penalty':['l1','l2','none','elasticnet'],

              'max_iter':[50,100,200,300,500],

              'multi_class':['auto','ovr','multinomial']}



grid=GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=4)

grid.fit(xx_train,yy_train)



print("Tuned Model Parameters: {}".format(grid.best_params_))
lg = LogisticRegression(C=5.0,max_iter=50,multi_class='multinomial',penalty='l2')

lg.fit(xx_train,yy_train)
param_grid = {'n_estimators':[50, 100,150,200,300],

              'criterion':['gini','entropy'],

              'max_depth':[5,10,50,100,200]}



grid=GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=4)

grid.fit(xx_train,yy_train)



print("Tuned Model Parameters: {}".format(grid.best_params_))
rfc = RandomForestClassifier(criterion='gini',max_depth=5,n_estimators=50)

rfc.fit(xx_train,yy_train)
print("Support Vector Machine Accuracy: {}".format(svm.score(xx_test, yy_test)))

print("DecisionTreeClassifier Accuracy: {}".format(dtc.score(xx_test, yy_test)))

print("LogisticRegression Accuracy: {}".format(lg.score(xx_test, yy_test)))

print("RandomForestClassifier Accuracy: {}".format(rfc.score(xx_test, yy_test)))