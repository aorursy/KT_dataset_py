import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.describe()
df.dtypes
plt.hist(df['age'], bins = [20,30,40,50,60,70,80], edgecolor = 'black')
plt.title('Age')
plt.hist(df['trestbps'], bins = [90,100,110,120,130,140,150,160,170,180,190,200], edgecolor = 'black')
plt.title('Resting Blood Pressure')
plt.hist(df['chol'], bins = 7, edgecolor = 'black')
plt.title('Cholesterol')
plt.hist(df['thalach'], bins = [70,80,90,100,110,120,130,140,150,160,170,180,190,200], edgecolor = 'black')
plt.title('Max Heart Rate')
plt.hist(df['oldpeak'], bins = 5, edgecolor = 'black')
plt.title('ST Depression')
plt.scatter(df['age'],df['trestbps'], s=30, c = '#b6eb7a', edgecolor = 'green', linewidth = 1, alpha = 0.8)
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.title('Age vs RBP')
plt.scatter(df['age'],df['chol'], s=30, c = '#9bdeac', edgecolor = 'green', linewidth = 1, alpha = 0.8)
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Age vs Cholesterol')
plt.scatter(df['age'],df['thalach'], s=30, c = '#b6eb7a', edgecolor = 'green', linewidth = 1, alpha = 0.8)
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.title('Age vs Max Heart Rate')
plt.scatter(df['age'],df['oldpeak'], s=30, c = '#a8df65', edgecolor = 'green', linewidth = 1, alpha = 0.8)
plt.xlabel('Age')
plt.ylabel('ST depression')
plt.title('Age vs ST depsression')
sns.jointplot(x=df['chol'], y=df['trestbps'], data=df, kind="kde")
sns.jointplot(x=df['thalach'], y=df['chol'], data=df)
f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df['chol'], df['oldpeak'], ax=ax)
sns.rugplot(df['chol'], color="g", ax=ax)
sns.rugplot(df['oldpeak'], vertical=True, ax=ax);
plt.scatter(df['trestbps'],df['thalach'], s=30, c = '#e2979c', edgecolor = 'red', linewidth = 1, alpha = 0.9)
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Max Heart Rate')
plt.title('RBP vs Max Heart Rate')
plt.scatter(df['trestbps'],df['oldpeak'], s=30, c = '#c70039', edgecolor = 'red', linewidth = 1, alpha = 0.8)
plt.xlabel('Resting Blood Pressure')
plt.ylabel('ST Depression')
plt.title('RBP vs ST Depression')
plt.scatter(df['thalach'],df['oldpeak'], s=40, c = '#ffbd69', edgecolor = 'orange', linewidth = 1, alpha = 1)
plt.xlabel('Max Heart Rate')
plt.ylabel('ST Depression')
plt.title('Max Heart Rate vs ST Depression')
X = df[['age','trestbps','chol','thalach','oldpeak']]
y = df['target']
sns.countplot(y)
yes, no = y.value_counts()
print('Number of Patients not diagnosed with Heart Disease:', no)
print('Number of Patients diagnosed with Heart Disease:', yes)
data = X
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std.iloc[:,:]], axis=1)
data = pd.melt(data, id_vars = 'target', var_name = 'features',
                value_name = 'value')
data.head()
plt.figure(figsize = (8,8))
sns.violinplot(x = 'features', y = 'value', hue = 'target', data = data, split = True, inner = 'quart', color='g')
plt.figure(figsize=(8,8))
sns.boxplot(x = 'features' , y='value', hue='target', data = data, color = 'r')
plt.figure(figsize = (8,8))
sns.swarmplot(x = 'features', y = 'value', hue = 'target', data = data)
sns.countplot(x="sex", data=df,hue='target')
male, fm = df['sex'].value_counts()
print('Number of Female Patients:', fm)
print('Number of Male Patients:', male)
sns.countplot(x="fbs", data=df,hue='target')
fbsno, fbsyes = df['fbs'].value_counts()
print('Fasting Blood Sugar > 120 :', fbsyes)
print('Fasting Blood Sugar < 120: ', fbsno)
sns.countplot(x="exang", data=df,hue='target')
no, yes = df['exang'].value_counts()
print('Exercise Induced Angina Yes: ', yes)
print('Exercise Induced Angina No:', no)
sns.catplot(x='exang',y='target',data=df,kind='point', hue = 'sex', color = '#e7305b')
sns.catplot(x='fbs',y='target',data=df,kind='point', hue = 'sex', color = '#436f8a')
sns.catplot(x='fbs',y='target',data=df,kind='point', hue = 'exang', color = '#79d70f')
sns.catplot(x = 'target',y='oldpeak',data=df,kind='violin',hue='sex', palette=sns.color_palette(['#ffdcb4', '#c060a1']))
sns.catplot(x = 'target',y='thalach',data=df,kind='box',hue='exang', palette=sns.color_palette(['#162447', '#74d4c0']))
sns.countplot(x="cp", data=df,hue='target')
sns.countplot(x="restecg", data=df,hue='target')
sns.countplot(x="slope", data=df,hue='target')
sns.countplot(x="ca", data=df,hue='target')
sns.countplot(x="thal", data=df,hue='target')
sns.catplot(x='cp',y='target',data=df,kind='point', color = 'g')
sns.catplot(x='restecg',y='target',data=df,kind='point', color = 'm' )
sns.catplot(x='slope',y='target',data=df,kind='point', color = '#ffa5b0')
sns.catplot(x='ca',y='target',data=df,kind='point',  color = '#1b6ca8')
plt.figure( figsize = (10,10))
sns.heatmap(df.corr(), annot = True)
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
df = pd.concat([df, a, b, c], axis = 1)
df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()
X = df.drop(columns = ['chol','fbs','age','sex','trestbps','restecg','target'], axis = 1)
y = df['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print('Training set shape: ', X_train.shape, y_train.shape)
print('Testing set shape: ', X_test.shape, y_test.shape)
# Function definition for fitting data
def model_fit(model,X, y,test):
    model.fit(X,y)
    y_pred = model.predict(test)
    return y_pred
# Function for calculating accuracy
from sklearn.metrics import accuracy_score
def accuracy(Y, y):
    return accuracy_score(Y,y)
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test = StandardScaler().fit(X_test).transform(X_test.astype(float))
model_accuracy = {}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn
y_knn = model_fit(knn, X_train, y_train, X_test)
knn_acc = accuracy(y_test, y_knn)
print('Test accuracy: ', knn_acc)
x = [0]
mean_acc = np.zeros(20)
mean_acc_train = np.zeros(20)
for i in range(1,21):
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat= knn.predict(X_test)
    yhat2 = knn.predict(X_train)
    mean_acc[i-1] = accuracy_score(y_test, yhat)
    mean_acc_train[i-1] = accuracy_score(y_train, yhat2)
    x.append(i)
plt.figure(figsize = (8,6))
plt.plot(np.arange(1,21), mean_acc, label = 'Test')
plt.plot(np.arange(1,21), mean_acc_train, label = 'Train')
plt.title('Test vs Train')
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.legend()
y_knn = model_fit(KNeighborsClassifier(n_neighbors = 13), X_train, y_train, X_test)
model_accuracy['KNN'] = accuracy(y_test, y_knn)
X = StandardScaler().fit(X).transform(X.astype(float))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X , y, cv = 5)
scores
scores.mean()
from sklearn import svm
clf2 = svm.SVC(C=1, kernel = 'rbf', gamma = 'auto')
clf2
y_svm = model_fit(clf2, X_train, y_train, X_test)
y2 = model_fit(clf2, X_train, y_train, X_train)
svm_acc = accuracy(y_test, y_svm)
svm2 = accuracy(y_train, y2)
model_accuracy['SVM'] = svm_acc
print('Train accuracy: ', svm2)
print('Test accuracy: ', svm_acc)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf2, X , y, cv = 5)
scores
scores.mean()
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf3 = DecisionTreeClassifier(random_state=42,criterion = 'entropy', max_depth = 3)
clf3
y_tree = model_fit(clf3,X_train,y_train, X_test)
y_tree2 = model_fit(clf3,X_train,y_train, X_train)
print("Train score: ", accuracy(y_train,y_tree2)," Test score: ",accuracy(y_test,y_tree))
scores = cross_val_score(clf3, X , y, cv = 5)
scores
scores.mean()
model_accuracy['Decision Tree'] = accuracy(y_test,y_tree)
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(random_state = 42, max_depth = 4, criterion = 'entropy')
clf4
y_rf = model_fit(clf4,X_train,y_train, X_test)
y_rf2 = model_fit(clf4,X_train,y_train, X_train)
print("Train score: ", accuracy(y_train,y_rf2)," Test score: ",accuracy(y_test,y_rf))
scores = cross_val_score(clf4, X , y, cv = 5)
scores
scores.mean()
model_accuracy['Random Forest'] = accuracy(y_test,y_rf)
from sklearn.linear_model import LogisticRegression
clf5 = LogisticRegression(C = 0.1, solver = 'newton-cg')
clf5
y_lr = model_fit(clf5,X_train,y_train, X_test)
y_lr2 = model_fit(clf5,X_train,y_train, X_train)
print("Train score: ", accuracy(y_train,y_lr2)," Test score: ",accuracy(y_test,y_lr))
scores = cross_val_score(clf5, X , y, cv = 5)
scores
scores.mean()
model_accuracy['Logistic Regression'] = accuracy(y_test,y_lr)
from sklearn.naive_bayes import GaussianNB
clf6 = GaussianNB()
clf6
y_nb = model_fit(clf6,X_train,y_train, X_test)
y_nb2 = model_fit(clf6,X_train,y_train, X_train)
print("Train score: ", accuracy(y_train,y_nb2)," Test score: ",accuracy(y_test,y_nb))
scores = cross_val_score(clf6, X , y, cv = 5)
scores
scores.mean()
model_accuracy['Naive Bayes'] = accuracy(y_test,y_nb)
from xgboost import XGBClassifier
clf7 = XGBClassifier(random_state=42, max_depth = 3, learning_rate = 0.01, n_estimators = 200)
clf7
y_xg = model_fit(clf7,X_train,y_train, X_test)
y_xg2 = model_fit(clf7,X_train,y_train, X_train)
print("Train score: ", accuracy(y_train,y_xg2)," Test score: ",accuracy(y_test,y_xg))
scores = cross_val_score(clf7, X , y, cv = 5)
scores
scores.mean()
model_accuracy['XGBoost'] = accuracy(y_test,y_xg)
plt.figure(figsize=(15,8))
plt.bar(model_accuracy.keys(),model_accuracy.values(), color = ['#87dfd6','#a6dcef','#ddf3f5','#111d5e','#111d5e','#a6dcef','#40bad5'])
plt.ylabel("Accuracy")
plt.xlabel("Classification Algorithm")
plt.show()
