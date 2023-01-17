import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset= pd.read_csv('../input/diabetes/diabetes.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
dataset.head()
dataset.info()
dataset.describe()
dataset.corr()
dataset.hist(figsize = (20,20))
# 0 - pink color scatter indicates No Diabetes
# 1 - blue color scatter indicates Has Diabetes
plt.rcParams['figure.figsize'] = (40, 41)
plt.style.use('dark_background')

sns.pairplot(dataset, hue = 'Outcome', palette = 'husl')
plt.title('Pair plot for the data', fontsize = 40)
plt.show()
dataset_copy = dataset.copy(deep = True)
dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of Nans
print(dataset_copy.isnull().sum())
dataset_copy['Glucose'].fillna(dataset_copy['Glucose'].mean(), inplace = True)
dataset_copy['BloodPressure'].fillna(dataset_copy['BloodPressure'].mean(), inplace = True)
dataset_copy['SkinThickness'].fillna(dataset_copy['SkinThickness'].median(), inplace = True)
dataset_copy['Insulin'].fillna(dataset_copy['Insulin'].median(), inplace = True)
dataset_copy['BMI'].fillna(dataset_copy['BMI'].median(), inplace = True)
dataset_copy.hist(figsize = (20,20))
# 0 - pink color scatter indicates No Diabetes
# 1 - blue color scatter indicates Has Diabetes
plt.rcParams['figure.figsize'] = (40, 41)
plt.style.use('dark_background')

sns.pairplot(dataset_copy, hue = 'Outcome', palette = 'husl')
plt.title('Pair plot for the data', fontsize = 40)
plt.show()
plt.figure(figsize=(12,10))  
p=sns.heatmap(dataset.corr(), annot=True,cmap ='RdYlGn') 
plt.figure(figsize=(12,10))  
p=sns.heatmap(dataset_copy.corr(), annot=True,cmap ='RdYlGn') 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(dataset_copy.drop(["Outcome"],axis = 1))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,C=1.0,max_iter=200)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#calculate the details Logistic Regression
print('train_score classifier',classifier.score(X_train,y_train))
print('test_score classifier',classifier.score(X_test,y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# drawing confusion matrix
sns.heatmap(cm,center=True)
plt.show(10,10)

from sklearn.svm import SVC
svcmodel = SVC(kernel='rbf',degree=3)
svcmodel.fit(X_train,y_train)

# Predicting the Test set results SVM
y_pred = svcmodel.predict(X_test)

#calculate the details SVM
print('train_score svcmodel', svcmodel.score(X_train,y_train))
print('test_score svcmodel',svcmodel.score(X_test,y_test))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# drawing confusion matrix
sns.heatmap(cm,center=True)
plt.show(10,10)

from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(hidden_layer_sizes=100 ,activation='relu',alpha=0.01,epsilon=1E-08)
mlp_model.fit(X_train,y_train)

# Predicting the Test set results NNClassifier Model
y_pred = mlp_model.predict(X_test)

#calculate the details NNClassifier Model
print('train_score mlp_model', mlp_model.score(X_train,y_train))
print('test_score mlp_model',mlp_model.score(X_test,y_test))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# drawing confusion matrix
sns.heatmap(cm,center=True)
plt.show(10,10)
from sklearn.neighbors import KNeighborsClassifier
Knnclassifier_model = KNeighborsClassifier(n_neighbors=11)
Knnclassifier_model.fit(X_train,y_train)

# Predicting the Test set results KNeighborsClassifier
y_pred = Knnclassifier_model.predict(X_test)

#calculate the details KNeighborsClassifier
print('train_score Knnclassifier_model', Knnclassifier_model.score(X_train,y_train))
print('test_score Knnclassifier_model',Knnclassifier_model.score(X_test,y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# drawing confusion matrix
sns.heatmap(cm,center=True)
plt.show(10,10)
from sklearn.tree import DecisionTreeClassifier
DT_model=DecisionTreeClassifier(criterion='entropy')
DT_model.fit(X_train,y_train)

# Predicting the Test set results DecisionTreeClassifier Model
y_pred = DT_model.predict(X_test)

#calculate the details DecisionTreeClassifier Model
print('train_score DT_model', DT_model.score(X_train,y_train))
print('test_score DT_model',DT_model.score(X_test,y_test))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# drawing confusion matrix
sns.heatmap(cm,center=True)
plt.show(10,10)
from sklearn.naive_bayes import GaussianNB
gussian_model = GaussianNB(priors=None, var_smoothing=1e-09)
gussian_model.fit(X_train,y_train)

# Predicting the Test set results Naive Bayes
y_pred = gussian_model.predict(X_test)

#calculate the details Naive Bayes
print('train_score gussian_model', gussian_model.score(X_train,y_train))
print('test_score gussian_model',gussian_model.score(X_test,y_test))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# drawing confusion matrix
sns.heatmap(cm,center=True)
plt.show(10,10)
from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier(criterion='gini',n_estimators=200,max_depth=3)
rfc.fit(X_train,y_train)

# Predicting the Test set results RandomForestClassifier Model
y_pred = rfc.predict(X_test)

#calculate the details RandomForestClassifier Model
print('train_score rfc', rfc.score(X_train,y_train))
print('test_score rfc',rfc.score(X_test,y_test))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# drawing confusion matrix
sns.heatmap(cm,center=True)
plt.show(10,10)
r=[[classifier.score(X_train,y_train),svcmodel.score(X_train,y_train),Knnclassifier_model.score(X_train,y_train),gussian_model.score(X_train,y_train)
   ,DT_model.score(X_train,y_train),mlp_model.score(X_train,y_train),rfc.score(X_train,y_train)],
   [classifier.score(X_test,y_test),svcmodel.score(X_test,y_test),Knnclassifier_model.score(X_test,y_test),gussian_model.score(X_test,y_test)
    ,DT_model.score(X_test,y_test),mlp_model.score(X_test,y_test),rfc.score(X_test,y_test)]]


X = np.arange(7)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.style.use('seaborn-notebook')
ax.bar(X+ 0.00,r[0], color = 'blue', width = 0.30,label = 'train score')
ax.bar(X+ 0.40, r[1], color = 'red', width = 0.30,label = 'test score')
for i,m in list(zip(X,r[0])):
  plt.text(x = i ,y = m,s = m)
for i,m in list(zip(X,r[1])):
  plt.text(x = i + 0.45 ,y = m,s = m)
ax.set_xlabel('Models')
ax.set_ylabel('score')
ax.set_xticklabels(('','classifier', 'SVC', 'KNN', 'Gussian', 'DT','MLP','RFC'))
plt.legend()
plt.show(10,10)