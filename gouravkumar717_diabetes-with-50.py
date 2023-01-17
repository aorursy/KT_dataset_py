import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
data=data = pd.read_csv('../input/diabetes.csv')
data.head()
data.info()
plt.figure(figsize=(8,5))
sns.heatmap(data.isnull() , yticklabels=False , cbar=False,cmap='viridis')
data['Outcome'].value_counts()
plt.figure(figsize=(8,9))
sns.pairplot(data , hue = 'Outcome')
data.describe()
corrmat = data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corrmat , vmax =.8 , square=True , annot = True)
data.columns
data.plot(kind = 'box' , subplots=True , layout=(3,3), sharex =False , sharey = False , figsize=(10,8))
f , axes = plt.subplots(nrows= 4 , ncols = 2 , figsize=(15,20))


sns.distplot(data.Pregnancies , kde = False , color = 'g' , ax = axes[0][0]).set_title('Pregnanices') 
axes[0][0].set_ylabel('Count')


sns.distplot(data.Glucose , kde = False , color ='r' , ax = axes[0][1]).set_title('Glucose')
axes[0][1].set_ylabel('Count')


sns.distplot(data.BloodPressure , kde = False , color ='b' , ax = axes[1][0]).set_title('BloodPressure')
axes[1][0].set_ylabel('Count')


sns.distplot(data.SkinThickness , kde = False , color ='y' , ax = axes[1][1]).set_title('SkinThickness')
axes[1][1].set_ylabel('Count')



sns.distplot(data.Insulin , kde = False , color ='r' , ax = axes[2][0]).set_title('Insulin')
axes[2][0].set_ylabel('Count')


sns.distplot(data.BMI , kde = False , color ='b' , ax = axes[2][1]).set_title('BMI')
axes[2][1].set_ylabel('Count')


sns.distplot(data.DiabetesPedigreeFunction , kde = False , color ='r' , ax = axes[3][0]).set_title('DiabetesPedigreeFunction')
axes[3][0].set_ylabel('Count')



sns.distplot(data.Age , kde = False , color ='black' , ax = axes[3][1]).set_title('Age')
axes[3][1].set_ylabel('Count')




pima_new= data
pima_new.info()
pima_new  =  pima_new[pima_new['Pregnancies']<13]
pima_new  =  pima_new[pima_new['Glucose']>30]



pima_new =pima_new[pima_new['BMI']>10]
pima_new = pima_new[pima_new['BMI']<50]

pima_new = pima_new[pima_new['DiabetesPedigreeFunction']<1.2]
pima_new = pima_new[pima_new['Age'] <65]
pima_new.plot(kind='box' , subplots=True , layout=(3,3) , sharex=False, sharey=False , figsize=(10,9))
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'Pregnancies'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'Pregnancies'] , color='r',shade=True, label='Yes')
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'Glucose'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'Glucose'] , color='r',shade=True, label='Yes')
ax.set(xlabel='Glucose', ylabel='Frequency')
plt.title('Glucose vs Yes or No')
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'BMI'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'BMI'] , color='r',shade=True, label='Yes')
ax.set(xlabel='BMI', ylabel='Frequency')
plt.title('BMI vs Yes or No')

fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'DiabetesPedigreeFunction'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'DiabetesPedigreeFunction'] , color='r',shade=True, label='Yes')
ax.set(xlabel='DiabetesPedigreeFunction', ylabel='Frequency')
plt.title('DiabetesPedigreeFunction vs Yes or No')
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'Age'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'Age'] , color='r',shade=True, label='Yes')
ax.set(xlabel='Age', ylabel='Frequency')
plt.title('Age vs Yes or No')
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(pima_new.drop('Outcome' , axis = 1) , pima_new['Outcome'], test_size=0.30 , random_state=123)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn import metrics




def classification_cv(model):
    kfold = model_selection.KFold(n_splits=10 , random_state=7 )
    scoring = 'accuracy'
    results = model_selection.cross_val_score(model , X_train_transformed,y_train,cv=kfold, scoring=scoring)
    
    return(print('Accuracy :%.3f (%.3f)')%(result.mean(), result.std()))
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)



dtree = tree.DecisionTreeClassifier(class_weight = 'balanced' , min_weight_fraction_leaf =0.01)
dtree = dtree.fit(X_train_transformed , y_train)

importances = dtree.feature_importances_
feat_names = pima_new.drop(['Outcome'] , axis=1).columns


indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))

plt.title('Feature importances by DecisionTreeClassifier')
plt.bar(range(len(indices)) , importances[indices] , color = 'blue' , align='center')
plt.step(range(len(indices)) , np.cumsum(importances[indices]),where='mid', label = 'Cumulative')
plt.xticks(range(len(indices)) , feat_names[indices] , rotation = 'vertical', fontsize=14)

plt.xlim([-1 , len(indices)])
plt.show()
def base_rate_model(X) :
    y = np.zeros(X.shape[0])
    return y
y_base_rate = base_rate_model(X_test_transformed)
from sklearn.metrics import accuracy_score
print ("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

print ("---Base Model---")
base_roc_auc = roc_auc_score(y_test, y_base_rate)
print ("Base Rate AUC = %2.2f" % base_roc_auc)
print(classification_report(y_test, y_base_rate))
print ("---Confusion Matrix---")
print(confusion_matrix(y_test, y_base_rate))





