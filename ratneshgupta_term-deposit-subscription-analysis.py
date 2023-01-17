# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings

import os

#Visulization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#importing Machine Learning parameters and classifiers 
import scipy.stats as stats
from scipy.stats import zscore

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,accuracy_score,confusion_matrix,recall_score,precision_score,f1_score, auc

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#Ensemble classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
df=pd.read_csv('/kaggle/input/bankfullcsv/bank-full.csv')
df.head()
df.shape
df.columns
df.info()
df.dtypes
df.isnull().sum()
cols = [col for col in df.columns]
col_with_unknown_value = []
for col in cols:
    if 'unknown' in df[col].values:
        col_with_unknown_value.append(col)
        
print("Columns with Unknown Values -",col_with_unknown_value)       

print("Unknown values count : \n")
for col in col_with_unknown_value:
    print(col," : ",df[df[col].str.contains('unknown')][col].count())
print("Other values count in attributes having unknown values -\n")
for col in col_with_unknown_value:
    print("===",col,"===")
    print(df.groupby(df[col])[col].count(),"\n")
for i in df.columns:
  print(i," :-")
  print(df[i].unique())
  print('==='*25)
df.apply(lambda x: len(x.unique()))
#df.describe()
#df.describe().transpose()
df.describe().T
### numerical 
numerical_cols = list(df.select_dtypes(exclude=['object']))
numerical_cols
df[numerical_cols].head()
### categorical
category_cols = list(df.select_dtypes(include=['object']))
category_cols
target='Target'
non_features=[target]
cat_features=[col for col in df.select_dtypes('object').columns if col not in non_features]
num_features=[col for col in df.select_dtypes(np.number).columns if col not in non_features]

print("Categorical Features :\n",cat_features,"\n")
print("Numerical Features :\n",num_features)
df[cat_features].describe()
#sns.pairplot(df)
#plt.show()

g = sns.pairplot(df )
g.set(xticklabels=[])
plt.show()
plt.figure(figsize=(15,15))
for i,col in enumerate(category_cols,start=1):
    plt.subplot(4,3,i);
    sns.barplot(df[col].value_counts().values, df[col].value_counts().index)
    plt.title(col)
f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#FA5858", "#64FE2E"]
labels ="Did not Open Term Suscriptions", "Opened Term Suscriptions"

plt.suptitle('Information on Term Suscriptions', fontsize=20)

df["Target"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=25)


# ax[0].set_title('State of Loan', fontsize=16)
ax[0].set_ylabel('% of Condition of Loans', fontsize=14)

# sns.countplot('loan_condition', data=df, ax=ax[1], palette=colors)
# ax[1].set_title('Condition of Loans', fontsize=20)
# ax[1].set_xticklabels(['Good', 'Bad'], rotation='horizontal')
palette = ["#64FE2E", "#FA5858"]

sns.barplot(x="education", y="balance", hue="Target", data=df, palette=palette, estimator=lambda x: len(x) / len(df) * 100)
ax[1].set(ylabel="(%)")
ax[1].set_xticklabels(df["education"].unique(), rotation=0, rotation_mode="anchor")
plt.show()
plt.style.use('seaborn-whitegrid')

df.hist(bins=20, figsize=(15,10), color='red')
plt.show()
df['Target'].value_counts()
plt.figure(figsize=(20,10))
for i,col in enumerate(num_features,start=1):
    plt.subplot(3,3,i);
    sns.boxplot(y=df[col],x=df[target]);
plt.show()
#df.education.replace('unknown',df.education.mode()[0],inplace=True)

df.loc[(df['age']>60) & (df['job']=='unknown'), 'job'] = 'retired'

df.loc[(df['education']=='unknown') & (df['job']=='management'), 'education'] = 'tertiary'
df.loc[(df['education']=='unknown') & (df['job']=='services'), 'education'] = 'secondary'
df.loc[(df['education']=='unknown') & (df['job']=='housemaid'), 'education'] = 'primary'

df.loc[(df['job'] == 'unknown') & (df['education']=='basic.4y'), 'job'] = 'blue-collar'
df.loc[(df['job'] == 'unknown') & (df['education']=='basic.6y'), 'job'] = 'blue-collar'
df.loc[(df['job'] == 'unknown') & (df['education']=='basic.9y'), 'job'] = 'blue-collar'
df.loc[(df['job']=='unknown') & (df['education']=='professional.course'), 'job'] = 'technician'
df['job'] = df.job.replace('unknown',df.job.mode()[0])
df['education'] = df.education.replace('unknown',df.education.mode()[0])
print("Other values count in attributes having unknown values -\n")
for col in col_with_unknown_value:
    print("===",col,"===")
    print(df.groupby(df[col])[col].count(),"\n")
fig, ax = plt.subplots(figsize=(13,10))

mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask, 1)] = True

sns.heatmap(df.corr(), annot=True,mask=mask, cmap='viridis',linewidths=0.5,ax=ax, fmt='.3f')

rotx = ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
roty = ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
df[num_features].plot(kind='box',subplots=True, layout=(4,4), fontsize=10, figsize=(16,16));
plt.show()

withOutliers = ['age', 'balance', 'duration', 'campaign','pdays','previous']

IQR=df[withOutliers].describe().T['75%']-df[withOutliers].describe().T['25%']

LW,UW = df[withOutliers].describe().T['25%']-(IQR*1.5),df[withOutliers].describe().T['75%']+(IQR*1.5)


for i in withOutliers:
    df[i][df[i]>UW[i]]=UW[i];
    df[i][df[i]<LW[i]]=LW[i]
df[withOutliers].plot(kind='box',subplots=True, layout=(4,4), fontsize=8, figsize=(14,14));
plt.style.use('seaborn-whitegrid')

df.hist(bins=20, figsize=(15,10), color='red')
plt.show() 
df['duration'] = df['duration'].apply(lambda n:n/60).round(2)
duration_campaign = sns.scatterplot(x='duration', y='campaign',data = df,
                     hue = 'Target')

plt.axis([0,65,0,65])
plt.ylabel('Number of Calls')
plt.xlabel('Duration of Calls (Minutes)')
plt.title('The Relationship between the Number and Duration of Calls')
# Annotation
plt.show()
print('Rows count having call duration less than 10 Sec -\t',df[df.duration < 10/60]['duration'].count())
# drop rows where call duration was less than 10 seconds
#dropped 342 rows
df = df.drop(df[df.duration < 10/60].index, axis = 0, inplace = False)
#putting age into bins
df.loc[df["age"] < 30,  'age'] = 20
df.loc[(df["age"] >= 30) & (df["age"] <= 39), 'age'] = 30
df.loc[(df["age"] >= 40) & (df["age"] <= 49), 'age'] = 40
df.loc[(df["age"] >= 50) & (df["age"] <= 59), 'age'] = 50
df.loc[df["age"] >= 60, 'age'] = 60
from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
df[category_cols] = df[category_cols].apply(LabelEncoder().fit_transform)
df.head()
df.corr()['Target'][:].plot.bar()
print("Target Attribute distribution \n")
print(df.Target.value_counts(),"\n")

fig,ax= plt.subplots()
fig.set_size_inches(20,5)
sns.countplot(x= "Target",data=df,ax= ax)
plt.show()
per_subs=round((df[df['Target'] == 1]['Target'].value_counts()[1]/df.Target.count())*100, 2)

print("% of clients subscribed for Term Deposite -\t",per_subs)
X = df.drop(['Target','contact','poutcome'],1)
y = df['Target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)

print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression

#Initialising Logistic Regression
lr_clf=LogisticRegression()

#Fitting on data
lr_clf.fit(X_train, y_train)

#Scoring the model on train data
print("Training Accuracy :\t ", lr_clf.score(X_train, y_train))

#Scoring the model on test_data
print("Testing Accuracy :\t  ",  lr_clf.score(X_test, y_test))

y_pred = lr_clf.predict(X_test)
metrics.confusion_matrix(y_pred, y_test)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

class_label = ["Positive", "Negative"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# Classification Report
print(classification_report(y_test, y_pred))
y_predictProb = lr_clf.predict_proba(X_train)

fpr, tpr, thresholds = roc_curve(y_train, y_predictProb[::,1])

roc_auc = auc(fpr, tpr)

print("auc :-",roc_auc)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

#Initialising Random Forest model
#knn_clf = KNeighborsClassifier(n_neighbors=5 , weights = 'distance' )
knn_clf = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn_clf, params_knn, cv=5)

#Fitting on data
knn_gs.fit(X_train, y_train)

#Scoring the model on train data
print("Training Accuracy :\t ", knn_gs.score(X_train, y_train))

#Scoring the model on test_data
print("Testing Accuracy :\t  ",  knn_gs.score(X_test, y_test))

#y_pred = knn_clf.predict(X_test)
#save best model
knn_best = knn_gs.best_estimator_
#check best n_neigbors value
print(knn_gs.best_params_)
# Confusion matrix
knn_cm=metrics.confusion_matrix(y_test, knn_gs.predict(X_test))
knn_cm
# Confusion Matrix
knn_cm=metrics.confusion_matrix(y_test, knn_gs.predict(X_test))

class_label = ["Positive", "Negative"]
df_cm = pd.DataFrame(knn_cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# Classification Report
print(classification_report(y_test, knn_gs.predict(X_test)))
y_predictProb = knn_gs.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[::,1])
roc_auc = auc(fpr, tpr)
print("auc :-",roc_auc)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
lr = LogisticRegression()

dt.fit(X_train,y_train)
knn.fit(X_train,y_train)
lr.fit(X_train,y_train)

dt_score=dt.score(X_test,y_test)
knn_score=knn.score(X_test,y_test)
lr_score=lr.score(X_test,y_test)

print("Accuracy Score of Decision Tree -\t",dt_score )
print("Accuracy Score of KNN  -\t",knn_score )
print("Accuracy Score of Logistic Regression -\t",lr_score )
    
from sklearn.ensemble import RandomForestClassifier

kfold = model_selection.KFold(n_splits=10, random_state=7)
#create a new random forest classifier
#rf = RandomForestClassifier()

model = RandomForestClassifier(n_estimators=100, max_features=3)
#create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [50, 100, 200]}

#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)

#fit model to training data
#rf_gs.fit(X_train, y_train)
#rf_score=rf_gs.score(X_test,y_test)

results = model_selection.cross_val_score(model, X, y, cv=kfold)

rf_score=results.mean()


print("Accuracy Score of Random Classifier -\t",rf_score)
rf_gs.fit(X_train, y_train)
y_pred=rf_gs.predict(X_test)
# Classification Report
print(classification_report(y_test, y_pred))
y_predictProb = rf_gs.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[::,1])
roc_auc = auc(fpr, tpr)
print("auc :-",roc_auc)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
from sklearn.ensemble import VotingClassifier
#model1 = LogisticRegression(random_state=1)
#model2 = DecisionTreeClassifier(random_state=1)
#model3 = LogisticRegression()
#model = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('knn', model3)], voting='hard')

model = VotingClassifier(estimators=[('lr', LogisticRegression(random_state=1)), ('dt', DecisionTreeClassifier(random_state=1)),('knn', KNeighborsClassifier())], voting='hard')
model.fit(X_train,y_train)
vt_score=model.score(X_test,y_test)

print('Accuracy score of Voting Classifier -\t',vt_score)
y_pred=model.predict(X_test)
# Classification Report
print(classification_report(y_test, y_pred))
kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()

model = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=7)

#model.fit(X_train,y_train)
#bg_clf=model.score(X_test,y_test)
#print('Accuracy score of Bagging Decision Tree Classifier -\t', bg_clf)


results = model_selection.cross_val_score(model, X, y, cv=kfold)
bg_score=results.mean()

print('Accuracy score of Bagging Decision Tree Classifier -\t', bg_score)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
# Classification Report
print(classification_report(y_test, y_pred))
y_predictProb = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[::,1])
roc_auc = auc(fpr, tpr)
print("auc :-",roc_auc)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import roc_auc_score

kfold = model_selection.KFold(n_splits=10, random_state=7)

model = AdaBoostClassifier(n_estimators=30, random_state=7)

#model.fit(X_train,y_train)
#ab_clf=model.score(X_test,y_test)
#print('Accuracy score of Adaboost Classifier -\t', ab_clf)

results = model_selection.cross_val_score(model, X, y, cv=kfold)
ab_score=results.mean()

print('Accuracy score of Adaboost Classifier -\t', ab_score)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
# Classification Report
print(classification_report(y_test, y_pred))
y_predictProb = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[::,1])
roc_auc = auc(fpr, tpr)
print("auc :-",roc_auc)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.1, random_state=22)
gbcl = gb_clf.fit(X_train, y_train)

pred_GB =gb_clf.predict(X_test)
gb_score = accuracy_score(y_test, pred_GB)

print('Accuracy score of GradientBoost Classifier -\t', gb_score)
