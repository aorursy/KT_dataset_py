import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('../input/mushrooms.csv')
data.head()
data.shape
data.columns
data.info()
lst=[]
for col in data.columns:
    lst.append(data[col].nunique())
x=dict(zip(data.columns,lst) )
x
data['class'].value_counts()
sns.countplot(x='class', palette='RdBu', data=data)
plt.title('Number of p and e mushrooms')
p_data=data.loc[data['class']=='p']
e_data=data.loc[data['class']=='e']
e_data.head(2)
lst=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
fig, axes = plt.subplots(11, 2, figsize=(15,45))
sns.set_style('white')
plt.subplots_adjust (hspace=0.4, wspace=0.2)
n=0
for i in lst:
    data_p_1=p_data.groupby(['class',i]).agg({i:'count'})
    data_p_1.columns=['number']
    data_p_1['perc']= (data_p_1['number']*100)/3916
    data_e_1=e_data.groupby(['class',i]).agg({i:'count'})
    data_e_1.columns=['number']
    data_e_1['perc']= (data_e_1['number']*100)/4208
    data_new_i=pd.concat([data_p_1,data_e_1],axis=0)
    data_new_i.drop(['number'],axis=1,inplace=True)
    data_new1=data_new_i.unstack(level=0).fillna(0)
    print(data_new1)
    data_new1.plot(kind='bar',cmap='rainbow', ax=axes[n//2, n%2])
    axes[n//2, n%2].set_ylabel('percentage')
    n+=1
data=data.drop(['veil-type'],axis=1)
data.shape
encoder=LabelEncoder()
lst=[]
for col in data.columns:
    if data[col].nunique()<=2:
        lst.append(str(col))
print(lst)
for i in lst:
    encoder.fit(data[i].drop_duplicates())
    data[i]=encoder.transform(data[i])

print(data.head(2))
#p=1
#e=0
data=pd.get_dummies(data)
data.shape
x=data.drop(['class'],axis=1)
X=x.values
y=data['class']
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=21)
pca=PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xlim(0,115,5)
pca=PCA(n_components=40)
pca.fit(X_train)
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)
X_train_pca.shape
logreg=LogisticRegression(random_state=1)
score = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='accuracy'))
p_scores = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='precision'))
r_scores = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='recall'))
print("Accuracy: %s" % '{:.2%}'.format(score))
print ('Precision : %s' %'{:.2%}' .format(p_scores))
print ('Recall score: %s' % '{:.2%}'.format(r_scores))
logreg=LogisticRegression(random_state=1)
param_grid = {'penalty': ['l1','l2'], 'C': [10,100,1000]}
logreg_cv = GridSearchCV(estimator = logreg, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 0)
logreg_cv.fit( X_train_pca, y_train)
print(logreg_cv.best_params_)
score=logreg_cv.best_score_
print("Accuracy: %s" % '{:.2%}'.format(score))
logreg2=LogisticRegression(random_state=1,penalty= 'l1',C=100)
logreg2.fit(X_train_pca, y_train)
y_pred=logreg2.predict(X_test_pca)
ascore=accuracy_score(y_test,y_pred)
pscore=precision_score(y_test,y_pred)
rscore=recall_score(y_test,y_pred)
matrix=confusion_matrix(y_test,y_pred)
print("Accuracy: %s" % '{:.2%}'.format(ascore))
print ('Precision : %s' %'{:.2%}' .format(pscore))
print ('Recall score: %s' % '{:.2%}'.format(rscore))

sns.heatmap(matrix,annot=True,fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
rf=RandomForestClassifier(random_state=21)
score_rf = np.mean(cross_val_score(rf,  X_train, y_train, scoring='accuracy'))
p_score_rf = np.mean(cross_val_score(rf,  X_train, y_train, scoring='precision'))
r_score_rf = np.mean(cross_val_score(rf,  X_train, y_train, scoring='recall'))
print("Accuracy for RandomForest: %s" % '{:.2%}'.format(score_rf))
print ('Precision RandomForest:: %s' %'{:.2%}' .format(p_score_rf))
print ('Recall score RandomForest:: %s' % '{:.2%}'.format(r_score_rf))
rf=RandomForestClassifier(random_state=21)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
ascore=accuracy_score(y_test,y_pred)
pscore=precision_score(y_test,y_pred)
rscore=recall_score(y_test,y_pred)
matrix=confusion_matrix(y_test,y_pred)
print("Accuracy: %s" % '{:.2%}'.format(ascore))
print ('Precision : %s' %'{:.2%}' .format(pscore))
print ('Recall score: %s' % '{:.2%}'.format(rscore))
print('Confusion matrix: ')
print(matrix)

sns.heatmap(matrix,annot=True,fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
fig, ax=plt.subplots(figsize=(15,25))
features = x.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Features Importance')
plt.barh(range(len(indices)), importances[indices], color='purple', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show() 