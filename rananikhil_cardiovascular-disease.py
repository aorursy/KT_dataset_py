from PIL import Image
Image.open('shutterstock_1248997882_web.jpg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)

from sklearn.model_selection import train_test_split
from sklearn import model_selection

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import lightgbm
from xgboost import XGBClassifier
heart=pd.read_csv('heart.csv')
heart.head(2)
print('Number of records in dataset are {} and Features are {}.  '.format(*heart.shape))
print("\nAny missing sample in set:",heart.isnull().values.any())
heart.describe()
#Some changes for better visualization

heart1=heart.copy()
heart1['target']=heart1['target'].map({1:'Unwell',0:'Healthy'})
heart1['cp']=heart1['cp'].map({0:'Typical Angina',1:'Atypical Angina',2:'Non-Anginal',3:'Asymptomatic'}).astype('object')
heart1['fbs']=heart1['fbs'].map({1:'>120 mg/dl',0:'<120 mg/dl'})
heart1['restecg']=heart1['restecg'].map({0:'normal',1:'ST-T abnormality',2:'probable_LVH'})
heart1['exang']=heart1['exang'].map({1:'Yes',0:'No'})
heart1['target'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')
plt.show()
heart[['age', 'trestbps','chol','thalach','oldpeak','target']].corr()['target'][:-1].plot.barh()
plt.title("Feature Relation with illness")
plt.xlabel('Correlation Coefficient with target')
plt.ylabel('Continous Features')
plt.show()
sns.kdeplot(heart1[heart1['target']=='Unwell']['chol'],shade=True,color="orange", label="Unwell", alpha=.7)
sns.kdeplot(heart1[heart1['target']=='Healthy']['chol'],shade=True,color="dodgerblue", label="Healthy", alpha=.7)
plt.title('Cholesterol in mg/d for both case')
plt.show()
sns.kdeplot(heart1[heart1['target']=='Unwell']['trestbps'],shade=True,color="orange", label="Unwell", alpha=.7)
sns.kdeplot(heart1[heart1['target']=='Healthy']['trestbps'],shade=True,color="dodgerblue", label="Healthy", alpha=.7)
plt.title('Resting Blood Pressure for both case')
plt.show()
Image.open('blood pressure.png')
sns.kdeplot(heart1[heart1['target']=='Unwell']['thalach'],shade=True,color="orange", label="Unwell", alpha=.7)
sns.kdeplot(heart1[heart1['target']=='Healthy']['thalach'],shade=True,color="dodgerblue", label="Healthy", alpha=.7)
plt.title('Maximum Heart Rate Achieved for both case')
plt.show()
sns.kdeplot(heart1[heart1['target']=='Unwell']['oldpeak'],shade=True,color="orange", label="Unwell", alpha=.7)
sns.kdeplot(heart1[heart1['target']=='Healthy']['oldpeak'],shade=True,color="dodgerblue", label="Healthy", alpha=.7)
plt.title('ST depression distribution for both case')
plt.show()
cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
plt.figure(figsize=(15,4))
for i,col in enumerate(cols,1):
   plt.subplot(1,5,i)
   sns.boxplot(heart[col])
   plt.xlabel(col)  
plt.show()
Categorical_Features=['sex','cp','exang','fbs','thal','restecg','slope','ca']
pd.DataFrame(heart[Categorical_Features].nunique(),columns=['Unique_counts'])
Image.open('angina.png')
plt.figure(figsize=(15,5))
plt.subplot(121)
heart1['cp'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.subplot(122)
sns.countplot(hue='target',x='cp',data=heart1,palette='husl')
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(121)
heart1['exang'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.subplot(122)
sns.countplot(hue='target',x='exang',data=heart1,palette='husl')
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(121)
heart1['fbs'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.subplot(122)
sns.countplot(hue='target',x='fbs',data=heart1,palette='husl')
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(121)
heart1['thal'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.subplot(122)
sns.countplot(hue='target',x='thal',data=heart1,palette='husl')
plt.show()
Image.open('The-description-of-the-ECG-curve.png')
plt.figure(figsize=(15,5))
plt.subplot(121)
heart1['restecg'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.subplot(122)
sns.countplot(hue='target',x='restecg',data=heart1,palette='husl')
plt.show()
heart1.groupby(['slope','target'])['slope','target'].size().unstack()
Image.open('st_depression.png')
plt.figure(figsize=(15,5))
plt.subplot(121)
heart1['slope'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.subplot(122)
sns.countplot(hue='target',x='slope',data=heart1,palette='husl')
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(121)
heart1['ca'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')

plt.subplot(122)
sns.countplot(hue='target',x='ca',data=heart1,palette='husl')
plt.show()
X=heart.drop('target',axis=1)
y=heart['target']

#Split the data into train and test set:(70/30)
x_trains,x_tests, y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=3)
models = []

models.append(('NB', BernoulliNB() ))
models.append(('LOGREG', LogisticRegression() ))
models.append(('DT', DecisionTreeClassifier() ))
models.append(('BAGGED_DT', BaggingClassifier() ))
models.append(('RFC', RandomForestClassifier() ))
#models.append(('KNN', KNeighborsClassifier() ))   #Use if data is scaled
models.append(('ADA', AdaBoostClassifier() ))
models.append(('SVM', SVC() ))
models.append(('LGBM', lightgbm.LGBMClassifier() ))
models.append(('XGB', XGBClassifier() ))

models
from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score

results_train = []
results_test = []
names = []
df_results=pd.DataFrame()

for name, model in models:
    names.append(name)
    kfold = model_selection.KFold(n_splits=10, random_state=3, shuffle=True)
    
    trainf1 =cross_val_score(model,x_trains,y_train,cv=kfold, scoring='f1_weighted')
    trainacc =cross_val_score(model,x_trains,y_train,cv=kfold, scoring='accuracy')
    trainpre =cross_val_score(model,x_trains,y_train,cv=kfold, scoring='precision_weighted')
    trainre =cross_val_score(model,x_trains,y_train,cv=kfold, scoring='recall_weighted')
    trainroc =cross_val_score(model,x_trains,y_train,cv=kfold, scoring='roc_auc')
    results_train.append(trainroc)
    
    
    
    
    testf1 = cross_val_score(model, x_tests, y_test,cv=kfold, scoring='f1_weighted')
    testacc =cross_val_score(model,x_tests,y_test,cv=kfold, scoring='accuracy')
    testpre =cross_val_score(model,x_tests,y_test,cv=kfold, scoring='precision_weighted')
    testre =cross_val_score(model,x_tests,y_test,cv=kfold, scoring='recall_weighted')
    testroc =cross_val_score(model,x_tests,y_test,cv=kfold, scoring='roc_auc')
    results_test.append(testroc)
    
    
    df_results = pd.concat([df_results,\
                            pd.DataFrame(np.array([name,round(trainf1.mean(),2),round(testf1.mean(),2),\
                                                   round(trainacc.mean(),2),round(testacc.mean(),2),\
                                                   round(trainpre.mean(),2),round(testpre.mean(),2),\
                                                   round(trainre.mean(),2),round(testre.mean(),2),\
                                                   round(trainroc.mean(),2),round(testroc.mean(),2) ]).reshape(1,-1),
                                         columns=['Description','F1score_train','F1score_test',\
                                                  'Accuracy_train','Accuracy_test','Precision_train','Precision_test',\
                                                  'Recall_train','Recall_test','Roc_auc_train','Roc_auc_test']  )
                           ], axis=0)
    
df_results
plt.figure(figsize=(20,4))

plt.subplot(121)
plt.title('Train: Algorithm Comparison on Roc-Auc-score')
plt.boxplot(results_train,labels=names)

plt.subplot(122)
plt.title('Test: Algorithm Comparison on Roc-Auc-score')
plt.boxplot(results_test,labels=names,manage_ticks=True)
plt.show()
