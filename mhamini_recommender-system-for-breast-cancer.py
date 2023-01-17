
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/fda_breast_cancer_kaggle.csv')
df.head()
plt.xlabel('weight (kg)') 
plt.ylabel('frequency') 
plt.title('Histogram of weight') 
#df['wt'].hist(figsize=[10,10] ,bins=25)
params = {'axes.titlesize':'25',
          
         'axes.labelsize': '25'}
plt.rcParams.update(params)
df['wt'].hist(figsize=[10,10],bins=[0,20,40,50,60,70,80,100,120,140,160,200])  
plt.grid(False)
plt.xlabel('age') 
plt.ylabel('frequency') 
plt.title('Histogram of age') 
params = {'axes.titlesize':'25',
          
         'axes.labelsize': '25'}
plt.rcParams.update(params)
df['age'].hist(figsize=[10,10],bins=[0,30,40,45,50,55,60,65,70,75,80,85,90,110])
plt.grid(False)
plt.xlabel('gender') 
plt.ylabel('frequency') 
plt.title('Histogram of gender') 
params = {'axes.titlesize':'25',
          
         'axes.labelsize': '25'}
plt.rcParams.update(params)
df['gndr_cod'].hist(figsize=[10,10])
plt.grid(False)
dummy_col = pd.get_dummies(df[['drug']])
df=df.drop(['drug','reporter_country','reaction'] , axis=1)
df = pd.concat([df, dummy_col], axis=1)
df.head()
y=df['de']
X=df.drop(['de','lt','ho','ds','ca','ri','ot'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
clf1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=0)    #building 100 decision trees
clf1.fit(X_train, y_train)
print ("Death model")
#print "oob score:", clf1.oob_score_
print (metrics.accuracy_score(y_test, clf1.predict(X_test)))

print (metrics.confusion_matrix(y_test, clf1.predict(X_test)))
TP=metrics.confusion_matrix(y_test, clf1.predict(X_test))[0,0]
TN=metrics.confusion_matrix(y_test, clf1.predict(X_test))[1,1]
FP=metrics.confusion_matrix(y_test, clf1.predict(X_test))[1,0]
FN=metrics.confusion_matrix(y_test, clf1.predict(X_test))[0,1]
print ("Sensitivity=", TP/(TP+FN))
print ("Specificity=", TN/(TN+FP))
print ("=======================================================")
y=df['lt']
X=df.drop(['de','lt','ho','ds','ca','ri','ot'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
clf2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=0)    #building 100 decision trees
clf2.fit(X_train, y_train)
print ("Life threating model")
#print "oob score:", clf2.oob_score_
print (metrics.accuracy_score(y_test, clf2.predict(X_test)))

print (metrics.confusion_matrix(y_test, clf2.predict(X_test)))
TP=metrics.confusion_matrix(y_test, clf2.predict(X_test))[0,0]
TN=metrics.confusion_matrix(y_test, clf2.predict(X_test))[1,1]
FP=metrics.confusion_matrix(y_test, clf2.predict(X_test))[1,0]
FN=metrics.confusion_matrix(y_test, clf2.predict(X_test))[0,1]
print ("Sensitivity=", TP/(TP+FN))
print ("Specificity=", TN/(TN+FP))
print ("=======================================================")

y=df['ho']
X=df.drop(['de','lt','ho','ds','ca','ri','ot'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
clf3 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=0)    #building 100 decision trees
clf3.fit(X_train, y_train)
print ("Hospitalization model")
#print "oob score:", clf3.oob_score_
print (metrics.accuracy_score(y_test, clf3.predict(X_test)))

print (metrics.confusion_matrix(y_test, clf3.predict(X_test)))
TP=metrics.confusion_matrix(y_test, clf3.predict(X_test))[0,0]
TN=metrics.confusion_matrix(y_test, clf3.predict(X_test))[1,1]
FP=metrics.confusion_matrix(y_test, clf3.predict(X_test))[1,0]
FN=metrics.confusion_matrix(y_test, clf3.predict(X_test))[0,1]
print ("Sensitivity=", TP/(TP+FN))
print ("Specificity=", TN/(TN+FP))
print ("=======================================================")

y=df['ds']
X=df.drop(['de','lt','ho','ds','ca','ri','ot'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
clf4 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=0)    #building 100 decision trees
clf4.fit(X_train, y_train)
print ("Disability model")
#print "oob score:", clf4.oob_score_
print (metrics.accuracy_score(y_test, clf4.predict(X_test)))

print (metrics.confusion_matrix(y_test, clf4.predict(X_test)))
TP=metrics.confusion_matrix(y_test, clf4.predict(X_test))[0,0]
TN=metrics.confusion_matrix(y_test, clf4.predict(X_test))[1,1]
FP=metrics.confusion_matrix(y_test, clf4.predict(X_test))[1,0]
FN=metrics.confusion_matrix(y_test, clf4.predict(X_test))[0,1]
print ("Sensitivity=", TP/(TP+FN))
print ("Specificity=", TN/(TN+FP))
print ("=======================================================")

y=df['ot']
X=df.drop(['de','lt','ho','ds','ca','ri','ot'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
clf5 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=0)    #building 100 decision trees
clf5.fit(X_train, y_train)
print ("Other Serious issues model")
#print "oob score:", clf5.oob_score_
print (metrics.accuracy_score(y_test, clf5.predict(X_test)))

print (metrics.confusion_matrix(y_test, clf5.predict(X_test)))
TP=metrics.confusion_matrix(y_test, clf5.predict(X_test))[0,0]
TN=metrics.confusion_matrix(y_test, clf5.predict(X_test))[1,1]
FP=metrics.confusion_matrix(y_test, clf5.predict(X_test))[1,0]
FN=metrics.confusion_matrix(y_test, clf5.predict(X_test))[0,1]
print ("Sensitivity=", TP/(TP+FN))
print ("Specificity=", TN/(TN+FP))
print ("=======================================================")

X_test=X_test.reset_index(drop=True)

def drug_recommender(age,wt,gndr):
    x=np.zeros(X_test.loc[0].shape)
    out=pd.DataFrame(columns={'drug','score'})
    for i in range(3,365):
        x = np.zeros(x.shape)
        prob=0.0
        x[0]=age
        x[1]=wt
        x[2]=gndr
        x[i]=1
        val1=clf1.predict_proba(x.reshape(1, -1))[0][1]
        val2=clf2.predict_proba(x.reshape(1, -1))[0][1]
        val3=clf3.predict_proba(x.reshape(1, -1))[0][1]
        val4=clf4.predict_proba(x.reshape(1, -1))[0][1]
        val5=clf5.predict_proba(x.reshape(1, -1))[0][1]
        for j in range(0,50):
            w1=round(np.random.triangular(5,9,10))
            w2=round(np.random.triangular(5,7,10))
            w3=round(np.random.triangular(3,5,7))
            w4=round(np.random.triangular(1,3,5))
            w5=round(np.random.triangular(1,1,5))
            pred=(w1*val1+w2*val2+w3*val3+w4*val4+w5*val5)
            prob=prob+pred
        out=out.append(pd.DataFrame({'drug':[X_test.columns[i]],'score':[prob/50]}))
    out=out.sort_values(['score'])
    out['drug']=out['drug'].str.replace("drug_", "")
    print ("==============================================")
    return out.head()
drug_recommender(57,90,0)