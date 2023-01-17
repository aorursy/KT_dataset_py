import pandas as pd





train = pd.read_csv('../input/santander-customer-satisfaction/train.csv',index_col='ID')

test = pd.read_csv('../input/santander-customer-satisfaction/test.csv',index_col='ID')

train.head()
train.shape
train.describe()
train.columns
train.dtypes
train.dtypes.unique()
train.isnull().sum()
train.isnull().sum().unique()
y = train.TARGET

y.head()
x = train.drop(['TARGET'],axis=1) 

x.head()
from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(x,y,train_size=0.65,test_size=0.35,random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error





model = DecisionTreeClassifier(random_state=1)

model.fit(X_train,Y_train)

preds = model.predict(X_val)
from sklearn import metrics



print(metrics.classification_report(preds,Y_val))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(random_state=1)



lr.fit(X_train,Y_train)

print(metrics.classification_report(lr.predict(X_val),Y_val))
print(y.value_counts())
train.TARGET.value_counts(normalize=True)
### General Function to randomly sample N*x , where N is the ratio of class 0 to class 1 points and 

### x is the number of class 1 points



def under_sampler(N,x = train.TARGET.value_counts()[1]):

    class_0 = train[train['TARGET']==0].sample( int(N*x) ,random_state=1)

    class_1 = train[train['TARGET']==1]

    return pd.concat([class_0,class_1],axis=0)



train_new = under_sampler(4)

train_new.shape
X = train_new.drop(columns='TARGET')

y = train_new.TARGET



X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=1,train_size=0.8)



model.fit(X_train,y_train)

preds = model.predict(X_val)

print(metrics.classification_report(preds,y_val))

print(metrics.roc_auc_score(preds,y_val))
import numpy as np
for N in np.arange(1.0,5.0,0.5):

    train_new = under_sampler(N)

    X = train_new.drop(columns='TARGET')

    y = train_new.TARGET



    X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=1,train_size=0.8)



    model.fit(X_train,y_train)

    preds = model.predict(X_val)

    print("The ratio of majority to minor class is ",N)

    print(metrics.classification_report(preds,y_val))

train.TARGET.value_counts()
train_new = train.copy()



train_new = pd.concat([train_new,train_new[train_new.TARGET == 1]],axis=0)

train_new.TARGET.value_counts()
X = train_new.drop(columns='TARGET')

y = train_new.TARGET



X_train,X_val,y_train,y_val = train_test_split(X,y,random_state=1,train_size=0.8)
model.fit(X_train,y_train)

preds = model.predict(X_val)

print(metrics.classification_report(preds,y_val))
metrics.roc_auc_score(preds,y_val)
from imblearn.over_sampling import SMOTE
from collections import Counter
print("Ratio of minority to majority class points",train.TARGET.value_counts()[1]/train.TARGET.value_counts()[0])
### CAUTION : Change n_jobs to match the number of threads that runs on your CPU. I have a CPU running 6 cores and 12 threads

### and hence assinged n_jobs = 12. If you aren't sure of the number of threads on your CPU, remove the parameter n_jobs.



sm = SMOTE(random_state=1,n_jobs=12,sampling_strategy = 0.25)# ratio of resampled minority class points to majority class points



X_resampled, y_resampled = sm.fit_resample(X,y)

print("Before sampling : ",Counter(y),"\nAfter Sampling : ",Counter(y_resampled))
X_train,X_val,y_train,y_val = train_test_split(X_resampled,y_resampled,random_state=1,train_size=0.8)



model.fit(X_train,y_train)

preds = model.predict(X_val)

print(metrics.classification_report(preds,y_val))
metrics.roc_auc_score(preds,y_val)
predictions = model.predict(test)
predictions
out =pd.DataFrame({'ID':test.index,'TARGET':predictions})

out.set_index('ID',inplace=True)
out
out.to_csv("Predictions1.csv",index=True)
train_new = under_sampler(6)



X = train_new.drop(columns='TARGET')

y = train_new.TARGET



y.value_counts()
sm = SMOTE(random_state=1,n_jobs=12,sampling_strategy = 0.4)



X_resampled, y_resampled = sm.fit_resample(X,y)

print("Before sampling : ",Counter(y),"\nAfter Sampling : ",Counter(y_resampled))
X_train,X_val,y_train,y_val = train_test_split(X_resampled,y_resampled,random_state=1,train_size=0.8)



model.fit(X_train,y_train)

preds = model.predict(X_val)

print(metrics.classification_report(preds,y_val))
metrics.roc_auc_score(preds,y_val)
predictions = model.predict(test)

out =pd.DataFrame({'ID':test.index,'TARGET':predictions})

out.set_index('ID',inplace=True)

out.to_csv('Predicitions2.csv',index=True)
train.TARGET.value_counts()
train_new = under_sampler(10)



X = train_new.drop(columns='TARGET')

y = train_new.TARGET
max_auc = 0

for i in np.arange(0.2,1,0.1):

    sm = SMOTE(random_state=1,n_jobs=12,sampling_strategy = i)

    X_resampled, y_resampled = sm.fit_resample(X,y)

    #print("Before sampling : ",Counter(y),"\nAfter Sampling : ",Counter(y_resampled))

    X_train,X_val,y_train,y_val = train_test_split(X_resampled,y_resampled,random_state=1,train_size=0.8)



    model.fit(X_train,y_train)

    preds = model.predict(X_val)

    #print(metrics.classification_report(preds,y_val))

    auc = metrics.roc_auc_score(preds,y_val)

    if auc>max_auc:

        max_auc = auc

        SMOTE_ratio = i

        

print("Optimal Soltution -\nAUC Score : ",max_auc,"\nSMOTE Ratio : ",SMOTE_ratio)
sm = SMOTE(random_state=1,n_jobs=12,sampling_strategy = 0.9)

X_resampled, y_resampled = sm.fit_resample(X,y)

print("Before sampling : ",Counter(y),"\nAfter Sampling : ",Counter(y_resampled))

X_train,X_val,y_train,y_val = train_test_split(X_resampled,y_resampled,random_state=1,train_size=0.8)



model.fit(X_train,y_train)

preds = model.predict(X_val)

print(metrics.classification_report(preds,y_val))
metrics.roc_auc_score(preds,y_val)
predictions = model.predict(test)



out =pd.DataFrame({'ID':test.index,'TARGET':predictions})

out.set_index('ID',inplace=True)

out.to_csv('Predicitions3.csv',index=True)
X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=0.8,random_state=1)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators = 100,n_jobs = 12,random_state=1)

rfc.fit(X_train,y_train)

preds = rfc.predict(X_val)

print(metrics.classification_report(preds,y_val))
metrics.roc_auc_score(preds,y_val)
train_new = under_sampler(6)



X = train_new.drop(columns='TARGET')

y = train_new.TARGET

sm = SMOTE(random_state=1,n_jobs=12,sampling_strategy = 0.4)



X_resampled, y_resampled = sm.fit_resample(X,y)

#print("Before sampling : ",Counter(y),"\nAfter Sampling : ",Counter(y_resampled))

X_train,X_val,y_train,y_val = train_test_split(X_resampled,y_resampled,random_state=1,train_size=0.8)



rfc.fit(X_train,y_train)

preds = rfc.predict(X_val)

print(metrics.classification_report(preds,y_val))
metrics.roc_auc_score(preds,y_val)
predictions = rfc.predict(test)



out =pd.DataFrame({'ID':test.index,'TARGET':predictions})

out.set_index('ID',inplace=True)

out.to_csv('Predicitions4.csv',index=True)