#import libraries
import pandas as pd #for reading the data set
import matplotlib.pyplot as plt#library to import plots
import seaborn as sns #library to import plots
from sklearn import preprocessing#for preprocessing the data set
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics import f1_score
test=pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')
train=pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv')
test.head()
train.isnull().sum()
test.isnull().sum()
#shape of the test data
test.shape
train.head()
#shape of train data
train.shape
#info of data frame
train.info()
#describing the data set
train.describe()
print("Number of anomalies : ",train['flag'].value_counts()[1])#counting the outcome  where there are anamolies
print("Number of no anomalies:",train['flag'].value_counts()[0])#counting the outcomes where there are no anamoulies
print("Pie Chart:")
fig, ax = plt.subplots(1, 1)
ax.pie(train.flag.value_counts(),autopct='%1.3f%%', labels=['anomalies','no anomalies'], colors=['yellowgreen','r'])
plt.axis('equal')
plt.ylabel('')
#elminatinng flag
v_features = train.drop('flag',1)
s=v_features.columns
#plotting histograms of all v_features (of anoumli+ no anamoui) to check which of them are useful
# the more diff b/w anoumli and not anoumli = more important for learning
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(train[s]):
    ax = plt.subplot(gs[i])
    sns.distplot(train[cn][train.flag == 1], bins=50,label="anamouli")
    sns.distplot(train[cn][train.flag == 0], bins=50,label="no anamouli")
    ax.set_xlabel('')
    ax.set_title('histogram of feature: '+cn)
plt.show()
#currentback
#motortemp back
#positionback
#currentfront
#motortempfron

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
cax = ax.matshow(train.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
plt.show()
train.describe()
sns.pairplot(train, hue='flag')
train_d=train[['currentBack','motorTempBack','positionBack','currentFront','motorTempFront']]
test_d=test[['timeindex','currentBack','motorTempBack','positionBack','currentFront','motorTempFront']]
train_d1=train[['currentBack','motorTempBack','positionBack','currentFront','motorTempFront']]
test_d1=test[['currentBack','motorTempBack','positionBack','currentFront','motorTempFront']]
def normalize(train_d):

    for feature in train_d.columns:
        train_d[feature] -= train_d[feature].mean()
        train_d[feature] /= train_d[feature].std()
        return train_d
def normalize(test_d):

    for feature in test_d.columns:
        test_d[feature] -= test_d[feature].mean()
        test_d[feature] /= test_d[feature].std()
        return test_d
max_total = 0 
max_AUC = 0

Reg_Model = RandomForestClassifier(criterion='entropy',n_estimators=40,max_depth=28)
Reg_Model = Reg_Model.fit(train_d1.values,train['flag'])
pred = Reg_Model.predict(test_d1)


submission=pd.DataFrame({
    "Sl.No": test_d['timeindex'],
    "flag": pred
})
submission.to_csv('submission3.csv',index=False)


def normalize(train_d1):
        for feature in train_d1.columns:
            min_max_scaler = preprocessing.MinMaxScaler(train_d[feature])
            return train_d1
def normalize(test_d1):
        for feature in test_d1.columns:
            min_max_scaler = preprocessing.MinMaxScaler(test_d[feature])
            return test_d1
max_total = 0 
max_AUC = 0

Reg_Model = RandomForestClassifier(criterion='entropy',n_estimators=40,max_depth=28)
Reg_Model = Reg_Model.fit(train_d1.values,train['flag'])
pred1 = Reg_Model.predict(test_d1)


submission=pd.DataFrame({
    "Sl.No": test_d['timeindex'],
    "flag": pred1
})
submission.to_csv('submission4.csv',index=False)

train_d2=train[['currentBack','motorTempBack','positionBack','currentFront','motorTempFront']]
test_d2=test[['currentBack','motorTempBack','positionBack','currentFront','motorTempFront']]
def normalize(train_d2):
        for feature in train_d2.columns:
            scaler = StandardScaler().fit(features) 
            rescaledX = scaler.transform(features) 
            return train_d2
def normalize(test_d2):
        for feature in test_d2.columns:
            scaler = StandardScaler().fit(features) 
            rescaledX = scaler.transform(features)
            return test_d2
max_total = 0 
max_AUC = 0

Reg_Model = RandomForestClassifier(criterion='entropy',n_estimators=40,max_depth=28)
Reg_Model = Reg_Model.fit(train_d2.values,train['flag'])
pred2 = Reg_Model.predict(test_d2)

submission=pd.DataFrame({
    "Sl.No": test_d['timeindex'],
    "flag": pred2
})
submission.to_csv('submission5.csv',index=False)

from sklearn.neighbors import KNeighborsClassifier 
  
knn = KNeighborsClassifier(n_neighbors = 1) 
  
knn.fit(train_d2,train['flag']) 
pred3 = knn.predict(test_d2) 
  

submission=pd.DataFrame({
    "Sl.No": test_d['timeindex'],
    "flag": pred3
})
submission.to_csv('submission7.csv',index=False)

