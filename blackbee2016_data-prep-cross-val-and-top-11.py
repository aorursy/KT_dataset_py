import pandas as pd
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(3)
train['Name']=train['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
train.Name[0:4]
train = pd.get_dummies(train,columns=['Embarked','Sex','Name'])
train.columns
del train['Cabin']
del train['Ticket']
for i in train.columns:
    mediane = train[i].median()
    train[i].fillna(mediane,inplace=True)

test['Name']=test['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
test = pd.get_dummies(test,columns=['Embarked','Sex','Name'])
del test['Cabin']
del test['Ticket']
for i in test.columns:
    mediane = test[i].median()
    test[i].fillna(mediane,inplace=True)
train.columns.shape
test.columns.shape
for i in train.columns:
    if i =='Survived':
        continue
    if i not in test.columns:
        del train[i]
        
for i in test.columns:
    if i not in train.columns:
        del test[i]
from sklearn.model_selection import train_test_split
score=[]
score_b = []
for i in range(10):
    rf = RandomForestClassifier(n_estimators=1000,max_depth=6,n_jobs=-1,criterion='entropy',max_features='sqrt',random_state=5)    
    TRAIN,TEST = train_test_split(train,test_size=0.2)
    y_TRAIN = TRAIN['Survived']
    del TRAIN['Survived']
    y_TEST = TEST['Survived']
    del TEST['Survived']
    rf.fit(TRAIN,y_TRAIN)

    a= sum(rf.predict(TEST)==y_TEST)/len(y_TEST)#Manually computing accuracy here
    b= sum(rf.predict(TRAIN)==y_TRAIN)/len(y_TRAIN) #Manually computing accuracy here
    score_b.append(b)
    score.append(a)
import matplotlib.pyplot as plt
plt.plot(score_b)
plt.plot(score)
plt.legend(['Train Data','Test Data'])
plt.title('Cross validation accurracy accross results')
y_train = train['Survived']
del train['Survived']
rf = RandomForestClassifier(n_estimators=1000,max_depth=6,n_jobs=-1,criterion='entropy',max_features='sqrt',random_state = 5)
rf.fit(train,y_train)
Y = rf.predict(test)
sub = pd.read_csv('../input/gender_submission.csv')
sub['Survived'] = Y
sub.to_csv('titanic_prediction_file.csv',index = False)
