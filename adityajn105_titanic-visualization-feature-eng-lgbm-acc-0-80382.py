import numpy as np
import pandas as pd

#import warnings
#warnings.filterwarnings('ignore')

from collections import Counter
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
print("Train Data Shape: {}".format(df_train.shape))
df_train.head()
print("Test Data Shape: {}".format(df_test.shape))
df_test.head()
%matplotlib inline
from matplotlib import pyplot as plt, style
style.use("fivethirtyeight")
print("NA Values in Train data")
for column in df_train.columns:
    print("{} : {}".format(column,df_train[column][df_train[column].isna()].size))
print("\nNA Values in Test data")
for column in df_test.columns:
    print("{} : {}".format(column,df_test[column][df_test[column].isna()].size))
"""
Age,Fare,cabin,embarked have NA values, we will find some way to find it
Feature : Sex lets visualize it
"""
print("Sex : {} ".format(dict(Counter(df_train['Sex']))))

fig = plt.figure(figsize=(15,3))
#ax1
ax1 = fig.add_subplot(1,2,1)
survived = [len(df_train[ (df_train['Survived']==1)&(df_train['Sex']=='male')]),len(df_train[(df_train['Survived']==1)&(df_train['Sex']=='female')])]
not_survived = [len(df_train[(df_train['Survived']==0)&(df_train['Sex']=='male')]),len(df_train[(df_train['Survived']==0)&(df_train['Sex']=='female')])]
ax1.bar(['Male','Female'],survived,label="Survived",color='green',alpha=0.5)
ax1.bar(['Male','Female'],not_survived,bottom=survived,label="Not Survived",color='red',alpha=0.5)
ax1.set_xlabel("Gender")
ax1.set_ylabel("Total")
ax1.legend()
#ax2
ax2 = fig.add_subplot(1,2,2)
survival_rate = [s/t for s,t in zip(survived,np.array(survived)+np.array(not_survived))]
ax2.bar(['Male','Female'],survival_rate,label="Survival Ratio",color='orange',alpha=0.5)
ax2_legend = ax2.legend()
"""
Looks like most of the woman have survived
Now lets look at Pclass
"""
print("Pclass : {} ".format(dict(Counter(df_train['Pclass']))))
pclass = sorted(df_train['Pclass'].unique())
survived = [len(df_train[(df_train['Survived']==1)&(df_train['Pclass']==x)]) for x in pclass ]
not_survived = [len(df_train[(df_train['Survived']==0)&(df_train['Pclass']==x)]) for x in pclass ]
fig = plt.figure(figsize=(15,3))
#ax1
ax1 = fig.add_subplot(1,2,1)
ax1.bar(['Pclass:1','Pclass:2','Pclass:3'],survived,color='green',label="Survived",alpha=0.5)
ax1.bar(['Pclass:1','Pclass:2','Pclass:3'],not_survived,bottom=survived,color='red',label="Not Survived",alpha=0.5)
ax1.set_xlabel("class")
ax1.set_ylabel("total")
ax1.legend()
#ax2
ax2 = fig.add_subplot(1,2,2)
survival_rate = [s/t for s,t in zip(survived,np.array(survived)+np.array(not_survived))]
ax2.bar(['Pclass:1','Pclass:2','Pclass:3'],survival_rate,color='orange',label="Survival Ratio",alpha=0.5)
ax2_legend = ax2.legend()
"""
Looks like passenger of class 3 are not able to survive 
Lets look at embarked
"""
print("Embarked : {} ".format(dict(Counter(df_train['Embarked']))))
# S is most common lets replace 2 nan value with that only
df_train.Embarked.fillna('S',inplace=True)
embarked = df_train['Embarked'].unique()

survived = [ len(df_train[ (df_train['Survived']==1)&(df_train['Embarked']==x)] ) for x in embarked ]
not_survived = [ len(df_train[ (df_train['Survived']==0)&(df_train['Embarked']==x)] ) for x in embarked ]
fig = plt.figure(figsize=(15,3))
#ax1
ax1 = fig.add_subplot(1,2,1)
ax1.bar(embarked,survived,label="Survived",color="g",alpha=0.5)
ax1.bar(embarked,not_survived,label="Not Survived",color="r",alpha=0.5,bottom=survived)
ax1.legend()
ax1.set_xlabel('Embarkment Point')
ax1.set_ylabel('Total')
#ax2
ax2 = fig.add_subplot(1,2,2)
survival_rate = [ s/t for s,t in zip(survived,np.array(survived)+np.array(not_survived))]
ax2.bar(embarked,survival_rate,color='orange',alpha=0.5,label='Survival Ratio')
ax2_legend = ax2.legend()
"""
People of C have survived most
'Q' and 'S' doesnt have much different survival rates, lets consider 'Q' same as 'S'
"""
df_train.Embarked.replace('S','Q',inplace=True)
df_test.Embarked.replace('S','Q',inplace=True)
"""
Lets look at fare
"""
df_test['Fare'].fillna(df_train['Fare'].median(),inplace=True)
median_fare = np.median(df_train['Fare'])
print("Median Fare : {}".format(median_fare))
fig= plt.figure(figsize=(15,3))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.set_title("Count vs Fare");
ax2.set_title("Survival Rate vs Fare")
#ax1
survived = [len(df_train[ (df_train['Survived']==1)&(df_train['Fare']<=median_fare)]),len(df_train[(df_train['Survived']==1)&(df_train['Fare']>median_fare)])] 
not_survived = [len(df_train[ (df_train['Survived']==0)&(df_train['Fare']<=median_fare)]),len(df_train[ (df_train['Survived']==0)&(df_train['Fare']>median_fare)])]
ax1.bar(['Low Fare','High Fare'],survived,label="Survived",color="g",alpha=0.5)
ax1.bar(['Low Fare','High Fare'],not_survived,bottom=survived,label="Not Survived",color="r",alpha=0.5)
ax1.legend()
#ax2
survival_rate = [s/t for s,t in zip(survived,np.array(survived)+np.array(not_survived))]
ax2.bar(['Low Fare','High Fare'],survival_rate,color='orange',label='Survival Ratio',alpha=0.5)
ax2_legend = ax2.legend()
"""
Lets create a new feature:
FamilySize
"""
df_train['FamilySize'] = df_train.SibSp + df_train.Parch + 1
df_test['FamilySize'] = df_test.SibSp + df_test.Parch + 1

"Lets visualize familysize"
fig = plt.figure(figsize=(15,3))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
#ax1
survived = [len(df_train[(df_train["Survived"]==1)&(df_train['FamilySize']==1)]),len(df_train[(df_train["Survived"]==1)&(df_train['FamilySize']>1)])]
not_survived = [len(df_train[(df_train["Survived"]==0)&(df_train['FamilySize']==1)]),len(df_train[(df_train["Survived"]==0)&(df_train['FamilySize']>1)])]
ax1.bar(['Single','Family'],survived,label="Survived",color='g',alpha=0.5)
ax1.bar(['Single','Family'],not_survived,bottom=survived,label="Not Survived",color='r',alpha=0.5)
ax1.legend()
#ax2
survival_ratio = [s/t for s,t in zip(survived,np.array(survived)+np.array(not_survived)) ]
ax2.bar(['Single','Family'],survival_ratio,label="Survival Ratio",color='orange',alpha=0.5)
ax2_legend = ax2.legend()
"""
Looks like Families are more likely to survive
Lets combine train and test data before moving further
"""
df = df_train.append(df_test,sort=False)
df.tail(5)
"""
Lets create another feature: 
title (denoting title )
Surname (will remove later)
"""
name = df.Name.str.split('[,.]',expand=True)
df['Title'] = name[1].str.strip()
df['Surname'] = name[0].str.strip()
print("titles : {} ".format(dict(Counter(df.Title))))
"""
These are similar
Mlle,Mme = Mlle
Capt,Don,Major,Sir = Sir
Dona, Lady, Jonkheer, the Countess
"""
df.Title.replace(['Mlle','Mme'],'Mlle',inplace=True)
df.Title.replace(['Capt','Don','Major','Sir'],'Sir',inplace=True)
df.Title.replace(['Dona','Lady','Jonkheer','the Countess'],'Lady',inplace=True)

print("\ntitles : {} ".format(dict(Counter(df.Title))))
"""replace age na values with mean of their correspoding Title"""
new_ages = []
for index in df[df.Age.isna()].index:
    title = df.iloc[index].Title
    new_ages.append(df[df.Title==title]['Age'].mean())
df.loc[df.Age.isna(),'Age']=np.array(new_ages)
"""Lets visualize age feature"""
fig = plt.figure(figsize=(15,3))
#ax1
ax1 = fig.add_subplot(1,2,1)
df_train  = df[:len(df_train)]

survived = df_train[df_train['Survived']==1]['Age'].astype('int')
not_survived = df_train[df_train['Survived']==0]['Age'].astype('int')

bins = [0,16,30,55,100]
hist_survived, bin_edges = np.histogram(survived,bins)
hist_not_survived, bin_edges = np.histogram(not_survived,bins)

xlabels = ['0-16','16-30','30-55','55+']

ax1.bar(xlabels,hist_survived,label="Survived",color="g",alpha=0.5)
ax1.bar(xlabels,hist_not_survived,bottom=hist_survived,label="Not Survived",color="r",alpha=0.5)
ax1.legend()
ax1.set_xlabel("Age")
ax1.set_ylabel("Total")

#ax2
ax2 = fig.add_subplot(1,2,2)
survival_rate = [s/t for s,t in zip(hist_survived,hist_survived+hist_not_survived)]
ax2.bar(xlabels,survival_rate,label="Survival Ratio",color="orange",alpha=0.5)
ax2.legend()
"""
Looks like children are most likely to survive
"""
survived = [len(df_train[(df_train['Survived']==1)&(df_train['Sex']=='male')&(df_train['Age']<18)]),
            len(df_train[(df_train['Survived']==1)&(df_train['Sex']=='female')&(df_train['Age']<18)]),
           len(df_train[(df_train['Survived']==1)&(df_train['Sex']=='male')&(df_train['Age']>18)]),
           len(df_train[(df_train['Survived']==1)&(df_train['Sex']=='female')&(df_train['Age']>18)])]
not_survived = [len(df_train[(df_train['Survived']==0)&(df_train['Sex']=='male')&(df_train['Age']<18)]),
            len(df_train[(df_train['Survived']==0)&(df_train['Sex']=='female')&(df_train['Age']<18)]),
           len(df_train[(df_train['Survived']==0)&(df_train['Sex']=='male')&(df_train['Age']>18)]),
           len(df_train[(df_train['Survived']==0)&(df_train['Sex']=='female')&(df_train['Age']>18)])]
x_labels = ["Male Child","Female Child","Male Adult","Female Adult"]
fig = plt.figure(figsize=(15,3))
#ax1
ax1 = fig.add_subplot(121)
ax1.bar(x_labels,survived,label="Survived",color="g",alpha=0.5,)
ax1.bar(x_labels,not_survived,bottom=survived,label="Not Survived",color="r",alpha=0.5,)
ax1.legend()

#ax2
ax2 = fig.add_subplot(122)
survival_ratio = [s/t for s,t in zip(survived,np.array(survived)+np.array(not_survived))]
ax2.bar(x_labels,survival_ratio,label="Survival ratio",color="orange",alpha=0.5)
ax2.legend()
"""
lets use age as categorical variable then continuos variable as
child,young,adult,old
"""
df["AgeGroup"] = 'Child'
df.loc[ (df['Age']>16) & (df['Age']<=30) , 'AgeGroup'] = 'Young'
df.loc[ (df['Age']>30) & (df['Age']<=55) , 'AgeGroup'] = 'Adult'
df.loc[df['Age']>55, 'AgeGroup'] = 'Old'
print("NA Values")
for column in df.columns:
    print("{} : {}".format(column,df[column][df[column].isna()].size))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.Sex = le.fit_transform(df.Sex)
df.Embarked = le.fit_transform(df.Embarked)
df.Title = le.fit_transform(df.Title)
df.AgeGroup = le.fit_transform(df.AgeGroup)

"""
Useful columns = PassengerId,Survived,Pclass,Sex,Fare,Embarked,FamilySize,Title,AgeGroup
"""
df_final = df.drop(['Name','Age','SibSp','Parch','Ticket','Cabin','Surname'],axis=1)
df_final.head()
#preparing train, test and validation data
train_df = df_final[:len(df_train)].drop(['PassengerId'],axis=1)
test_df = df_final[len(df_train):]

Y = train_df['Survived'].astype('int')
X = train_df.drop(['Survived'],axis=1)
X_test_PassengerId = test_df['PassengerId']
X_test = test_df.drop(['PassengerId','Survived'],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=0.2)
print("Train Size : {}\nValidation Size : {}\nTest Size : {}".format(len(X_train),len(X_validation),len(X_test)))
X.describe()
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
params={
    'num_leaves':[5,10,30],
    'n_estimators':[20,30,50,100],
    'learning_rate':[0.01,0.05],
    'n_jobs':[-1]
}
lgbm = LGBMClassifier()
gridcv = GridSearchCV(lgbm,param_grid=params,scoring='accuracy',n_jobs=1,cv=4,verbose=2)
gridcv.fit(X_train,Y_train)
clf = gridcv.best_estimator_
print(clf)
print("Validation Accuracy : {}".format(clf.score(X_validation,Y_validation)))
#lets train classifier with full data
clf.fit(X,Y)

predictions = clf.predict(X_test)
#predictions = np.argmax(predictions,axis=1)
sol = pd.DataFrame({
    "PassengerId":X_test_PassengerId,
    "Survived":predictions
})
sol.to_csv("sol.csv",header=True,index=False)
"""
Accuraccy - 0.77033
Keras ANN - 50-relu,50-relu
            loss - binary_crossentropy
            optimizer - adam

from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils
batch_size =  1
num_classes = 2

Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_validation = np_utils.to_categorical(Y_validation, num_classes)

clf = Sequential()
clf.add(Dense(50,activation='relu',input_shape=(7,)))
clf.add(Dense(50,activation='relu',input_shape=(50,1)))
clf.add(Dropout(0.2))
clf.add(Dense(num_classes,activation='softmax',input_shape=(50,1)))

from keras import losses
from keras import optimizers
clf.compile(loss=losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

clf.fit(
    X_train,
    Y_train,
    validation_data = (X_validation,Y_validation),
    batch_size=batch_size,
    nb_epoch=50,
    verbose=2
)
"""