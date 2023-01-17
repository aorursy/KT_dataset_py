import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # generally required for plotting, as well as seaborn

import seaborn as sns # seaborn for data visualization
titanic_data = pd.read_csv('../input/train.csv')
titanic_data[titanic_data['Parch']!=0][:5]
titanic_data.head(n=7)
titanic_data.info()
sns.heatmap(titanic_data.isnull(),cbar=False,yticklabels=False,cmap='plasma')
fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(12,8))

sns.distplot(titanic_data[(titanic_data['Sex']=='female') & (titanic_data['Age'].isnull()==0)]['Age'],bins=30,color='red',ax=axes[0])

axes[0].set_title('Distribution of age for females')

sns.distplot(titanic_data[(titanic_data['Sex']=='male') & (titanic_data['Age'].isnull()==0)]['Age'],bins=30,color='blue',ax=axes[1])

axes[1].set_title('Distribution of age for males')
print("Mean age of males: %d"%np.round(np.nanmean(titanic_data[titanic_data['Sex']=='male']['Age'])))

print("Mean age of females: %d" %np.round(np.nanmean(titanic_data[titanic_data['Sex']=='female']['Age'])))      
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12,12))

for i in range(3):

    aax = axes[0,i]

    sns.distplot(titanic_data[(titanic_data['Sex']=='female') & (titanic_data['Age'].isnull()==0) & (titanic_data['Pclass']==i+1)]['Age'],color='red',bins=20, ax=aax)

    aax.set_title('Female age in class %d'%(i+1))

    aax = axes[1,i]

    sns.distplot(titanic_data[(titanic_data['Sex']=='male') & (titanic_data['Age'].isnull()==0) & (titanic_data['Pclass']==i+1)]['Age'],color='blue',bins=20, ax=aax)

    aax.set_title('Male age in class %d'%(i+1))
male_mean_age_class = []

female_mean_age_class = []

for i in range(3):

    mage = np.nanmean(titanic_data[(titanic_data['Sex']=='male') & (titanic_data['Pclass']==(i+1))]['Age'])

    male_mean_age_class.append(np.round(mage))

    print("The mean age of males in Passenger class %d"%(i+1)+" is %d"%np.round(mage))

for i in range(3):

    fage = np.nanmean(titanic_data[(titanic_data['Sex']=='female') & (titanic_data['Pclass']==(i+1))]['Age'])

    female_mean_age_class.append(np.round(fage))

    print("The mean age of females in Passenger class %d"%(i+1)+" is %d"%np.round(fage))
def sub_age(data,male_age,female_age):

    age = data[0] # age of passenger

    s = data[1] # sex of passenger

    pclass = data[2] # Travel class of passenger

    if pd.isnull(age)==1:

        if s == 'male':

            age_subs = male_age[pclass-1]

        else:

            age_subs = female_age[pclass-1]

    else:

        age_subs = age

    return age_subs
titanic_data['Age'] = titanic_data[['Age','Sex','Pclass']].apply(sub_age,axis=1,args=(male_mean_age_class,female_mean_age_class))
sns.heatmap(titanic_data.isnull(),cbar=False,yticklabels=False,cmap='plasma')
titanic_data.drop('Cabin',axis=1,inplace=True)
SexBin = pd.get_dummies(titanic_data['Sex'],drop_first=True)
titanic_data = pd.concat([titanic_data,SexBin],axis=1)
titanic_data.head()
sns.heatmap(titanic_data.corr(),annot=True,cmap='YlGnBu')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

titanic_lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True)
y_train = titanic_data['Survived']

X_train = titanic_data[['Fare','Pclass','male','Parch','SibSp','Age']]
titanic_lda.fit(X_train,y_train)
titanic_test_data = pd.read_csv("../input/test.csv")

titanic_test_data.head()
sns.heatmap(titanic_test_data.isnull(),cbar=False,yticklabels=False,cmap='plasma')
titanic_test_data['Age'] = titanic_test_data[['Age','Sex','Pclass']].apply(sub_age,axis=1,args=(male_mean_age_class,female_mean_age_class))
sns.heatmap(titanic_test_data.isnull(),cbar=False,yticklabels=False,cmap='plasma')
SexBint = pd.get_dummies(titanic_test_data['Sex'],drop_first=True)

titanic_test_data = pd.concat([titanic_test_data,SexBint],axis=1)

titanic_test_data.info()
titanic_test_data.head()

fare_mean_per_class = []

for i in range(3):

    mfare = np.nanmean(titanic_test_data[titanic_test_data['Pclass']==i+1]['Fare'])

    mfare= (np.around(mfare,decimals=4))

    fare_mean_per_class.append(mfare)
def sub_faref(data,fare_mean,i):

    farep = data[0]

    pclass = int(data[1])

    if pd.isnull(farep)==True:        

        subs_fare = fare_mean[pclass-1]        

    else:

        subs_fare = farep        

    return subs_fare
titanic_test_data['Fare'] = titanic_test_data[['Fare','Pclass']].apply(sub_faref,axis=1,args=(fare_mean_per_class,1))
X_test = titanic_test_data[['Fare','Pclass','male','Parch','SibSp','Age']]
X_test.head()
y_pred = titanic_lda.predict(X_test)

y_pred = np.array(y_pred)
survivor_pred = pd.DataFrame({'Survived':y_pred,

                             'PassengerId':titanic_test_data['PassengerId']})

survivor_pred.head(n=10)

true_survive = pd.read_csv("../input/gender_submission.csv")

true_survive.head(n=10)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(true_survive['Survived'],survivor_pred['Survived'],target_names=['Not Survived','Survived']))
fig,ax=plt.subplots(figsize=(12,8))

sns.heatmap(confusion_matrix(true_survive['Survived'],survivor_pred['Survived']),annot=True)

ax.set_title('Confusion Matrix for LDA')