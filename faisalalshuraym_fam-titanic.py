# import necessary libraries 

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt



# import models 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV



# set display rows to 1000

pd.set_option('display.max_rows',1000)

pd.set_option('display.max_columns',1000)



# Lines below are just to ignore warnings

import warnings

warnings.filterwarnings('ignore')
train =pd.read_csv('../input/titanic/train.csv')
test =pd.read_csv('../input/titanic/test.csv')
test.head()
train.head()
train.info()
train.Embarked.unique()
test.info()
# filling the age 

train.describe().transpose()
test.describe().transpose()
# clean Age by breaking by fliing them with mean of each class 

mean_of_class =train[['Age','Pclass']].groupby('Pclass').mean()

mean_of_class
# group by age and Pclass and take the mean 

mean_of_class =test[['Age','Pclass']].groupby('Pclass').mean()
mean_of_class
train.head()
#defining a function 'impute_age'

def impute_age(age_pclass): # passing age_pclass as ['Age', 'Pclass']

    

    # Passing age_pclass[0] which is 'Age' to variable 'Age'

    Age = age_pclass[0]

    

    # Passing age_pclass[2] which is 'Pclass' to variable 'Pclass'

    Pclass = age_pclass[1]

    

    #applying condition based on the Age and filling the missing data respectively 

    if pd.isnull(Age):



        if Pclass == 1:

            return 38



        elif Pclass == 2:

            return 30



        else:

            return 25



    else:

        return Age
# (for train) grab age and apply the impute_age, our custom function 

train['Age']= train.apply(lambda i : impute_age(i[['Age','Pclass']]),axis=1 )
#(for test) grab age and apply the impute_age, our custom function 

test['Age']= test.apply(lambda i : impute_age(i[['Age','Pclass']]),axis=1 )
train.head()
train.isnull().sum()
test.isnull().sum()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
## clean cabin and fill it with zero and 1

train['Cabin'].value_counts()
# fill in nan with 0 

train.Cabin.fillna(0, inplace=True) 
# fill in 1 when we have string 

train.Cabin = train.Cabin.apply(lambda x: 0 if x !=0 else 1)
test.head()
test.Cabin.fillna(0, inplace=True) 
# fill in 1 for test datset

test.Cabin=test.Cabin.apply(lambda x: 0 if x !=0 else 1)
# dummy Embark dataset 

train['Embarked'].fillna('S', inplace=True)
test.isnull().sum()
# fill in test with the mean 

test['Fare'].fillna(test.Fare.mean(), inplace=True)
test.isnull().sum()
train.head(10)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
import seaborn as sns

pal = {'male':"blue", 'female':"Pink"}

sns.set(style="darkgrid")

plt.subplots(figsize = (15,8))

ax = sns.barplot(x = "Sex", 

                 y = "Survived", 

                 data=train, 

                 palette = pal,

                 linewidth=5,

                 order = ['female','male'],

                 capsize = .05,



                )



plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25,loc = 'center', pad = 40)

plt.ylabel("% of passenger survived", fontsize = 15, )

plt.xlabel("Sex",fontsize = 15);
pal = {1:"seagreen", 0:"gray"}

sns.set(style="darkgrid")

plt.subplots(figsize = (15,8))

ax = sns.countplot(x = "Sex", 

                   hue="Survived",

                   data = train, 

                   linewidth=4, 

                   palette = pal

)



## Fixing title, xlabel and ylabel

plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25, pad=40)

plt.xlabel("Sex", fontsize = 15);

plt.ylabel("# of Passenger Survived", fontsize = 15)



## Fixing xticks

#labels = ['Female', 'Male']

#plt.xticks(sorted(train.Sex.unique()), labels)



## Fixing legends

leg = ax.get_legend()

leg.set_title("Survived")

legs = leg.texts

legs[0].set_text("No")

legs[1].set_text("Yes")



plt.show()
plt.subplots(figsize = (15,10))

sns.barplot(x = "Pclass", 

            y = "Survived", 

            data=train, 

            linewidth=5,

            capsize = .1



           )

plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25, pad=40)

plt.xlabel("Socio-Economic class", fontsize = 15);

plt.ylabel("% of Passenger Survived", fontsize = 15);

labels = ['Upper', 'Middle', 'Lower']

#val = sorted(train.Pclass.unique())

val = [0,1,2] ## this is just a temporary trick to get the label right. 

plt.xticks(val, labels);
 #Kernel Density Plot

fig = plt.figure(figsize=(15,8),)

## I have included to different ways to code a plot below, choose the one that suites you. 

ax=sns.kdeplot(train.Pclass[train.Survived == 0] , 

               color='gray',

               shade=True,

               label='not survived')

ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] , 

               color='g',

               shade=True, 

               label='survived', 

              )

plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25, pad = 40)

plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)

plt.xlabel("Passenger Class", fontsize = 15,labelpad =20)

## Converting xticks into words for better understanding

labels = ['Upper', 'Middle', 'Lower']

plt.xticks(sorted(train.Pclass.unique()), labels);
# Kernel Density Plot

fig = plt.figure(figsize=(15,8),)

ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')

ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')

plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25, pad = 40)

plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)

plt.xlabel("Fare", fontsize = 15, labelpad = 20);
train[train.Fare > 280]

# (train )dummies for 'Embarked','Sex' columns 

train =pd.get_dummies(train,columns=['Embarked','Sex'],drop_first=True)
#(test) dummies for 'Embarked','Sex' columns

test =pd.get_dummies(test,columns=['Embarked','Sex'],drop_first=True)
train.head()
train.Name
test.head()
train.columns
# corrlation heat map to the realtionship between the coulmns 

## heatmeap to see the correlation between features. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)

import numpy as np

mask = np.zeros_like(train.corr(), dtype=np.bool)

#mask[np.triu_indices_from(mask)] = True



plt.subplots(figsize = (15,12))

sns.heatmap(train.corr(), 

            annot=True,

            mask = mask,

            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"

            linewidths=.9, 

            linecolor='gray',

            fmt='.2g',

            center = 0,

            square=True)

plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40);
# feutures  

X_train= train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked_Q', 'Embarked_S', 'Sex_male']]
# target 

y_train= train['Survived']
# fetures

X_test= test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked_Q', 'Embarked_S', 'Sex_male']]
# target 

y_test = train['Survived']
# importation 

from sklearn.preprocessing import StandardScaler 
# scale

ss = StandardScaler().fit(X_train)
# apply tranformation to dataset

ss.transform(X_train)
# descalring Random forest Classifier 

from sklearn.model_selection import GridSearchCV
# descalring Random forest Classifier 

rf = RandomForestClassifier() # bootstrap=True by default #max_features='auto',
# fit our model to random forest 

rf.fit(X_train,y_train)
rf.score(X_train,y_train) 
cross_val_score(rf,X_train,y_train).mean()
from sklearn.model_selection import RandomizedSearchCV
# specify parameters and distributions to sample from

rf_params = {

    'n_estimators': [10, 50, 100],

    'max_features':[2, 3, 5, 7, 8],

    'max_depth': [1,2,3,4,5,6,7,8,9,10],

    'criterion':['gini', 'entropy'],

}
gs = GridSearchCV(rf, param_grid=rf_params, cv=5, verbose= 1)
gs.fit(X_train,y_train)

pred_gs = gs.predict(X_test) 
gs.best_estimator_
gs.best_score_
cross_val_score(gs,X_train,y_train,).mean()
sub = pd.DataFrame()

sub['PassengerId'] = test.PassengerId

sub['Survived'] = pred_gs

sub.to_csv('submission_Titanic_final.csv',index=False)