# Import General Libraries 

import pandas as pd

import numpy as np

import random

import seaborn as sns

import matplotlib.pyplot as plt
# Loading Data-Set

path1 = '../input/train.csv'

path2 = '../input/test.csv'

path3 = '../input/genderclassmodel.csv'



train = pd.read_csv(path1)

test = pd.read_csv(path2)

test_label = pd.read_csv(path3)
# Data-Set Priliminery Review 



print(train.info())

print('------------------------')

print(train.isnull().sum())

print('------------------------')

print(train.describe())



print('########################')



#print(test.info())

#print('------------------------')

#print(test.isnull().sum())

#print('------------------------')

#print(test.describe())
# Gender visual EDA (Figure 1)

%matplotlib inline

plt.figure(figsize=(5,5))



men = train['Sex'][train['Sex']=='male'].count()

women = train['Sex'][train['Sex']=='female'].count()



slices = [men,women]; labels = ['Men','Women']; colors = ['b','m']



plt.pie(slices,labels = labels, colors = colors, startangle = 90, autopct = '%1.1f%%',explode = [0.01,0])

plt.title('Fig.1 : Passenger Gender Ratio',fontweight="bold", size=12)

plt.show()
# Gender visual EDA (Figure 2)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)

men_survived = train['Sex'][(train['Sex']=='male')&(train['Survived']==1)].count()

men_not_survived = men - men_survived

slices = [men_survived,men_not_survived];labels = ['Survived','Not Survived'];colors = ['b','lightgray']



plt.pie(slices,labels = labels, colors = colors, startangle = 90, autopct = '%1.1f%%',explode = [0.01,0])

plt.title('Men')



plt.subplot(1,2,2)

women_survived = train['Sex'][(train['Sex']=='female')&(train['Survived']==1)].count()

women_not_survived = women - women_survived

slices = [women_survived,women_not_survived];labels = ['Survived','Not Survived'];colors = ['m','lightgray']



plt.pie(slices,labels = labels, colors = colors, startangle = 90, autopct = '%1.1f%%',explode = [0.01,0])

plt.title('Women')

plt.suptitle('Fig.2 : Survaival Rate by Gender',fontweight="bold", size=12)

plt.subplots_adjust(top=0.75)



plt.show()

# Gender visual EDA (Figure 3)

plt.figure(figsize=(5,5))



slices = [men_survived,men_not_survived,women_survived,women_not_survived]

labels = ['Men - Survived','Men - Not Survived','Women - Survived','Women - Not Survived']

colors = ['b','lightgray','m','lightgray']

plt.pie(slices,labels = labels, colors = colors, startangle = 90, autopct = '%1.1f%%',explode = [0.02,0,0.02,0])

plt.title('Fig.3 : Survival Ratio devided by Gender',fontweight="bold", size=12)

plt.show()
# Sex is another object column in the dataset, which needs to be converted to categorial type



train['Sex']=train['Sex'].astype('category')

train['Sex_cat']=train['Sex'].cat.codes



test['Sex']=test['Sex'].astype('category')

test['Sex_cat']=test['Sex'].cat.codes
# Visual EDA on Age column

train['Age'].dropna().hist(bins=8, color='m',alpha=0.5,label='Onboard') # All passengers onboard 

train['Age'][train['Survived']==1].dropna().hist(bins=8, color='b',alpha=0.75,label='Survived') # Survived passengers



plt.xlabel('Age'); plt.ylabel('Number of Passenger')

plt.title('Fig.4 : Passengers Age Distribution',fontweight="bold", size=12)



plt.legend()

plt.tight_layout()
# Age distibution based on class and sex

fig = sns.FacetGrid(train, row='Pclass', col='Sex', size=2.0, aspect=2.0)

fig.map(plt.hist, 'Age', alpha=.5, bins=10, color = 'darkslateblue')

plt.subplots_adjust(top=0.85)



fig.fig.suptitle('Fig.5 : Age Distribution based on Gender & Class',fontweight="bold", size=12)

fig.add_legend()

plt.show()
# Claculating mean for 'Age' column based on gender and class

# Neat impelementation of this session is inspired by nice work of "Manav Sehgal". 

# You can find his original notebook at following URL:

# https://www.kaggle.com/startupsci/titanic-data-science-solutions



age_train = np.zeros((2,3))

age_test = np.zeros((2,3))



for i in range(0,2):

    for j in range(0,3):

        age_train[i,j] = train['Age'][(train['Sex_cat'] == i) & (train['Pclass'] == j+1)].mean()

        age_test[i,j] = test['Age'][(test['Sex_cat'] == i) & (test['Pclass'] == j+1)].mean()



for i in range(0,2):

    for j in range(0,3):

        train.loc[(train['Age'].isnull())&(train['Sex_cat'] == i)&(train['Pclass'] == j+1),'Age'] = age_train[i,j] 

        test.loc[(test['Age'].isnull())&(test['Sex_cat'] == i)&(test['Pclass'] == j+1),'Age'] = age_test[i,j]   
# Alternative method (common practice)       



#ave_age_train = train['Age'].mean()

#std_age_train = train['Age'].std()

#ave_age_test = test['Age'].mean()

#std_age_test = test['Age'].std()

        

#random.seed(42)

#train['Age']=train['Age'].fillna(ave_age_train + random.uniform(-1,1) * std_age_train)

#test['Age']=test['Age'].fillna(ave_age_test + random.uniform(-1,1) * std_age_test)
# Visualization of survival rate based on fare

bins = np.arange(0, 550, 10)

index = np.arange(54)

train['fare_bin'] = pd.cut(train.Fare,bins,right=False)

total_bin = train.groupby(['fare_bin']).size().values

survived_bin = train[train['Survived']==1].groupby(['fare_bin']).size().values



np.seterr(divide='ignore', invalid='ignore') # ignoring "divide by zero" or "divide by NaN"

survived_fare = survived_bin*100/total_bin

###############

fig, ax = plt.subplots(1,1,figsize=(10,6))

colormap=plt.cm.get_cmap('jet')

ax.scatter(index*10,survived_fare,marker='o', edgecolor='black', c=survived_fare**0.2,cmap=colormap,alpha=0.75,s = 7*total_bin )

plt.xlabel('Fare'); plt.ylabel('Survival Rate (%)')

plt.title('Fig.6 : Survival Rate vs. Fare',fontweight="bold", size=12)

plt.xticks([10,50,100,150,200,250,300,350,400,450,500,550])

plt.xlim([-50,550]);plt.ylim([0,110])  

plt.tight_layout()

plt.show()
# There is one missing value in Fare column of test dataset, which is required to 

# be replaced by proper value. Preliminary review also revealed some rows in both training

# and test datasets have zero values. Therfore:



train['Fare'].replace('0',None,inplace=True)

test['Fare'].replace('0',None,inplace=True)



train_fare_trans = train['Fare'].groupby(train['Pclass'])

test_fare_trans = test['Fare'].groupby(test['Pclass'])



f = lambda x : x.fillna(x.mean())

train['Fare'] = train_fare_trans.transform(f)

test['Fare'] = test_fare_trans.transform(f)

###############

# similar to age, fare values are also needed to be scaled

###############

train = train.drop(['fare_bin'],axis=1) # 'fare_bin' only generatad for visualization purposes
# Missing data regarding port of embarkation is replaced by most frequent value

mode_emb_train = train['Embarked'].mode()

train['Embarked']=train['Embarked'].fillna("S")

train['Embarked']=train['Embarked'].astype('category')

train['Embarked_cat']=train['Embarked'].cat.codes

###############

mode_emb_test = test['Embarked'].mode()

test['Embarked']=test['Embarked'].fillna("S")

test['Embarked']=test['Embarked'].astype('category')

test['Embarked_cat']=test['Embarked'].cat.codes
# Dropping unnecessary columns

X_train = train.drop(['Survived','Sex','Embarked','PassengerId','Name','Ticket','Cabin'],axis=1)

Y_train = train['Survived']

###############

X_test  = test.drop(['Sex','Embarked','PassengerId','Name','Ticket','Cabin'],axis=1)

Y_test = test_label['Survived']

# Making correlation coefficients pair plot of all feature in order to identify degenrate features

ax = plt.axes()

sns.heatmap(X_train.corr(), vmax=1.0,vmin=-1.0, square=True, annot=True, cmap='Blues',linecolor="white", linewidths=0.01, ax=ax)

ax.set_title('Fig.7 : Correlation Coefficient Pair Plot',fontweight="bold", size=12)

plt.show()
# Import ML Libraries 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
# Logistic Regression

steps = [('scaler', StandardScaler()),('logreg', LogisticRegression())]

pipeline = Pipeline(steps)

pipeline.fit(X_train, Y_train)

Y_pred1 = pipeline.predict(X_test)

print('Logistic Regression')

print('==========================')

print('Test score:',pipeline.score(X_test, Y_test))

print('==========================')

print('Final Report on Prediction Result')

print(classification_report(Y_test, Y_pred1))
# KNeighborsClassifier

import warnings

warnings.filterwarnings('ignore') # Updated version of GridSearchCV is not available

steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameters = {'knn__n_neighbors':np.arange(1, 100)}

cv = GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, Y_train)

Y_pred2 = cv.predict(X_test)

print('K Neighbors Classifier')

print('==========================')

d = cv.best_params_

print('Optimum Number of Neighbors:',d.get('knn__n_neighbors'))

print('Test score:', cv.score(X_test, Y_test))

print('==========================')

print('Final Report on Prediction Result')

print(classification_report(Y_test, Y_pred2))

# RandomForestClassifier

steps = [('scaler', StandardScaler()),('randfor', RandomForestClassifier())]

pipeline = Pipeline(steps)



parameters = {'randfor__n_estimators':np.arange(1, 100)}

cv = GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, Y_train)

Y_pred3 = cv.predict(X_test)



print('Random Forest Classifier')

print('==========================')

d = cv.best_params_

print('Optimum Number of Estimators:',d.get('randfor__n_estimators'))

print('Test score:', cv.score(X_test, Y_test))

print('==========================')

print('Final Report on Prediction Result')

print(classification_report(Y_test, Y_pred3))

# Final submission



submission = pd.DataFrame({

        'PassengerId': test['PassengerId'],

        'Survived': Y_pred1 })

submission.to_csv('titanic.csv', index=False)