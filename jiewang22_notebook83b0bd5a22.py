import pandas as pd

%pylab inline

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score

from imblearn.over_sampling import SMOTE


titanic_path = "../input/train.csv"

df = pd.read_csv(titanic_path)

test_path = "../input/test.csv"

df_test = pd.read_csv(test_path)
hist2d(df['Fare'], df['Survived'],bins=(20,2), range=((0,100),(0,1)));

ylabel('Survived')

xlabel('Fare')
# In the training data, Age has 19.8% missing values, Cabin has 77.1%, Embarked has 0.2%

ratio_of_missing = df.isnull().sum()/len(df)

print(ratio_of_missing)
# In the testing data, Age has 20.5% missing values, Cabin has 78.2%, Fare has 0.2%

test_ratio_of_missing = df_test.isnull().sum()/len(df_test)

print(test_ratio_of_missing)
# There are 11 predictors, 6 of them are numerical (Passeneger ID, Pclass,Age,Sibsp, Parch,Fare) 

# and  5 categorical predictors (Tickets, Name,Sex,Cabin,Embarked)

df.info()
## Correlation matrix between numerical values (Survived, SibSp Parch Age and Fare and Pclass values) and Survived 

sns.heatmap(df[["Survived","SibSp","Parch","Age","Fare","Pclass"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.figure(figsize=(15,5))

plt.subplot(1, 4, 1)

Pclass_survive_ratio=(df.groupby(['Pclass']).sum()/df.groupby(['Pclass']).count())['Survived']

Pclass_survive_ratio.plot(kind='bar')

plt.subplot(1, 4, 2)

age_train_p=df[~np.isnan(df['Age'])] #remove missing values

ages=np.arange(0,80,10) #categorize the age

age_cut=pd.cut(age_train_p.Age,ages) 

age_cut_grouped=age_train_p.groupby(age_cut) 

age_Survival_Rate=(age_cut_grouped.sum()/age_cut_grouped.count())['Survived']  

age_Survival_Rate.plot(kind='bar')



plt.subplot(1, 4, 3)

Parch_survive_ratio=(df.groupby(['Parch']).sum()/df.groupby(['Parch']).count())['Survived']

Parch_survive_ratio.plot(kind='bar')



plt.subplot(1, 4, 4)

SibSp_survive_ratio=(df.groupby(['SibSp']).sum()/df.groupby(['SibSp']).count())['Survived']

SibSp_survive_ratio.plot(kind='bar')



Fare_plot = sns.distplot(df["Fare"], color="b", label="Skewness : %.2f"%(df["Fare"].skew()))

Fare_plot.legend(loc="best")
#Apply log to Fare to reduce skewness distribution

df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
log_Fare_plot  = sns.distplot(df["Fare"], color="b", label="Skewness : %.2f"%(df["Fare"].skew()))

log_Fare_plot.legend(loc="best")
Embarked_survive_ratio=(df.groupby(['Embarked']).sum()/df.groupby(['Embarked']).count())['Survived']

Embarked_survive_ratio.plot(kind='bar')
sex_survive_ratio = (df.groupby(['Sex']).sum()/df.groupby(['Sex']).count())['Survived']

sex_survive_ratio.plot(kind='bar')
# drop Cabin from both training and testing dataset

df.drop('Cabin',axis=1)

df_test.drop('Cabin',axis=1)
# Extract Mr., Miss,Master,Mrs from the name

def title_extract(data):

    name_arrange= data.Name.str.split('.').apply(lambda x: x[0])

    sur = name_arrange.str.split(',').apply(lambda x: x[1])

    data['Title'] = name_arrange.str.split(',').apply(lambda x: x[1])
title_extract(df)

title_extract(df_test)
df.groupby('Title')['Title'].count()
df_test.groupby('Title')['Title'].count()
# sevral titles like Col, Dona os very rare, so I combined them together, give a new title "Rare_Title

def redefine_title(data):

    data.loc[data.Title ==' Mlle', 'Title'] = ' Miss' 

    data.loc[data.Title ==' Ms', 'Title'] = ' Miss'

    data.loc[data.Title ==' Mme', 'Title'] =' Mrs' 

    data['Title'] =data['Title'].replace([' Lady', ' Countess',' Capt', ' Col',' Don', ' Dr',\

                                       ' Major', ' Rev', ' Sir', ' Jonkheer',' the Countess',' Dona'], ' Rare_Title')

redefine_title(df)

df['Title'].value_counts()
redefine_title(df_test)

df_test['Title'].value_counts()
# Create a groupby object: by_Tile_Sex_class

df_group_age = df.groupby(['Title','Sex'])

test_group_age = df_test.groupby(['Title','Sex'])

# Write a function that imputes median

def impute_median(series):

    return series.fillna(series.median())



# Impute age and assign to titanic.Imputed_age

df['Imputed_age'] = df_group_age.Age.transform(impute_median)

print(df['Imputed_age'].isnull().sum())



df_test['Imputed_age'] = test_group_age.Age.transform(impute_median)

print(df_test['Imputed_age'].isnull().sum())
#Imputation of missing values for categories predictor Embarked

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().index[0])

df['Embarked'].isnull().sum()
#Imputation of missing values for categories predictor Embarked

df_test['Fare'] = df_test['Fare'].fillna(df['Fare'].mean())

df['Fare'].isnull().sum()
import seaborn as sns

p1=sns.kdeplot(df['Imputed_age'], shade=True, color="r")

p1=sns.kdeplot(df['Age'], shade=True, color="b")
def newfeature(data):

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1    

newfeature(df)

newfeature(df_test)
familysize_ratio = (df.groupby(['FamilySize']).sum()/df.groupby(['FamilySize']).count())['Survived']

familysize_ratio.plot(kind='bar')

#The family size seems to play an important role, survival probability is worst for large families.
Title_survive_ratio = (df.groupby(['Title']).sum()/df.groupby(['Title']).count())['Survived']

Title_survive_ratio.plot(kind='bar')
def cat_num(data):

    title_mapping = {" Mr": 1, " Miss": 2, " Mrs": 3, " Master": 4, " Rare_Title": 5}

    data["TitleCat"] = data.loc[:,'Title'].map(title_mapping)

    sex_mapping = {"female": 1, "male": 0}

    data["Gender"] = data.loc[:,'Sex'].map(sex_mapping)

    emabrked_mapping = {"S": 0, "C": 1, "Q": 2}

    data["EmabrkedCat"] = data.loc[:,'Embarked'].map(emabrked_mapping)

    data.loc[df['Fare'] <= 10.4625, 'Fare'] = 0

    data.loc[(df['Fare'] > 10.4625) & (df['Fare'] <= 73.5), 'Fare'] = 1

    data.loc[df['Fare'] > 73.5, 'Fare'] = 2

    data['Single'] = data['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    data['SmallF'] = data['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

    data['MedF'] = data['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

    data['LargeF'] = data['FamilySize'].map(lambda s: 1 if s >= 5 else 0)



    

cat_num(df)

cat_num(df_test)
# 0 indicated dead, 1 indicated survive

df.groupby('Survived')['Survived'].count()
variables = ['Pclass','Gender','Imputed_age','SibSp', 'Parch','Fare','TitleCat','EmabrkedCat','Survived','Single',

'SmallF','MedF','LargeF']

data_relevent_num = df[variables]



#Split data into 2 parts: training data and testing data and balance the data using SMOTE

training_features, test_features, \

training_target, test_target, = train_test_split(data_relevent_num.drop(['Survived'], axis=1),

                                               data_relevent_num['Survived'],

                                               test_size = .25,

                                               random_state=12)

sm = SMOTE(random_state=12,ratio = 'minority')

x_train_balance, y_train_balance =sm.fit_sample(training_features, training_target)

#print (training_target.value_counts(), np.bincount(y_train_balance))



#clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)

#clf_rf.fit(x_train_balance, y_train_balance)

#y_predict = clf_rf.predict(test_features)

#random_confusion = confusion_matrix(test_target, y_predict)

#random_confusion



#print(accuracy_score(test_target,y_predict))
# Cross validate model with Kfold stratified cross val

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



kfold = StratifiedKFold(n_splits=10)

# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 8],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(x_train_balance,y_train_balance)

RFC_best = gsRFC.best_estimator_

# Best score

print(gsRFC.best_score_)

print(RFC_best)

#y_predict = gsRFC.predict(test_features)

#RFC_accuracy = accuracy_score(test_target,y_predict)

#RFC_accuracy
# Predict the survival using the best tunning parameter

RFCbest = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',

            max_depth=None, max_features=1, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=10,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

RFCbest.fit(x_train_balance, y_train_balance)

y_predict = RFCbest.predict(test_features)

print(accuracy_score(test_target,y_predict))
# Gradient boosting tunning

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(x_train_balance,y_train_balance)



GBC_best = gsGBC.best_estimator_

print(GBC_best)

# Best score

gsGBC.best_score_
# Predict the survival using the best tunning parameter

gsGBCbest = GradientBoostingClassifier(criterion='friedman_mse', init=None,

              learning_rate=0.1, loss='deviance', max_depth=4,

              max_features=0.3, max_leaf_nodes=None,

              min_impurity_decrease=0.0, min_impurity_split=None,

              min_samples_leaf=100, min_samples_split=2,

              min_weight_fraction_leaf=0.0, n_estimators=300,

              presort='auto', random_state=None, subsample=1.0, verbose=0,

              warm_start=False)

gsGBCbest.fit(x_train_balance, y_train_balance)

y_predict = gsGBCbest.predict(test_features)

print(accuracy_score(test_target,y_predict))
lsr = LogisticRegression()

lsr_trained_model = lsr.fit(x_train_balance,y_train_balance)

lsr_y_pred = lsr_trained_model.predict(test_features)

lsr_accuracy = accuracy_score(test_target,lsr_y_pred)

lsr_accuracy
RFCbest.fit(x_train_balance, y_train_balance)

y_predict = RFCbest.predict(test_features)

random_confusion = confusion_matrix(test_target, y_predict)

random_confusion



print(accuracy_score(test_target,y_predict))



imp= RFCbest.feature_importances_

names = training_features.columns

imp, names = zip(*sorted(zip(imp,names)))

plt.barh(range(len(names)), imp, align = 'center')

plt.yticks(range(len(names)),names)

plt.xlabel('Importance of Features')

plt.ylabel('Features')

plt.title('Importance of each feature')

plt.show()

#print (importances)

variables = ['Pclass','Gender','Imputed_age','SibSp', 'Parch','Fare','TitleCat','EmabrkedCat','Single','SmallF','MedF','LargeF']

data_relevent_test = df_test[variables]

df_test['Survived'] = RFCbest.predict(data_relevent_test)

submission = pd.DataFrame(df_test)

submission.to_csv("prediction_rf.csv")