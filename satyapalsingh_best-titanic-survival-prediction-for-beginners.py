# importing libraires

%matplotlib inline                 



import pandas as pd                # Implemennts milti-dimensional array and matrices

import numpy as np                 # For data manipulation and analysis

import matplotlib.pyplot as plt    # Plotting library for Python programming language and it's numerical mathematics extension NumPy

import seaborn as sns              # Provides a high level interface for drawing attractive and informative statistical graphics





from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()







from catboost import CatBoostClassifier   # import algorithms for model training 
from sklearn.preprocessing import Imputer



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



train_df.head()
#As test has only one missing value so lets fill it..

test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)

data_df = train_df.append(test_df) # The entire data: train + test.

passenger_id=test_df['PassengerId']



## We will drop PassengerID and Ticket since it will be useless for our data. 

train_df.drop(['PassengerId'], axis=1, inplace=True)

test_df.drop(['PassengerId'], axis=1, inplace=True)

test_df.shape
print (train_df.isnull().sum())

print (''.center(20, "*"))

print (test_df.isnull().sum())

sns.boxplot(x='Survived',y='Fare',data=train_df)
train_df=train_df[train_df['Fare']<400]
train_df['Sex'] = train_df.Sex.apply(lambda x: 0 if x == "female" else 1)

test_df['Sex'] = test_df.Sex.apply(lambda x: 0 if x == "female" else 1)
train_df
pd.options.display.max_columns = 99

test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)

train_df.head()

for name_string in data_df['Name']:

    data_df['Title']=data_df['Name'].str.extract('([A-Za-z]+)\.',expand=True)





mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)
data_df.groupby('Title')['Age'].median()
# imputing with the mean() strategy.



for name_string in data_df['Name']:

    data_df['Title']=data_df['Name'].str.extract('([A-Za-z]+)\.',expand=True)

    

#replacing the rare title with more common one.

# here the mapping for different title is done through finding relation between two columns. (Sex,Title)



#pd.crosstab(data_train['Title'],data_train['Sex']) (try this out)



mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)



data_df['Title'].value_counts()

train_df['Title']=data_df['Title'][:891]

test_df['Title']=data_df['Title'][891:]



titles=['Mr','Miss','Mrs','Master','Rev','Dr']

for title in titles:

    age_to_impute = data_df.groupby('Title')['Age'].mean()[titles.index(title)]

    #print(age_to_impute)

    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

data_df.isnull().sum()







train_df['Age']=data_df['Age'][:891]

test_df['Age']=data_df['Age'][891:]

test_df.isnull().sum()
train_df.describe()
train_df.groupby('Survived').mean()
train_df.groupby('Sex').mean()
train_df.corr()
plt.subplots(figsize = (15,8))

sns.heatmap(train_df.corr(), annot=True,cmap="PiYG")

plt.title("Correlations Among Features", fontsize = 20)
plt.subplots(figsize = (15,8))

sns.barplot(x = "Sex", y = "Survived", data=train_df, edgecolor=(0,0,0), linewidth=2)

plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)

labels = ['Female', 'Male']

plt.ylabel("% of passenger survived", fontsize = 15)

plt.xlabel("Gender",fontsize = 15)

plt.xticks(sorted(train_df.Sex.unique()), labels)



# 1 is for male and 0 is for female.
sns.set(style='darkgrid')

plt.subplots(figsize = (15,8))

ax=sns.countplot(x='Sex',data=train_df,hue='Survived',edgecolor=(0,0,0),linewidth=2)

train_df.shape

## Fixing title, xlabel and ylabel

plt.title('Passenger distribution of survived vs not-survived',fontsize=25)

plt.xlabel('Gender',fontsize=15)

plt.ylabel("# of Passenger Survived", fontsize = 15)

labels = ['Female', 'Male']

#Fixing xticks.

plt.xticks(sorted(train_df.Survived.unique()),labels)

## Fixing legends

leg = ax.get_legend()

leg.set_title('Survived')

legs=leg.texts

legs[0].set_text('No')

legs[1].set_text('Yes')

plt.subplots(figsize = (8,8))

ax=sns.countplot(x='Pclass',hue='Survived',data=train_df)

plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)

leg=ax.get_legend()

leg.set_title('Survival')

legs=leg.texts



legs[0].set_text('No')

legs[1].set_text("yes")
plt.subplots(figsize=(10,8))

sns.kdeplot(train_df.loc[(train_df['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')

ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived' )



labels = ['First', 'Second', 'Third']

plt.xticks(sorted(train_df.Pclass.unique()),labels)
plt.subplots(figsize=(15,10))



ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 0),'Fare'],color='r',shade=True,label='Not Survived')

ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 1),'Fare'],color='b',shade=True,label='Survived' )

plt.title('Fare Distribution Survived vs Non Survived',fontsize=25)

plt.ylabel('Frequency of Passenger Survived',fontsize=20)

plt.xlabel('Fare',fontsize=20)
train_df.head()
#fig,axs=plt.subplots(nrows=2)

fig,axs=plt.subplots(figsize=(10,8))

sns.set_style(style='darkgrid')

sns.kdeplot(train_df.loc[(train_df['Survived']==0),'Age'],color='r',shade=True,label='Not Survived')

sns.kdeplot(train_df.loc[(train_df['Survived']==1),'Age'],color='b',shade=True,label='Survived')

train_df.head()
## Family_size seems like a good feature to create

train_df['family_size'] = train_df.SibSp + train_df.Parch+1

test_df['family_size'] = test_df.SibSp + test_df.Parch+1

def family_group(size):

    a = ''

    if (size <= 1):

        a = 'loner'

    elif (size <= 4):

        a = 'small'

    else:

        a = 'large'

    return a



train_df['family_group'] = train_df['family_size'].map(family_group)

test_df['family_group'] = test_df['family_size'].map(family_group)
train_df['is_alone'] = [1 if i<2 else 0 for i in train_df.family_size]

test_df['is_alone'] = [1 if i<2 else 0 for i in test_df.family_size]
## We are going to create a new feature "age" from the Age feature. 

train_df['child'] = [1 if i<16 else 0 for i in train_df.Age]

test_df['child'] = [1 if i<16 else 0 for i in test_df.Age]

train_df.child.value_counts()
train_df.head()

#test_df.head()
train_df['calculated_fare'] = train_df.Fare/train_df.family_size

test_df['calculated_fare'] = test_df.Fare/test_df.family_size

train_df.calculated_fare.mean()
train_df.calculated_fare.mode()
def fare_group(fare):

    a= ''

    if fare <= 4:

        a = 'Very_low'

    elif fare <= 10:

        a = 'low'

    elif fare <= 20:

        a = 'mid'

    elif fare <= 45:

        a = 'high'

    else:

        a = "very_high"

    return a

train_df['fare_group'] = train_df['calculated_fare'].map(fare_group)

test_df['fare_group'] = test_df['calculated_fare'].map(fare_group)
train_df = pd.get_dummies(train_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)

test_df = pd.get_dummies(test_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)

train_df.drop(['Cabin', 'family_size','Ticket','Name', 'Fare'], axis=1, inplace=True)

test_df.drop(['Ticket','Name','family_size',"Fare",'Cabin'], axis=1, inplace=True)

pd.options.display.max_columns = 99

def age_group_fun(age):

    a = ''

    if age <= 1:

        a = 'infant'

    elif age <= 4: 

        a = 'toddler'

    elif age <= 13:

        a = 'child'

    elif age <= 18:

        a = 'teenager'

    elif age <= 35:

        a = 'Young_Adult'

    elif age <= 45:

        a = 'adult'

    elif age <= 55:

        a = 'middle_aged'

    elif age <= 65:

        a = 'senior_citizen'

    else:

        a = 'old'

    return a

        
train_df['age_group'] = train_df['Age'].map(age_group_fun)

test_df['age_group'] = test_df['Age'].map(age_group_fun)
train_df = pd.get_dummies(train_df,columns=['age_group'], drop_first=True)

test_df = pd.get_dummies(test_df,columns=['age_group'], drop_first=True)

#Lets try all after dropping few of the column.

train_df.drop(['Age','calculated_fare'],axis=1,inplace=True)

test_df.drop(['Age','calculated_fare'],axis=1,inplace=True)
#age=pd.cut(data_df['Age'],4)

#data_df['Age2']=label.fit_transform(age)

#fare=pd.cut(data_df['Fare'],4)

#data_df['Fare2']=label.fit_transform(fare)

#train_df['Age']=data_df['Age2'][:891]

#train_df['Fare']=data_df['Fare2'][:891]

#test_df['Age']=data_df['Age2'][891:]

#test_df['Fare']=data_df['Fare2'][891:]

#train_df = pd.get_dummies(train_df,columns=['Age','Fare'], drop_first=True)

#test_df = pd.get_dummies(test_df,columns=['Age','Fare'], drop_first=True)

#print(test_df.shape)

#print(train_df.shape)

train_df.head()



train_df.drop(['Title_Rev','age_group_old','age_group_teenager','age_group_senior_citizen','Embarked_Q'],axis=1,inplace=True)

test_df.drop(['Title_Rev','age_group_old','age_group_teenager','age_group_senior_citizen','Embarked_Q'],axis=1,inplace=True)
X = train_df.drop('Survived', 1)

y = train_df['Survived']

#testing = test_df.copy()

#testing.shape
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    svm.SVC(probability=True),

    DecisionTreeClassifier(),

    CatBoostClassifier(),

    XGBClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]

    





log_cols = ["Classifier", "Accuracy"]

log= pd.DataFrame(columns=log_cols)

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit





# SSplit=StratifiedShuffleSplit(test_size=0.3,random_state=7)

# acc_dict = {}



# for train_index,test_index in SSplit.split(X,y):

#     X_train,X_test=X.iloc[train_index],X.iloc[test_index]

#     y_train,y_test=y.iloc[train_index],y.iloc[test_index]

    

#     for clf in classifiers:

#         name = clf.__class__.__name__

          

#         clf.fit(X_train,y_train)

#         predict=clf.predict(X_test)

#         acc=accuracy_score(y_test,predict)

#         if name in acc_dict:

#             acc_dict[name]+=acc

#         else:

#             acc_dict[name]=acc



# log['Classifier']=acc_dict.keys()

# log['Accuracy']=acc_dict.values()

# #log.set_index([[0,1,2,3,4,5,6,7,8,9]])

# %matplotlib inline

# sns.set_color_codes("muted")

# ax=plt.subplots(figsize=(10,8))

# ax=sns.barplot(y='Classifier',x='Accuracy',data=log,color='b')

# ax.set_xlabel('Accuracy',fontsize=20)

# plt.ylabel('Classifier',fontsize=20)

# plt.grid(color='r', linestyle='-', linewidth=0.5)

# plt.title('Classifier Accuracy',fontsize=20)

## Necessary modules for creating models. 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix


std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

testframe = std_scaler.fit_transform(test_df)

testframe.shape

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1000)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score,recall_score,confusion_matrix

logreg = LogisticRegression(solver='liblinear', penalty='l1')

logreg.fit(X_train,y_train)

predict=logreg.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))

print(precision_score(y_test,predict))

print(recall_score(y_test,predict))
C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]

penalties = ['l1','l2']



param = {'penalty': penalties, 'C': C_vals, }

grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=10,shuffle=True), n_jobs=1,scoring='accuracy')
grid.fit(X_train,y_train)

print (grid.best_params_)

print (grid.best_score_)

print(grid.best_estimator_)
#grid.best_estimator_.fit(X_train,y_train)

#predict=grid.best_estimator_.predict(X_test)

#print(accuracy_score(y_test,predict))

logreg_grid = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])

logreg_grid.fit(X_train,y_train)

y_pred = logreg_grid.predict(X_test)

logreg_accy = round(accuracy_score(y_test, y_pred), 3)

print (logreg_accy)

print(confusion_matrix(y_test,y_pred))

print(precision_score(y_test,y_pred))

print(recall_score(y_test,y_pred))
ABC=AdaBoostClassifier()



ABC.fit(X_train,y_train)

predict=ABC.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))

print(precision_score(y_test,predict))

from sklearn.tree import DecisionTreeClassifier

n_estimator=[50,60,100,150,200,300]

learning_rate=[0.001,0.01,0.1,0.2,]

hyperparam={'n_estimators':n_estimator,'learning_rate':learning_rate}

gridBoost=GridSearchCV(ABC,param_grid=hyperparam,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=15,shuffle=True), n_jobs=1,scoring='accuracy')
gridBoost.fit(X_train,y_train)

print(gridBoost.best_score_)

print(gridBoost.best_estimator_)
gridBoost.best_estimator_.fit(X_train,y_train)

predict=gridBoost.best_estimator_.predict(X_test)

print(accuracy_score(y_test,predict))

xgb=XGBClassifier(max_depth=2, n_estimators=700, learning_rate=0.009,nthread=-1,subsample=1,colsample_bytree=0.8)

xgb.fit(X_train,y_train)

predict=xgb.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))

print(precision_score(y_test,predict))

print(recall_score(y_test,predict))
lda=LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)

predict=lda.predict(X_test)

print(accuracy_score(y_test,predict))

print(precision_score(y_test,predict))

print(recall_score(y_test,predict))
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



dectree = DecisionTreeClassifier( criterion="entropy",

                                 max_depth=5,

                                class_weight = 'balanced',

                                min_weight_fraction_leaf = 0.009,

                                random_state=2000)

dectree.fit(X_train, y_train)

y_pred = dectree.predict(X_test)

dectree_accy = round(accuracy_score(y_pred, y_test), 3)

print(dectree_accy)

print(confusion_matrix(y_test,y_pred))

print(precision_score(y_test,y_pred))

print(recall_score(y_test,y_pred))

#from sklearn.ensemble import RandomForestClassifier

#from sklearn.metrics import precision_score,recall_score,confusion_matrix

#randomforest = RandomForestClassifier(n_estimators=100,max_depth=9,min_samples_split=6, min_samples_leaf=4)

##randomforest = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

#randomforest.fit(X_train, y_train)

#y_pred = randomforest.predict(X_test)

#random_accy = round(accuracy_score(y_pred, y_test), 3)

#print (random_accy)

#print(confusion_matrix(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=20,max_features=0.2, min_samples_leaf=8,random_state=20)

#randomforest = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

randomforest.fit(X_train, y_train)

y_pred = randomforest.predict(X_test)

random_accy = round(accuracy_score(y_pred, y_test), 3)

print (random_accy)

print(precision_score(y_test,y_pred))

print(recall_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

from sklearn.ensemble import BaggingClassifier

BaggingClassifier = BaggingClassifier()

BaggingClassifier.fit(X_train, y_train)

y_pred = BaggingClassifier.predict(X_test)

bagging_accy = round(accuracy_score(y_pred, y_test), 3)

print(bagging_accy)
from sklearn.ensemble import VotingClassifier



voting_classifier = VotingClassifier(estimators=[

    ('logreg',logreg), 

    ('random_forest', randomforest),

    ('decision_tree',dectree), 

    ('XGB Classifier', xgb),

    ('BaggingClassifier', BaggingClassifier)])

voting_classifier.fit(X_train,y_train)

y_pred = voting_classifier.predict(X_test)

voting_accy = round(accuracy_score(y_pred, y_test), 3)

print(voting_accy)
#y_predict=randomforest.predict(testframe)
# Prediction with catboost algorithm.

from catboost import CatBoostClassifier

model = CatBoostClassifier(verbose=False, one_hot_max_size=3)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

acc = round(accuracy_score(y_pred, y_test), 3)

print(acc)
y_predict=model.predict(testframe)
temp = pd.DataFrame(pd.DataFrame({

        "PassengerId": passenger_id,

        "Survived": y_predict

    }))





temp.to_csv("../working/submission3.csv", index = False)