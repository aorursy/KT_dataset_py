# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle table-like data and matrices

import numpy as np

import pandas as pd



# Visualisation

import matplotlib as mpl 

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns





#Modeling

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier,  VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



from sklearn.model_selection import  cross_val_score, KFold, learning_curve

from sklearn.model_selection import GridSearchCV



%matplotlib inline
#get titanic data

train = pd.read_csv("../input/train.csv") #(891,12)

test = pd.read_csv("../input/test.csv") #(418, 11)



#combine train & test

total = train.append(test, ignore_index = True)
total.info() 
total.head()
total.describe(include='all')
#unique value

print('col_name'.center(15),'count','value'.center(20))

for col in total.columns:

    length=len(total[col].dropna().unique())

    if length <=10:

        print('##',col.center(11),':' ,length,' ,',total[col].dropna().unique())

    else:

        print('##',col.center(11),':' ,length)
total.isnull().sum()
a=train.Survived.value_counts()

plt.pie(a,labels=a.index,autopct='%1.1f%%')
sns.distplot(train.Age.dropna())
#Survival(%) by Age Interval

fig,ax=plt.subplots(3,3)

fig.subplots_adjust(hspace=0.8,wspace=0.4)

for interval in range(2,11):

    age_dict0={(i,i+interval):0 for i in range(0,int(train.Age.max()+interval),interval)}

    age_dict1={(i,i+interval):0 for i in range(0,int(train.Age.max()+interval),interval)}

    

    def survive_age1(age):

        global age_dict0; value=age//interval

        age_dict0[(interval*value,interval*(value+1))]+=1

                                        

    def survive_age2(age):

        global age_dict1; value=age//interval

        age_dict1[(interval*value,interval*(value+1))]+=1

                      

    train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())].apply(survive_age1)

    train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())].apply(survive_age2)

    age_list=[round(age_dict1[i]*100/(age_dict1[i]+age_dict0[i])) for i in age_dict1.keys() if age_dict0[i]+age_dict1[i]!=0]

    print('###interval=%d###'%(interval))

    a,b=divmod(interval-2,3)

    ax[a][b].plot(age_list,marker='.')

    ax[a][b].set_title("interval:{}".format(interval))

plt.xlabel("Age",x=-1,y=0)

plt.show()

    
a=train['Sex'].value_counts()

print(a)
plt.pie(a,labels=a.index,autopct='%1.1f%%')
sns.barplot(x="Sex", y="Survived", data=train)
a=train['Embarked'].value_counts()

print(a)
plt.pie(a,labels=a.index,autopct='%1.1f%%')
sns.barplot(x='Embarked', y='Survived', data=train)
train[["Embarked","Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
a=train['Pclass'].value_counts()

print(a)

plt.pie(a,labels=a.index,autopct='%1.1f%%')
sns.catplot(x='Pclass',y='Survived',data=train,kind='bar',size=6)
grid = sns.FacetGrid(total, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
a=train['Parch'].value_counts()

print(a)
plt.pie(a,labels=a.index,autopct='%1.1f%%')
sns.catplot(x='Parch',y='Survived',data=train,kind='bar',size=6)
a=train['SibSp'].value_counts()

print(a)
plt.pie(a,labels=a.index,autopct='%1.1f%%')
sns.barplot(x='SibSp',y='Survived',data=train)
sns.distplot(train.Fare.dropna()) 
sns.catplot(x='Survived',y='Fare',data=train,kind='box',size=6)
#Fare Distribution according to Survived

grid = sns.FacetGrid(total, col='Survived', height=3, aspect=1.6)

grid.map(plt.hist, 'Fare', alpha=.5, bins=20)

grid.add_legend();
sns.heatmap(train.corr(),annot=True)
sns.catplot(x='Embarked',y='Age',size=6,kind='box',data=total)

#Conclusion : no special differenciation
sns.factorplot(x='Sex',y='Fare',size=6,kind='box',data=total)
#Age Distirbution according to Title

total['Title'] = total.Name.str.extract('([A-Za-z]+)\.', expand=True)

print(list(total.Title.unique()))

print(total.Title.value_counts())



total['Title'] = total['Title'].replace('Mlle', 'Miss')

total['Title'] = total['Title'].replace(['Capt', 'Col','Countess',

    'Don','Dona', 'Dr', 'Major','Mme','Ms','Lady','Sir', 'Rev', 'Jonkheer' ],'Rare')

print(total.Title.value_counts())
total['FamilySize'] = total['SibSp'] + total['Parch'] + 1
f,ax=plt.subplots(1,1,figsize=(15,4))

sns.barplot('FamilySize','Survived',data=total,ax=ax)

ax.set_title('FamilySize vs Survived')

total.loc[total['FamilySize'] == 1, 'FamilySize'] = 0

total.loc[(total['FamilySize'] > 1) & (total['FamilySize'] <= 4), 'FamilySize'] = 1

total.loc[(total['FamilySize'] > 4), 'FamilySize']   = 2
total[['FamilySize', 'Survived']].groupby(['FamilySize']).mean().sort_values(by='Survived',ascending=False)
sns.heatmap(total.corr(),annot=True)
#Parchm SibSp del

total = total.drop(['Parch','SibSp'], axis=1)
total[total.Fare.isnull()]
sns.distplot(total.Fare[(total.Pclass==3) & (total.Fare.notnull())])
total['Fare'] = total.Fare.fillna(total[total['Pclass']==3]['Fare'].median())
total.Fare.isnull().sum()
total[total.Embarked.isnull()]
sns.catplot(x='Embarked',y='Fare',size=6,kind='box',data=total)
total['Embarked'] = total.Embarked.fillna('C')
total.Embarked.isnull().sum()
TotalAge = total[total.Age.isnull()==False]

grid = sns.FacetGrid(TotalAge, col="Title", hue="Title",col_wrap=4)

grid.map(sns.distplot, "Age")
total[['Title', 'Age']].groupby(['Title']).median().sort_values(by='Title',ascending=False)
total['Age']=total.groupby('Title').transform(lambda x:x.fillna(x.median()))
total = total.drop(['Cabin','Name','Ticket'], axis=1)
total.head()
total['Sex'] = total['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

total.head()
total["Embarked"] = total["Embarked"].astype("category")

total = pd.get_dummies(total, columns = ["Embarked"],prefix="Embarked")

total.head()
total["Pclass"] = total["Pclass"].astype("category")

total = pd.get_dummies(total, columns = ["Pclass"],prefix="Pclass")

total.head()
total["Title"] = total["Title"].astype("category")

total = pd.get_dummies(total, columns = ["Title"],prefix="Title")
total[ 'Family_Single' ] = total[ 'FamilySize' ].map( lambda s : 1 if s == 0 else 0 )

total[ 'Family_Small' ]  = total[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )

total[ 'Family_Large' ]  = total[ 'FamilySize' ].map( lambda s : 1 if s == 2 else 0 )



total = total.drop(['FamilySize'], axis=1)

total.head()
del train,test

train=total[:891];test=total[891:];test_id=test['PassengerId'];test=test.drop(['Survived','PassengerId'],1)

y_train=train['Survived'];x_train=train.drop(['Survived','PassengerId'],1)



x_train1, x_test, y_train1, y_test = train_test_split(x_train, y_train, random_state=0)

del total,train
random_state = 0

kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)



classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))





cv_means = [];cv_stds= []

for classifier in classifiers :

    result=cross_val_score(classifier, x_train, y_train, scoring = "accuracy", cv = kfold)

    cv_means.append(result.mean());cv_stds.append(result.std())

    

cv_df= pd.DataFrame({"Means":cv_means,"Stds": cv_stds,"Algorithm":["SVC","DecisionTree","RandomForest","KNeighboors","LogisticRegression"]})



g = sns.barplot("Means","Algorithm",data = cv_df,orient = "h",**{'xerr':cv_stds})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
#Logistic Regression

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}



grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5,n_jobs=-1,return_train_score=True)



grid_search.fit(x_train1, y_train1)

print("Test score: {}".format(grid_search.score(x_test,y_test)))

print("Best Parameter : {}".format(grid_search.best_params_))

print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))



print("Best score model: \n{}".format(grid_search.best_estimator_))
logi=LogisticRegression(C=0.1)

logi.fit(x_train,y_train)

y_pred=logi.predict(test).astype(int)

test_score = round(logi.score(x_train, y_train) * 100, 2)

print(test_score)



submission = pd.DataFrame({

        "PassengerId": test_id,

        "Survived": y_pred

    })



submission.to_csv('submission.csv', index=False)