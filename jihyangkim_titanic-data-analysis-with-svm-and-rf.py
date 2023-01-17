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

testID = test['PassengerId']



del train, test

train=total[:891]
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
train.Survived.value_counts()/train.Survived.count()
sns.distplot(total.Age.dropna())
#Survival(%) by Age Interval

fig,ax=plt.subplots(3,3)

fig.subplots_adjust(hspace=0.8,wspace=0.4)

for interval in range(2,11):

    age_dict0={(i,i+interval):0 for i in range(0,int(total.Age.max()+interval),interval)}

    age_dict1={(i,i+interval):0 for i in range(0,int(total.Age.max()+interval),interval)}

    

    def survive_age1(age):

        global age_dict0; value=age//interval

        age_dict0[(interval*value,interval*(value+1))]+=1

                                        

    def survive_age2(age):

        global age_dict1; value=age//interval

        age_dict1[(interval*value,interval*(value+1))]+=1

                      

    total["Age"][(total["Survived"] == 0) & (total["Age"].notnull())].apply(survive_age1)

    total["Age"][(total["Survived"] == 1) & (total["Age"].notnull())].apply(survive_age2)

    age_list=[round(age_dict1[i]*100/(age_dict1[i]+age_dict0[i])) for i in age_dict1.keys() if age_dict0[i]+age_dict1[i]!=0]

    print('###interval=%d###'%(interval))

    a,b=divmod(interval-2,3)

    ax[a][b].plot(age_list,marker='.')

    ax[a][b].set_title("interval:{}".format(interval))

plt.xlabel("Age",x=-1,y=0)

plt.show()

    
sns.catplot('Sex',data=total,kind='count',size=6)
sns.barplot(x="Sex", y="Survived", data=total)
sns.catplot('Embarked',data=total,kind='count',size=6)
sns.catplot(x='Embarked',y='Survived',data=total,kind='bar',size=6)
total[["Embarked","Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.catplot('Pclass',data=train,kind='count',size=6)
sns.catplot(x='Pclass',y='Survived',data=train,kind='bar',size=6)
grid = sns.FacetGrid(total, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
sns.catplot('Parch',data=total,kind='count',size=6)
sns.catplot(x='Parch',y='Survived',data=total,kind='bar',size=6)
sns.catplot('SibSp',data=total,kind='count',size=6)
sns.factorplot(x='SibSp',y='Survived',data=total,kind='bar',size=6)
sns.distplot(total.Fare.dropna()) 
sns.catplot(x='Survived',y='Fare',data=total,kind='box',size=6)
#Fare Distribution according to Survived

grid = sns.FacetGrid(total, col='Survived', height=3, aspect=1.6)

grid.map(plt.hist, 'Fare', alpha=.5, bins=20)

grid.add_legend();
sns.heatmap(total.corr(),annot=True)


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
f,ax=plt.subplots(1,2,figsize=(15,4))

sns.barplot('FamilySize','Survived',data=total,ax=ax[0])

ax[0].set_title('FamilySize vs Survived')

total.loc[total['FamilySize'] == 1, 'FamilySize'] = 0

total.loc[(total['FamilySize'] > 1) & (total['FamilySize'] <= 4), 'FamilySize'] = 1

total.loc[(total['FamilySize'] > 4), 'FamilySize']   = 2
total[['FamilySize', 'Survived']].groupby(['FamilySize']).mean().sort_values(by='Survived',ascending=False)
sns.heatmap(total.corr(),annot=True)
#Parchm SibSp del

total = total.drop(['Parch','SibSp'], axis=1)
total[total.Fare.isnull()]
sns.distplot(total.Fare[(total.Pclass==3) & (total.Fare.notnull())])
total['Fare'] = total.Fare.fillna(total.Fare.median())
total[total.Embarked.isnull()]
sns.catplot(x='Embarked',y='Fare',size=6,kind='box',data=total)
total['Embarked'] = total.Embarked.fillna('C')
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
train = total[ :891] ;y_train=train['Survived'];x_train=train.drop('Survived',1)

test=total[891:];test=test.drop('Survived',1)

#testID = test['PassengerId']

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=0)

del total,train
random_state = 0

kfold = KFold(n_splits=8, shuffle=True, random_state=random_state)



classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))





cv_means = [];cv_stds= []

for classifier in classifiers :

    result=cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold)

    cv_means.append(result.mean());cv_stds.append(result.std())

    

cv_df= pd.DataFrame({"Means":cv_means,"Stds": cv_stds,"Algorithm":["SVC","DecisionTree","RandomForest","KNeighboors","LogisticRegression"]})



g = sns.barplot("Means","Algorithm",data = cv_df,orient = "h",**{'xerr':cv_stds})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
#SVM

param_grid_svm = [{'kernel': ['rbf'],

'C': [0.001, 0.01, 0.1, 1, 10, 100],

'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},

{'kernel': ['linear'],

'C': [0.001, 0.01, 0.1, 1, 10, 100]}]



grid_search = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5,n_jobs=-1)

grid_search.fit(x_train, y_train)

print("Best parameters: {}".format(grid_search.best_params_))

print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))



model_svm = grid_search.best_estimator_
#RamdomForest

param_grid_rf = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}



grid_search = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)

grid_search.fit(x_train, y_train)

print("Best parameters: {}".format(grid_search.best_params_))

print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))



model_rf = grid_search.best_estimator_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(model_svm,"SVM learning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(model_rf,"RF learning curves",x_train,y_train,cv=kfold)
classifier = model_rf

indices = np.argsort(classifier.feature_importances_)[::-1][:40]

g = sns.barplot(y=x_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40])

g.set_xlabel("Relative importance",fontsize=12)

g.set_ylabel("Features",fontsize=12)

g.tick_params(labelsize=9)

g.set_title('RandomForest' + " feature importance")
votingC = VotingClassifier(estimators=[('rfc', model_rf),('svc', model_svm)], voting='soft', n_jobs=4)



votingC = votingC.fit(x_train, y_train)
test_Survived = pd.Series(votingC.predict(test), name="Survived").astype(int)



results = pd.concat([testID, test_Survived], axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)

results.head()