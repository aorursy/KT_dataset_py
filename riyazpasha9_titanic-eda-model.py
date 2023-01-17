# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
print(train.shape)

print(test.shape)
train.isna().sum()
test.isna().sum()
train.columns
train['Survived'].value_counts(normalize=True)
sns.countplot(train['Survived'])
train['Fare'].describe()
pd.crosstab(train['Sex'],train['Survived']).plot.bar(stacked=True)

plt.ylabel('Frequency')

plt.show()

sns.boxplot(y=train['Age'])
plt.figure(figsize=(10,5))

sns.boxplot(train['Pclass'],train['Age'])

plt.show()
train[train['Age']<1]
fix,ax = plt.subplots(1,3,figsize=(12,5))

sns.boxplot(y=train['Fare'],ax=ax[0])

sns.distplot(train['Fare'],ax=ax[1])

sns.distplot(train['Age'].dropna(),ax=ax[2])

plt.tight_layout()

plt.show()
train['Age'].skew()
pd.crosstab(train['Pclass'],train['Survived']).plot.bar(stacked=True)

plt.xlabel('Frequency')

plt.show()
pd.crosstab(train['Parch'],train['Survived']).plot.bar(stacked=True)

plt.show()
pd.crosstab(train['Embarked'],train['Survived'],).plot.bar(stacked=True)

plt.show()
sns.heatmap(pd.crosstab(train['SibSp'],train['Survived']),annot=True,cmap='Blues')

plt.show()
from wordcloud import WordCloud

for col in ['Name','Cabin']:

    

    text = " ".join(review for review in train[col].dropna())

    word = WordCloud(width=1000,height=800,margin=0,max_font_size=150,background_color='white').generate(text)



    plt.figure(figsize=[8,8])

    plt.imshow(word,interpolation='bilinear')

    plt.axis('off')

    plt.show()
#IQR method

for col in ['Age','Fare']:

    

    q1= train[col].quantile(0.25)

    q3 = train[col].quantile(0.75)

    iqr=q3-q1

    print(col,'IQR',iqr)



    upper_limit = q3+1.5*iqr

    lower_limit = q1-1.5*iqr

    print(col,'Upper limit for age',upper_limit)

    print(col,'Lower limit for age',lower_limit)
sns.heatmap(train.corr(),cmap='Blues',annot=True)

plt.show()
train_1 = train.copy()
age_median=train_1['Age'].median()

train_1['Age'] = train_1['Age'].fillna(age_median)
for col in train.columns:

    

    print(col,'Percentage of missing values',train[col].isna().sum()/train.shape[0]*100)
train_1.drop(columns=['Cabin'],inplace=True)
from sklearn_pandas import CategoricalImputer

imputer = CategoricalImputer()

train_1['Embarked']=imputer.fit_transform(train['Embarked'])
train_1.isna().sum()
#we will drop passengerid,Ticket,Name

train_1.drop(columns=['Name','Ticket','PassengerId'],inplace=True)
train_1 = pd.get_dummies(data=train_1,columns=['Sex','Embarked'],drop_first=True)

train_1.head()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier

from sklearn.naive_bayes import BernoulliNB,GaussianNB

from sklearn.svm import SVC
X = train_1.drop(columns=['Survived'],axis=1)

y= train_1['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
model = DecisionTreeClassifier(max_depth=6,class_weight='balanced',random_state=0)

model.fit(X_train,y_train)

acc_decision_tree=model.score(X_train,y_train)*100

print(model.score(X_train,y_train))

print(model.score(X_test,y_test))
model.feature_importances_
sns.barplot(x=model.feature_importances_,y=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',

       'Embarked_S'])

plt.title('Feature Importance Plot')

plt.show()
model1  = LogisticRegression(class_weight='balanced',C=4.5,random_state=0)

model1.fit(X_train,y_train)

acc_logistic=model1.score(X_train,y_train)*100

print(model1.score(X_train,y_train))

print(model1.score(X_test,y_test))
model2  = RandomForestClassifier(n_estimators=21,max_depth=6,criterion='gini',random_state=0,class_weight='balanced',

                                min_samples_split=2)

model2.fit(X_train,y_train)

acc_random_forest=model2.score(X_train,y_train)*100

print(model2.score(X_train,y_train))

print(model2.score(X_test,y_test))
rf  = RandomForestClassifier(n_estimators=1000,min_samples_split=30,min_samples_leaf=5,random_state=42,warm_start=True)

rf.fit(X_train,y_train)

acc_random_forest=rf.score(X_train,y_train)*100

print(rf.score(X_train,y_train))

print(rf.score(X_test,y_test))
model2.feature_importances_
sns.barplot(x=model2.feature_importances_,y=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',

       'Embarked_S'])

plt.title('Feature Importance Plot')

plt.show()
#Learning rate =0.01 got from the grid serach cv

model3 = AdaBoostClassifier(base_estimator=model2,random_state=0,learning_rate=0.001)

model3.fit(X_train,y_train)

acc_ada_boost=model3.score(X_train,y_train)*100

print(model3.score(X_train,y_train))

print(model3.score(X_test,y_test))
#from the Grid Searchcv I have got the parameters as 36,1,1

model4 =GradientBoostingClassifier(random_state=1,n_estimators=36,max_depth=1,learning_rate=1)

model4.fit(X_train,y_train)

acc_gradient_boost=model4.score(X_train,y_train)*100

print(model4.score(X_train,y_train))

print(model4.score(X_test,y_test))
# By gridsearch we have got the values for the hyperparameters

model5 = SVC(kernel='rbf',C=10,gamma=0.1,random_state=1)

model5.fit(X_train,y_train)

acc_svc=model5.score(X_train,y_train)*100

print(model5.score(X_train,y_train))

print(model5.score(X_test,y_test))
model6 = VotingClassifier(estimators=[('DT',model),('LR',model1),('RF',model2),('AD',model3),('GB',model4),('SVC',model5)],

                          voting='hard')

model6.fit(X_train,y_train)

acc_voting_classifier=model6.score(X_train,y_train)*100

print(model6.score(X_train,y_train))

print(model6.score(X_test,y_test))
from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier(base_estimator=model2,n_estimators=20,random_state=1)

bc.fit(X_train,y_train)

acc_Bagging_classifier=bc.score(X_train,y_train)*100

print(bc.score(X_train,y_train))

print(bc.score(X_test,y_test))
from sklearn.model_selection import GridSearchCV



parameters = [{'learning_rate':[0.01,0.1,0.001,1,5,10,20]}]

grid_search = GridSearchCV(estimator = model3,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)

                           #n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print('Best Accuracy',best_accuracy)

print('Best Parameters',best_parameters)               
from sklearn.model_selection import cross_val_score

for models  in [model,model1,model2,model3,model4,model5,model6,bc]:

    

    accuracies = cross_val_score(estimator = models, X = X_train, y = y_train, cv = 10)

    print(models,accuracies.mean())

    print(accuracies.std())
models = pd.DataFrame({

    'Model': ['Decision Tree','Logistic Regression','Random Forest','Adaboost Classifier','Gradient boost',

             'Support Vector Classifier','Voting Classifier','Bagging Classifier'],

    'Score': [acc_decision_tree,acc_logistic,acc_random_forest,acc_ada_boost,acc_gradient_boost,

             acc_svc,acc_voting_classifier,acc_Bagging_classifier]})

models.sort_values(by='Score', ascending=False)
test_1 = test.copy()

test_1['Age'] = test_1['Age'].fillna(test_1['Age'].median())

test_1['Fare'] = test_1['Fare'].fillna(test_1['Fare'].mean())

test_1.drop(columns=['Name','Ticket','PassengerId','Cabin'],inplace=True)
test_1.head()
test_1 = pd.get_dummies(data=test_1,columns=['Sex','Embarked'],drop_first=True)

test_1.head()
y_pred = rf.predict(test_1)

y_pred
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)