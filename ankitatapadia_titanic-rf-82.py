#import all requiring libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#loading input data sets
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")
#print the first several lines
train_raw.head()
#brief information
train_raw.info()
#percentage of survival
train_raw["Survived"].value_counts(normalize=True)
#countplot of survival
sns.countplot(train_raw['Survived'])
#crosstab the survival in Pclass
pd.crosstab(train_raw["Survived"], train_raw["Pclass"], margins=True)
#count the mean of survival in Pclass
train_raw['Survived'].groupby(train_raw['Pclass']).mean()
#countplot the survival in Pclass
sns.countplot(train_raw['Pclass'], hue=train_raw['Survived'])
#display the name
train_raw['Name'].head()
#distribution of the titles of name
train_raw['Name_Title'] = train_raw['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
train_raw['Name_Title'].value_counts()
#average survival rate of title of name
train_raw['Survived'].groupby(train_raw['Name_Title']).mean()
#observing the survival in age
train_raw['Survived'].groupby(pd.qcut(train_raw['Age'],5)).mean()
#count by age
pd.qcut(train_raw['Age'],5).value_counts()
#crosstab the survival in gender
pd.crosstab(train_raw.Survived, train_raw.Sex, margins=True)
#selecting the major part according to gender
answer_everyone_died = pd.DataFrame({"PassengerId":test_raw.PassengerId, "Survived":0})
answer_everyone_died.to_csv("submission_zeros.csv", index=False)
#selected by gender
test_gender_submission = test_raw.copy()
test_gender_submission["Survived"] = 0
test_gender_submission.loc[(test_gender_submission["Sex"]=="female"), "Survived"] = 1
my_gender_submission = pd.DataFrame({"PassengerId":test_gender_submission["PassengerId"], "Survived":test_gender_submission["Survived"]})
my_gender_submission.to_csv("my_gender_submission.csv", index=False)
#crosstab the survival in gender and children
train_2 = train_raw.copy()
train_2["MightSurvive"] = 0
train_2.loc[(train_2["Sex"]=="female") | (train_2["Age"] < 10), "MightSurvive"] = 1
pd.crosstab(train_2["Survived"], train_2["MightSurvive"], margins=True)
#selected by gender and children
test_gender_children_submission = test_raw.copy()
test_gender_children_submission["MightSurvive"] = 0
test_gender_children_submission.loc[(test_gender_children_submission["Sex"]=="female") | (test_gender_children_submission["Age"]<10), "MightSurvive"] = 1
gender_children_submission = pd.DataFrame({"PassengerId":test_gender_children_submission["PassengerId"], "Survived":test_gender_children_submission["MightSurvive"]})
gender_children_submission.to_csv("gender_children_submission.csv", index=False)
#crosstab the survival in sibsp and parch
train_2.loc[(train_2.Parch >= 4), "MightSurvive"] = 0
train_2.loc[(train_2.SibSp >= 4), "MightSurvive"] = 0
pd.crosstab(train_2["Survived"], train_2["MightSurvive"], margins=True)
#selected by sibsp and parch
test_tweaked_submission = test_raw.copy()
test_tweaked_submission["MightSurvive"] = 0
test_tweaked_submission.loc[(test_tweaked_submission["Sex"]=="female")|(test_tweaked_submission["Age"]<10), "MightSurvive"] = 1
test_tweaked_submission.loc[(test_tweaked_submission["Parch"]>=4), "MightSurvive"] = 0
test_tweaked_submission.loc[(test_tweaked_submission["SibSp"]>=4), "MightSurvive"] = 0
tweaked_submission = pd.DataFrame({"PassengerId":test_tweaked_submission["PassengerId"], "Survived":test_tweaked_submission["MightSurvive"]})
tweaked_submission.to_csv("manually_tweaked_submission.csv", index=False)
#extracting name feature
def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test
#imputing age feature
def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test
#form the family size
def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test
#ticket column
def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test
#carbin column
def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test
def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test
#embarked column
def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test
#dummy variables
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test
#drop column
def drop(train, test, bye = ['PassengerId']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test
#prepare data
train_new = train_raw.copy()
test_new = test_raw.copy()
train_new, test_new = names(train_new, test_new)
train_new, test_new = age_impute(train_new, test_new)
train_new, test_new = cabin_num(train_new, test_new)
train_new, test_new = cabin(train_new, test_new)
train_new, test_new = embarked_impute(train_new, test_new)
train_new, test_new = fam_size(train_new, test_new)
test_new['Fare'].fillna(train_new['Fare'].mean(), inplace = True)
train_new, test_new = ticket_grouped(train_new, test_new)
train_new, test_new = dummies(train_new, test_new, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train_new, test = drop(train_new, test_new)
#random forest model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

gs = gs.fit(train_new.iloc[:, 1:], train_new.iloc[:, 0])

#print(gs.bestscore)
#print(gs.bestparams)
#print(gs.cvresults)

#fit the model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train_new.iloc[:, 1:], train_new.iloc[:, 0])
print("%.4f" % rf.oob_score_)
#variable importance
pd.concat((pd.DataFrame(train_new.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
#prediction
predictions = rf.predict(test_new)
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test_raw.iloc[:, 0], predictions), axis = 1)
predictions.to_csv('random_forest_model.csv', sep=",", index = False)
