import numpy as np

import pandas as pd
df_train = pd.read_csv('/kaggle/input/titanic/train.csv', decimal=',')

df_train.Age = df_train['Age'].astype(float)

df_test = pd.read_csv('/kaggle/input/titanic/test.csv', decimal=',')

df_test.Age = df_test['Age'].astype(float)



df_all = pd.concat([df_train, df_test], sort=False, copy=False, ignore_index=True)
print(df_train.info())

df_train.head()
print(df_test.info())

df_test.head()
df_all.info()
tmp = df_train.groupby('Survived').Pclass

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()
tmp = df_train.groupby('Survived').Sex

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()
tmp = df_train.groupby('Survived').Age

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar(figsize=[20,5])

#tmp.plot.hist(alpha=0.7, legend=True, bins=25)
tmp = df_train.groupby('Survived').Parch

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()
tmp = df_train.groupby('Survived').SibSp

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()
tmp = df_train.groupby('Survived').Fare

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar(figsize=[20,5])
tmp = df_train.groupby('Survived').Embarked

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()

def makeTitle(df):

    for index, row in df.iterrows():

        pos_comma = row['Name'].find(',')

        pos_point = row['Name'].find('.')

        df.at[index, 'Title'] = (row['Name'][pos_comma+2:pos_point])

        

    titles = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Mme":        "Mrs",

        "Mlle":       "Miss",

        "Ms":         "Mrs",

        "Mr" :        "Mr",

        "Mrs" :       "Mrs",

        "Miss" :      "Miss",

        "Master" :    "Master",

        "Lady" :      "Royalty"

    }

    df.Title = df.Title.map(titles)

    

    return df

        

tmp = makeTitle(df_train)

tmp = tmp.groupby(['Survived', 'Title'])

(tmp.size()[1] / (tmp.size()[1] + tmp.size()[0])).fillna(0).plot.bar()
import re



def getTicketNumber(df):

    df = df.copy()

    for index,row in df.iterrows():

        if (row['Ticket'] == 'LINE'):

            df.at[index, 'Ticket'] = -1

        else:

            ticketNum = re.sub('[^0-9]','', row['Ticket'])

            df.at[index, 'Ticket'] = float(ticketNum)

    df.astype({'Ticket': 'int32'}).dtypes

    return df



tmp = getTicketNumber(df_train)

tmp = tmp.groupby(['Ticket']).agg({'PassengerId':'count', 'Survived':'mean'})

tmp = tmp.rename(columns={"PassengerId": "Count", "Survived": "Survived %"})
print('Traveling in Group')

print('Average group size: ' + str(tmp[tmp['Count'] > 1].mean()['Count']))

print('Chances to survive in a group: ' + str(tmp[tmp['Count'] > 1].mean()['Survived %']))

print()

print('Traveling alone')

print('Average group size: ' + str(tmp[tmp['Count'] == 1].mean()['Count']))

print('Chances to survive in a group: ' + str(tmp[tmp['Count'] == 1].mean()['Survived %']))
def fillingGaps(df):

    for index, row in df.iterrows():

        if (np.isnan(row.Age)):

            df.loc[index, 'Age'] = df[(df.Sex == row.Sex) & (df.Pclass == row.Pclass)].Age.mean()



    

    return df



df_all = fillingGaps(df_all)

df_all.info()
df_all = makeTitle(df_all)

df_all = getTicketNumber(df_all)

df_all['travelingInGroup'] = df_all.duplicated(['Ticket'], keep=False)

df_all.head(1)
def tranformData(df):  

    df = df.copy()

    for index in df.index:   

        

        #PClass

        pClass = df.iloc[index].Pclass

        if pClass == 1:

            df.at[index,'Pclass_1'] = 1

        elif pClass == 2:

            df.at[index,'Pclass_2'] = 1

        elif pClass == 3:

            df.at[index,'Pclass_3'] = 1

      

        #Sex

        sex = df.iloc[index].Sex

        if(sex == 'male'):

            df.at[index, 'Sex'] = 0

        elif(sex == 'female'):

            df.at[index, 'Sex'] = 1

        

        #Embarked

        embarked = df.iloc[index].Embarked

        if embarked == 'Q':

            df.at[index, 'EmbarkedQ'] = 1

        elif embarked == 'C':

            df.at[index, 'EmbarkedC'] = 1

        elif embarked == 'S':

            df.at[index, 'EmbarkedS'] = 1



    

        #Title

        title = df.iloc[index].Title

        df.at[index, title] = 1

        

            

    df = df.drop(columns=['Pclass', 'Title', 'Embarked'])

    df = df.fillna(0);

    

    return df



tmp = tranformData(df_all)

df_all = tmp

df_all.info()
df_all.head(1)
df_sub = df_all.drop(['Name', 'Cabin', 'PassengerId', 'Survived', 'Ticket'], 1)

df_sub.head(1)
y = df_train[:891].Survived
X_train = df_sub[:891]

X_test = df_sub[891:]
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



def testRandomForestClassifier(X, y):

    clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0)

    result = cross_val_score(clf , X, y , cv=10)

    print(result)

    print(result.mean())

    clf.fit(X,y)

    return pd.DataFrame(clf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance',ascending=False)
testRandomForestClassifier(X_train, y)
X_train = X_train.drop(['Pclass_1','Pclass_2', 'Royalty', 'Officer', 'Master', 'Parch','EmbarkedC', 'EmbarkedQ', 'EmbarkedS'],1)

X_test = X_test.drop(['Pclass_1','Pclass_2', 'Royalty', 'Officer', 'Master', 'Parch','EmbarkedC', 'EmbarkedQ', 'EmbarkedS'],1)
testRandomForestClassifier(X_train, y)
test_params = dict(     

    min_samples_split = [2,3,4,5], 

    min_samples_leaf = [2,3,4], 

    max_depth = [10,20,30],

    n_estimators = [10,20,30],

)



clf = RandomForestClassifier(n_jobs=-1)

clf_cv = GridSearchCV(estimator=clf,param_grid=test_params, cv=5) 

clf_cv.fit(X_train, y)



print('Best score: ' + str(clf_cv.best_score_))

print('Optimal params: ' + str(clf_cv.best_estimator_))
def predictAndSave(X, clf, name): 

    result = clf.predict(X)

    df_result = pd.DataFrame(data=result, columns=['Survived'])



    df_result = pd.concat([df_test, df_result], axis=1)

    df_result = df_result[['PassengerId', 'Survived']]

    df_result.head(6)

    df_result.to_csv('submission-' + name + '.csv', index=False)

    return df_result
result = predictAndSave(X_test, clf_cv.best_estimator_, 'randomForest')

result.head(5)