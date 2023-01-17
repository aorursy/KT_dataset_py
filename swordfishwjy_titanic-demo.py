import numpy as np

import pandas as pd

# import matplotlib.pyplot as plt
df_train = pd.read_csv('/kaggle/input/titanic/train.csv', decimal=',')

# df_train.Age = df_train['Age'].astype(float)

df_test = pd.read_csv('/kaggle/input/titanic/test.csv', decimal=',')

# df_test.Age = df_test['Age'].astype(float)



df_all = pd.concat([df_train, df_test], sort=False, copy=False, ignore_index=True)
# show the infomation include column names, number of entries, number of non-null value, data type

print(df_train.info())

# show the first 5 entries

df_train.head(6)
# show some basic statistical values, note that only the features with "int64" data type are discribed.

df_train.describe()
print(df_test.info())

df_test.head()
df_all.info()
tmp = df_train.groupby('Survived').Pclass

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()
tmp = df_train.groupby('Survived').Sex

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()
df_train.Age = df_train['Age'].astype(float)

tmp = df_train.groupby('Survived').Age

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot(kind='bar', figsize=[20,5], sort_columns=True)

#tmp.plot.hist(alpha=0.7, legend=True, bins=25)
tmp = df_train.groupby('Survived').Parch

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()
tmp = df_train.groupby('Survived').SibSp

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()
tmp = df_train.groupby('Survived').Fare

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar(figsize=[20,5])
tmp = df_train.groupby('Survived').Embarked

(tmp.value_counts()[1] / (tmp.value_counts()[1] + tmp.value_counts()[0])).fillna(0).plot.bar()

df_train.head(10)
# create title columns

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

# Since Age is in "object" type, we cannot calculate the mean value. We have cast the object to new type "float"

df_all.Age = df_all['Age'].astype(float)

df_all = fillingGaps(df_all)

df_all.info()
df_all = makeTitle(df_all)

df_all = getTicketNumber(df_all)

df_all['travelingInGroup'] = df_all.duplicated(['Ticket'], keep=False) # Mark all duplicate ticket as True, otherwise false.

df_all.head(5)
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
df_all.head(3)
df_sub = df_all.drop(['Name', 'Cabin', 'PassengerId', 'Survived', 'Ticket'], 1)

df_sub.head(1)
df_sub.info()

df_sub.Fare = df_sub['Fare'].astype(float)

df_sub.info()
y_train = df_train[:891].Survived

X_train = df_sub[:891]

X_test = df_sub[891:]
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



def testRandomForestClassifier(X, y):

    clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0)

    result = cross_val_score(clf , X, y , cv=10) 

    print(result)

    print(result.mean()) # output average 

    clf.fit(X,y)

    return pd.DataFrame(clf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance',ascending=False)
testRandomForestClassifier(X_train, y_train)
X_train = X_train.drop(['Master', 'Parch','EmbarkedC', 'EmbarkedQ', 'EmbarkedS'],1)

X_test = X_test.drop(['Master', 'Parch','EmbarkedC', 'EmbarkedQ', 'EmbarkedS'],1)
X_train.info()
testRandomForestClassifier(X_train, y_train)
clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0)

clf.fit(X_train,y_train)

result = clf.predict(X_test)

print(result[0:10])
