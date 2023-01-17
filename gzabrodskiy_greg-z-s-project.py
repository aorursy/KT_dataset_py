#

# Code by Gregory Zabrodskiy & Poome 

#

#





# loading nessasary packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Common Model Algorithms

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn import tree

from collections import Counter



from sklearn.naive_bayes import GaussianNB

from sklearn import svm



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#from sklearn.naive_bayes import GaussianNB



#from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import LabelEncoder

#from sklearn import feature_selection

#from sklearn import model_selection

from sklearn import metrics





#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,12



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_all = [df_train, df_test]

for df in df_all:

    print(df.isna().sum())
def titleType(row):

    if 'Miss.' in row['Name']:

        return 1

    elif 'Mlle.' in row['Name']:

        return 1

    elif 'Master.' in row['Name']:

        return 2

    elif 'Rev.' in row['Name']:

        return 3

    elif 'Rev.' in row['Name']:

        return 3

    elif 'Col.' in row['Name']:

        return 3

    elif 'Capt.' in row['Name']:

        return 3

    elif 'Major.' in row['Name']:

        return 3

    elif 'Dr.' in row['Name']:

        return 3   

    elif 'Mrs.' in row['Name']:

        return 4

    else:

        return 0 #others

for df in df_all:

    df['TitleType'] = df.apply(titleType, axis=1)







for df in df_all:   

    

    #df['Embarked'].fillna('S', inplace = True)

    df['Fare'] = df[['Fare', 'Pclass']].groupby('Pclass').transform(lambda x: x.fillna(x.median()))    

    #df['Age'] = df[['Age', 'Sex', 'Pclass', 'Parch']].groupby(['Sex', 'Pclass', 'Parch']).transform(lambda x: x.fillna(x.mean()))

    df['Age'] = df[['Age', 'Sex', 'TitleType']].groupby(['Sex', 'TitleType']).transform(lambda x: x.fillna(x.median()))

    df['Age'].fillna(df['Age'].median(), inplace = True)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)



for df in df_all:

    df['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'U' for i in df['Cabin'] ])

    df['Cabin'] = LabelEncoder().fit_transform(df['Cabin'])
def ageGroup(row):

    if np.isnan(row['Age']):

        return -1 # unknown

    elif row['Age'] < 12:

        return 0 # child

    elif row ['Age'] < 20:

        return 1 # teen

    else:

        return 2 # adult 



    

for df in df_all:

    

    df['AgeGroup'] = df.apply(ageGroup, axis=1)

    #df['AgeGroup'] = pd.qcut(df['Age'], 4)

    #df['AgeGroup'] = LabelEncoder().fit_transform(df['AgeGroup'])

    df['FareGroup'] = pd.qcut(df['Fare'], 5)

    df['FareGroup'] = LabelEncoder().fit_transform(df['FareGroup'])

    df.loc[df.Fare == 0, 'FareGroup'] = -1   

    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

    





def familyGroup(row):

    if row['SibSp'] + row['Parch'] == 0:

        return 0 # alone

    elif row['SibSp'] + row['Parch']  < 3:

        return 1 # small

    else:

        return 2 # large

    #return row['SibSp'] + row['Parch'] 



def isAlone(row):

    return 1 if row['SibSp'] + row['Parch'] == 0 else 0

    

for df in df_all:

    df['FamilyGroup'] = df.apply(familyGroup, axis=1)

#    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    df['isAlone'] = df.apply(isAlone, axis=1)

for df in df_all:

    df.drop(['Name', 'Ticket'], axis=1, inplace = True)

#df_train.drop('PassengerId', axis=1, inplace = True)



df_train.describe()
print(df_test.isnull().sum())

df_test.describe()
sns.heatmap(df_train.corr(), cmap =  sns.diverging_palette(255, 0), annot=True, fmt='0.2f', linewidths=0.1,vmax=1.0, linecolor='white', annot_kws={'fontsize': 7 })

for col in df_train:

    if df_train[col].dtype == 'int64':

        if col not in ['Survived', 'PassengerId']:

            print(df_train[[col, 'Survived']].groupby(col).agg(['mean', 'count']))
g = sns.catplot(x="Pclass",y="Survived",data=df_train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.catplot(x="Sex",y="Survived",data=df_train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.catplot(x="Embarked",y="Survived",data=df_train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.catplot(x="AgeGroup",y="Survived",data=df_train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.catplot(x="FamilyGroup",y="Survived",data=df_train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.catplot(x="FareGroup",y="Survived",data=df_train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.catplot(x="TitleType",y="Survived",data=df_train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.catplot(x="isAlone",y="Survived",data=df_train,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.catplot(x="Cabin",y="Survived",data=df_train,kind="bar", height = 6 , palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 0) & (df_train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 1) & (df_train["Age"].notnull())], ax =g, color="Green", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 0) & (df_train["Fare"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 1) & (df_train["Fare"].notnull())], ax =g, color="Green", shade= True)

g.set_xlabel("Fare")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
print(df_train[['Pclass', 'Sex', 'Embarked', 'Survived']].groupby(['Pclass', 'Sex', 'Embarked']).agg(['mean', 'count']))
print(df_train[['Pclass', 'Sex', 'FamilyGroup', 'Survived']].groupby(['Pclass', 'Sex', 'FamilyGroup']).agg(['mean', 'count']))

print(df_train[['Pclass', 'Fare', 'Survived']].groupby(['Pclass', 'Survived']).agg(['mean', 'median', 'min', 'max', 'count']))

dt = tree.DecisionTreeClassifier()

rf = RandomForestClassifier(criterion='gini', n_estimators=1000, random_state=1000)



nb = GaussianNB()

svm = svm.SVC(gamma='scale')



vc = VotingClassifier(estimators=[('nb', nb), ('rf', rf), ('svm', svm)], voting='hard')

x = df_train.drop(columns=['Survived', 'PassengerId'])

y = df_train['Survived']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1000)



x_train1 = x_train[['Sex', 'Fare']]

x_test1 = x_test[['Sex', 'Fare']]

y_pred = dt.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (DT): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))



y_pred = rf.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (RF): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))



x_train1 = x_train[['Sex', 'Fare', 'Age']]

x_test1 = x_test[['Sex',  'Fare', 'Age']]

y_pred = dt.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (DT): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))



y_pred = rf.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (RF): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))
final_features = ['Sex', 'FareGroup', 'Pclass', 'Embarked', 'AgeGroup', 'isAlone']



x_train1 = x_train[final_features]

x_test1 = x_test[final_features]



y_pred = nb.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (NB): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))



y_pred = svm.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (SVM): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))





y_pred = dt.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (DT): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))



y_pred = vc.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (VC): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))





y_pred = rf.fit(x_train1, y_train).predict(x_test1)

print('Accuracy (RF): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))



cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index = [i for i in ['not survived', 'survived']], columns = [i for i in ['not survived', 'survived']])

plt.figure(figsize = (3,3))

a = sns.heatmap(cm, annot=True, fmt='g')

a.set (ylabel='True label', xlabel='Predicted label')

plt.setp(a.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")

plt.setp(a.get_yticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")
x_train = df_train[final_features]

y_train = df_train['Survived']

x_test = df_test[final_features]





y_pred = rf.fit(x_train, y_train).predict(x_test)





df_feature_importance = pd.DataFrame()

df_feature_importance['feature'] = x_train.columns

df_feature_importance['importance'] = rf.feature_importances_



plt.figure(figsize=(5, 5))

sns.barplot(x='importance', y='feature', data=df_feature_importance.sort_values(by='importance', ascending=False))

plt.show()

sdf = pd.DataFrame(columns=['PassengerId', 'Survived'])

sdf['PassengerId'] = df_test['PassengerId']

sdf['Survived'] = y_pred

sdf.to_csv('submissions.csv', header=True, index=False)

print(sdf.head(10))

print(sdf.groupby('Survived').count())