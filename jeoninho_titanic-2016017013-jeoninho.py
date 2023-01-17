



import numpy as np 

import pandas as pd 

import os



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.cross_validation import KFold

from sklearn.ensemble import (AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier,VotingClassifier)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, LogisticRegression, PassiveAggressiveClassifier,RidgeClassifierCV

from sklearn.metrics import accuracy_score,auc,classification_report,confusion_matrix,mean_squared_error, precision_score, recall_score,roc_curve

from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate,train_test_split,GridSearchCV,KFold,learning_curve,RandomizedSearchCV,StratifiedKFold

from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier



from sklearn import ensemble, linear_model,neighbors, svm, tree



from scipy.stats import randint

from xgboost import XGBClassifier





import warnings

warnings.filterwarnings('ignore')





print(os.listdir("../input"))







df_train=pd.read_csv('../input/train.csv',sep=',')

df_test=pd.read_csv('../input/test.csv',sep=',')

df_data = df_train.append(df_test) 



PassengerId = df_test['PassengerId']

Submission=pd.DataFrame()

Submission['PassengerId'] = df_test['PassengerId']
print(df_train.shape)

print("----------------------------")

print(df_test.shape)
df_train.columns
df_data.info()
print(pd.isnull(df_data).sum())
df_train.describe()
df_test.describe()
df_train.head(5)
df_train.tail(5)
grid = sns.FacetGrid(df_train, col = "Pclass", row = "Sex", hue = "Survived", palette = 'seismic')

grid = grid.map(plt.scatter, "PassengerId", "Age")

grid.add_legend()

grid
grid = sns.FacetGrid(df_train, col = "Embarked", row = "Sex", hue = "Survived", palette = 'seismic')

grid = grid.map(plt.scatter, "PassengerId", "Age")

grid.add_legend()

grid
grid = sns.FacetGrid(df_train, col = "SibSp", row = "Sex", hue = "Survived", palette = 'seismic')

grid = grid.map(plt.scatter, "PassengerId", "Age")

grid.add_legend()

grid
grid = sns.FacetGrid(df_train, col = "Parch", row = "Sex", hue = "Survived", palette = 'seismic')

grid = grid.map(plt.scatter, "PassengerId", "Age")

grid.add_legend()

grid
g = sns.pairplot(df_train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked']], hue='Survived', palette = 'seismic',size=4,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=50) )

g.set(xticklabels=[])
NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Fare']



test = df_test[NUMERIC_COLUMNS].fillna(-1000)

data_to_train = df_train[NUMERIC_COLUMNS].fillna(-1000)

y=df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(data_to_train, y, test_size=0.3,random_state=21, stratify=y)



clf = SVC()

clf.fit(X_train, y_train)



print("Accuracy: {}".format(clf.score(X_test, y_test)))
Submission['Survived']=clf.predict(test)

print(Submission.head())

print('predictions generated')
print('file created')
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=(12,6))

sns.boxplot(data = df_data, x = "Pclass", y = "Fare",ax=ax1);

plt.figure(1)

sns.boxplot(data = df_data, x = "Embarked", y = "Fare",ax=ax2);

plt.show()
embarked = ['S', 'C', 'Q']

for port in embarked:

    fare_to_impute = df_data.groupby('Embarked')['Fare'].median()[embarked.index(port)]

    df_data.loc[(df_data['Fare'].isnull()) & (df_data['Embarked'] == port), 'Fare'] = fare_to_impute

df_train["Fare"] = df_data['Fare'][:891]

df_test["Fare"] = df_data['Fare'][891:]

print('Missing Fares Estimated')
for x in range(len(df_train["Fare"])):

    if pd.isnull(df_train["Fare"][x]):

        pclass = df_train["Pclass"][x] #Pclass = 3

        df_train["Fare"][x] = round(df_train[df_train["Pclass"] == pclass]["Fare"].mean(), 8)

        

      

for x in range(len(df_test["Fare"])):

    if pd.isnull(df_test["Fare"][x]):

        pclass = df_test["Pclass"][x] #Pclass = 3

        df_test["Fare"][x] = round(df_test[df_test["Pclass"] == pclass]["Fare"].mean(), 8)

        

df_data["FareBand"] = pd.qcut(df_data['Fare'], 8, labels = [1, 2, 3, 4,5,6,7,8]).astype('int')

df_train["FareBand"] = pd.qcut(df_train['Fare'], 8, labels = [1, 2, 3, 4,5,6,7,8]).astype('int')

df_test["FareBand"] = pd.qcut(df_test['Fare'], 8, labels = [1, 2, 3, 4,5,6,7,8]).astype('int')

df_train[["FareBand", "Survived"]].groupby(["FareBand"], as_index=False).mean()

print('FareBand feature created')
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

df_data["Embarked"] = df_data["Embarked"].map(embarked_mapping)

df_train["Embarked"] = df_data["Embarked"][:891]

df_test["Embarked"] = df_data["Embarked"][891:]

print('Embarked feature created')

df_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
fareband = [1,2,3,4]

for fare in fareband:

    embark_to_impute = df_data.groupby('FareBand')['Embarked'].median()[fare]

    df_data.loc[(df_data['Embarked'].isnull()) & (df_data['FareBand'] == fare), 'Embarked'] = embark_to_impute

df_train["Embarked"] = df_data['Embarked'][:891]

df_test["Embarked"] = df_data['Embarked'][891:]

print('Missing Embarkation Estimated')
dummies=pd.get_dummies(df_train[['Sex']], prefix_sep='_') #Gender

df_train = pd.concat([df_train, dummies], axis=1) 

testdummies=pd.get_dummies(df_test[['Sex']], prefix_sep='_') #Gender

df_test = pd.concat([df_test, testdummies], axis=1) 

print('Gender Feature added ')
gender_mapping = {"female": 0, "male": 1}

df_data["Sex"] = df_data['Sex'].map(gender_mapping)

df_data["Sex"]=df_data["Sex"].astype('int')



df_train["Sex"] = df_data["Sex"][:891]

df_test["Sex"] = df_data["Sex"][891:]

print('Gender Category created')
df_data['NameLen'] = df_data['Name'].apply(lambda x: len(x))

print('Name Length calculated')



df_train["NameLen"] = df_data["NameLen"][:891]

df_test["NameLen"] = df_data["NameLen"][891:]



df_train["NameBand"] = pd.cut(df_train["NameLen"], bins=5, labels = [1,2,3,4,5])

df_test["NameBand"] = pd.cut(df_test["NameLen"], bins=5, labels = [1,2,3,4,5])



dummies=pd.get_dummies(df_train[["NameBand"]].astype('category'), prefix_sep='_') #Embarked

df_train = pd.concat([df_train, dummies], axis=1) 

dummies=pd.get_dummies(df_test[["NameBand"]].astype('category'), prefix_sep='_') #Embarked

df_test = pd.concat([df_test, dummies], axis=1)

print("Name Length categories created")



pd.qcut(df_train['NameLen'],5).value_counts()
df_data["Title"] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



df_data["Title"] = df_data["Title"].replace('Mlle', 'Miss')

df_data["Title"] = df_data["Title"].replace('Master', 'Master')

df_data["Title"] = df_data["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')

df_data["Title"] = df_data["Title"].replace(['Jonkheer','Don'],'Mr')

df_data["Title"] = df_data["Title"].replace(['Capt','Major', 'Col','Rev','Dr'], 'Millitary')

df_data["Title"] = df_data["Title"].replace(['Lady', 'Countess','Sir'], 'Honor')



df_train["Title"] = df_data['Title'][:891]

df_test["Title"] = df_data['Title'][891:]



titledummies=pd.get_dummies(df_train[['Title']], prefix_sep='_') #Title

df_train = pd.concat([df_train, titledummies], axis=1) 

ttitledummies=pd.get_dummies(df_test[['Title']], prefix_sep='_') #Title

df_test = pd.concat([df_test, ttitledummies], axis=1) 

print('Title categories added')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Millitary": 5, "Honor": 6}

df_data["TitleCat"] = df_data['Title'].map(title_mapping)

df_data["TitleCat"] = df_data["TitleCat"].astype(int)

df_train["TitleCat"] = df_data["TitleCat"][:891]

df_test["TitleCat"] = df_data["TitleCat"][891:]

print('Title Category created')
titles = ['Master', 'Miss', 'Mr', 'Mrs', 'Millitary','Honor']

for title in titles:

    age_to_impute = df_data.groupby('Title')['Age'].median()[title]

    df_data.loc[(df_data['Age'].isnull()) & (df_data['Title'] == title), 'Age'] = age_to_impute

df_train["Age"] = df_data['Age'][:891]

df_test["Age"] = df_data['Age'][891:]

print('Missing Ages Estimated')
bins = [0,12,24,45,60,np.inf]

labels = ['Child', 'Young Adult', 'Adult','Older Adult','Senior']

df_train["AgeBand"] = pd.cut(df_train["Age"], bins, labels = labels)

df_test["AgeBand"] = pd.cut(df_test["Age"], bins, labels = labels)

print('Age Feature created')



dummies=pd.get_dummies(df_train[["AgeBand"]], prefix_sep='_') #Embarked

df_train = pd.concat([df_train, dummies], axis=1) 

dummies=pd.get_dummies(df_test[["AgeBand"]], prefix_sep='_') #Embarked

df_test = pd.concat([df_test, dummies], axis=1)

print('AgeBand feature created')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Training Age values - Titanic')

axis2.set_title('Test Age values - Titanic')



df_train['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



df_test['Age'].hist(bins=70, ax=axis2)



facet = sns.FacetGrid(df_train, hue="Survived",palette = 'seismic',aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, df_train['Age'].max()))

facet.add_legend()
sns.boxplot(data = df_train, x = "Title", y = "Age");
df_train["Alone"] = np.where(df_train['SibSp'] + df_train['Parch'] + 1 == 1, 1,0) # People travelling alone

df_test["Alone"] = np.where(df_test['SibSp'] + df_test['Parch'] + 1 == 1, 1,0) # People travelling alone

print('Lone traveller feature created')
df_data['Mother'] = (df_data['Title'] == 'Mrs') & (df_data['Parch'] > 0)

df_data['Mother'] = df_data['Mother'].astype(int)



df_train["Mother"] = df_data["Mother"][:891]

df_test["Mother"] = df_data["Mother"][891:]

print('Mother Category created')
df_train["Family Size"] = (df_train['SibSp'] + df_train['Parch'] + 1)

df_test["Family Size"] = df_test['SibSp'] + df_test['Parch'] + 1

print('Family size feature created')
df_data["Last_Name"] = df_data['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5

df_data["Family_Survival"] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in df_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):

        

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 0



print("Number of passengers with family survival information:", 

      df_data.loc[df_data['Family_Survival']!=0.5].shape[0])



for _, grp_df in df_data.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 0

                        

print("Number of passenger with family/group survival information: " 

      +str(df_data[df_data['Family_Survival']!=0.5].shape[0]))



df_train["Family_Survival"] = df_data['Family_Survival'][:891]

df_test["Family_Survival"] = df_data['Family_Survival'][891:]
df_data["HadCabin"] = (df_data["Cabin"].notnull().astype('int'))

df_train["HadCabin"] = df_data["HadCabin"][:891]

df_test["HadCabin"] = df_data["HadCabin"][891:]

print('Cabin feature created')
# Extract Deck

df_data["Deck"] = df_data.Cabin.str.extract('([A-Za-z])', expand=False)

df_data["Deck"] = df_data["Deck"].fillna("N")

# Map Deck

deck_mapping = {"N":0,"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

df_data['Deck'] = df_data['Deck'].map(deck_mapping)

#Split to training and test

df_train["Deck"] = df_data["Deck"][:891]

df_test["Deck"] = df_data["Deck"][891:]

print('Deck feature created')



#Map and Create Deck feature for training

df_data["Deck"] = df_data.Cabin.str.extract('([A-Za-z])', expand=False)

deck_mapping = {"0":0,"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

df_data['Deck'] = df_data['Deck'].map(deck_mapping)

df_data["Deck"] = df_data["Deck"].fillna("0")

df_data["Deck"]=df_data["Deck"].astype('int')



df_train["Deck"] = df_data['Deck'][:891]

df_test["Deck"] = df_data['Deck'][891:]

print('Deck feature created')



# convert categories to Columns

dummies=pd.get_dummies(df_train[['Deck']].astype('category'), prefix_sep='_') #Gender

df_train = pd.concat([df_train, dummies], axis=1) 

dummies=pd.get_dummies(df_test[['Deck']].astype('category'), prefix_sep='_') #Gender

df_test = pd.concat([df_test,dummies], axis=1)

print('Deck Categories created')
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 



Ticket = []

for i in list(df_data.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

df_data["Ticket"] = Ticket

df_data["Ticket"].head()



df_train["Ticket"] = df_data["Ticket"][:891]

df_test["Ticket"] = df_data["Ticket"][891:]

print('Ticket feature created')
# ticket prefix



df_data['TicketRef'] = df_data['Ticket'].apply(lambda x: str(x)[0])

df_data['TicketRef'].value_counts()

#df_data["ticketBand"] = pd.qcut(df_data['ticket_ref'], 5, labels = [1, 2, 3, 4,5]).astype('int')



# split to test and training

df_train["TicketRef"] = df_data["TicketRef"][:891]

df_test["TicketRef"] = df_data["TicketRef"][891:]



# convert AgeGroup categories to Columns

dummies=pd.get_dummies(df_train[["TicketRef"]].astype('category'), prefix_sep='_') #Embarked

df_train = pd.concat([df_train, dummies], axis=1) 

dummies=pd.get_dummies(df_test[["TicketRef"]].astype('category'), prefix_sep='_') #Embarked

df_test = pd.concat([df_test, dummies], axis=1)

print("TicketBand categories created")
# convert AgeGroup categories to Columns

dummies=pd.get_dummies(df_train[["Pclass"]].astype('category'), prefix_sep='_') #Embarked

df_train = pd.concat([df_train, dummies], axis=1) 

dummies=pd.get_dummies(df_test[["Pclass"]].astype('category'), prefix_sep='_') #Embarked

df_test = pd.concat([df_test, dummies], axis=1)

print("pclass categories created")
# create free feature based on fare = 0 

df_data["Free"] = np.where(df_data['Fare'] ==0, 1,0)

df_data["Free"] = df_data['Free'].astype(int)



df_train["Free"] = df_data["Free"][:891]

df_test["Free"] = df_data["Free"][891:]

print('Free Category created')
Pclass = [1,2,3]

for aclass in Pclass:

    fare_to_impute = df_data.groupby('Pclass')['Fare'].median()[aclass]

    df_data.loc[(df_data['Fare'].isnull()) & (df_data['Pclass'] == aclass), 'Fare'] = fare_to_impute

        

df_train["Fare"] = df_data["Fare"][:891]

df_test["Fare"] = df_data["Fare"][891:]        



#map Fare values into groups of numerical values

df_train["FareBand"] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4]).astype('category')

df_test["FareBand"] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4]).astype('category')



# convert FareBand categories to Columns

dummies=pd.get_dummies(df_train[["FareBand"]], prefix_sep='_') #Embarked

df_train = pd.concat([df_train, dummies], axis=1) 

dummies=pd.get_dummies(df_test[["FareBand"]], prefix_sep='_') #Embarked

df_test = pd.concat([df_test, dummies], axis=1)

print("Fareband categories created")
# convert Embarked categories to Columns

dummies=pd.get_dummies(df_train[["Embarked"]].astype('category'), prefix_sep='_') #Embarked

df_train = pd.concat([df_train, dummies], axis=1) 

dummies=pd.get_dummies(df_test[["Embarked"]].astype('category'), prefix_sep='_') #Embarked

df_test = pd.concat([df_test, dummies], axis=1)

print("Embarked feature created")
#check for any other unusable values

print(len(df_test.columns))

print(pd.isnull(df_test).sum())
df_train.describe()
# Groupby title

df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# plot age distribution by title

facet = sns.FacetGrid(data = df_train, hue = "Title", legend_out=True, size = 5)

facet = facet.map(sns.kdeplot, "Age")

facet.add_legend();
grid = sns.FacetGrid(df_train, col = "FareBand", row = "Sex", hue = "Survived", palette = 'seismic')

grid = grid.map(plt.scatter, "PassengerId", "Age")

grid.add_legend()

grid
grid = sns.FacetGrid(df_train, col = "Deck", row = "Sex", hue = "Survived", palette = 'seismic')

grid = grid.map(plt.scatter, "PassengerId", "Age")

grid.add_legend()

grid
grid = sns.FacetGrid(df_train, col = "Family Size", row = "Sex", hue = "Survived", palette = 'seismic')

grid = grid.map(plt.scatter, "PassengerId", "Age")

grid.add_legend()

grid
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Training Age values - Titanic')

axis2.set_title('Test Age values - Titanic')



x1=df_train[df_train["Survived"]==0]

x2=df_train[df_train["Survived"]==1]



# Set up the matplotlib figure

plt.figure(1)

sns.jointplot(x="Family Size", y="Pclass", data=x1, kind="kde", color='b');

plt.figure(2)

sns.jointplot(x="Family Size", y="Pclass", data=x2, kind="kde", color='r');

plt.show()
sns.jointplot(data=x1, x='PassengerId', y='Age', kind='scatter',color='b')

plt.figure(4)

sns.jointplot(data=x2, x='PassengerId', y='Age', kind='scatter',color='r')

# sns.plt.show()
df_train.columns
df_train.head()
# Create list of interesting columns

SIMPLE_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

INTERESTING_COLUMNS=['Survived','Pclass','Age','SibSp','Parch','Title','Alone','Mother','Family Size','Family_Survival','Embarked','FareBand','TicketRef']

CATEGORY_COLUMNS=['Family Size','Family_Survival','Alone','Mother','Sex_female','Sex_male','AgeBand_Child',

       'AgeBand_Young Adult', 'AgeBand_Adult', 'AgeBand_Older Adult',

       'AgeBand_Senior','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','NameBand_1',

       'NameBand_2', 'NameBand_3', 'NameBand_4', 'NameBand_5','Embarked','TicketRef_A', 'TicketRef_C', 'TicketRef_F', 'TicketRef_L',

       'TicketRef_P', 'TicketRef_S', 'TicketRef_W', 'TicketRef_X','Pclass_1', 'Pclass_2', 'Pclass_3','HadCabin','Free','FareBand_1', 'FareBand_2', 'FareBand_3', 'FareBand_4'] 
# create test and training data

test = df_test[CATEGORY_COLUMNS].fillna(-1000)

data_to_train = df_train[CATEGORY_COLUMNS].fillna(-1000)

X_train, X_test, y_train, y_test = train_test_split(data_to_train, df_train['Survived'], test_size=0.3,random_state=21, stratify=df_train['Survived'])



RandomForest = RandomForestClassifier(random_state = 0)

RandomForest.fit(X_train, y_train)

print('Evaluation complete')

# Print the accuracy# Print  

print("Accuracy: {}".format(RandomForest.score(X_test, y_test)))
#map feature correlation

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(df_train[INTERESTING_COLUMNS].corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)
RandomForest_checker = RandomForestClassifier()

RandomForest_checker.fit(X_train, y_train)

importances_df = pd.DataFrame(RandomForest_checker.feature_importances_, columns=['Feature_Importance'],

                              index=X_train.columns)

importances_df.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)

print(importances_df)
Submission['Survived']=RandomForest.predict(test)

print(Submission.head())

print('Submission created')
# write data frame to csv file

# Submission.set_index('PassengerId', inplace=True)

Submission.to_csv('randomforestcat01.csv',sep=',')

print('file created')
REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

SIMPLE_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

INTERESTING_COLUMNS=['Survived','Pclass','Age','SibSp','Parch','Title','Alone','Mother','Family Size','Family_Survival','Embarked','FareBand','TicketRef']

CATEGORY_COLUMNS=['Pclass','SibSp','Parch','Family Size','Family_Survival','Alone','Mother','Sex_female','Sex_male','AgeBand_Child',

       'AgeBand_Young Adult', 'AgeBand_Adult', 'AgeBand_Older Adult',

       'AgeBand_Senior','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','NameBand_1',

       'NameBand_2', 'NameBand_3', 'NameBand_4', 'NameBand_5','Embarked','TicketRef_A', 'TicketRef_C', 'TicketRef_F', 'TicketRef_L',

       'TicketRef_P', 'TicketRef_S', 'TicketRef_W', 'TicketRef_X','HadCabin','Free'] 

CATEGORY_COLUMNS=['Pclass','SibSp','Parch','Family Size','Family_Survival','Alone','Mother','Sex_female','Sex_male','AgeBand_Child',

       'AgeBand_Young Adult', 'AgeBand_Adult', 'AgeBand_Older Adult',

       'AgeBand_Senior','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','NameBand_1',

       'NameBand_2', 'NameBand_3', 'NameBand_4', 'NameBand_5','Embarked','TicketRef_A', 'TicketRef_C', 'TicketRef_F', 'TicketRef_L',

       'TicketRef_P', 'TicketRef_S', 'TicketRef_W', 'TicketRef_X','HadCabin','Free'] 



#print(df_test.columns)

# create test and training data

data_to_train = df_train[CATEGORY_COLUMNS].fillna(-1000)

prediction = df_train["Survived"]

test = df_test[CATEGORY_COLUMNS].fillna(-1000)

X_train, X_val, y_train, y_val = train_test_split(data_to_train, prediction, test_size = 0.3,random_state=21, stratify=y)

print('Data split')
adaboost=AdaBoostClassifier()

adaboost.fit(X_train, y_train)

y_pred = adaboost.predict(X_val)

acc_adaboost = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_adaboost)
bagging=BaggingClassifier()

bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_val)

acc_bagging = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_bagging)
#Decision Tree

decisiontree = DecisionTreeClassifier()

decisiontree.fit(X_train, y_train)

y_pred = decisiontree.predict(X_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
# ExtraTreesClassifier

et = ExtraTreesClassifier()

et.fit(X_train, y_train)

y_pred = et.predict(X_val)

acc_et = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_et)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
# Gradient Boosting Classifier

gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

y_pred = gbk.predict(X_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
# KNN or k-Nearest Neighbors

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
linear_da=LinearDiscriminantAnalysis()

linear_da.fit(X_train, y_train)

y_pred = linear_da.predict(X_val)

acc_linear_da = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_da)
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

y_pred = linear_svc.predict(X_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
MLP = MLPClassifier()

MLP.fit(X_train, y_train)

y_pred = MLP.predict(X_val)

acc_MLP= round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_MLP)
passiveaggressive = PassiveAggressiveClassifier()

passiveaggressive.fit(X_train, y_train)

y_pred = passiveaggressive.predict(X_val)

acc_passiveaggressive = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_passiveaggressive)
# Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
# Random Forest

randomforest = RandomForestClassifier(random_state = 0)

randomforest.fit(X_train, y_train)

y_pred = randomforest.predict(X_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
ridge = RidgeClassifierCV()

ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_val)

acc_ridge = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_ridge)
# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
# instanciate model

clf = SVC()

# fit model

clf.fit(X_train, y_train)

# predict results

y_pred = clf.predict(X_val)

# check accuracy

acc_clf = round(accuracy_score(y_pred, y_val) * 100, 2)

#print accuracy

print(acc_clf)
# xgboost

xgb = XGBClassifier(n_estimators=10)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_val)

acc_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_xgb)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Ridge Classifier',

              'Random Forest', 'Naive Bayes', 'Linear SVC', 'MLP','AdaBoost','Linear discriminant','Passive Aggressive',

              'Decision Tree', 'Gradient Boosting Classifier','Extra Trees','Stochastic Gradient Descent','Perceptron','xgboost'],

    'Score': [acc_clf, acc_knn, acc_logreg,acc_ridge,acc_randomforest, acc_gaussian,acc_linear_svc, acc_MLP,acc_adaboost,acc_linear_da,acc_passiveaggressive,acc_decisiontree,acc_gbk,acc_et,acc_sgd,acc_perceptron,acc_xgb]})

models.sort_values(by='Score', ascending=False)
Submission['Survived']=ridge.predict(test)

print(Submission.head(5))

print('Prediction complete')
# write data frame to csv file

Submission.set_index('PassengerId', inplace=True)

Submission.to_csv('ridgesubmission02.csv',sep=',')

print('File created')
REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

SIMPLE_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

INTERESTING_COLUMNS=['Survived','Pclass','Age','SibSp','Parch','Title','Alone','Mother','Family Size','Family_Survival','Embarked','FareBand','TicketRef']

CATEGORY_COLUMNS=['Pclass','SibSp','Parch','Family Size','Family_Survival','Alone','Mother','Sex_female','Sex_male','AgeBand_Child',

       'AgeBand_Young Adult', 'AgeBand_Adult', 'AgeBand_Older Adult',

       'AgeBand_Senior','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','NameBand_1',

       'NameBand_2', 'NameBand_3', 'NameBand_4', 'NameBand_5','Embarked','TicketRef_A', 'TicketRef_C', 'TicketRef_F', 'TicketRef_L',

       'TicketRef_P', 'TicketRef_S', 'TicketRef_W', 'TicketRef_X','HadCabin','Free'] 



#print(df_test.columns)

# create test and training data

data_to_train = df_train[CATEGORY_COLUMNS].fillna(-1000)

prediction = df_train["Survived"]

test = df_test[CATEGORY_COLUMNS].fillna(-1000)

X_train, X_val, y_train, y_val = train_test_split(data_to_train, prediction, test_size = 0.3,random_state=21, stratify=prediction)

print('Data split')
# Support Vector Classifier parameters 

param_grid = {'C':np.arange(1, 7),

              'degree':np.arange(1, 7),

              'max_iter':np.arange(0, 12),

              'kernel':['rbf','linear'],

              'shrinking':[0,1]}



clf = SVC()

svc_cv=GridSearchCV(clf, param_grid, cv=10)

svc_cv.fit(X_train, y_train)



print("Tuned SVC Parameters: {}".format(svc_cv.best_params_))

print("Best score is {}".format(svc_cv.best_score_))

acc_svc_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc_cv)
# Logistic Regression



from sklearn.linear_model import LogisticRegression

# create parameter grid as a dictionary where the keys are the hyperparameter names and the values are lists of values that we want to try.

param_grid = {"solver": ['newton-cg','lbfgs','liblinear','sag','saga'],'C': [0.01, 0.1, 1, 10, 100]}



# instanciate classifier

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



logreg_cv = GridSearchCV(logreg, param_grid, cv=30)

logreg_cv.fit(X_train, y_train)



y_pred = logreg_cv.predict(X_val)

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))

print("Best score is {}".format(logreg_cv.best_score_))

acc_logreg_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg_cv)
# KNN or k-Nearest Neighbors with GridSearch



# create parameter grid as a dictionary where the keys are the hyperparameter names and the values are lists of values that we want to try.

param_grid = {"n_neighbors": np.arange(1, 50),

             "leaf_size": np.arange(20, 40),

             "algorithm": ["ball_tree","kd_tree","brute"]

             }

# instanciate classifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_cv = GridSearchCV(knn, param_grid, cv=10)

knn_cv.fit(X_train, y_train)

y_pred = knn_cv.predict(X_val)

print("Tuned knn Parameters: {}".format(knn_cv.best_params_))

print("Best score is {}".format(knn_cv.best_score_))

acc_knn_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn_cv)
# DecisionTree with RandomizedSearch



# Setup the parameters and distributions to sample from: param_dist

param_dist = {"random_state" :  np.arange(0, 10),

              "max_depth": np.arange(1, 10),

              "max_features": np.arange(1, 10),

              "min_samples_leaf": np.arange(1, 10),

              "criterion": ["gini","entropy"]}



# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv

tree_cv = RandomizedSearchCV(tree, param_dist, cv=30)

# Fit it to the data

tree_cv.fit(X_train,y_train)

y_pred = tree_cv.predict(X_val)

# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))

print("Best score is {}".format(tree_cv.best_score_))

acc_tree_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_tree_cv)
# Random Forest



# Setup the parameters and distributions to sample from: param_dist

param_dist = {"random_state" :  np.arange(0, 10),

              "n_estimators" :  np.arange(1, 20),

              "max_depth": np.arange(1, 10),

              "max_features": np.arange(1, 10),

              "min_samples_leaf": np.arange(1, 10),

              "criterion": ["gini","entropy"]}



# Instantiate a Decision Tree classifier: tree

randomforest = RandomForestClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

randomforest_cv = RandomizedSearchCV(randomforest, param_dist, cv=30)



# Fit it to the data

randomforest_cv.fit(X_train,y_train)

y_pred = randomforest_cv.predict(X_val)

# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(randomforest_cv.best_params_))

print("Best score is {}".format(randomforest_cv.best_score_))

acc_randomforest_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest_cv)
# Gradient Boosting Classifier



# Setup the parameters and distributions to sample from: param_dist

param_dist = {'max_depth':np.arange(1, 7),

              'min_samples_leaf': np.arange(1, 6),

              "max_features": np.arange(1, 10),

             }



# Instantiate Classifier

gbk = GradientBoostingClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

gbk_cv = RandomizedSearchCV(gbk, param_dist, cv=30)



gbk_cv.fit(X_train, y_train)

y_pred = gbk_cv.predict(X_val)



print("Tuned Gradient Boost Parameters: {}".format(gbk_cv.best_params_))

print("Best score is {}".format(gbk_cv.best_score_))

acc_gbk_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk_cv)
# xgboost



# Setup the parameters and distributions to sample from: param_dist

param_dist = {'learning_rate': [.01, .03, .05, .1, .25], #default: .3

            'max_depth': np.arange(1, 10), #default 2

            'n_estimators': [10, 50, 100, 300], 

            'booster':['gbtree','gblinear','dart']

            #'seed': 5  

             }

# Instantiate Classifier

xgb = XGBClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

xgb_cv = RandomizedSearchCV(xgb, param_dist, cv=20)



# Fit model

xgb_cv.fit(X_train, y_train)



# Make prediction

y_pred = xgb_cv.predict(X_val)



# Print results

print("xgBoost Parameters: {}".format(xgb_cv.best_params_))

print("Best score is {}".format(xgb_cv.best_score_))

acc_xgb_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_xgb_cv)
optmodels = pd.DataFrame({

    'optModel': ['SVC','KNN','Decision Tree','Gradient Boost','Logistic Regression','xgboost'],

    'optScore': [svc_cv.best_score_,knn_cv.best_score_,tree_cv.best_score_,gbk_cv.best_score_,logreg_cv.best_score_,xgb_cv.best_score_]})

optmodels.sort_values(by='optScore', ascending=False)
optmodels = pd.DataFrame({

    'optModel': ['Linear Regression','KNearestNieghbours','Decision Tree','Gradient Boost','Logistic Regression','xgboost'],

    'optScore': [acc_svc_cv,acc_knn_cv,acc_tree_cv,acc_gbk_cv,acc_logreg_cv,acc_xgb_cv]})

optmodels.sort_values(by='optScore', ascending=False)
# define function to plot test and training curves

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



# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)

# Plot chart for each model

g = plot_learning_curve(svc_cv.best_estimator_,"linear regression learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(logreg_cv.best_estimator_,"logistic regression learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(knn_cv.best_estimator_,"knn learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(tree_cv.best_estimator_,"decision tree learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(randomforest_cv.best_estimator_,"random forest learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gbk_cv.best_estimator_,"gradient boosting learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(xgb_cv.best_estimator_,"xg boost learning curves",X_train,y_train,cv=kfold)
# Select columns

X_train = df_train[CATEGORY_COLUMNS].fillna(-1000)

y_train = df_train["Survived"]

X_test = df_test[CATEGORY_COLUMNS].fillna(-1000)



from sklearn.tree import DecisionTreeClassifier

test = df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)

# select classifier

#tree = DecisionTreeClassifier(random_state=0,max_depth=5,max_features=7,min_samples_leaf=2,criterion="entropy") #85,87

#tree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,max_features=7, max_leaf_nodes=None, min_impurity_decrease=0.0,min_impurity_split=None, min_samples_leaf=9,min_samples_split=2, min_weight_fraction_leaf=0.0,presort=False, random_state=8, splitter='best')

#tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,max_features=7, max_leaf_nodes=None, min_impurity_decrease=0.0,min_impurity_split=None, min_samples_leaf=9,min_samples_split=2, min_weight_fraction_leaf=0.0,presort=False, random_state=9, splitter='best')

#knn = KNeighborsClassifier(algorithm='kd_tree',leaf_size=20,n_neighbors=5)

#logreg = LogisticRegression(solver='newton-cg')

#xgboost=XGBClassifier(n_estimators= 300, max_depth= 10, learning_rate= 0.01)

#gbk=GradientBoostingClassifier(min_samples_leaf=1,max_features=4,max_depth=5)

#logreg=LogisticRegression(solver='newton-cg',C= 10)

#gboost=GradientBoostingClassifier(random_state= 7,n_estimators=17,min_samples_leaf= 4, max_features=9,max_depth=5, criterion='gini')

randomf=RandomForestClassifier(random_state= 7,n_estimators=17,min_samples_leaf= 4, max_features=9,max_depth=5, criterion='gini')



# train model

randomf.fit(data_to_train, prediction)

# make predictions

Submission['Survived']=randomf.predict(X_test)

#Submission.set_index('PassengerId', inplace=True)

Submission.to_csv('randomforestcats01.csv',sep=',')

print(Submission.head(5))

print('File created')
# knn Hyper Tunning with confusion Matrix

REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

SIMPLE_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

INTERESTING_COLUMNS=['Survived','Pclass','Age','SibSp','Parch','Title','Alone','Mother','Family Size','Family_Survival','Embarked','FareBand','TicketRef']

CATEGORY_COLUMNS=['Pclass','SibSp','Parch','Family Size','Family_Survival','Alone','Mother','Sex_female','Sex_male','AgeBand_Child',

       'AgeBand_Young Adult', 'AgeBand_Adult', 'AgeBand_Older Adult',

       'AgeBand_Senior','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','NameBand_1',

       'NameBand_2', 'NameBand_3', 'NameBand_4', 'NameBand_5','Embarked','TicketRef_A', 'TicketRef_C', 'TicketRef_F', 'TicketRef_L',

       'TicketRef_P', 'TicketRef_S', 'TicketRef_W', 'TicketRef_X','HadCabin','Free']  



# create test and training data

data_to_train = df_train[CATEGORY_COLUMNS].fillna(-1000)

X_test2= df_test[CATEGORY_COLUMNS].fillna(-1000)

prediction  = df_train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(data_to_train, prediction, test_size = 0.3,random_state=21, stratify=prediction)

print('Data Split')



hyperparams = {'algorithm': ['auto'], 'weights': ['uniform', 'distance'] ,'leaf_size': list(range(1,50,5)), 

               'n_neighbors':[6,7,8,9,10,11,12,14,16,18,20,22]}

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, cv=10, scoring = "roc_auc")

gd.fit(X_train, y_train)



gd.best_estimator_.fit(X_train,y_train)

y_pred=gd.best_estimator_.predict(X_test)

Submission['Survived']=gd.best_estimator_.predict(X_test2)



# Print the results

print('Best Score')

print(gd.best_score_)

print('Best Estimator')

print(gd.best_estimator_)

acc_gd_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print('Accuracy')

print(acc_gd_cv)



# Generate the confusion matrix and classification report

print('Confusion Matrrix')

print(confusion_matrix(y_test, y_pred))

print('Classification_report')

print(classification_report(y_test, y_pred))

#Submission.set_index('PassengerId', inplace=True)

print('Sample Prediction')

print(Submission.head(10))

#Submission.to_csv('knngridsearch03.csv',sep=',')

print('KNN prediction created')
REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

SIMPLE_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

INTERESTING_COLUMNS=['Survived','Pclass','Age','SibSp','Parch','Title','Alone','Mother','Family Size','Family_Survival','Embarked','FareBand','TicketRef']

CATEGORY_COLUMNS=['Pclass','SibSp','Parch','Family Size','Family_Survival','Alone','Mother','Sex_female','Sex_male','AgeBand_Child',

       'AgeBand_Young Adult', 'AgeBand_Adult', 'AgeBand_Older Adult',

       'AgeBand_Senior','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','NameBand_1',

       'NameBand_2', 'NameBand_3', 'NameBand_4', 'NameBand_5','Embarked','TicketRef_A', 'TicketRef_C', 'TicketRef_F', 'TicketRef_L',

       'TicketRef_P', 'TicketRef_S', 'TicketRef_W', 'TicketRef_X','HadCabin','Free']  



# create test and training data

data_to_train = df_train[CATEGORY_COLUMNS].fillna(-1000)

X_test2= df_test[CATEGORY_COLUMNS].fillna(-1000)

prediction  = df_train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(data_to_train, prediction, test_size = 0.3,random_state=21, stratify=prediction)

print('Data Split')



hyperparams = {"random_state" :  np.arange(0, 10),

              "max_depth": np.arange(1, 10),

              "max_features": np.arange(1, 10),

              "min_samples_leaf": np.arange(1, 10),

              "criterion": ["gini","entropy"]}



gd=GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = hyperparams, verbose=True, cv=10, scoring = "roc_auc")

gd.fit(X_train, y_train)



gd.best_estimator_.fit(X_train,y_train)

y_pred=gd.best_estimator_.predict(X_test)

Submission['Survived']=gd.best_estimator_.predict(X_test2)



print('Best Score')

print(gd.best_score_)

print('Best Estimator')

print(gd.best_estimator_)

acc_gd_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print('Accuracy')

print(acc_gd_cv)



print('Confusion Matrrix')

print(confusion_matrix(y_test, y_pred))

print('Classification_report')

print(classification_report(y_test, y_pred))

print(Submission.head(10))

Submission.to_csv('Treegridsearch03.csv',sep=',')

print('Decision Tree prediction created')


REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] 

SIMPLE_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked']

INTERESTING_COLUMNS=['Survived','Pclass','Age','SibSp','Parch','Title','Alone','Mother','Family Size','Family_Survival','Embarked','FareBand','TicketRef']

CATEGORY_COLUMNS=['Pclass','SibSp','Parch','Family Size','Family_Survival','Alone','Mother','Sex_female','Sex_male','AgeBand_Child',

       'AgeBand_Young Adult', 'AgeBand_Adult', 'AgeBand_Older Adult',

       'AgeBand_Senior','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','NameBand_1',

       'NameBand_2', 'NameBand_3', 'NameBand_4', 'NameBand_5','Embarked','TicketRef_A', 'TicketRef_C', 'TicketRef_F', 'TicketRef_L',

       'TicketRef_P', 'TicketRef_S', 'TicketRef_W', 'TicketRef_X','HadCabin','Free']  



data_to_train = df_train[CATEGORY_COLUMNS].fillna(-1000)

X_test2= df_test[CATEGORY_COLUMNS].fillna(-1000)

prediction  = df_train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(data_to_train, prediction, test_size = 0.3,random_state=21, stratify=prediction)

print('Data Split')



hyperparams = {'solver':["newton-cg", "lbfgs", "liblinear", "sag", "saga"],

              'C': [0.01, 0.1, 1, 10, 100]}



gd=GridSearchCV(estimator = LogisticRegression(), param_grid = hyperparams, verbose=True, cv=10, scoring = "roc_auc")

gd.fit(X_train, y_train)



gd.best_estimator_.fit(X_train,y_train)

y_pred=gd.best_estimator_.predict(X_test)

Submission['Survived']=gd.best_estimator_.predict(X_test2)



print('Best Score')

print(gd.best_score_)

print('Best Estimator')

print(gd.best_estimator_)

acc_gd_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print('Accuracy')

print(acc_gd_cv)



print('Confusion Matrrix')

print(confusion_matrix(y_test, y_pred))

print('Classification_report')

print(classification_report(y_test, y_pred))

print(Submission.head(10))

Submission.to_csv('Logregwithconfusion01.csv',sep=',')

print('Logistic Regression prediction created')
df_train.columns


X_train = df_train[CATEGORY_COLUMNS].fillna(-1000)

y_train = df_train["Survived"]

X_test = df_test[CATEGORY_COLUMNS].fillna(-1000)



randomf=RandomForestClassifier(criterion='gini', n_estimators=700, min_samples_split=10,min_samples_leaf=1,max_features='auto',oob_score=True,random_state=1,n_jobs=-1)

randomf.fit(X_train, y_train)

Submission['Survived']=randomf.predict(X_test)



acc_gd_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print('Accuracy')

print(acc_gd_cv)

print(Submission.head(10))

Submission.to_csv('finalrandomforest01.csv',sep=',')

print('Random Forest prediction created')
MLA = [

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    

    linear_model.LogisticRegressionCV(),

    

    neighbors.KNeighborsClassifier(),

    

    svm.SVC(probability=True),

    

    ]



index = 1

for alg in MLA:

    predicted = alg.fit(X_train, y_train).predict(X_test)

    fp, tp, th = roc_curve(y_test, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()
REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] 

SIMPLE_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked']

INTERESTING_COLUMNS=['Survived','Pclass','Age','SibSp','Parch','Title','Alone','Mother','Family Size','Family_Survival','Embarked','FareBand','TicketRef']

CATEGORY_COLUMNS=['Pclass','SibSp','Parch','Family Size','Family_Survival','Alone','Mother','Sex_female','Sex_male','AgeBand_Child',

       'AgeBand_Young Adult', 'AgeBand_Adult', 'AgeBand_Older Adult',

       'AgeBand_Senior','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','NameBand_1',

       'NameBand_2', 'NameBand_3', 'NameBand_4', 'NameBand_5','Embarked','TicketRef_A', 'TicketRef_C', 'TicketRef_F', 'TicketRef_L',

       'TicketRef_P', 'TicketRef_S', 'TicketRef_W', 'TicketRef_X','HadCabin','Free']  



data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)

data_to_test = df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)

prediction = df_train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(data_to_train, prediction, test_size = 0.3,random_state=21, stratify=prediction)

print('Data Split')
logreg = LogisticRegression(C=10, solver='newton-cg')

logreg.fit(X_train, y_train)

y_pred_train_logreg = cross_val_predict(logreg,X_val, y_val)

y_pred_test_logreg = logreg.predict(X_test)

print('logreg first layer predicted')



tree = DecisionTreeClassifier(random_state=8,min_samples_leaf=6, max_features= 7, max_depth= 4, criterion='gini', splitter='best')

tree.fit(X_train, y_train)

y_pred_train_tree = cross_val_predict(tree,X_val,y_val)

y_pred_test_tree = tree.predict(X_test)

print('decision tree first layer predicted')



randomforest = RandomForestClassifier(random_state=8, n_estimators=15, min_samples_leaf= 4, max_features= 6, max_depth=4,criterion='gini')

randomforest.fit(X_train, y_train)

y_pred_train_randomforest = cross_val_predict(randomforest, X_val, y_val)

y_pred_test_randomforest = randomforest.predict(X_test)

print('random forest first layer predicted')



gbk = GradientBoostingClassifier(min_samples_leaf=3, max_features= 3, max_depth= 3)

gbk.fit(X_train, y_train)

y_pred_train_gbk = cross_val_predict(gbk, X_val, y_val)

y_pred_test_gbk = gbk.predict(X_test)

print('gbk first layer predicted')



knn = KNeighborsClassifier(algorithm='auto', leaf_size=36, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=12, p=2,weights='uniform')

knn.fit(X_train, y_train)

y_pred_train_knn = cross_val_predict(knn, X_val, y_val)

y_pred_test_knn = gbk.predict(X_test)

print('knn first layer predicted')



clf = SVC(C=3, degree=1, kernel='linear', max_iter=1, shrinking=0)

clf.fit(X_train, y_train)

y_pred_train_clf = cross_val_predict(clf, X_val, y_val)

y_pred_test_clf = clf.predict(X_test)

print('clf first layer predicted')
from sklearn.ensemble import VotingClassifier



votingC = VotingClassifier(estimators=[('logreg', logreg_cv.best_estimator_), ('gbk', gbk_cv.best_estimator_),

('tree', tree_cv.best_estimator_), ('randomforest',randomforest_cv.best_estimator_),('knn',knn_cv.best_estimator_) ], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train, y_train)



Submission['Survived'] = votingC.predict(X_test)

Submission.to_csv('Votingclassifier02.csv',sep=',')

print('Voting Classifier Ensemble File created')

print(Submission.head())
second_layer_train = pd.DataFrame( {'Logistic Regression': y_pred_train_logreg.ravel(),

                                    'Gradient Boosting': y_pred_train_gbk.ravel(),

                                    'Decision Tree': y_pred_train_tree.ravel(),

                                    'Random Forest': y_pred_train_randomforest.ravel()

                                    } )



X_train_second = np.concatenate(( y_pred_train_logreg.reshape(-1, 1), y_pred_train_gbk.reshape(-1, 1), 

                                  y_pred_train_tree.reshape(-1, 1), y_pred_train_randomforest.reshape(-1, 1)),

                                  axis=1)

X_test_second = np.concatenate(( y_pred_test_logreg.reshape(-1, 1), y_pred_test_gbk.reshape(-1, 1), 

                                 y_pred_test_tree.reshape(-1, 1), y_pred_test_randomforest.reshape(-1, 1)),

                                 axis=1)

tree = DecisionTreeClassifier(random_state=8,min_samples_leaf=6, max_depth= 4, criterion='gini').fit(X_train_second,y_val)



Submission['Survived'] = tree.predict(X_test_second)

print(Submission.head())

print('Tuned Ensemble model prediction complete')
Submission.to_csv('tunedensemblesubmission04.csv',sep=',')

print('tuned Ensemble File created')