#import modules needed for analysis

import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt



#List files in the data folder

import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Load the training data and print the first 5 rows.

train_data = pd.read_csv('../input/titanic/train.csv')

train_data.head()
#Load the testing data and print the first 5 rows.

test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()

#Preliminary stats on the training data.

train_data.describe()
#To explore the data, append the test and train data together. This will allow us to impute features and do data anlysis on the whole data set.

titanic =  train_data.append(test_data, ignore_index = True, sort = True)

titanic.describe()

titanic.info()
#Percentage of survival by Gender

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



#Format as %

print("% of women who survived: " + "{:.2%}".format(rate_women))

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print("% of men who survived: " + "{:.2%}".format(rate_men))
#Categorical factors for plotting

factors_plot = ['Age','Survived','Pclass','SibSp','Parch','sex_code']



#Establish Sex as a Categorical field for plotting

sex_cat = pd.Categorical(train_data['Sex'])

train_data['sex_code'] = sex_cat.codes+1



#Set Correlation

corr = train_data.corr()



#Plot the correlation with a heatmap.

plt.figure(figsize=(10, 10))

sns.heatmap(corr, annot = True, square = True, linewidths=0.01)

plt.title('Feature Correlation');



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show() # ta-da!



#Append the Sex_Code to the full data set

sex_cat = pd.Categorical(titanic['Sex'])

titanic['sex_code'] = sex_cat.codes+1
#Plot the Passenger Class and Survival by Sex.

sns.set(font_scale=1)

g = sns.catplot(x="Sex", y="Survived", col="Pclass",

                    data=train_data, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Men", "Women"])

    .set(ylim=(0, 1))

    .despine(left=True))



plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many Men and Women Survived by Passenger Class');

#Survival by Ticket Class and Age

g = sns.FacetGrid(train_data, col="Pclass", row="Survived", margin_titles=True)

g.map(plt.hist, "Age", color="green");

#Survival by Ticket Class and Sex

g = sns.FacetGrid(train_data, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age", color="green");

#What are the unique deck types based on the first letter of the cabin.

print(train_data['Cabin'].str[0].unique())

#Set the column Deck as the first letter of the cabin.

train_data["Deck"] = train_data['Cabin'].str[0]

#Establish the Deck field as a category.

train_data["Deck"] = train_data["Deck"].astype('category')
#Plot Survival by Deck

sns.catplot("Survived", col="Deck", col_wrap=4,

                    data=train_data[train_data.Deck.notnull()],

                    kind="count", height=2.5, aspect=.8);
#Plot Sex by Deck

sns.catplot("Sex", col="Deck", col_wrap=4,

                    data=train_data[train_data.Deck.notnull()],

                    kind="count", height=2.5, aspect=.8);
#Overall Survival by Deck

sns.catplot("Deck", col="Survived",

                    data=train_data[train_data.Deck.notnull()],

                    kind="count", height=2.5, aspect=.8);
#Find Null Values and store null columns into a variable

null_columns=titanic.columns[titanic.isnull().any()]

titanic.isnull().sum()

#Survived is not in the test data set, therefore nulls exists
titanic[titanic['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=titanic);
#np.median(titanic[titanic.Pcclass]==1)

#Median Fare for First Class, Embarking from Cherbourg

titanic[(titanic.Pclass == 1) & (titanic.Embarked == 'C')]["Fare"].median()

#Grouped median and mean

grouped_median = titanic.groupby(['Embarked','Pclass'])[['Fare']].apply(np.median)

print(grouped_median)

grouped_mean = titanic.groupby(['Embarked','Pclass'])[['Fare']].apply(np.mean)

print(grouped_mean)
#Fill in the missing fare classes with 'C'



titanic["Embarked"] = titanic["Embarked"].fillna('C')

print(titanic["Embarked"].describe())



print("\n There are ",titanic["Embarked"].isna().sum(),"null records")

#Null Fare Records

titanic["Fare"].isna().sum()
titanic[titanic['Fare'].isnull()]
#Fill in the missing Fares

#fills fares, based on ticket class, and embarkation location



def fill_missing_fare(df, pclass, embarkation):

    median_fare=df[(df['Pclass'] == pclass) & (df['Embarked'] == embarkation)]['Fare'].median()



    df["Fare"] = df["Fare"].fillna(median_fare)

    return df



titanic=fill_missing_fare(titanic, 3, 'S')



#Verify the known fare has been filled

print(titanic.loc[1043])
titanic['Age'].fillna(titanic['Age'].median(), inplace = True)



plt.hist(titanic['Age'])
#Bucket age into discrete bins

titanic['AgeBin'] = pd.cut(titanic['Age'].astype(int), 5)



ax = titanic['AgeBin'].value_counts().plot(kind='barh')
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]+1

print(titanic["FamilySize"].value_counts())
#Make discrete family sizes

titanic.loc[titanic["FamilySize"] == 1, "FsizeD"] = 'solo'

titanic.loc[(titanic["FamilySize"]>1) & (titanic["FamilySize"] < 5 ), "FsizeD"] = 'small'

titanic.loc[titanic["FamilySize"]>4, "FsizeD"] = 'large'



print(titanic["FsizeD"].unique())

print(titanic["FsizeD"].value_counts())
fig, saxis = plt.subplots(1, 2)



sns.pointplot(kind = 'point', x = "FsizeD", y = "Survived", data = titanic[titanic.Survived.notnull()], ax = saxis[0])

sns.pointplot(kind = 'point', x = 'FamilySize',  y= 'Survived', data = titanic[titanic.Survived.notnull()], ax = saxis[1])
#get regex

import re



#define the formula

def get_title(name):

    #title search is the regex search of a space, any capital or lowercase letter, as many times as needed, and ending with a period

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #if a title exists, extract and return it

    if title_search:

        return title_search.group(1)

    return ""



titles = titanic["Name"].apply(get_title)

print(pd.value_counts(titles))

#Set titles column

titanic["Title"] = titles
#Establish Rare Titles

rare_titles = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

#Add a new column Title2 where the persons title is a rare title

titanic.loc[titanic["Title"].isin(rare_titles), "Title2"] = 'Rare Title'



women_titles = ['Mlle', 'Ms', 'Mme', 'Dona']

#fill Title2 with overwritten women's titles

titanic.loc[titanic["Title"].isin(women_titles), "Title2"] = 'Miss'



#fill in the leftovers with the original titles

titanic["Title2"] = titanic["Title2"].fillna(titanic["Title"])

#Records by Title

print(titanic["Title2"].value_counts())

#Total Titanic Records

print(len(titanic))
#factors_plot = ['Age','Survived','Pclass','SibSp','Parch','sex_code','Title2','Deck','FsizeD']

titanic['Cabin'].str[0].unique()

titanic["Deck"] = titanic['Cabin'].str[0]

titanic["Deck"] = titanic["Deck"].astype('category')



#Establish Sex as a Categorical field for plotting

train_data = titanic[titanic.Survived.notnull()]





#Set Correlation

corr = train_data.corr()

plt.figure(figsize=(10, 10))



sns.heatmap(corr, annot = True, square = True, linewidths=0.01)

plt.title('Feature Correlation');



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show() # ta-da!

train_data.head()
titanic.info()
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(titanic[['Age', 'Fare']])

titanic[['Age', 'Fare']] = std_scale.transform(titanic[['Age', 'Fare']])



std_scale = preprocessing.StandardScaler().fit(test_data[['Age', 'Fare']])

test_data[['Age', 'Fare']] = std_scale.transform(test_data[['Age', 'Fare']])
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



lab_encoder = LabelEncoder()



cols = ['Embarked','Title','FsizeD','AgeBin']

newcols = ['Embarked_Code','Title_Code','FsizeD_Code','AgeBin_Code']



for col, ncols  in zip(cols, newcols):

        titanic[ncols] = lab_encoder.fit_transform(titanic[col])
#Outcome selected        

target = ['Survived']



#Variables uncoded

vars_uncoded = ['Sex','Embarked','Title','FsizeD','Pclass','AgeBin']



#Variables Coded

vars_coded = ['sex_code','Embarked_Code','Title_Code','FsizeD_Code','Pclass','AgeBin_Code']
titanic.columns
titanic.head()
#Begin testing with a Random Forest

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]





#Test with reduced features in original format

features = ["Pclass","Sex","SibSp","Parch"]



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)

model.fit(X,y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.head(10)
#Split encoded data into test & train data sets.

train_dat = titanic[titanic.Survived.notnull()]

test_dat = titanic[titanic.Survived.isnull()]
#Set new training random forest with coded variables of new data set.

y = train_dat['Survived']

X = pd.get_dummies(train_dat[vars_coded])

X_test = pd.get_dummies(test_dat[vars_coded])



model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)

model.fit(X,y)

predictions = model.predict(X_test)
#Create new gender submission DF

gender_submission = pd.DataFrame({'PassengerId': test_dat.PassengerId, 'Survived': predictions.astype(int)})

gender_submission.head(10)
#Save Gender Submission

save_file = 'gender_submission.csv'

gender_submission.to_csv(save_file, sep = ',',index=False)

gender_submission.shape