import pandas as pd

import numpy as np



titanic = pd.read_csv("../input/train.csv")

titanic_test = pd.read_csv("../input/test.csv")

titanic.shape
titanic.describe()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)



pd.options.display.mpl_style = 'default'

titanic.hist(bins=10,figsize=(9,7),grid=False)
g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age",color="purple")
g = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g = sns.FacetGrid(titanic, hue="Survived", col="Sex", margin_titles=True,

                palette="Set1",hue_kws=dict(marker=["^", "v"]))

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare')
titanic.Embarked.value_counts().plot(kind='bar', alpha=0.55)

plt.title("Passengers per boarding location")
sns.factorplot(x = 'Embarked',y="Survived", data = titanic,color="r")
sns.set(font_scale=1)

g = sns.factorplot(x="Sex", y="Survived", col="Pclass",

                    data=titanic, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Men", "Women"])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many Men and Women Survived by Passenger Class')
ax = sns.boxplot(x="Survived", y="Age", 

                data=titanic)

ax = sns.stripplot(x="Survived", y="Age",

                   data=titanic, jitter=True,

                   edgecolor="gray")

sns.plt.title("Survival by Age",fontsize=12)
titanic.Age[titanic.Pclass == 1].plot(kind='kde')    

titanic.Age[titanic.Pclass == 2].plot(kind='kde')

titanic.Age[titanic.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
corr=titanic.corr()#["Survived"]

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between features')
titanic.corr()["Survived"]
g = sns.factorplot(x="Age", y="Embarked",

                    hue="Sex", row="Pclass",

                    data=titanic[titanic.Embarked.notnull()],

                    orient="h", size=2, aspect=3.5, 

                   palette={'male':"purple", 'female':"blue"},

                    kind="violin", split=True, cut=0, bw=.2)
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=titanic)
titanic["Embarked"] = titanic["Embarked"].fillna('C')
def fill_missing_fare(df):

    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()

#'S'

       #print(median_fare)

    df["Fare"] = df["Fare"].fillna(median_fare)

    return df



titanic_test=fill_missing_fare(titanic_test)
titanic["Deck"]=titanic.Cabin.str[0]

titanic_test["Deck"]=titanic_test.Cabin.str[0]

titanic["Deck"].unique() # 0 is for null values
g = sns.factorplot("Survived", col="Deck", col_wrap=4,

                    data=titanic[titanic.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8)
titanic = titanic.assign(Deck=titanic.Deck.astype(object)).sort("Deck")

g = sns.FacetGrid(titanic, col="Pclass", sharex=False,

                  gridspec_kws={"width_ratios": [5, 3, 3]})

g.map(sns.boxplot, "Deck", "Age");
titanic.Deck.fillna('Z', inplace=True)

titanic_test.Deck.fillna('Z', inplace=True)

titanic["Deck"].unique() # Z is for null values
# Create a family size variable including the passenger themselves

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]+1

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1

print(titanic["FamilySize"].value_counts())
# Discretize family size

titanic.loc[titanic["FamilySize"] == 1, "FsizeD"] = 'singleton'

titanic.loc[(titanic["FamilySize"] > 1)  &  (titanic["FamilySize"] < 5) , "FsizeD"] = 'small'

titanic.loc[titanic["FamilySize"] >4, "FsizeD"] = 'large'



titanic_test.loc[titanic_test["FamilySize"] == 1, "FsizeD"] = 'singleton'

titanic_test.loc[(titanic_test["FamilySize"] >1) & (titanic_test["FamilySize"] <5) , "FsizeD"] = 'small'

titanic_test.loc[titanic_test["FamilySize"] >4, "FsizeD"] = 'large'

print(titanic["FsizeD"].unique())

print(titanic["FsizeD"].value_counts())
sns.factorplot(x="FsizeD", y="Survived", data=titanic)
#Create feture for length of name 

# The .apply method generates a new series

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))



titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

#print(titanic["NameLength"].value_counts())

'''

titanic.loc[titanic["NameLength"]>37 , "NlengthD"] = 'long'

titanic.loc[titanic["NameLength"]<38 , "NlengthD"] = 'short'



titanic_test.loc[titanic_test["NameLength"]>37 , "NlengthD"] = 'long'

titanic_test.loc[titanic_test["NameLength"]<38 , "NlengthD"] = 'short'

'''



bins = [0, 20, 40, 57, 85]

group_names = ['short', 'okay', 'good', 'long']

titanic['NlengthD'] = pd.cut(titanic['NameLength'], bins, labels=group_names)

titanic_test['NlengthD'] = pd.cut(titanic_test['NameLength'], bins, labels=group_names)



sns.factorplot(x="NlengthD", y="Survived", data=titanic)

print(titanic["NlengthD"].unique())
import re



#A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



#Get all the titles and print how often each one occurs.

titles = titanic["Name"].apply(get_title)

print(pd.value_counts(titles))





#Add in the title column.

titanic["Title"] = titles



# Titles with very low cell counts to be combined to "rare" level

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Also reassign mlle, ms, and mme accordingly

titanic.loc[titanic["Title"] == "Mlle", "Title"] = 'Miss'

titanic.loc[titanic["Title"] == "Ms", "Title"] = 'Miss'

titanic.loc[titanic["Title"] == "Mme", "Title"] = 'Mrs'

titanic.loc[titanic["Title"] == "Dona", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Lady", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Countess", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Capt", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Col", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Don", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Major", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Rev", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Sir", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Jonkheer", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Dr", "Title"] = 'Rare Title'



#titanic.loc[titanic["Title"].isin(['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

#                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']), "Title"] = 'Rare Title'



#titanic[titanic['Title'].isin(['Dona', 'Lady', 'Countess'])]

#titanic.query("Title in ('Dona', 'Lady', 'Countess')")



titanic["Title"].value_counts()





titles = titanic_test["Name"].apply(get_title)

print(pd.value_counts(titles))



#Add in the title column.

titanic_test["Title"] = titles



# Titles with very low cell counts to be combined to "rare" level

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Also reassign mlle, ms, and mme accordingly

titanic_test.loc[titanic_test["Title"] == "Mlle", "Title"] = 'Miss'

titanic_test.loc[titanic_test["Title"] == "Ms", "Title"] = 'Miss'

titanic_test.loc[titanic_test["Title"] == "Mme", "Title"] = 'Mrs'

titanic_test.loc[titanic_test["Title"] == "Dona", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Lady", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Countess", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Capt", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Col", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Don", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Major", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Rev", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Sir", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Jonkheer", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Dr", "Title"] = 'Rare Title'



titanic_test["Title"].value_counts()
titanic["TicketNumber"] = titanic["Ticket"].str.extract('(\d{2,})', expand=True)

titanic["TicketNumber"] = titanic["TicketNumber"].apply(pd.to_numeric)





titanic_test["TicketNumber"] = titanic_test["Ticket"].str.extract('(\d{2,})', expand=True)

titanic_test["TicketNumber"] = titanic_test["TicketNumber"].apply(pd.to_numeric)
titanic.TicketNumber.fillna(titanic["TicketNumber"].median(), inplace=True)

titanic_test.TicketNumber.fillna(titanic_test["TicketNumber"].median(), inplace=True)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



labelEnc=LabelEncoder()



cat_vars=['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck']

for col in cat_vars:

    titanic[col]=labelEnc.fit_transform(titanic[col])

    titanic_test[col]=labelEnc.fit_transform(titanic_test[col])



titanic.head()
with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(titanic["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="red")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")
from sklearn.ensemble import RandomForestRegressor

#predicting missing values in age using Random Forest

def fill_missing_age(df):

    

    #Feature set

    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',

                 'TicketNumber', 'Title','Pclass','FamilySize',

                 'FsizeD','NameLength',"NlengthD",'Deck']]

    # Split sets into train and test

    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values

    test = age_df.loc[ (df.Age.isnull()) ]# null Ages

    

    # All age values are stored in a target array

    y = train.values[:, 0]

    

    # All the other values are stored in the feature array

    X = train.values[:, 1::]

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(test.values[:, 1::])

    

    # Assign those predictions to the full data set

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df
titanic=fill_missing_age(titanic)

titanic_test=fill_missing_age(titanic_test)
with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(titanic["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="tomato")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")

    plt.xlim((15,100))
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(titanic[['Age', 'Fare']])

df_std = std_scale.transform(titanic[['Age', 'Fare']])





std_scale = preprocessing.StandardScaler().fit(titanic_test[['Age', 'Fare']])

df_std = std_scale.transform(titanic_test[['Age', 'Fare']])

titanic.corr()["Survived"]
# Import the linear regression class

from sklearn.linear_model import LinearRegression

# Sklearn also has a helper that makes it easy to do cross validation

from sklearn.cross_validation import KFold



# The columns we'll use to predict the target

predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",

              "Embarked","NlengthD", "FsizeD", "Title","Deck"]

target="Survived"

# Initialize our algorithm class

alg = LinearRegression()



# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.

# We set random_state to ensure we get the same splits every time we run this.

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)



predictions = []
for train, test in kf:

    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.

    train_predictors = (titanic[predictors].iloc[train,:])

    # The target we're using to train the algorithm.

    train_target = titanic[target].iloc[train]

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

    # We can now make predictions on the test fold

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0





accuracy=sum(titanic["Survived"]==predictions)/len(titanic["Survived"])

accuracy
from sklearn.metrics import accuracy_score

from sklearn import metrics

from sklearn.metrics import classification_report



accuracy_score(titanic["Survived"], predictions)
metrics.confusion_matrix(titanic["Survived"], predictions)
classification_report(titanic["Survived"], predictions, target_names=['Not Survived', 'Survived'])
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.

# We set random_state to ensure we get the same splits every time we run this.

kf1 = KFold(titanic_test.shape[0], n_folds=3, random_state=1)



predictionsF = []





for train, t in kf:

    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.

    train_predictors = (titanic[predictors].iloc[train,:])

    # The target we're using to train the algorithm.

    train_target = titanic[target].iloc[train]

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

for i, test in kf1:

    # We can now make predictions on the test fold

    test_predictions1 = alg.predict(titanic_test[predictors].iloc[test,:])

    predictionsF.append(test_predictions1)

titanic_test


predictionsF = np.concatenate(predictionsF, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictionsF[predictionsF > .5] = 1

predictionsF[predictionsF <=.5] = 0

predictionsF.size
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictionsF.astype(int)

    })

submission.to_csv("titanic_submission.csv", index=False)