import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#visualization

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

#plt.style.available (you can use this code to learn different template of seaborn library, I will use whitegrid)

import plotly.express as px



from collections import Counter



#to close warnings

import warnings

warnings.filterwarnings("ignore")
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.describe()
#train_df.columns
#train_df.info()
fig, axarr =plt.subplots(2,3, figsize=(30, 10), sharey=True)



sns.countplot(train_df['Survived'], ax=axarr[0][0])

sns.countplot(train_df['Sex'], ax=axarr[0][1])

sns.countplot(train_df['Pclass'], ax=axarr[0][2])

sns.countplot(train_df['Embarked'], ax=axarr[1][0])

sns.countplot(train_df['SibSp'], ax=axarr[1][1])

sns.countplot(train_df['Parch'], ax=axarr[1][2])

fig.show()
fig, ax =plt.subplots(1,2, figsize=(30, 10), sharey=False)



sns.distplot(train_df['Age'], ax=ax[0])

sns.distplot(train_df['Fare'], ax=ax[1])

fig.show()

train_df[['Sex','Survived']].groupby(['Sex'],as_index = False).mean().sort_values(by='Survived',ascending=False)
fig = px.sunburst(train_df, path=['Sex', 'Survived'], values='PassengerId')

fig.show()
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index = False).mean().sort_values(by='Survived',ascending=False)
g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)

g.set_ylabels("Survived Probability")

plt.show()
train_df[['Embarked','Survived']].groupby(['Embarked'],as_index = False).mean().sort_values(by='Survived',ascending=False)
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index = False).mean().sort_values(by='Survived',ascending=False)
g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)

g.set_ylabels("Survived Probability")

plt.show()
train_df[['Parch','Survived']].groupby(['Parch'],as_index = False).mean().sort_values(by='Survived',ascending=False)
g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)

g.set_ylabels("Survived Probability")

plt.show()
#train_df.Age.value_counts(ascending=False, bins=4)

# I use this code try to find right bins number.
bin = [0,10,20,40,60,80] 

#use pd.cut function can attribute the values into its specific bins 

category = pd.cut(train_df.Age,bin) 

category = category.to_frame() 

category.columns = ['AgeRange'] 

#concatenate age and its bin 

df_new = pd.concat([train_df,category],axis = 1) 
df_new[['AgeRange','Survived']].groupby(['AgeRange'],as_index = False).mean()
g = sns.FacetGrid(train_df, col = "Survived")

g.map(sns.distplot, "Age", bins = 25)

plt.show()
train_df.Fare.value_counts(ascending=False, bins=6)

# I use this code try to find right bins number.
bin = [ 0,40,80,170, 255,340,513] 

#use pd.cut function can attribute the values into its specific bins 

category = pd.cut(train_df.Fare,bin) 

category = category.to_frame() 

category.columns = ['FareRange'] 

#concatenate age and its bin 

df_new = pd.concat([train_df,category],axis = 1) 
df_new[['FareRange','Survived']].groupby(['FareRange']).mean()
g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 3)

g.map(plt.hist, "Age", bins = 25)

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked", size = 3)

g.map(sns.pointplot, "Pclass","Survived","Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 2.3)

g.map(sns.barplot, "Sex", "Fare")

g.add_legend()

plt.show()
#function that finds outliers



def outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2) #to find if sample has 2 or more outliers

    

    return multiple_outliers
#detect outliers

train_df.loc[outliers(train_df,["Age","SibSp","Parch","Fare"])]
# drop outliers

train_df = train_df.drop(outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
train_df_len = len(train_df) # not to lose train data set

train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
#finding missing values

train_df.columns[train_df.isnull().any()]
train_df.isnull().sum() # it shows how many missing values we have each column
train_df[train_df["Embarked"].isnull()] #the rows Embarked has missing values.
import plotly.express as px



fig = px.box(train_df, x="Embarked", y="Fare",template='plotly_white' )

fig.show()
train_df["Embarked"] = train_df["Embarked"].fillna("S")

train_df[train_df["Embarked"].isnull()] #to check




fig = px.box(train_df, x="Fare",template='plotly_white' )

fig.show()
train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"] == 3])
train_df["Fare"] = train_df["Fare"].fillna(24.816367)
train_df[train_df["Fare"].isnull()] #to check
train_df[train_df["Age"].isnull()]
sns.factorplot(x = "Sex", y = "Age", hue = "Pclass",data = train_df, kind = "box")

plt.show()
sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")

sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")

plt.show()
train_df.head()
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]] # to convert categorical sex variable to numerical one.
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True)

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index) #make list that inludes all mising values in age column 

for i in index_nan_age: # lets fill these values under the below conditions

    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()#fill same median with Parch,Pclass and SibSp variables are same

    age_med = train_df["Age"].median()# if not find any same variable, use age median

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i] = age_pred

    else:

        train_df["Age"].iloc[i] = age_med
train_df['CabinN'] = train_df['Cabin'].str[:1] #first I extracted first letter of Cabin
f = sns.countplot(x="CabinN", data = train_df)

plt.show()
train_df['Deck'] = train_df['CabinN']

train_df['Deck'] = train_df['Deck'].fillna('M')
train_df['Deck'] = train_df['Deck'].replace(['A', 'B', 'C'], 'ABC')

train_df['Deck'] = train_df['Deck'].replace(['D', 'E'], 'DE')

train_df['Deck'] = train_df['Deck'].replace(['F', 'G'], 'FG')



train_df['Deck'].value_counts()

train_df.drop(['Cabin','CabinN'],axis=1, inplace=True)
train_df.info()
train_df["Name"].head(10)
# example to understand separation:

#s = " Allen, Mr. William Henry"

#s.split(".") # this separete with dot

#s.split(".")[0] # and choose first one like 'Allen, Mr'

#s.split(".")[0].split(',') # and we separe new object with comma, so we have 'Allen' and ' Mr'

#s.split(".")[0].split(',')[-1] # and we choose last object of it, it is : ' Mr'

#s.split(".")[0].split(',')[-1].strip() # and we use .strip() if we have any space to get rid of it.
name = train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df.Title.value_counts()
train_df["Title"] = train_df["Title"].replace(["Capt","Col","Don","Dr","Major","Rev","Jonkheer","Dona"],"other")

train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" or i == "Lady" or i == "the Countess"  else 2 if i == "Mr" or i == "Sir" else 3 for i in train_df["Title"]]
f = sns.countplot(x="Title", data = train_df)

f.set_xticklabels(["Master","Female's","Male's","Other"])



plt.show()
g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")

g.set_xticklabels(["Master","Female's","Male's","Other"])

g.set_ylabels("Survival Probability")

plt.show()
train_df.drop(labels = ["Name"], axis = 1, inplace = True)
train_df = pd.get_dummies(train_df,columns=["Title"])
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
g = sns.factorplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]
train_df = pd.get_dummies(train_df, columns= ["family_size"])
train_df = pd.get_dummies(train_df, columns=["Embarked"])
train_df["Ticket"].head()
tickets = [] # new empty list

for i in list(train_df.Ticket): # to look each data in Ticket column

    if not i.isdigit(): # if data is not just a digit number

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])# split with '.' and '/' and drop all space than append the list the first item

    else: # if just a digit number, write 'x'

        tickets.append("x")

train_df["Ticket"] = tickets
sns.countplot(x="Ticket", data = train_df)

plt.xticks(rotation = 90)

plt.show()
g = sns.factorplot(x = "Ticket", y = "Survived", data = train_df, kind = "bar")

g.set_ylabels("Survival")

plt.xticks(rotation = 90)

plt.show()
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")# prefix = 'T' means that each dummy column begins with 'T' not 'Tickets'
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns= ["Pclass"])
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"])
train_df.head()
train_df = pd.get_dummies(train_df, columns= ["Deck"])
train_df.drop(labels = ["PassengerId"], axis = 1, inplace = True)


train_df['Infants'] = train_df['Age'].apply(lambda x: 1 if x < 1 else 0)

train_df['Child'] = train_df['Age'].apply(lambda x: 1 if x < 18 else 0)

train_df['Adult'] = train_df['Age'].apply(lambda x: 1 if x >= 18 and x < 65 else 0)

train_df['Elderly'] = train_df['Age'].apply(lambda x: 1 if x >= 65 else 0)

 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
train_df_len
test = train_df[train_df_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
train = train_df[:train_df_len]

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2) #to see percentage of regression score, we multiple 100 and getting two decimal

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Testing Accuracy: % {}".format(acc_log_test))
random_state = 42 # random state model number, we choose exact number to find same result each of run

#our models;

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier(),

             XGBClassifier(random_state=random_state),

             GradientBoostingClassifier(random_state=random_state),

             AdaBoostClassifier(random_state=random_state)]



# decision tree parameters

dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



#support vector classifier parameters

svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



#random forest parameter

rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,5,30],

                "bootstrap":[False],

                "n_estimators":[100,600],

                "criterion":["entropy"]}



#logistic regression parameters

logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



#k-nearest neighboor parameters

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}



xgb_param_grid = {"max_depth": range(1,20,2)}



gbc_param_grid = {"n_estimators":[10,30]}



adab_param_grid = {"n_estimators":[10,30]}









classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid,

                   xgb_param_grid,

                   gbc_param_grid,

                   adab_param_grid]

cv_result = [] # cross validation result: it will add each result

best_estimators = [] #choosing best results

for i in range(len(classifier)):  #this loop try to choose best results according to above models

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1) # verbose show us all results when code is running

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier", "XGBClassifier", "GradientBoostingClassifier", "AdaBoostClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
print(cv_results)
votingC = VotingClassifier(estimators = [("rfc",best_estimators[2]),("gb",best_estimators[6])],voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("submission.csv", index = False)