import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("ggplot")



import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%javascript

IPython.OutputArea.auto_scroll_threshold = 999;
dfTrain = pd.read_csv("/kaggle/input/titanic/train.csv")

dfTest = pd.read_csv("/kaggle/input/titanic/test.csv")



print("\nTrain dataframe info\n")

dfTrain.info()

print("\nTest dataframe info\n")

dfTest.info()
dfTrain.head(10)
dfTrain.describe()
import plotly

from plotly.offline import iplot

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



def PlotPieChart(df,label):

    trace = go.Pie(labels=df[label])

    layout = dict(title = str(label))

    fig = dict(data=[trace], layout=layout)

    iplot(fig)
categoricalLabels = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked"]

for label in categoricalLabels:

    PlotPieChart(dfTrain,label)
import plotly.express as px

def PlotHistogram(df,label):

    fig = px.histogram(df, x=label)

    fig.show()
numericalLabels = ["Age", "Fare"]

for label in numericalLabels:

    PlotHistogram(dfTrain,label)
import plotly.express as px

def relationPieChart(df,value,name):

    fig = px.pie(df, values=value, names=name, title=str(value+" -- "+name))

    fig.show()
print("Surviving probability of Pclasses")

print(dfTrain[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending=False))

relationPieChart(dfTrain,"Survived","Pclass")
print("Surviving probability of genders")

print(dfTrain[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending=False))

relationPieChart(dfTrain,"Survived","Sex")
print("Surviving probability of Sibsps")

print(dfTrain[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending=False))

relationPieChart(dfTrain,"Survived","SibSp")
print("Surviving probability of Parchs")

print(dfTrain[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending=False))

relationPieChart(dfTrain,"Survived","Parch")
def detectOutlier(df,features, minOutlierCount):

    outlierList = []

    

    for feature in features:

        #1st quartile

        Q1 = np.percentile(df[feature],25)

        #3rd quartile

        Q3 = np.percentile(df[feature],75)

        #IQR

        IQR = Q3 - Q1

        #Outlier Step

        outlierStep = IQR * 1.5

        #detect outlier and their indices

        outlierListCol = df[(df[feature] < Q1 - outlierStep) | (df[feature] > Q3 + outlierStep)].index

        #store indices

        outlierList.extend(outlierListCol)

    

    outlierIndices = Counter(outlierList)

    multipleOutliers = list(i for i,v in outlierIndices.items() if v > minOutlierCount)

    

    return multipleOutliers
dfTrain.loc[detectOutlier(dfTrain,["Age","SibSp","Parch","Fare"],2)]
dfTrain.drop(detectOutlier(dfTrain,["Age","SibSp","Parch","Fare"],2),axis=0,inplace=True)

dfTrain.reset_index(inplace=True,drop=True)
correlationList = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]

sns.heatmap(dfTrain[correlationList].corr(), annot=True, fmt=".2f")
g = sns.factorplot(x = "SibSp", y = "Survived", kind="bar", data=dfTrain,size = 5)

g.set_ylabels("Survival Probablity")

plt.show()
g = sns.factorplot(x = "Parch", y = "Survived", kind="bar", data=dfTrain,size = 5)

g.set_ylabels("Survival Probablity")

plt.show()
g = sns.factorplot(x = "Pclass", y = "Survived", kind="bar", data=dfTrain,size = 5)

g.set_ylabels("Survival Probablity")

plt.show()
g = sns.FacetGrid(dfTrain,col="Survived",size = 5)

g.map(sns.distplot,"Age",bins=30)

plt.show()
import plotly.express as px

fig = px.histogram(dfTrain[dfTrain.Survived == 1], x="Age", color="Sex", marginal="violin", title ="Survived passengers by their ages", color_discrete_map={"male": "#187196","female": "#fab7cc"})

fig.show()

fig2 = px.histogram(dfTrain[dfTrain.Survived == 0], x="Age", color="Sex", marginal="violin", title ="Couldnt survived passengers by their ages", color_discrete_map={"male": "#187196","female": "#fab7cc"})

fig2.show()
fig = px.histogram(dfTrain[dfTrain.Survived == 1], x="Pclass", color="Sex", marginal="violin", title ="Survived passengers by their ticket class", color_discrete_map={"male": "#187196","female": "#fab7cc"})

fig.show()

fig2 = px.histogram(dfTrain[dfTrain.Survived == 0], x="Pclass", color="Sex", marginal="violin", title ="Couldnt survived passengers by their ticket class", color_discrete_map={"male": "#187196","female": "#fab7cc"})

fig2.show()
print("\nTrain dataframe columns that include null values:\n")

print(dfTrain.columns[dfTrain.isna().any()])

print("\nTrain dataframe null rows count:\n")

print(dfTrain.isna().sum())

print("\n================================================")

print("\nTest dataframe columns that include null values:\n")

print(dfTest.columns[dfTest.isna().any()])

print("\nTest dataframe null rows count:\n")

print(dfTest.isna().sum())
dfTrain[dfTrain.Embarked.isna()]
dfTest[dfTest.Fare.isna()]
dfCombined = pd.concat([dfTrain, dfTest], axis=0)

dfCombined.info()
fig = px.box(dfCombined, x="Embarked", y="Fare", points="all")

fig.show()
dfTrain[dfTrain.Embarked.isna()][["Fare","Embarked"]]
dfTrain["Embarked"].fillna("C",inplace=True)

dfTrain.iloc[[60,821]]
dfTest[dfTest.Fare.isna()]
np.mean(dfCombined[(dfCombined["Pclass"] == 3) & (dfCombined["Embarked"] == "S")]["Fare"])
dfTest["Fare"].fillna(np.mean(dfCombined[(dfCombined["Pclass"] == 3) & (dfCombined["Embarked"] == "S")]["Fare"]) , inplace=True)

dfTest.iloc[[152]]
dfTrain[dfTrain["Age"].isna()]
dfTest[dfTest["Age"].isna()]
dfCombined["Gender"] = [1 if i == "male" else 0 for i in dfCombined["Sex"]] # make sex variable numerical and store them in gender column to show in heatmap.

correlationList = ["Age", "Gender", "Pclass", "SibSp", "Parch", "Fare"]

sns.heatmap(dfCombined[correlationList].corr(), annot=True, fmt=".2f")

plt.show()

dfCombined.drop(["Gender"],axis=1,inplace=True) # drop the Gender column, because it was necessary for only heatmap.
fig = px.box(dfCombined, x = "Sex", y = "Age", color="Pclass", points="all", title="Correlation between Sex --- Age --- Pclass")

fig.show()
fig = px.box(dfCombined, x = "SibSp", y = "Age", points="all", title="Correlation between Age and SibSp")

fig.show()
fig = px.histogram(dfCombined, x = "Fare", y = "Age", histfunc='avg', title="Correlation between Average Age and Fare")

fig.show()
fig = px.box(dfCombined, x = "Parch", y = "Age", points="all", title="Correlation between Age and Parch")

fig.show()
trainIndexNanAge = list(dfTrain[dfTrain["Age"].isna()].index)

print("number of nan age train indexes : {}".format(len(trainIndexNanAge)))

testIndexNanAge = list(dfTest[dfTest["Age"].isna()].index)

print("number of nan age test indexes : {}".format(len(testIndexNanAge)))

combinedIndexNanAge = list(dfCombined[dfCombined["Age"].isna()].index)

print("number of total nan age indexes : {}".format(len(combinedIndexNanAge)))
for index in trainIndexNanAge:

    age_pred = dfCombined["Age"][((dfCombined["SibSp"] == dfTrain.iloc[index]["SibSp"]) & (dfCombined["Parch"] == dfTrain.iloc[index]["Parch"]) & (dfCombined["Pclass"] == dfTrain.iloc[index]["Pclass"]))].median()

    if not np.isnan(age_pred):

        dfTrain["Age"].iloc[index] = age_pred
for index in testIndexNanAge:

    age_pred = dfCombined["Age"][((dfCombined["SibSp"] == dfTest.iloc[index]["SibSp"]) & (dfCombined["Parch"] == dfTest.iloc[index]["Parch"]) & (dfCombined["Pclass"] == dfTest.iloc[index]["Pclass"]))].median()

    if not np.isnan(age_pred):

        dfTest["Age"].iloc[index] = age_pred
trainIndexNanAge = list(dfTrain[dfTrain["Age"].isna()].index)

print("number of nan age train indexes : {}".format(len(trainIndexNanAge)))

testIndexNanAge = list(dfTest[dfTest["Age"].isna()].index)

print("number of nan age test indexes : {}".format(len(testIndexNanAge)))

dfCombined = pd.concat([dfTrain, dfTest], axis=0)

combinedIndexNanAge = list(dfCombined[dfCombined["Age"].isna()].index)

print("number of total nan age indexes : {}".format(len(combinedIndexNanAge)))
age_med = dfCombined["Age"].median()

for index in trainIndexNanAge:

    dfTrain["Age"].iloc[index] = age_med

for index in testIndexNanAge:

    dfTest["Age"].iloc[index] = age_med
trainIndexNanAge = list(dfTrain[dfTrain["Age"].isna()].index)

print("number of nan age train indexes : {}".format(len(trainIndexNanAge)))

testIndexNanAge = list(dfTest[dfTest["Age"].isna()].index)

print("number of nan age test indexes : {}".format(len(testIndexNanAge)))

dfCombined = pd.concat([dfTrain, dfTest], axis=0)

combinedIndexNanAge = list(dfCombined[dfCombined["Age"].isna()].index)

print("number of total nan age indexes : {}".format(len(combinedIndexNanAge)))

del dfCombined, combinedIndexNanAge, testIndexNanAge, trainIndexNanAge
def find_title(name):

    return name.split(",")[1].split(".")[0].strip()
dfTrain["Title"] = dfTrain["Name"].apply(find_title)

dfTest["Title"] = dfTest["Name"].apply(find_title)

print("Used Different Titles Are:")

print(pd.concat([dfTrain,dfTest],axis=0)["Title"].value_counts())
other_list = ["Rev", "Dr", "Col", "Major", "Ms", "Mlle", "Jonkheer", "Lady", "Mme", "Dona", "Capt", "the Countess", "Sir", "Don"]

dfTrain["Title"] = dfTrain["Title"].replace(other_list, "Other")

dfTest["Title"] = dfTest["Title"].replace(other_list, "Other")

print("Used Different Titles Are:")

print(pd.concat([dfTrain,dfTest],axis=0)["Title"].value_counts())
g = sns.factorplot(x = "Title", y = "Survived", kind="bar", data=dfTrain, size = 5)

g.set_ylabels("Survival Probablity")

plt.show()
dfTrain.drop(["Name"], axis=1, inplace=True)

dfTest.drop(["Name"], axis=1, inplace=True)



dfTrain["Title"] = dfTrain["Title"].astype("category")

dfTrain = pd.get_dummies(dfTrain,columns=["Title"])

dfTest["Title"] = dfTest["Title"].astype("category")

dfTest = pd.get_dummies(dfTest,columns=["Title"])



print("Train dataframe columns: {}".format(dfTrain.columns.values))

print("Test dataframe columns: {}".format(dfTest.columns.values))
dfTrain["Pclass"] = dfTrain["Pclass"].astype("category")

dfTrain = pd.get_dummies(dfTrain,columns=["Pclass"])



dfTest["Pclass"] = dfTest["Pclass"].astype("category")

dfTest = pd.get_dummies(dfTest,columns=["Pclass"])



print("Train dataframe columns: {}".format(dfTrain.columns.values))

print("Test dataframe columns: {}".format(dfTest.columns.values))
dfTrain["Embarked"] = dfTrain["Embarked"].astype("category")

dfTrain = pd.get_dummies(dfTrain,columns=["Embarked"])



dfTest["Embarked"] = dfTest["Embarked"].astype("category")

dfTest = pd.get_dummies(dfTest,columns=["Embarked"])



print("Train dataframe columns: {}".format(dfTrain.columns.values))

print("Test dataframe columns: {}".format(dfTest.columns.values))
dfTrain["Sex"] = dfTrain["Sex"].astype("category")

dfTrain = pd.get_dummies(dfTrain,columns=["Sex"], prefix="S")



dfTest["Sex"] = dfTest["Sex"].astype("category")

dfTest = pd.get_dummies(dfTest,columns=["Sex"], prefix="S")



print("Train dataframe columns: {}".format(dfTrain.columns.values))

print("Test dataframe columns: {}".format(dfTest.columns.values))
dfTrain.drop(["Ticket","Cabin","PassengerId"], axis=1, inplace=True)

dfTest.drop(["Ticket","Cabin"], axis=1, inplace=True)



print("Train dataframe columns: {}".format(dfTrain.columns.values))

print("Test dataframe columns: {}".format(dfTest.columns.values))
dfTrain.head()
dfTest.head()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
dfTrain = (dfTrain - np.min(dfTrain)) / (np.max(dfTrain) - np.min(dfTrain)).values

dfTestTemp = dfTest["PassengerId"]

dfTest = (dfTest - np.min(dfTest)) / (np.max(dfTest) - np.min(dfTest)).values

dfTest["PassengerId"] = dfTestTemp

del dfTestTemp

dfTest.describe()
xTrain = dfTrain.drop(["Survived"], axis = 1)

yTrain = dfTrain["Survived"]

xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size = 0.33, random_state = 50)

print("sizes: ")

print("xTrain: {}, xValidation: {}, yTrain: {}, yValidation: {}".format(len(xTrain),len(xVal),len(yTrain),len(yVal)))

print("\t\t\t test: {}".format(len(dfTest)))
logreg = LogisticRegression()

logreg.fit(xTrain,yTrain)

acc_logreg_train = logreg.score(xTrain,yTrain)*100

acc_logreg_validation = logreg.score(xVal,yVal)*100

print("Train data accuracy: {}".format(acc_logreg_train))

print("Validation data accuracy: {}".format(acc_logreg_validation))
classifiers = [DecisionTreeClassifier(random_state = 50),

               SVC(random_state = 50),

               RandomForestClassifier(random_state = 50),

               KNeighborsClassifier(),

               LogisticRegression(random_state = 50)]



decisionTree_params_grid = {"min_samples_split":range(10,500,20),

                            "max_depth":range(1,20,2)}

svc_params_grid = {"kernel":["rbf"],

                   "gamma":[0.001,0.01,0.1,1],

                   "C":[1,10,50,100,200],

                   "probability":[True]}

randomForest_params_grid = {"max_features":[1,3,10],

                            "min_samples_split":[2,3,10],

                            "min_samples_leaf":[1,3,10],

                            "bootstrap":[False],

                            "n_estimators":[100,300],

                            "criterion":["gini"]}

knn_params_grid = {"n_neighbors":np.linspace(1,19,10,dtype=int).tolist(),

                   "weights":["uniform","distance"],

                   "metric":["euclidean","manhattan"]}

logisticRegression_params_grid = {"C":np.logspace(-3,3,7),

                                  "penalty":["l1","l2"]}





classifier_params = [decisionTree_params_grid,

                     svc_params_grid,

                     randomForest_params_grid,

                     knn_params_grid,

                     logisticRegression_params_grid]
cvResult = []

bestEstimators = []

for classifierIndex in range(len(classifiers)):

    classifier = GridSearchCV(classifiers[classifierIndex], param_grid=classifier_params[classifierIndex], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs=-1 , verbose=1)

    classifier.fit(xTrain,yTrain)

    cvResult.append(classifier.best_score_)

    bestEstimators.append(classifier.best_estimator_)

    print("current best score = {}".format(cvResult[classifierIndex]))
cvResult = pd.DataFrame({"Cross Validation Means": cvResult, "ML Models":[

    "Decision Tree Classifier",

    "SVM",

    "Random Forest Classifier",

    "K Neighbors Classifier",

    "Logistic Regression"]})
fig = px.bar(cvResult, x='ML Models', y='Cross Validation Means', title="Cross Validation Scores")

fig.show()
votingClassifier = VotingClassifier(estimators = [("decissionTree",bestEstimators[0]),

                                                  ("randomForest",bestEstimators[2]),

                                                  ("logisticRegression",bestEstimators[4])],

                                    voting = "soft",

                                    n_jobs = -1)
votingClassifier = votingClassifier.fit(xTrain,yTrain)

print("Train data accuracy: {}".format(accuracy_score(votingClassifier.predict(xTrain),yTrain)))

print("Validation data accuracy: {}".format(accuracy_score(votingClassifier.predict(xVal),yVal)))
survivedTest = pd.Series(votingClassifier.predict(dfTest.drop(["PassengerId"],axis=1)),name="Survived").astype(int)

results = pd.concat([dfTest["PassengerId"], survivedTest],axis = 1)

results.head(10)
results.info()
results.describe()
results.to_csv("titanic.csv", index = False)