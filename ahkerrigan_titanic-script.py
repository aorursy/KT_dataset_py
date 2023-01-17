import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



from scipy import stats



sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

IDtest = test["PassengerId"]
## In order to make changes to all data, we need to combine train and test for the time being

train_len = len(train)

dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
# When it comes to cleaning data, what is the very first thing you should do?

# Always manage your missing values. 



# Fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)



# Check for Null values

dataset.isnull().sum()



# The reason why there is a significant number of null values in survived is only because we combiend the train and the test, and obviously we don't have the solutions for the test
# If we want to check what just the train looks like, we still have it in memory



train.info()

train.isnull().sum()
train.head()
train.describe()
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 

g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# Using a heatmap, we can check to see how our numerical values correlate with each other. As you can see, the only numerical value that seemst o really correlate with 

# survival is Fare. That's not to say that the others are useless, but intuitively you can imagine bigger fair = rich = surivived
# What definately is possible is that the other features have subpopulations that have actual correlation. That is to say``````````````````````
# Explore SibSp feature vs Survived

g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# What does thisd tell us about number of siblings?

# It tells us that while 0-2 siblings have a fairly average chance of survival, 3-4 have a dramatically smaller chance.

# This means that while sibling count isn't good by itself, knowing whether they have 3-4 is important
# Explore Parch feature vs Survived

g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# What this tells us is that smaller families (1-2) have better chance of survival than people without families, or people wtih large families

# Why am I not saying that families of 3 have the best chance of surivial?

# While sure, the average seems marginally higher, the uncertainty is much greater around families of 3, meaning the values are more spread out.

# Families of 1-2 are much more certain to have more than 50% chance of survival.
# Now, lets take a look at a Facet graph of Age

# Explore Age vs Survived

g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")
# What does this tell us

# Before in our heatmap, we noted that Age does not seem correlated with survival, but this graph tells a different story. 

# The distribution seems somewhat normal, except for the unusual jump in survivability of infants

# This means it might be valauble to classify passenger into age groups, rather than leave their age

# let's try superimposing these to get a clearer picture



# Explore Age distibution 

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
# Let's move onto Fare

dataset["Fare"].isnull().sum()
# Theres only a single missing value of fare, what should we do with it?

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
# Explore Fare distribution 

g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")
# Oof, this is extremly skewed. This means that our model is going to massively overweight values on the right end. Therefore, we should probobly transform it into a log function

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")

# That looks much, much better right?

# Let's move onto sex

g = sns.barplot(x="Sex",y="Survived",data=train)

g = g.set_ylabel("Survival Probability")
# Oh wow, men only have about a 20% chance of survival, while women have 70%

# You could actually stop everything right now and just predict survival based on gender and get ~75% accuracy wiht these numbers
train[["Sex","Survived"]].groupby('Sex').mean()
# Explore Pclass vs Survived

g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# As you would expect, chance of survival is pretty much directly correlated with class

# Explore Pclass vs Survived by Sex

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# As we saw earlier with our exploration of Age, there is clearly some information hidden within

# However there are 300 values missing. This is way too many to simply replace with the mean (because 300 missing is enough to shift the mean to a false value)

# The solution: Let's see if there is some other variable that correlates with Age, and use that to predict what the missing age might be

# Explore Age vs Sex, Parch , Pclass and SibSP

g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box")

g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")

g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")
# What can we conclude from these graphs?
# The distribution of age seems to be the same across male and femlae,  the higher the class of passenger, the older they are

# Parch (parents/children) seem to be postively correlated wtih age, while sibling count engativly correlated. T

# That is to say, older passengers tend to have more children/parents, while younger passengers have more siblings
# Let's go ahead and convert sex to a numerical value since we missed it

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
# Let's look directly at the correlation between numerical features

g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)

# This is a good example of our visual intuition being wrong. We were correct that Age and Sex have nothing to do with each other

# But parent/childen is actually negativly correlated with age, as is class and sibling count

# Heres how we're going to do it. For every missing Age value, we are going to find rows with the same Sibling, parch, and class values as that row with missing age, and average those ages
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = dataset["Age"].median()

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med
# Let's see how things changed

g = sns.factorplot(x="Survived", y = "Age",data = train, kind="box")

g = sns.factorplot(x="Survived", y = "Age",data = train, kind="violin")
# So theres still not correlation between ages and survival, except for that little blip at the buttom of the survived violen
# Now it's time for the best part: Feature engineering

# Quesion: Should we keep the name of passengers? How could the name be useful?
# Get Title from Name

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)

dataset["Title"].head()
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45) 
# So it would be a waste of time to worry about titles that only appear once or twice, so let's just change them to "rare"

# Then, let's map each title to 

# Convert to categorical values for title Title 

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)
g = sns.countplot(dataset["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")
# Well, we already know how important sex was to survival, but now "master" is revealing the importance of children as well

# As we know "Women and children first"

# Drop Name variable

dataset.drop(labels = ["Name"], axis = 1, inplace = True)



# If you recall, the only useful information from Age was that children had a slight advantage from

# survival. However, most Ages were actually missing. There were no missing titles, so we're instead

# going to rely on the existing of the "master" title to figure out who is a child.

# Did we waste a ton of time cleaning up the Ages? Not really, a lot of the time when working on data

# you find that previous approaches were not as good as you originally expected

dataset.drop(labels = ["Age"], axis = 1, inplace = True)
dataset.head()
# Let's go over to family size again, since we talked about that a lot

# Let's examine size of family, including the passenger themselves

# Create a family size descriptor from SibSp and Parch

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.factorplot(x="Fsize",y="Survived",data = dataset)

g = g.set_ylabels("Survival Probability")
# Lets break these into 4 categories

# Create new feature of family size

dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
# We've essentially turned family size into a binary value, since theres no clear smooth correlation`

g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")
# We're gong to convert Title and embarked to binary values (a different column for each possible title/embark point)

# dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset.head()
# What about cabin?

dataset["Cabin"].isnull().sum()

# In an old version of this, we tried this with Cabin, and in fact using it drops our score my 2 percent. Try running it and see what happens

# Can you explain why this is actually suboptimal?

"""



# That is...a lot of missing values```````````````````````````````````````````````

# Let's just replace missing values with an X, indicating no cabin listed

dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])

"""
#g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])

#g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])

#g = g.set_ylabels("Survival Probability")
#dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")

#

#dataset.head()

# Ticket may have some information, buts its very likely to be similar to fair, so lets just drop it. Let's drop passengerID as well

dataset.drop(labels = ["PassengerId", "Ticket", "Cabin"], axis = 1, inplace = True)
# Alright, time for some machine learning!

## Separate train dataset and test dataset



train = dataset[:train_len]

test = dataset[train_len:]

test.drop(labels=["Survived"],axis = 1,inplace=True)
## Separate train features and label 



train["Survived"] = train["Survived"].astype(int)



Y_train = train["Survived"]



X_train = train.drop(labels = ["Survived"],axis = 1)
# We need a cross-validator for our hyperparameter searcher

kfold = StratifiedKFold(n_splits=10)
# RFC Parameters tunning 

RFC = RandomForestClassifier()



## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,Y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train,Y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,Y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(X_train,Y_train)



ada_best = gsadaDTC.best_estimator_
# Let's see how they all do

# Note: I chose these 4 classifiers for you because I already know they do well for this problem

# As an excercise, look into more than 10 classifiers and look up how to rank their cross-validation score

test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")

test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")

test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")

test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")





# Concatenate all classifier results

ensemble_results = pd.concat([test_Survived_RFC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)





g= sns.heatmap(ensemble_results.corr(),annot=True)
# The final step: We will have these 4 models vote on each possible prediction

votingC = VotingClassifier(estimators=[('rfc', RFC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)





votingC = votingC.fit(X_train, Y_train)
# Finally, let's send our submission to a CSV

test_Survived = pd.Series(votingC.predict(test), name="Survived")



results = pd.concat([IDtest,test_Survived],axis=1)



results.to_csv("TitanicSubmission.csv",index=False)