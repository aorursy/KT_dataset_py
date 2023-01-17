# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


from sklearn.neural_network import MLPClassifier
# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
titanic_df.head()
titanic_df.info()
print("----------------------------")
test_df.info()
# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId'], axis=1)
#test_df    = test_df.drop(['Ticket'], axis=1)
#Name
titanic_df_title = [i.split(",")[1].split(".")[0].strip() for i in titanic_df["Name"]]
test_df_title = [i.split(",")[1].split(".")[0].strip() for i in test_df["Name"]]

titanic_df["Title"] = pd.Series(titanic_df_title)
test_df["Title"] = pd.Series(test_df_title)

titanic_df.head()
g = sns.countplot(x="Title",data=titanic_df)
# easy to read
g = plt.setp(g.get_xticklabels(), rotation=45)
titanic_df["Title"] = titanic_df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df["Title"] = test_df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

titanic_df["Title"] = titanic_df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
test_df["Title"] = test_df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

titanic_df["Title"] = titanic_df["Title"].astype(int)
test_df["Title"] = test_df["Title"].astype(int)
g = sns.countplot(titanic_df["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=titanic_df,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")
titanic_df.head()
# convert to indicator values Title
titanic_df = pd.get_dummies(titanic_df, columns = ["Title"])
test_df = pd.get_dummies(test_df, columns = ["Title"])
titanic_df.head()
# Drop Name variable
titanic_df.drop(labels = ["Name"], axis = 1, inplace = True)
test_df.drop(labels = ["Name"], axis = 1, inplace = True)
titanic_df.head()
test_df.head()
# Embarked

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# plot
sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)
# Fare

# only for test_df, since there is a missing "Fare" values
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
# Age 

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# axis3.set_title('Original Age values - Test')
# axis4.set_title('New Age values - Test')

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)
        
# plot new Age Values
titanic_df['Age'].hist(bins=70, ax=axis2)
# test_df['Age'].hist(bins=70, ax=axis4)
# .... continue with plot Age column

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
# Cabin&Ticket
titanic_len = len(titanic_df)
all_dataset =  pd.concat(objs=[titanic_df, test_df], axis=0).reset_index(drop=True)
# Replace the Cabin number by the type of cabin 'X' if not
all_dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in all_dataset['Cabin'] ])
Ticket = []
for i in list(all_dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
all_dataset["Ticket"] = Ticket



#g = sns.countplot(all_dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
#g = sns.factorplot(y="Survived",x="Cabin",data=all_dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
#g = g.set_ylabels("Survival Probability")

all_dataset = pd.get_dummies(all_dataset, columns = ["Cabin"],prefix="Cabin")
all_dataset = pd.get_dummies(all_dataset, columns = ["Ticket"], prefix="T")

titanic_df = all_dataset[:titanic_len]
titanic_df = titanic_df.drop(["PassengerId"],axis=1)
titanic_df["Survived"] = titanic_df["Survived"].astype(int)

test_df = all_dataset[titanic_len:]
test_df = test_df.drop(["Survived"],axis=1)
test_df["PassengerId"] = test_df["PassengerId"].astype(int)

# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]+1
test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]+1

# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

g = sns.factorplot(x="Family",y="Survived",data = titanic_df)
g = g.set_ylabels("Survival Probability")


# Create new feature of family size
titanic_df['Single'] = titanic_df['Family'].map(lambda s: 1 if s == 1 else 0)
titanic_df['SmallF'] = titanic_df['Family'].map(lambda s: 1 if  s == 2  else 0)
titanic_df['MedF'] = titanic_df['Family'].map(lambda s: 1 if 3 <= s <= 4 else 0)
titanic_df['LargeF'] = titanic_df['Family'].map(lambda s: 1 if s >= 5 else 0)

test_df['Single'] = test_df['Family'].map(lambda s: 1 if s == 1 else 0)
test_df['SmallF'] = test_df['Family'].map(lambda s: 1 if  s == 2  else 0)
test_df['MedF'] = test_df['Family'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test_df['LargeF'] = test_df['Family'].map(lambda s: 1 if s >= 5 else 0)


g = sns.factorplot(x="Single",y="Survived",data=titanic_df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=titanic_df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=titanic_df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=titanic_df,kind="bar")
g = g.set_ylabels("Survival Probability")
# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=titanic_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)
# Pclass

# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])
sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)
# define training and testing sets

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()
titanic_df.info()
print("----------------------------")
test_df.info()
# Logistic Regression
#class_weight ='balanced',
logreg = LogisticRegression(penalty='l2',solver='liblinear',multi_class='ovr')
logreg.fit(X_train, Y_train)
Y_pred_logreg = logreg.predict(X_test)
logreg.score(X_train, Y_train)
#GradientBoosting
GradientBoostingTree = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0).fit(X_train, Y_train)
GradientBoostingTree_score=GradientBoostingTree.score(X_train, Y_train)
GradientBoostingTree_score
# Random Forests
#数据集比较简单，模型较为复杂，设置max_depth和min_samples_split参数，防止过拟合
random_forest = RandomForestClassifier(n_estimators=100,max_features=9,max_depth = 6, min_samples_split=20)
random_forest.fit(X_train, Y_train)
Y_pred_random_forest = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
#SVC
SVC_Model = SVC(C=2.5,cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
SVC_Model.fit(X_train, Y_train)
SVC_Model_score=SVC_Model.score(X_train, Y_train)
SVC_Model_score
#MLP
#MLP_model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)
#MLP_model = MLPClassifier(activation='relu', solver='lbfgs', alpha=0.0001)
#MLP_model.fit(X_train, Y_train)
#MLP_model_score=SVC_Model.score(X_train, Y_train)
#MLP_model_score
#voting_final = VotingClassifier(estimators=[('GB', GradientBoostingTree), ('RF', random_forest),('LR',logreg),('SVC',SVC_Model),('MLP',MLP_model)], voting='hard', n_jobs=1)
voting_final = VotingClassifier(estimators=[('GB', GradientBoostingTree), ('RF', random_forest),('LR',logreg),('SVC',SVC_Model)], voting='hard', n_jobs=1)

voting_final = voting_final.fit(X_train, Y_train)
votingY_pred = voting_final.predict(X_test)
voting_final.score(X_train, Y_train)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": votingY_pred
    })
submission.to_csv('titanic.csv', index=False)