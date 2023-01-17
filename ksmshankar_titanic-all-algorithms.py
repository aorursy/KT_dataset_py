import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic_df = pd.read_csv("../input//train.csv")

test_df = pd.read_csv("../input/test.csv")

print(titanic_df.head())
print(test_df.head())
print(titanic_df.info())

print("....................................")

print(test_df.info())
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'],axis=1)

test_df = test_df.drop(['Name','Ticket'],axis=1)
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



# Either to consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.



# OR, don't create dummy variables for Embarked column, just drop it, 

# because logically, Embarked doesn't seem to be useful in prediction.



embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



titanic_df = titanic_df.join(embark_dummies_titanic)

test_df    = test_df.join(embark_dummies_test)



titanic_df.drop(['Embarked'], axis=1,inplace=True)

test_df.drop(['Embarked'], axis=1,inplace=True)

# Fare



# only for test_df, since there is a missing "Fare" values

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



titanic_df['Fare'] = titanic_df['Fare'].astype('int')

test_df['Fare'] = test_df['Fare'].astype('int')



fare_survived = titanic_df['Fare'][titanic_df['Survived'] == 1]

fare_not_survived = titanic_df['Fare'][titanic_df['Survived'] == 0]



average_fare = pd.DataFrame([fare_survived.mean(),fare_not_survived.mean()])



std_fare = pd.DataFrame([fare_survived.std(),fare_not_survived.std()])



# plot

titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



average_fare.index.names = std_fare.index.names = ["Survived"]

average_fare.plot(yerr=std_fare,kind='bar',legend=False)

#Age 



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



# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, titanic_df['Age'].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
# Cabin

# It has a lot of NaN values, so it won't cause a remarkable impact on prediction

titanic_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)
# Family



# Instead of having two columns Parch & SibSp, 

# we can have only one column represent if the passenger had any family member aboard or not,

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1

titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0



test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



# drop Parch & SibSp

titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)



# plot

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)



# average of survived for those who had/didn't have any family member

family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)

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



X = titanic_df.drop("Survived",axis=1)

Y = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()



from sklearn import cross_validation

seed =112

X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=seed)



print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train,Y_train)



#r2 score

trainlogscore = logreg.score(X_train,Y_train)

print(trainlogscore)

Y_pred = logreg.predict(X_train)



testlogscore = logreg.score(X_val,Y_val)

print(testlogscore)



#coefficients

print(logreg.coef_)



#confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_train,Y_pred,labels=[1,0])

print(cm)



#plot cm

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

fig.colorbar(cax)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()



#train AUC

from sklearn import metrics

print(metrics.roc_auc_score(Y_train,Y_pred))

#Validation AUC

Y_val_pred = logreg.predict(X_val)

print(metrics.roc_auc_score(Y_val,Y_val_pred))



#roc surve

fpr,tpr,thresholds= metrics.roc_curve(Y_train,Y_pred)

print(fpr,tpr,thresholds)

plt.plot(fpr,tpr)

plt.show



metrics.auc(fpr,tpr)

#fscore

metrics.f1_score(Y_train,Y_pred)

#precision

metrics.precision_score(Y_train,Y_pred)



#precison,recall

pre,rec,thre = metrics.precision_recall_curve(Y_train,logreg.predict_proba(X_train)[:,0])



print(metrics.classification_report(Y_train,Y_pred))



#stats logit function

import statsmodels.discrete.discrete_model as sm

logit = sm.Logit(Y_train,X_train)

logit.fit().params

#coefficients

print(logreg.coef_)
# KNN



neighbors = np.array(np.arange(1,20,2))

for k in neighbors:

    knnmodel = KNeighborsClassifier(n_neighbors=k,n_jobs=1)

    knnmodel.fit(X_train,Y_train)

    acc = knnmodel.score(X_train,Y_train)

    acc_val = knnmodel.score(X_val,Y_val)

    print(k,acc,acc_val)

#k = 15, variance is less and accuracy = 72.3%
# DecisionTreeClassifier



deplist = np.array(np.arange(5,30,5))

for d in deplist:

    decmodel = DecisionTreeClassifier(max_depth=d)

    decmodel.fit(X_train,Y_train)

    acc = decmodel.score(X_train,Y_train)

    acc_val = decmodel.score(X_val,Y_val)

    print(d,acc,acc_val)

#best depth = 5 with 81%





decmodel.feature_importances_



X_train.columns





# #visualize tree as graph

from sklearn.tree import export_graphviz



with open("fruit_classifier.txt", "w") as f:

    f = export_graphviz(decmodel, out_file=f)



X_train.columns



# or



import graphviz

export_graphviz(decmodel, out_file="mytree.dot")

with open("mytree.dot") as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)
# Random forest

nestimators = np.array([20,40,60,80,100,150,200])

for est in nestimators:

    rfmodel = RandomForestClassifier(est)

    rfmodel.fit(X_train,Y_train)

    acc = rfmodel.score(X_train,Y_train)

    acc_val = rfmodel.score(X_val,Y_val)

    print(est,acc,acc_val)



rfmodel.feature_importances_



X_train.columns



depthlist = np.array([20,40,60,80,100])

est = 80

seed=0

for d in depthlist:

    rfmodel = RandomForestClassifier(max_depth=d,n_estimators=est)

    rfmodel.fit(X_train,Y_train)

    acc = rfmodel.score(X_train,Y_train)

    acc_val = rfmodel.score(X_val,Y_val)

    print(d,acc,acc_val)
#Extra tress Classifier

from sklearn.ensemble import ExtraTreesClassifier

depthlist = np.array([20,40,60,80,100])

est = 80

for d in depthlist:

    xtratreesmodel = ExtraTreesClassifier(max_depth=d,n_estimators=est)

    xtratreesmodel.fit(X_train,Y_train)

    acc = xtratreesmodel.score(X_train,Y_train)

    acc_val = xtratreesmodel.score(X_val,Y_val)

    print(d,acc,acc_val)

# Support vector machines



from sklearn.svm import SVC



penlist = np.array([0.001,0.01,0.1,0.5,0.1])

for i in penlist:

    svcmodel = SVC(C=i)

    svcmodel.fit(X_train,Y_train)

    acc = svcmodel.score(X_train,Y_train)

    acc_val = svcmodel.score(X_val,Y_val)

    print(i,acc,acc_val)
# Naive Bayes



from sklearn.naive_bayes import GaussianNB

naivemodel = GaussianNB()

naivemodel.fit(X_train,Y_train)

acc = naivemodel.score(X_train,Y_train)

acc_val = naivemodel.score(X_val,Y_val)

print(acc,acc_val)
# Neural Network



from sklearn.neural_network import MLPClassifier

for i in np.array([100]):

    for j in np.array([2,3,4,5,6,7,8,9]):

        nnmodel = MLPClassifier(hidden_layer_sizes=(i,j))

        nnmodel.fit(X_train,Y_train)

        acc = nnmodel.score(X_train,Y_train)

        acc_val = nnmodel.score(X_val,Y_val)

        print(i,j,acc,acc_val)
# adaboost



from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score

for i in np.array([100,200,300,400,500,600]):

    adabmaodel = AdaBoostClassifier(n_estimators=i)

    scores = cross_val_score(adabmaodel,X_train,Y_train)

    #acc = adabmaodel.score(X_train,Y_train)

    #acc_val = adabmaodel.score(X_val,Y_val)

    #print(i,acc,acc_val)

    print(i,scores.mean())
# GradientBoostingClassifier



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score

for i in np.array([100,200,300,400,500,600]):

    gradbmaodel = GradientBoostingClassifier(n_estimators=i)

    gradbmaodel.fit(X_train,Y_train)

    acc = gradbmaodel.score(X_train,Y_train)

    acc_val = gradbmaodel.score(X_val,Y_val)

    print(i,acc,acc_val)
# Stochastic Gradient classifier



from sklearn.linear_model import SGDClassifier

sgdmodel = SGDClassifier()

sgdmodel.fit(X_train,Y_train)

acc = sgdmodel.score(X_train,Y_train)

acc_val = sgdmodel.score(X_val,Y_val)

print(acc,acc_val)
# Bagging Classifier



from sklearn.ensemble import BaggingClassifier

baggingmodel = BaggingClassifier(base_estimator=RandomForestClassifier(max_depth=20,n_estimators=100),max_features=0.5,max_samples=0.5)

baggingmodel.fit(X_train,Y_train)

acc = baggingmodel.score(X_train,Y_train)

acc_val = baggingmodel.score(X_val,Y_val)

print(acc,acc_val)
# XGboost



import xgboost as xgb

for i in np.array([100,200,300,400,500]):

    for j in np.array([0.1]):

        xgbmodel = xgb.XGBClassifier(n_estimators=i,learning_rate=j)

        xgbmodel.fit(X_train,Y_train)

        acc = xgbmodel.score(X_train,Y_train)

        acc_val = xgbmodel.score(X_val,Y_val)

        print(i,acc,acc_val)





xgbmodel.feature_importances_
# LGBoost



import lightgbm as lgb

for i in np.array([100,200,300,400,500]):

    for j in np.array([0.1]):

        lgbmodel = lgb.LGBMClassifier(n_estimators=i,learning_rate=j)

        lgbmodel.fit(X_train,Y_train)

        acc = lgbmodel.score(X_train,Y_train)

        acc_val = lgbmodel.score(X_val,Y_val)

        print(i,acc,acc_val)



#importnat features 

print("\n")

print(lgbmodel.feature_importances_)

X_train.columns