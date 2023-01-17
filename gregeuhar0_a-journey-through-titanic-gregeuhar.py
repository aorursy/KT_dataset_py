## Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
## get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

titanic_df.tail()
## Plot Functions for data exploration



# Low cardinality X

def plots_catY_catX(df,variable,column_Y):

	# plot

	sns.factorplot(variable,column_Y, data=df,size=4,aspect=3)

	fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

	sns.countplot(x=variable, data=df, ax=axis1)

	sns.countplot(x=column_Y, hue=variable, data=df, order=[1,0], ax=axis2)



	var_perc = df[[variable, column_Y]].groupby([variable],as_index=False).mean()

	sns.barplot(x=variable, y=column_Y, data=var_perc,ax=axis3)



	return None



# High cardinality X

def plots_catY_binnedX(df,variable,column_Y):

	# Histogram per outcome

	variable_false = df[variable][df[column_Y] == 0]

	variable_true    = df[variable][df[column_Y] == 1]

	avg_variable = DataFrame([variable_false.mean(), variable_true.mean()])

	std_variable      = DataFrame([variable_false.std(), variable_true.std()])



	df[variable].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,100))



	avg_variable.index.names = std_variable.index.names = [column_Y]

	avg_variable.plot(yerr=std_variable,kind='bar',legend=False)





	# Curves

	facet = sns.FacetGrid(df, hue=column_Y,aspect=4)

	facet.map(sns.kdeplot,variable,shade= True)

	facet.set(xlim=(0, df[variable].max()))

	facet.add_legend()



	# Histogram per variable

	fig, axis1 = plt.subplots(1,1,figsize=(18,4))

	average = df[[variable, column_Y]].groupby([variable],as_index=False).mean()

	sns.barplot(x=variable, y=column_Y, data=average)



	return None



# 2 Xs relationship

def plot_catY_catsXs_stacked_bar(df,variable_1,variable_2,column_Y):

    ## Stacked bars

    cross_tab = pd.crosstab(df[variable_1], df[variable_2])

    dummy = cross_tab.div(cross_tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

    return None

    

def plot_catY_catsXs_violin(df,variable_1,variable_2,column_Y):

    ## Violin

    sns.violinplot(x=variable_1, y=variable_2, hue=column_Y, data=df, split=True)

    return None
def fill_na(df,variable,group_by_column=''):

	if group_by_column!='':

		med = df.groupby(group_by_column)[variable].transform('median')

		df[variable]=df[variable].fillna(med)

	else:

		df[variable]=df[variable].fillna(df[variable].median())



	return df[variable]
## Feature engineering Training

# Cabin First Letter

titanic_df['Cabin']=titanic_df['Cabin'].astype(str)

titanic_df['cabin_flag']=titanic_df.Cabin.apply(lambda x:1 if len(x)>0 and x[0] in ('B','C','D','E','F') else 0)



# Family Size

titanic_df['family_size'] =  titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df['alone']=titanic_df.apply(lambda x:1 if x['family_size']==0 else 0,axis=1)

titanic_df['oversized_flag']=titanic_df.apply(lambda x:1 if x['family_size']>=4 else 0,axis=1)



# Ticket Shared

shared_tickets=pd.DataFrame(titanic_df.groupby('Ticket').Survived.count())

shared_tickets.columns=['Shared_tickets']

shared_tickets['Ticket']=shared_tickets.index

titanic_df=pd.merge(titanic_df, shared_tickets, on='Ticket', how='left')



# Age

titanic_df['Age']=fill_na(titanic_df,'Age')

titanic_df['Age']=titanic_df['Age'].astype(int)



# Sex

titanic_df['female_flag']=titanic_df.apply(lambda x:1 if x['Sex']=='female' else 0,axis=1)



## Drops unncessary columns

titanic_df=titanic_df.drop(['PassengerId','Name','Cabin','Embarked','Ticket','Sex','family_size','SibSp','Parch'], axis=1)
## Feature engineering Test

# Cabin First Letter

test_df['Cabin']=test_df['Cabin'].astype(str)

test_df['cabin_flag']=test_df.Cabin.apply(lambda x:1 if len(x)>0 and x[0] in ('B','C','D','E','F') else 0)



# Family Size

test_df['family_size'] =  test_df["Parch"] + test_df["SibSp"]

test_df['alone']=test_df.apply(lambda x:1 if x['family_size']==0 else 0,axis=1)

test_df['oversized_flag']=test_df.apply(lambda x:1 if x['family_size']>=4 else 0,axis=1)



# Ticket Shared

shared_tickets=pd.DataFrame(test_df.groupby('Ticket').Ticket.count())

shared_tickets.columns=['Shared_tickets']

shared_tickets['Ticket']=shared_tickets.index

test_df=pd.merge(test_df, shared_tickets, on='Ticket', how='left')



# Age

test_df['Age']=fill_na(test_df,'Age')

test_df['Age']=test_df['Age'].astype(int)



# Fare

test_df['Fare']=fill_na(test_df,'Fare')



# Sex

test_df['female_flag']=test_df.apply(lambda x:1 if x['Sex']=='female' else 0,axis=1)



## Drops unncessary columns

test_df=test_df.drop(['Name','Cabin','Embarked','Ticket','Sex','family_size','SibSp','Parch'], axis=1)
titanic_df.info()

print("----------------------------")

test_df.info()
# define training and testing sets

X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test = test_df.drop("PassengerId",axis=1)

#X_train, X_test, y_train, y_test = train_test_split(titanic_df.drop("Survived",axis=1),

#                                                    Y_train,

#                                                    test_size=0.2,

#                                                    random_state=0)



X_train_scaled=(X_train-X_train.min())/(X_train.max()-X_train.min())

X_test_scaled=(X_test-X_test.min())/(X_test.max()-X_test.min())

# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train_scaled, Y_train)



Y_pred_logreg = logreg.predict(X_test)



logreg.score(X_train_scaled, Y_train)



scores = cross_val_score(logreg, 

                         titanic_df.drop("Survived",axis=1),

                         Y_train,

                         cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Support Vector Machines



svc = SVC()



svc.fit(X_train_scaled, Y_train)



Y_pred = svc.predict(X_test)



svc.score(X_train_scaled, Y_train)



scores = cross_val_score(svc, 

                         titanic_df.drop("Survived",axis=1),

                         Y_train,

                         cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred_forest = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



scores = cross_val_score(random_forest, 

                         titanic_df.drop("Survived",axis=1),

                         Y_train,

                         cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
knn = KNeighborsClassifier(n_neighbors = 3)



knn.fit(X_train, Y_train)



Y_pred = knn.predict(X_test)



knn.score(X_train, Y_train)



scores = cross_val_score(knn, 

                         titanic_df.drop("Survived",axis=1),

                         Y_train,

                         cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Gaussian Naive Bayes



gaussian = GaussianNB()



gaussian.fit(X_train, Y_train)



Y_pred = gaussian.predict(X_test)



gaussian.score(X_train, Y_train)



scores = cross_val_score(gaussian, 

                         titanic_df.drop("Survived",axis=1),

                         Y_train,

                         cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = DataFrame(titanic_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred_logreg

    })

submission.to_csv('titanic.csv', index=False)