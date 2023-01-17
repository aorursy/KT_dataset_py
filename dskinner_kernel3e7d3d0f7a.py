# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#!/usr/bin/python

# -*- coding: utf-8 -*- 

# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib import style

#%matplotlib inline



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score





def split_name(name):

	last_name, remainder = name.split(', ')

	title = remainder.split('.')[0]



	return pd.Series({

	    'Title': title,

	    'Surname': last_name

	})





def getTitleField(df):

	df_new = df['Name'].apply(split_name)

	df = pd.concat([df,df_new],axis=1)

	return df



def split_age(age):

	age_u_11 = 0

	age_11_18 = 0 

	age_18_22 = 0

	age_22_27 = 0

	age_27_33 = 0

	age_33_40 = 0

	age_40_50 = 0

	age_50_66 = 0

	age_o_66 = 0

	if age<=11: age_u_11=1

	if age>11&age<=18: age_11_18=1

	if age>18&age<=22: age_18_22=1

	if age>22&age<=27: age_22_27=1

	if age>27&age<=33: age_27_33=1

	if age>33&age<=40: age_33_40=1

	if age>40&age<=50: age_40_50=1

	if age>50&age<=66: age_50_66=1

	if age>66: age_o_66=1



	return pd.Series({

	    'age_u_11': age_u_11,

	    'age_11_18': age_11_18,

	    'age_18_22': age_18_22,

	    'age_22_27': age_22_27,

	    'age_27_33': age_27_33,

	    'age_33_40': age_33_40,

	    'age_40_50': age_40_50,

	    'age_50_66': age_50_66,

	    'age_o_66': age_o_66

	})



def getAgeFields(df):

	df_new = df['Age'].apply(split_age)

	df = pd.concat([df,df_new],axis=1)

	df = df.drop(['Age'], axis=1)

	return df



def split_Pclass(pClass):

	first = 0

	second = 0 

	third = 0

	if pClass==1: first=1

	if pClass==2: second=1

	if pClass==3: third=1



	return pd.Series({

	    'firstClass': first,

	    'secondClass': second,

	    'thirdClass': third

	})



def getPclassFields(df):

	df_new = df['Pclass'].apply(split_Pclass)

	df = pd.concat([df,df_new],axis=1)

	df = df.drop(['Pclass'], axis=1)

	return df



def split_gender(gender):

	male = 0

	female = 0

	if gender == "male": male=1

	if gender == "female": female=1

	return pd.Series({

	    'Male': male,

	    'Female': female

	})	



def getGenderFields(df):

	df_new = df['Sex'].apply(split_gender)

	df = pd.concat([df,df_new],axis=1)

	df = df.drop(['Sex'], axis=1)

	return df



def split_embarked(embarked):

	southampton = 0

	cherbourg = 0 

	queenstown = 0

	if embarked=="S": southampton=1

	if embarked=="C": cherbourg=1

	if embarked=="Q": queenstown=1



	return pd.Series({

	    'EmbarkSouthampton': southampton,

	    'EmbarkCherbourg': cherbourg,

	    'EmbarkQueenstown': queenstown

	})



def getEmbarkedFields(df):

	common_value = 'S'

	df['Embarked'] = df['Embarked'].fillna(common_value)

	df_new = df['Embarked'].apply(split_embarked)

	df = pd.concat([df,df_new],axis=1)

	df = df.drop(['Embarked'], axis=1)

	return df



def getNotAloneField(df):

	df['relatives'] = df['SibSp'] + df['Parch']

	df.loc[df['relatives'] > 0, 'NotAlone'] = 0

	df.loc[df['relatives'] == 0, 'NotAlone'] = 1

	df = df.drop(['relatives'], axis=1)

	df = df.drop(['SibSp'], axis=1)

	df = df.drop(['Parch'], axis=1)

	df['NotAlone'] = df['NotAlone'].astype(int)

	return df





def fillAges(df):

	titles = ["Miss", "Mrs", "Master", "Mr"]



	for title in titles:

		title_df = df[df['Title']==title]

		other_df = df[df['Title']!=title]



		mean = title_df.Age.mean()

		std = title_df.Age.std()

		is_null =title_df.Age.isnull().sum()

		#print ("here")

		# compute random numbers between the mean, std and is_null

		rand_age = np.random.randint(mean - std, mean + std, size = is_null)

		# fill NaN values in Age column with random values generated



		age_slice = title_df.Age.copy()

		age_slice[np.isnan(age_slice)] = rand_age

		title_df.Age = age_slice

		title_df.Age = title_df["Age"].astype(int)



		df = pd.concat([title_df,other_df])



	mean = df["Age"].mean()

	std = df["Age"].std()

	is_null = df["Age"].isnull().sum()

	# compute random numbers between the mean, std and is_null

	rand_age = np.random.randint(mean - std, mean + std, size = is_null)

	# fill NaN values in Age column with random values generated

	age_slice = df["Age"].copy()

	age_slice[np.isnan(age_slice)] = rand_age

	df["Age"] = age_slice

	df["Age"] = df["Age"].astype(int)



	return df



def dropSurplusFields(df,fields):

	for field in fields:

		df = df.drop([field], axis=1)

	return df







#def preprocessData(df):



def getRandomForest(X_train, 

					Y_train,

					X_test):

	random_forest = RandomForestClassifier(n_estimators=100)

	random_forest.fit(X_train, Y_train)

	Y_prediction = random_forest.predict(X_test)



	random_forest.score(X_train, Y_train)

	acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

	print ("Random Forest = ", acc_random_forest)

	scores = cross_val_score(random_forest, X_train, Y_train, cv=10, scoring = "accuracy")

	print("Scores:", scores)

	print("Mean:", scores.mean())

	print("Standard Deviation:", scores.std())

	print ("**********************************************")





	importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

	importances = importances.sort_values('importance',ascending=False).set_index('feature')

	#print (importances.head(15))

	rf_data = pd.read_csv('../input/titanic/test.csv')

	rf_data.insert((rf_data.shape[1]),'Survived',Y_prediction)

	rf_data = dropSurplusFields(rf_data,

								['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

	rf_data.to_csv('RandomForest_submission.csv', index = False)



def getLogReg(X_train, 

					Y_train,

					X_test):

	logreg = LogisticRegression()

	logreg.fit(X_train, Y_train)



	Y_pred = logreg.predict(X_test)



	acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



	print ("Log Reg = ",acc_log)

	scores = cross_val_score(logreg, X_train, Y_train, cv=10, scoring = "accuracy")

	print("Scores:", scores)

	print("Mean:", scores.mean())

	print("Standard Deviation:", scores.std())

	print ("**********************************************")



def getDecisionTree(X_train, 

					Y_train,

					X_test):

	decision_tree = DecisionTreeClassifier() 

	decision_tree.fit(X_train, Y_train)  

	Y_pred = decision_tree.predict(X_test)  

	acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

	print ("Decision Tree = ",acc_decision_tree)

	scores = cross_val_score(decision_tree, X_train, Y_train, cv=10, scoring = "accuracy")

	print("Scores:", scores)

	print("Mean:", scores.mean())

	print("Standard Deviation:", scores.std())

	print ("**********************************************")



def getKNN(X_train, 

					Y_train,

					X_test):

	knn = KNeighborsClassifier(n_neighbors = 3) 

	knn.fit(X_train, Y_train)  

	Y_pred = knn.predict(X_test)  

	acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

	print ("KNN = ",acc_knn)

	scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring = "accuracy")

	print("Scores:", scores)

	print("Mean:", scores.mean())

	print("Standard Deviation:", scores.std())

	print ("**********************************************")



def getSGD(X_train, 

					Y_train,

					X_test):

	sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

	sgd.fit(X_train, Y_train)

	Y_pred = sgd.predict(X_test)



	sgd.score(X_train, Y_train)



	acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

	print ("SGD = ",acc_sgd)

	scores = cross_val_score(sgd, X_train, Y_train, cv=10, scoring = "accuracy")

	print("Scores:", scores)

	print("Mean:", scores.mean())

	print("Standard Deviation:", scores.std())

	print ("**********************************************")



def getGNB(X_train, 

					Y_train,

					X_test):

	gaussian = GaussianNB() 

	gaussian.fit(X_train, Y_train)  

	Y_pred = gaussian.predict(X_test)  

	acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

	print ("GNB = ",acc_gaussian)

	scores = cross_val_score(gaussian, X_train, Y_train, cv=10, scoring = "accuracy")

	print("Scores:", scores)

	print("Mean:", scores.mean())

	print("Standard Deviation:", scores.std())

	print ("**********************************************")

	gnb_data = pd.read_csv('../input/titanic/test.csv')

	gnb_data.insert((gnb_data.shape[1]),'Survived',Y_pred)

	gnb_data = dropSurplusFields(gnb_data,

								['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

	gnb_data.to_csv('gnb_submission.csv', index = False)







test_df = pd.read_csv("../input/titanic/test.csv")

train_df = pd.read_csv("../input/titanic/train.csv")	



'''print ("train_df.info()")

train_df.info()

print ("********************************************************")

print ("train_df.describe()")

print ( train_df.describe())

print ( "********************************************************")

print ("train_df.head(8)")



print (train_df.head(8))

print ("********************************************************")

print ("missing_data.head(6)")



total = train_df.isnull().sum().sort_values(ascending=False)

print (total)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

#percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_1], axis=1, keys=['Total', '%'])

print (missing_data.head(6))

print ("********************************************************")'''





# df_new has the new columns

train_df = getTitleField(train_df)

test_df = getTitleField(test_df)

print (train_df.head(8))



'''print ("Total Misss = ",train_df[train_df['Title']=="Miss"].Fare.count())

print ("Total Misss with age = ",train_df[train_df['Title']=="Miss"].Age.dropna().count())

print ("Total Mr's = ",train_df[train_df['Title']=="Mrs"].Fare.count())

print ("Total Mr's with age = ",train_df[train_df['Title']=="Mrs"].Age.dropna().count())



print (train_df[train_df['Title']=="Miss"].Age.mean())

print (train_df[train_df['Title']=="Mrs"].Age.mean())'''



train_df = fillAges(train_df)

test_df = fillAges(test_df)





'''total = train_df.isnull().sum().sort_values(ascending=False)

print (total)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

#percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_1], axis=1, keys=['Total', '%'])

print (missing_data.head(6))

print ("********************************************************")'''



'''survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

master = train_df[train_df['Title']=='Miss']

mister = train_df[train_df['Title']=='Mrs']

ax = sns.distplot(master[master['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(master[master['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Miss')

ax = sns.distplot(mister[mister['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(mister[mister['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Mrs')

plt.show()'''



#title_counts = train_df['Title'].value_counts()

#print (title_counts)



#cabin_counts = train_df['Cabin'].value_counts()

#print (cabin_counts)



train_df = getPclassFields(train_df)

test_df = getPclassFields(test_df)



train_df = getGenderFields(train_df)

test_df = getGenderFields(test_df)



train_df = getEmbarkedFields(train_df)

test_df = getEmbarkedFields(test_df)



train_df = getNotAloneField(train_df)

test_df = getNotAloneField(test_df)



#train_df = getAgeFields(train_df)

#test_df = getAgeFields(test_df)

train_df['Age'] = train_df['Age']/80

test_df['Age'] = test_df['Age']/80



dropped_fields = ['Name',

				  'Ticket',

				  'Fare',

				  'Cabin',

				  'Surname',

				  'Title',

				  'PassengerId']

train_df = dropSurplusFields(train_df,dropped_fields)

test_df = dropSurplusFields(test_df,dropped_fields)



'''train_numerical_features = list(train_df.select_dtypes(include=['int64', 'float64', 'int32']).columns)

ss_scaler = StandardScaler()

train_df_ss = pd.DataFrame(data = train_df)

train_df_ss[train_numerical_features] = ss_scaler.fit_transform(train_df_ss[train_numerical_features])'''



X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df#.drop("PassengerId", axis=1).copy()



print (train_df.head(8))



train_df.info()



getDecisionTree(X_train, 

				Y_train,

				X_test)



getRandomForest(X_train, 

					Y_train,

					X_test)



getLogReg(X_train, 

				Y_train,

				X_test)



getKNN(X_train, 

				Y_train,

				X_test)





getSGD(X_train, 

				Y_train,

				X_test)



getGNB(X_train, 

				Y_train,

				X_test)

	


