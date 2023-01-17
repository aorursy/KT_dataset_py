# Import necessaries, read files.

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Load the dataset

try:

    train_data = pd.read_csv("../input/train.csv")

    train_label = train_data['Survived']

    train_data.drop(['Survived'], axis = 1, inplace = True)

    concat_set = pd.concat((train_data, pd.read_csv("../input/test.csv"))).reset_index()

    concat_set.drop(['PassengerId'], axis = 1, inplace = True) 

    print("Dataset has been loaded.")

except:

    print("Dataset could not be loaded. Is the dataset missing?")
# Dictionary for title mapping using names (Thanks to Ahmed BESBES).

Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"

                    }
# Data cleaning, by random generating, filling medians for missing rows, etc.

age_avg 	   = concat_set['Age'].mean()

age_std 	   = concat_set['Age'].std()

age_null_count = concat_set['Age'].isnull().sum()



age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

concat_set['Age'][np.isnan(concat_set['Age'])] = age_null_random_list

concat_set['Age'] = concat_set['Age'].astype(int)    

concat_set['CategoricalAge'] = pd.cut(concat_set['Age'], 5)



concat_set['Fare'].fillna((concat_set['Fare'].median()), inplace=True)

concat_set['Embarked'].fillna('S', inplace=True)

concat_set['Title'] = concat_set['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

concat_set['Title'] = concat_set.Title.map(Title_Dictionary)

concat_set['FamilySize'] = concat_set['SibSp'] + concat_set['Parch'] + 1

concat_set['IsAlone'] = 0

concat_set.loc[concat_set['FamilySize'] == 1, 'IsAlone'] = 1

concat_set['CategoricalFare'] = pd.qcut(concat_set['Fare'], 4)



# Feature Dropping

drop_elements = ['index','Name', 'Ticket', 'Cabin', 'SibSp','FamilySize', 'Parch', 'Fare', 'Age']

concat_set.drop(drop_elements, axis = 1, inplace=True)



# Let's make it all numerical values (it's just my preference)

for feature in concat_set.keys():

    concat_set[feature] = pd.Categorical(concat_set[feature]).codes 
# This part is for classifier comparison (for model selection)

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log      = pd.DataFrame(columns=log_cols)





sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X, y = concat_set[:891].values, train_label.values



acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="g")
# splitting the train & test data set

train_X, train_Y = concat_set[:891], train_label

test_X = concat_set[891:]
# picked the best performing classifier.

logreg = SVC(probability=True)

logreg.fit(train_X, train_Y)
# performance check

from sklearn.model_selection import cross_val_score

cross_val_score(logreg, train_X, train_Y, cv=10).mean()
# write a result file

f = open('SVC_DataCleaning_FeatureTransformation_Numericalized.csv', 'w')

f.write('PassengerId,Survived\n')

for i, v in enumerate(logreg.predict(test_X)):

    f.write(str(i+892)+','+str(v)+'\n')

f.close()