"""

Created on Sun Mar 24 14:34:32 2019



@author: joshuapeterson

"""



#Define the problem

## On April 15, 1912, during her maiden voyage, the Titanic sank after

## colliding with an iceberg killing 67.5% of the people on board or 1502 out

## of 2224.



## One of the reasons for the loass of life was due to the fact that there

## were not enough lifeboats on board.



## There was some element of luck involved however, most of the survivors

## appeared to be women, children and the upper-class patrons



# Tools necessary for data analysis and wrangling



import pandas as pd

import numpy as np

import random as rnd



# Tools necessary for visualization



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.plotly as py



# Tools necessary for machine learning



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier





# Import data in to the Python environment



train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

combine = [train_df, test_df]



# The data dictionary is as follows



## survival = 	Survival, response is binary 	0 = No, 1 = Yes

## pclass = 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd

## sex = 	Sex	

## Age = 	Age in years	

## sibsp = 	# of siblings / spouses aboard the Titanic	

## parch = 	# of parents / children aboard the Titanic	

## ticket = 	Ticket number	

## fare = 	Passenger fare	

## cabin = 	Cabin number	

## embarked = 	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton







# I next want to check out the features of the data set



print(train_df.columns.values)



# Categorical features include: Survived, Sex, and Embarked.  Ordinal features

# include Pclass



# Continuous numerical features include Age, Fare while discrete numerical

# features include SibSP and Parch



# I next want to get a preview of the data by looking at the head and tail



train_df.head()

train_df.tail()



# There are data that contain null or empty values that will need addressing

## Cabin, Age & Embarked contain null values in the training data set

## Cabin & Age are incomplete in the test data set



# Features of the data set

## (2) are float, (5) are int and (5) are string objects



# Distribution of numerical values across the sample

## We do this to determine how representative the training data set is compared

## to the actual problem



## Total samples are 891 or 40% of the actual number of passengers on board the

## Titanic (2224)



train_df.info()

print("-"*40)

test_df.info()





train_df.describe()



## Approximately 38.3% of the passengers survived compared to the overall rate

## of 32.5%

## Largely, the passengers where 3rd class looking at the 75th percentile.  

## This is also indicative of the fare cost in the 75th percentile ($31)

## Most people did not travel with a parent or child (0 at 75th percentile)

## Most people either traveled with 1 sibling or spouse (at 75th percentile)



train_df.describe(include=['O'])



# Next I would like to analyze how each independent variable correlates with

# the dependent variable survival.



# The following variables should be analyzed given the completness and the 

# assumption of correlation between the variables



## pclass

## sex

## Age

## sibsp

## parch

## fare

## embarked



# The first correlation analysis will be by the variable pclass



class_cor = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False

        ).mean().sort_values(by='Survived', ascending=False)

class_cor



## From this analysis, we can see that there is a high correlation between

## passenger class and survival rate as appprox. 63% of 1st class passengers

## survived



# The next correlation analysis will be the variable sex



survive_cor = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False

        ).mean().sort_values(by='Survived', ascending=False)

survive_cor



## Here again, we see a high correlation between sex and survival as 74%

## of the survivors were female



# Next, we will analysis Age.  



age_cor = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False

        ).mean().sort_values(by='Survived', ascending=False)

age_cor



## There appears to be a higher correlation between younger passengers.  This

## will be better observed using a histogram



# Next, sibsp



sibsp_cor = train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False

        ).mean().sort_values(by='Survived', ascending=False)

sibsp_cor



## Interestingly, there is a higher correlation of survival if the person was

## accompanied by a sibling or spouse



# Next, parch



parch_cor = train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False

        ).mean().sort_values(by='Survived', ascending=False)

parch_cor



## This is an interesting data point as those who had traveled with between 1

## and 3 parents or children had a 50% or better rate of survival.  Anything

## less or more resulted in a low survival rate



# Next, fare



fare_cor = train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False

        ).mean().sort_values(by='Survived', ascending=False)

fare_cor



## This variable seems to be a little more scattered.  Once again, the story

## would be better told through a histogram or scatter chart



# Next, embarked



emb_cor = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False

        ).mean().sort_values(by='Survived', ascending=False)

emb_cor





# Next I will create a histogram of the chosen variables, age and fare



# Age variable



## This histogram plots out the binary outcome, survived = 0 and survived = 1

## against age.  The non-survivor plot is heavily skewed to the right exhibiting

## positive skew.  This suggests a higher number of younger people did not

## survive that fateful night.  The survivor plot also exhbits some positive

## skewness but appears to be more platykurtic.  



age = sns.FacetGrid(train_df, col="Survived")

age.map(plt.hist, "Age", bins=20)



## I've created a grid looking at survival based on age vs. pclass



grid_age = sns.FacetGrid(train_df, col="Survived", row="Pclass", size=2.2, aspect=1.6)

grid_age.map(plt.hist, "Age", alpha=.5, bins=20)

grid_age.add_legend();



## This is an interesting look at the survival rate based on age and class.  The

## first distribution curve that stands out is the p-class 3 non-survivor.

## The curve is positive skewing suggesting that if you are young and 3rd class   

## your chances of survival are less compared to the pclass=1 curve which

## appears flatter and more platykurtic.  If you happened to be younger

## AND 1st class, your chances of survival were greater than if you were

## 2nd or 3rd class.



# Fare variable



fare = sns.FacetGrid(train_df, col="Survived")

fare.map(plt.hist, "Fare", bins=20)



## Here we see both heavily, positive skewing curves for non-survived and

## survived classes when it comes to fare.  You will see that if you paid

## a cheaper fare, or assumed to be 2nd or 3rd class, you had a greater chance

## of not surviving.  However, the same is true for those who survived.



## I've created a grid looking at fare vs. pclass



grid_fare = sns.FacetGrid(train_df, col="Survived", row="Pclass", size=2.2, aspect=1.6)

grid_fare.map(plt.hist, "Fare", alpha=.5, bins=20)

grid_fare.add_legend();



## Once again this breakdown supports the higher instances of lower fare 

## survival rate which correlates to age.



# Data Wrangling



# Next we want to drop un-needed variables



## We will drop Cabin and Ticket



print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape



# Next we want to see if there is a correlation with Title and survival.



for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])  



# Next we can replace titles that are less common



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()  



# Convert categorical titles to ordinal



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()



# Drop the name feature



train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape



# Convert features which contain strings to numerical values.



for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()



# We need to now estimate and complete features with mising or null values.



## We can either generate randome numbers.

## Or we could guess values based off of correlated features.  We note that

## there is a correlation between Age, Gender and Pclass.

## We can combine the two methods using randome number between the mean and

## standard deviation



grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()



# We will fix the empty array cells by guessing age values based on pclass

# and gender combinations.



guess_ages = np.zeros((2,3))

guess_ages



#Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed 

#values of Age for the six combinations.



for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()



# Age bands to determine correlations with survived



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)  



# Next we will replace age with ordinals based on the bands



for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()



train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()



# Next I want to create a new variable by combining existing variables



## We can create a new variable called FamilySize by combining SibSp and Parch.



for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



# Next we can create another variable called IsAlone



for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()



# We are going to replace IsAlone with SibSp, Parch and FamilySize



for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

    

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].dropna().head(10)



# Next, we shall fill in the missing values for point of embarkment by

# filling out the most common occurence



freq_port = train_df.Embarked.dropna().mode()[0]

freq_port    



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



# And then convert the variable from a character to numeric feature



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()



# We will now do the same thing for the Fare variable



test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()



# We will now create the Fare band



train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',ascending=True)



# We next want to convert the FareBand to ordinal values



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

# Here is the train data set



train_df = train_df.dropna()

train_df.head(10)



# Here is the test data set



test_df = test_df

test_df.head(10)



# Next we are going to use the following algorithms and/or packages to determine

# the optimal model for prediction.  These are



## 1) Logistic regression

## 2) KNN

## 3) SVM

## 4) Naive Bayes

## 5) Decision trees

## 6) Random forest

## 7) Perceptron

## 8) Stochastic Gradient Descent

## 9) Linear SVC



X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape



# 1) Logistic regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) *100, 2)

acc_log



## The logistic regression algorithm returns a score of 81.09



# Next, I will check the coefficients of each of the variables



coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)



## Sex has the highest positive correlation followed by age, two very good

## variables for prediction.



## Age*Class has the most inversely correlated variable followed by IsAlone

## Once again, these are two very good variables to use for prediction



# 2) KNN



knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn



## The k-nearest neighbors algorithm returns a score of 80.81





# 3) SVM (Support Vector Machines)



svm = SVC()

svm.fit(X_train, Y_train)

Y_pred = svm.predict(X_test)

acc_svm = round(svm.score(X_train, Y_train) *100, 2)

acc_svm



## The SVM algorithm returns a score of 84.45



# 4) Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian



## The Gaussian Naive Bayes algorithm returns a score of 72.83



# 5) Decision Trees



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree



## The Decision Trees model returns a score of 89.64



# 6) Random Forest



random_forest = RandomForestClassifier()

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest



## The Random Forest algorithm returns a score of 89.5



# 7) Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron



## The Perceptron model returns a score of 73.81



# 8) Stochastic Gradient Descent



sgd =  SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd



## The SGD Classifier model returns a score of 78.99



# 9) Linear SVC



svc = LinearSVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc



## The Linear SVC model returns a score of 81.51



# 10) Gradient Boosting Classifier



gbc = GradientBoostingClassifier()

gbc.fit(X_train, Y_train)

Y_pred = gbc.predict(X_test)

acc_gbc = round(gbc.score(X_train, Y_train) * 100, 2)

acc_gbc



## Gradient Boosting Classifier returns a score of 86.13



# 11) Neural network



mlp = MLPClassifier()

mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test)

acc_mlp = round(mlp.score(X_train, Y_train) * 100, 2)

acc_mlp



## The MLP returns a score of 83.47





# Next we want to evaluate the models to determine the optimal algorithm



models = pd.DataFrame({

    'Model': ['Logistic Regression', 'KNN', 'Support Vector Machines', 

              'Naive Bayes', 'Decision Tree', 'Random Forest',  'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 'GBC', 'MLP', 

              ],

    'Score': [acc_log, acc_knn, acc_svm, 

              acc_gaussian, acc_decision_tree, acc_random_forest,  acc_perceptron, 

              acc_sgd, acc_svc, acc_gbc, acc_mlp]})

models.sort_values(by='Score', ascending=False)



## The mose optimial model is the Decision Tree with a score of 89.64



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

        })

submission.to_csv('/Users/joshuapeterson/petersonTitanic_submission.csv',index=False) 