# Common Imports

import pandas as pd

from pandas import Series, DataFrame



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns # This affects things like the color of the axes, whether a grid is

                      # enabled by default, and other aesthetic elements.

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



import warnings

warnings.filterwarnings('ignore')
# Get train and test csv files as a DataFrame

titanic_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# see the data

titanic_df.head() # Returns first n rows (default is n=5)
titanic_df.info()

test_df.info()
# Need to remove the columns not necessary for prediction

titanic_df.drop(['Name','PassengerId','Ticket'], axis=1)

test_df.drop(['Name','Ticket'], axis=1).head(5)
# Only in titanic_df fill the empty cells of Embarked with most occured status

titanic_df['Embarked'] = titanic_df['Embarked'].fillna("S")



#Plot

sns.factorplot('Embarked','Survived', data=titanic_df, size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize = (15,5))



# CountPlot

# A special case for the bar plot is when you want to show the number of observations in each 

# category rather than computing a statistic(generally mean) for a second variable.

sns.countplot(x = 'Embarked',data=titanic_df, ax= axis1)

sns.countplot(x='Survived', hue='Embarked', data=titanic_df, order = [1,0], ax=axis2)



# group by embarked, and get the mean of survived passengers

embark_perc = titanic_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()

sns.barplot(x='Embarked', y='Survived',data=embark_perc, order=['S','Q','C'], ax=axis3)



# Either to consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.



embark_dummies_titanic = pd.get_dummies(titanic_df['Embarked'])

embark_dummies_titanic.drop(['S'],axis = 1, inplace=True)



embark_dummies_test = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'],axis =1, inplace = True)



titanic_df = titanic_df.join(embark_dummies_titanic)

test_df    = test_df.join(embark_dummies_test)



titanic_df.drop(['Embarked'], axis =1, inplace = True)

test_df.drop(['Embarked'], axis = 1, inplace = True)
# only in test data we have missing fare values

test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)



# convert it to int 

titanic_df['Fare'] = titanic_df['Fare'].astype(int)

test_df['Fare'] = test_df['Fare'].astype(int)



# get fares for passengers survied and not-survived

fare_survived = titanic_df['Fare'][titanic_df['Survived'] == 1]

fare_notsurvived = titanic_df['Fare'][titanic_df['Survived'] == 0]



# get average and std

average_fare = DataFrame([fare_notsurvived.mean(), fare_survived.mean()])

std_fare = DataFrame([fare_notsurvived.std(), fare_survived.std()])



# Plot

titanic_df['Fare'].plot(kind = 'hist', bins = 100, figsize= (15,3), xlim = (0,50))

average_fare.index.names = std_fare.index.names = ['Survived']

average_fare.plot( yerr = std_fare, legend=False, kind = 'bar') # yerr = yerror
fig, (axis1,axis2) = plt.subplots(1,2, figsize = (15,4))

axis1.set_title('Original Age Value - Titanic')

axis2.set_title('New Age Value - Titanic')



average_age = titanic_df['Age'].mean()

std_age = titanic_df['Age'].std()

nan_age = titanic_df['Age'].isnull().sum()



average_age_test = test_df['Age'].mean()

std_age_test = test_df['Age'].std()

nan_age_test = test_df['Age'].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age - std_age, average_age + std_age, size= nan_age)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size= nan_age_test)



# Plot original Age value

# Drop null values

titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# fill NaN values

titanic_df['Age'][np.isnan(titanic_df['Age'])] = rand_1

test_df['Age'][np.isnan(test_df['Age'])] = rand_2



# convert to integer

titanic_df['Age'] = titanic_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)



# Plot new age values

titanic_df['Age'].hist(bins = 70, ax=axis2)
# Peaks for survived/non-survived passengers by their age

facet = sns.FacetGrid(titanic_df, hue = 'Survived', aspect=4)

facet.map(sns.kdeplot, 'Age', shade = True) # Apply a plotting function to each facet's subset of the data.

facet.set(xlim = (0, titanic_df['Age'].max()))

facet.add_legend()

# sns.kdeplot (Fit and plot a univariate or bivariate kernel density estimate.)



# Average survived passengers by age

fig, axis1 = plt.subplots(1,1, figsize = (18,4))

Average_age = titanic_df[['Age', 'Survived']].groupby('Age', as_index = False).mean()

sns.barplot(data=Average_age, x='Age', y='Survived')
# large number of nan values, so will not cause any harm to prediction if we remove it

titanic_df.drop('Cabin', axis=1, inplace = True)

test_df.drop('Cabin', axis=1, inplace = True)
# Instead of having two columns Parch & Sibsp, we can have only one column whether the passenger 

# had any family member or not.

titanic_df['Family'] = titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1

titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0



test_df['Family'] = test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



# Drop Parch and SibSp

titanic_df = titanic_df.drop(['Parch', 'SibSp'], axis= 1)

test_df = test_df.drop(['Parch', 'SibSp'], axis= 1)



# Plot

fig, (axis1, axis2) = plt.subplots(1,2, sharex= True, figsize = (10,5))

sns.countplot(x = 'Family', data= titanic_df, order=[1,0], ax=axis1)



# average of survived for those who had/didn't have any family member

Family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'], as_index = False).mean()

sns.barplot(x='Family', y = 'Survived', data=Family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child

def get_person (passenger):

    age, sex = passenger

    return 'child' if age < 16 else sex



titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person, axis = 1)

test_df['Person'] = test_df[['Age','Sex']].apply(get_person, axis = 1)



# drop sex

titanic_df.drop(['Sex'], axis=1, inplace= True)

test_df.drop(['Sex'], axis = 1, inplace= True)



# create dummies and drop Male

person_dummies_titanic = pd.get_dummies(titanic_df['Person'])

person_dummies_titanic.columns = ['Child', 'Female', 'Male']

person_dummies_titanic.drop(['Male'], axis = 1, inplace=True)



person_dummies_test = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child', 'Female', 'Male']

person_dummies_test.drop(['Male'], axis =1, inplace = True)



titanic_df = titanic_df.join(person_dummies_titanic)

test_df    = test_df.join(person_dummies_test)



fig, (axis1, axis2) = plt.subplots(1,2, figsize = (10,5))

sns.countplot(x = 'Person', data=titanic_df, ax = axis1)



# average of survived for each person

person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'], as_index = False).mean()

sns.barplot(x='Person', y='Survived', data= person_perc, ax=axis2, order=['male', 'female', 'child'])



titanic_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)

sns.factorplot('Pclass', 'Survived', data= titanic_df, order= [1,2,3], size=5)



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
Titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)

Test_df = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
X_train = Titanic_df.drop("Survived", axis = 1)

y_train = Titanic_df["Survived"]

X_test = Test_df.copy()
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

#y_pred = logreg.predict(X_test)

logreg.score(X_train, y_train)
svc = SVC()

svc.fit(X_train, y_train)

#y_pred = svc.predict(X_test)

svc.score(X_train, y_train)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.cross_validation import StratifiedKFold

from sklearn.model_selection import train_test_split

from scipy.stats import randint



seed = 123
x_data, x_test_data, y_data, y_test_data = train_test_split(X_train, y_train, test_size = 0.2)

# cv = StratifiedKFold(y_train, n_folds=5, random_state=seed, shuffle=True)

rf_para = {

    'max_depth': [1, 5, 10],

    'min_samples_leaf': [1, 2],

    'n_estimators': randint(1, 1001), # uniform discrete random distribution

    'min_samples_split': [2,3,4],

    'min_weight_fraction_leaf': [0,0.05,0.5]

    }
bst_grid = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=rf_para, scoring='accuracy',n_jobs=1,\

                       cv = 10)
bst_grid.fit(x_data.values, y_data.values)

bst_grid.best_params_

bst_grid.grid_scores_
print ('Best accuracy obtained: {}'.format(bst_grid.best_score_))

print ('Parameters:')

for key, value in bst_grid.best_params_.items():

    print('\t{}:{}'.format(key,value))
from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(model,title, X, y,n_jobs = 1, ylim = None, cv = None,train_sizes = np.linspace(0.1, 1, 5)):

    

    # Figrue parameters

    plt.figure(figsize=(10,8))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel('Training Examples')

    plt.ylabel('Score')

    

    train_sizes, train_score, test_score = learning_curve(model, X, y, cv = cv, n_jobs=n_jobs, train_sizes=train_sizes)

    

    # Calculate mean and std

    train_score_mean = np.mean(train_score, axis=1)

    train_score_std = np.std(train_score, axis=1)

    test_score_mean = np.mean(test_score, axis=1)

    test_score_std = np.std(test_score, axis=1)

    

    plt.grid()

    plt.fill_between(train_sizes, train_score_mean - train_score_std, train_score_mean + train_score_std,\

                    alpha = 0.1, color = 'r')

    plt.fill_between(train_sizes, test_score_mean - test_score_std, test_score_mean + test_score_std,\

                    alpha = 0.1, color = 'g')

    

    plt.plot(train_sizes, train_score_mean, 'o-', color="r", label="Training score")

    plt.plot(train_sizes, test_score_mean, 'o-', color="g", label="Cross-validation score")

    

    plt.legend(loc = "best")

    return plt
# random forest

random_forest = RandomForestClassifier(n_estimators=200, max_depth=8, max_features=0.2, bootstrap=False, \

                min_samples_leaf=2, min_samples_split=3)

random_forest.fit(X_train, y_train)

# y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
# Plotting learning curve

title = 'Learning Curve (Random Forest)'

# cross validation with 50 iterations to have a smoother curve

cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

model = random_forest

plot_learning_curve(model,title,X_train, y_train, n_jobs=-1,ylim=None,cv=cv)

plt.show()
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

#y_pred = knn.predict(X_test)

knn.score(X_train, y_train)
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

#y_pred = gaussian.predict(X_test)

gaussian.score(X_train, y_train)
yy_pred = random_forest.predict(X_test)

solution = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':yy_pred }, columns=['PassengerId', 'Survived'])

solution.to_csv('Titanic.csv', index = False)