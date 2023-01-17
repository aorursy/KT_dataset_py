# Data Manipulattion

import numpy as np

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Importing Dependencies

%matplotlib inline



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')

# Read and preview the test data from csv file.

train = pd.read_csv("../input/titanic/train.csv")

train.head()
train.shape
# Read and preview the train data from csv file.

test = pd.read_csv("../input/titanic/test.csv")

test.head()
test.shape
merge = pd.concat([train,test],sort = False)

merge.head()
# Let's see the shape of the combined data

merge.shape
print(merge.columns.values)
# data types of different variables

merge.info()
# Visualization of Missing variables

plt.figure(figsize=(8,4))

sns.heatmap(merge.isnull(), yticklabels=False, cbar=False, cmap = 'RdGy')
# Count of missing variables

merge.isnull().sum()
#Let's see the Name column.

merge['Name'].head(10)
# Extracting title from Name and create a new variable Title.

merge['Title'] = merge['Name'].str.extract('([A-Za-z]+)\.')

merge['Title'].head()
# let's see the different categories of Title from Name column.

merge['Title'].value_counts()
# Replacing  Dr, Rev, Col, Major, Capt with 'Officer'

merge['Title'].replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace=True)



# Replacing Dona, Jonkheer, Countess, Sir, Lady with 'Aristocrate'

merge['Title'].replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)



#  Replace Mlle and Ms with Miss. And Mme with Mrs.

merge['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
# let's see how Tittle looks now

merge['Title'].value_counts()
# adding parents and siblings data to get family members data

merge['Family Members'] = merge.SibSp + merge.Parch
#Converting binary to numeric

merge.Sex = merge.Sex.map({'male':1,'female':0})
#For Embarked there are 2 rows that are having null values

#imputing Embarked with mode because Embarked is a categorical variable.

merge['Embarked'].value_counts()
# Here S is the most frequent

merge['Embarked'].fillna(value = 'S', inplace = True)
# Impute missing values of Fare. Fare is a numerical variable and only one rows that are having null values. Hence it will be imputed by median.'''

merge['Fare'].fillna(value = merge['Fare'].median(), inplace = True)
# Let's plot correlation heatmap to see which variable is highly correlated with Age. We need to convert categorical variable into numerical to plot correlation heatmap. So convert categorical variables into numerical.

df = merge.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Parch', 'SibSp', 'Ticket', 'Fare']]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df = df.apply(le.fit_transform) # data is converted.

df.head()
 # Inserting Age in variable correlation.

df['Age'] = merge['Age']

# Move Age at index 0.

df = df.set_index('Age').reset_index()

df.head(2)
# Now create the heatmap correlation of df

plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), cmap ='BrBG',annot = True)

plt.title('Variables correlated with Age')

plt.show()
# Create a boxplot to view the correlated and medium of the Pclass and Title variables with Age.

# Boxplot b/w Pclass and Age

sns.boxplot(y='Age', x='Pclass', data=merge)
# Boxplot b/w Title and Age

sns.boxplot(y='Age', x='Title', data=merge)
# Impute Age with median of respective columns (i.e., Title and Pclass)

merge['Age'] = merge.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
#Since for cabin there are only 204 rows that have values we can say that this coloum is better dropped

merge.drop('Cabin',axis=1,inplace=True)
# creating 'Age Groups' category column from Age column

def age_group(x):

    if (x > 0) and (x <=5):

        return 'infant'

    elif (x > 5) and (x <=12):

        return 'children'

    elif (x > 12) and (x <=18):

        return 'teenager'

    elif (x > 18) and (x <=35):

        return 'yound adults'

    elif (x > 35) and (x <=60):

        return 'adults'

    elif (x > 60) and (x <=80):

        return 'aged'

    else:

        return 'unknown'



merge['Age Groups'] = merge['Age'].apply(age_group)
# creating 'fare Groups' category column from fare column

def fare_group(x):

    if (x > -1) and (x <=130):

        return 'low'

    elif (x > 130) and (x <=260):

        return 'medium'

    elif (x > 260) and (x <=390):

        return 'high'

    elif (x > 390) and (x <=520):

        return 'very high'

    else:

        return 'unknown'



merge['fare Groups'] = merge['Fare'].apply(fare_group)
# plotting count plot to know the number of male and female

sns.countplot(merge['Sex'])

plt.show()
# filtering only survived passengers in the dataset

df_survived = merge[merge['Survived']==1]
# plotting count plot to know the number of male and female in the survived passengers

sns.countplot(df_survived['Sex'])

plt.show()
# bin by age group and analyse which age group survived

plt.figure(figsize=(10,6))

sns.boxplot(x = 'Survived', y = 'Age', data = merge)

plt.title("Age v/s Survived")

plt.show()
plt.figure(figsize=(10,6))

sns.barplot(y = 'Survived', x = 'Age Groups', data = merge)

plt.title("Age Groups v/s Survived")

plt.show()
by_pclass_segment_group = merge.pivot_table(values='Survived',index='Pclass',aggfunc='mean')

by_pclass_segment_group.reset_index(inplace=True)

by_pclass_segment_group['Survived'] = 100*by_pclass_segment_group['Survived']

plt.figure(figsize=(8,4))

sns.barplot(x='Pclass',y='Survived', data=by_pclass_segment_group)

plt.xlabel("Pclass")

plt.ylabel("Percentage of passenger survived")

plt.title("% of passenger survived vs Class of the passenger")

plt.show()
# droping the feature that would not be useful anymore

merge.drop(columns = ['Name', 'Age','Ticket','Fare','Parch','SibSp'], inplace = True, axis = 1)

merge.columns
merge.head()
# convert categotical data into dummies variables

merge = pd.get_dummies(merge, drop_first=True)

merge.tail()
merge.shape
#Let's split the train and test set to feed machine learning algorithm.

train = merge.iloc[:891, :]

test  = merge.iloc[891:, :]
train['Survived'] = train['Survived'].astype('int')
train.head()
#Drop passengerid from train set and Survived from test set.'''

train = train.drop(columns = ['PassengerId'], axis = 1)

test = test.drop(columns = ['Survived'], axis = 1)
# setting the data as input and output for machine learning models

X_train = train.drop(columns = ['Survived'], axis = 1) 

y_train = train['Survived']



# Extract test set

X_test  = test.drop("PassengerId", axis = 1).copy()
X_test.shape
# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
# Support Vector Machines

from sklearn.svm import SVC, LinearSVC

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_svc
# K nearest neighbours

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
# Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian,acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
y_test_pred = random_forest.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test_pred

    })

submission
submission.to_csv("submission.csv", index=False)