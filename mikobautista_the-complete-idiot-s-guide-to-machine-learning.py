# Information Processing

import numpy as np

import pandas as pd

import re



# Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

sns.set_palette(sns.color_palette('pastel'))



# Machine Learning

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
# Import the dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.info()

print("======================================")

test.info()
print(train.isnull().sum()) # Null values for Age, Cabin, Embarked

print("===================")

print(test.isnull().sum()) # Null values for Age, Fare, Cabin
# Ticket is an alphanumeric string indicating the passenger's ticket

# number. There does not seem to be any inherent pattern in them.

print(train['Ticket'].sample(10))



# Let's drop this feature. It is unlikely to contain any useful data

# for our purposes.

def drop_ticket(df):

    df.drop('Ticket', axis=1, inplace=True)

    

drop_ticket(train)

drop_ticket(test)
# Sex is a no-brainer feature to keep. Females have a significantly better chance

# of survival than males. The entire population's survival rate is only 38.4%, but

# it jumps all the way to 74.2% for females!

_, ax = plt.subplots(1, 2, figsize=(20,10))

train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0])

ax[0].set_title('Survival Rate of Population')

sns.barplot('Sex', 'Survived', data=train, ax=ax[1])

ax[1].set_title('Survival Rate by Sex')

plt.show()
# Embarked seems to have a strong correlation with survival.

# Those who departed from Cherbourg ("C"), for example, were more

# likely to survive than those who departed from Queenstown ("Q").

_, ax = plt.subplots(1, 2, figsize=(20,10))

sns.countplot('Embarked', data=train, ax=ax[0]);

ax[0].set_title('Passenger Count by Embarked')

ax[0].set_ylabel('Passenger Count')

sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Survival Count by Embarked')

ax[1].set_ylabel('Survival Count')

plt.show()
# From the chart above, we observe that a vast majority of the passengers

# embarked from Southampton ("S"). It would be reasonable to use this value

# to fill the 2 entries with null Embarked.

def fill_null_embarked(df):

    df.Embarked.fillna('S', inplace=True)

    

fill_null_embarked(train)

fill_null_embarked(test)
# Age is a very powerful indicator. The probability of survival is

# the highest for young passengers. On the flipside, it

# is the lowest for the elderly and those in their prime years of

# strength (perhaps because they were preoccupied helping other people).

_, ax = plt.subplots(figsize=(20,5))

sns.distplot(train[train['Survived'] == 0]['Age'].dropna(), hist=False, kde_kws={"shade": True})

sns.distplot(train[train['Survived'] == 1]['Age'].dropna(), hist=False, kde_kws={"shade": True})

ax.set_title("Population Distribution by Age and Survival Rate")

ax.set_ylabel('% of Population')

ax.legend(['Survived = 0', 'Survived = 1'])

ax.set_xlim(0, 85)

plt.show()
# Unlike Embarked, there are a substantial number of missing values for Age. It

# would be ill-advised for us to blindly fill these in. Fortunately, we can

# utilize the salutations in the Name feature to make a more educated guess.

def fill_null_age(df_train, df_test):

    # Calculate the average age for all passengers with the same salutation.

    df_combined = pd.concat([train[['Age', 'Name']], test[['Age', 'Name']]])

    df_combined['Salutation'] = df_combined.Name.str.extract(', ([A-Za-z]+)\.', expand=False)

    average_ages = df_combined.groupby('Salutation')['Age'].mean()

    df_combined.drop('Salutation')

    # Fill in null Age values using these averages. 

    df_train.loc[df_train.Age.isnull(), 'Age'] = df_train[df_train.Age.isnull()].apply(lambda row: average_ages[re.search(', ([A-Za-z]+)\.', row.Name).group(1)], axis=1)

    df_test.loc[df_test.Age.isnull(), 'Age'] = df_test[df_test.Age.isnull()].apply(lambda row: average_ages[re.search(', ([A-Za-z]+)\.', row.Name).group(1)], axis=1)

    return df_train, df_test



train, test = fill_null_age(train, test)
# Age bands simplify the data we pass into the model. They

# allow us to capture core information with lower overhead.

def create_age_bands(df):

    bins = [np.NINF, 15, 30, 60, np.inf]

    labels = ['0_Young', '1_Prime', '2_Mature', '3_Elderly']

    df['AgeBand'] = pd.cut(df.Age, bins, labels = labels)

    df.drop('Age', axis=1, inplace=True)

    

create_age_bands(train)

create_age_bands(test)



_, ax = plt.subplots(1, 2, figsize=(20,10))

sns.countplot(x="AgeBand", data=train, ax=ax[0]);

ax[0].set_title('Passenger Count by AgeBand')

ax[0].set_ylabel('Passenger Count')

sns.countplot('AgeBand', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Survival Count by AgeBand')

ax[1].set_ylabel('Survival Count')

plt.show()
# Money talks. Pclass, a proxy for socio-economic status, is another feature

# that we should keep. 1 denotes upper class, 2 denotes middle, 3 denotes lower.

# Clearly, those with more money were likelier to survive the tragedy.

# In fact, nearly *all* females from upper class survived!

_, ax = plt.subplots(1, 2, figsize=(20,10))

train['Pclass'].value_counts().plot.pie(explode=[0.025,0.025,0.025], autopct='%1.1f%%', ax=ax[0])

ax[0].set_title('Population Pclass Breakdown')

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train, ax=ax[1]);

ax[1].set_title('Survival Rate by Pclass and Sex')

plt.show()
# Although a person's name might not be very indicative of his/her survival,

# the Name field also contains the passenger's salutation. Individuals with

# titles such as Don, Rev, Dr, etc. have a low likelihood of survival. These

# salutations are typically found in occupations where having a special love for

# community is essential (medical/military/royalty). I hypothesize that they

# felt compelled to save others before themselves, thus explaining the low

# survival rate.

train['Salutation'] = train.Name.str.extract(', ([A-Za-z\s]+)\.', expand=False)

_, ax = plt.subplots(figsize=(20,5))

sns.barplot('Salutation', 'Survived', data=train, ax=ax)

ax.set_title('Survival Rate by Salutation')

plt.show()
# Create a new Salutation feature, combining all of the "Rescuer" prefixes.

# In addition, we also group synonymous prefixes together.

def simplify_name(df):

    df['Salutation'] = df.Name.str.extract(', ([A-Za-z\s]+)\.', expand=False)

    df['Salutation'].replace(['Dr', 'Rev', 'Major', 'Col','Capt', 'Sir', 'Don', 'Jonkheer'], 'Rescuer', inplace = True)

    df['Salutation'].replace(['Lady', 'Dona', 'the Countess', 'Mme'], 'Mrs', inplace = True)

    df['Salutation'].replace(['Mlle', 'Ms'], 'Miss', inplace = True)

    df.drop('Name', axis=1, inplace=True)



simplify_name(train)

simplify_name(test)
# The data indicates that the Cabin feature did have an impact on survival.

# It didn't matter too much which type of cabin you had though. It seems that

# as long as you *had* one, you were already much better off than those without.

train['CabinLetter'] = train['Cabin'].str[0]

train['CabinLetter'].fillna('None', inplace=True)

_, ax = plt.subplots(figsize=(20,5))

sns.barplot('CabinLetter', 'Survived', data=train, ax=ax)

ax.set_title("Survival Rate by CabinLetter")

train.drop('CabinLetter', axis=1, inplace=True)

plt.show()



# Hence, we can simplify the cabin feature by converting it to a WithCabin bool.

def simplify_cabin(df):

    df.loc[df.Cabin.notnull(), 'WithCabin'] = True

    df.loc[df.Cabin.isnull(), 'WithCabin'] = False

    df.Cabin.unique()

    df.drop('Cabin', axis=1, inplace=True)



simplify_cabin(train)

simplify_cabin(test)
# The trends in SibSp (sibling + spouse) and Parch (parent + children) are identical.

# Someone who is traveling alone is less likely to survive than those with 1-3

# companions. However, he/she would be better off than those with 4+ companions.

# To simplify both features together, we will create a new one called FamilySize.

_, ax = plt.subplots(figsize=(20,5))

sns.pointplot('SibSp', 'Survived', data=train, ax=ax, errwidth=0, color='lightskyblue',)

sns.pointplot('Parch', 'Survived', data=train, ax=ax, errwidth=0, color='mediumaquamarine')

ax.set_title('Survival Rate by SibSp and Parch')

ax.legend(['SibSp', 'Parch'])



def create_family_size(df):

    bins = [np.NINF, 0, 3, np.inf]

    labels = ['0_Alone', '1_Few', '2_Many']

    df['FamilySize'] = pd.cut(df.SibSp + df.Parch, bins, labels = labels)

    df.drop('SibSp', axis=1, inplace=True)

    df.drop('Parch', axis=1, inplace=True)



create_family_size(train)

create_family_size(test)



_, ax = plt.subplots(figsize=(20,5))

sns.barplot('FamilySize', 'Survived', data=train, ax=ax)

ax.set_title("Survival Rate by FamilySize")

plt.show()
# Did I mention money talks? Very few passengers who paid over $80 died. Meanwhile,

# your odds of survival were terrible if your ticket cost less than $20.

_, ax = plt.subplots(figsize=(20,5))

sns.distplot(train[train['Survived'] == 0]['Fare'].dropna(), hist=False, kde_kws={"shade": True})

sns.distplot(train[train['Survived'] == 1]['Fare'].dropna(), hist=False, kde_kws={"shade": True})

ax.set_title("Population Distribution by Fare and Survival Rate")

ax.set_ylabel("% of Population")

ax.set_xlim(0, 160)

ax.legend(['Survived = 0', 'Survived = 1'])

plt.show()
# In the entire dataset, there is only one null value for Fare.

# We can fill this in with the average without much concern.

def fill_null_fare(df_train, df_test):

    df_combined = pd.concat([train['Fare'], test['Fare']])

    avg_fare = df_combined.mean()

    df_train.Fare.fillna(avg_fare, inplace=True)

    df_test.Fare.fillna(avg_fare, inplace=True)



fill_null_fare(train, test)
# FareBands will simplify the processing of this feature.

def create_fare_bands(df):

    bins = [np.NINF, 20, 80, np.inf]

    labels = ['0_Low', '1_Medium', '2_High']

    df['FareBand'] = pd.cut(df.Fare, bins, labels = labels)

    df.drop('Fare', axis=1, inplace=True)

    

create_fare_bands(train)

create_fare_bands(test)



_, ax = plt.subplots(figsize=(20,5))

sns.barplot('FareBand', 'Survived', data=train, ax=ax)

ax.set_title("Survival Rate by FareBand")

plt.show()
train.head()
# Integer encode the ordinal features.

def encode_ordinal_features(df_train, df_test):

    ordinal_features = ['FamilySize', 'AgeBand', 'FareBand']

    df_combined = pd.concat([df_train[ordinal_features], df_test[ordinal_features]])

    

    for feature in ordinal_features:

        le = LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

train, test = encode_ordinal_features(train, test)

train.head()
# Binary encode the categorical features.

def encode_categorical_features(df_train, df_test):

    categorical_features = ['Salutation', 'Sex', 'Embarked', 'WithCabin']

    df_combined = pd.concat([df_train[categorical_features], df_test[categorical_features]])

    for feature in categorical_features:

        # First, perform integer encoding.

        le = LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

        df_combined[feature] = le.transform(df_combined[feature])

        combined_integer_encoded = df_combined[feature].values.reshape(len(df_combined[feature]), 1)

        # Then, perform binary encoding.

        ohe = OneHotEncoder(sparse=False)

        ohe = ohe.fit(combined_integer_encoded)

        train_binary_encoded = ohe.transform(df_train[feature].values.reshape(len(df_train[feature]), 1))

        test_binary_encoded = ohe.transform(df_test[feature].values.reshape(len(df_test[feature]), 1))

        num_unique_values = len(df_combined[feature].unique())

        for i in range(num_unique_values):

            if (i > 0): # Avoid the dummy variable trap.

                col_name = feature + "_" + str(le.inverse_transform(i))

                train_col_data = train_binary_encoded[:, i].astype(int)

                test_col_data = test_binary_encoded[:, i].astype(int)

                df_train[col_name] = train_col_data

                df_test[col_name] = test_col_data

        df_train.drop(feature, axis=1, inplace=True)

        df_test.drop(feature, axis=1, inplace=True)

    return df_train, df_test



train, test = encode_categorical_features(train, test)



train.head()
# Split the training data among X_train, X_val, Y_train, and Y_val.

predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

X_train, X_val, Y_train, Y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_val)

acc_log = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print('Logistic Regression: ' + str(acc_log))



# KNN

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_val)

acc_knn = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print('KNN: ' + str(acc_knn))



# Support Vector Machines (SVM)

svc = SVC(kernel = 'linear', random_state = 0)

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_val)

acc_svc = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print('SVC: ' + str(acc_svc))



# Kernel SVM

kernel_svc = SVC(kernel = 'rbf', random_state = 0)

kernel_svc.fit(X_train, Y_train)

Y_pred = kernel_svc.predict(X_val)

acc_kernel_svc = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print('Kernel SVC: ' + str(acc_kernel_svc))



# Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_val)

acc_gaussian = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print('Gaussian: ' + str(acc_gaussian))



# Decision Tree Classification

decision_tree = DecisionTreeClassifier(random_state = 0)

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_val)

acc_decision_tree = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print('Decision Tree: ' + str(acc_decision_tree))



# Random Forest Classification

random_forest = RandomForestClassifier(n_estimators=100, random_state = 0)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_val)

random_forest.score(X_train, Y_train)

acc_random_forest = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print('Random Forest: ' + str(acc_random_forest))
# Perform K-Fold Cross-Validation with 10 splits.

kfold = KFold(n_splits=10, random_state=0)

mean=[]

accuracy=[]

std=[]

classifiers=['Logistic Regression',

             'KNN',

             'SVC',

             'Kernel SVC',

             'Gaussian',

             'Decision Tree',

             'Random Forest']

models=[logreg,

        knn,

        svc,

        kernel_svc,

        gaussian,

        decision_tree,

        random_forest]

for i in models:

    model = i

    cv_result = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = "accuracy")

    mean.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

    

print(pd.DataFrame({'CV Mean': mean,'Std': std}, index=classifiers))

_, ax = plt.subplots(figsize=(20,10))

sns.boxplot(classifiers, accuracy, ax=ax)

ax.set_title("Model Accuracy")

ax.set_xlabel("Model")

ax.set_ylabel("Accuracy")
# Tune the hyper-parameters for Kernel SVC, our best performing model.

C = np.logspace(-6, 5, 12)

gamma = np.logspace(-6, 5, 12)

param_grid = {'C': C, 'gamma': gamma}

gd = GridSearchCV(estimator=kernel_svc, param_grid=param_grid, verbose=True)

gd.fit(X_train, Y_train)

print('Hyper-Parameter SVC (Best Score): ' + str(gd.best_score_))

print('Hyper-Parameter SVC (Best Estimator): ' + str(gd.best_estimator_))
# For good measure, do the same for Logistic Regression, our second best performing model.

C = np.logspace(-6, 5, 12)

param_grid = {'C': C, 'penalty': ['l1', 'l2']}

gd = GridSearchCV(estimator=logreg, param_grid=param_grid, verbose=True)

gd.fit(X_train, Y_train)

print('Hyper-Parameter Logistic Regression (Best Score): ' + str(gd.best_score_))

print('Hyper-Parameter Logistic Regression (Best Estimator): ' + str(gd.best_estimator_))
# Make our final predictions and export the CSV file.

ids = test['PassengerId']

hyper_param_kernel_svc = SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',

  max_iter=-1, probability=False, random_state=0, shrinking=True,

  tol=0.001, verbose=False)

hyper_param_kernel_svc.fit(X_train, Y_train)

predictions = hyper_param_kernel_svc.predict(test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)