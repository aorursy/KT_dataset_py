# Importing the basic libraries

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# importing the train dataset

df1 = pd.read_csv('../input/train.csv')



# importing the test dataset

df2 = pd.read_csv('../input/test.csv')



# And let's check how the data looks like!

df1.sample(10)
# Creating a list of the 2 dataframes so we perform operations on both

dfs = [df1, df2]



# Intializing the output dataframe

output = df2[['PassengerId']]
df1.info()
# Checking missing values

print(df1.isnull().sum())

print('-' * 20)

print(df2.isnull().sum())
# Addressing missing values

for df in dfs:

    

    # Let's drop the Cabin and some unnecessary attributes

    df.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True)

    

    # Filling Embarked with the mode

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    

    # Filling Fare with the median

    df['Fare'].fillna(df['Fare'].median(), inplace=True)



df1.head()
for df in dfs:

    # Let's create a family size attribute based on SibSp and Parch

    df['FamilySize'] = df['SibSp'].astype('int') + df['Parch'].astype('int') + 1

    

    # Now based on family size, we might be able to check if the person is alone

    df['IsAlone'] = (df['FamilySize'] == 1).astype('int')

    

    # Based on the name we might be able to check the title

    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    

df1.head()
df1['Title'].value_counts()
# Let's call "Other" every Title that is not our top 4

top_5_titles = df1['Title'].value_counts().head(4).index.tolist()



for df in dfs:

    df['Title'] = df['Title'].apply(lambda x: x if x in top_5_titles else 'Other')

    

df1['Title'].value_counts()
df1.sample(5)
# Let's get a copy of the dataset only with given ages

df_age = df1[df1['Age'].notna()]

df_age.isnull().sum()
# Let's first check the correlation between Age and Fare

sns.regplot(x='Fare', y='Age', data=df_age)
# Let's see Pclass

sns.boxplot(x='Pclass', y='Age', data=df_age)
# And FamilySize?

print(df_age[['FamilySize', 'Age']].corr())

df_age[['FamilySize', 'Age']].groupby('FamilySize').median().plot()
# How about sex?

sns.boxplot(x=df_age['Sex'], y=df_age['Age'])
# Finally Title

sns.boxplot(x=df_age['Title'], y=df_age['Age'])
# Defing X matrix and target y for Age regression

X_age = df_age[['Pclass', 'FamilySize']]

y_age = df_age['Age']



age_dummies = pd.get_dummies(df_age[['Title']])

age_dummies.drop(columns=['Title_Other'], inplace=True)



X_age = pd.concat([X_age, age_dummies], axis=1)

X_age.head()
# Let's split the dataset into train and test so we can evaluate the performance of our age predictor

from sklearn.model_selection import train_test_split



X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X_age.values, y_age, test_size=0.2, random_state=4)
# Let's check how a multiple linear regression fits our data

from sklearn.linear_model import LinearRegression

age_predictor = LinearRegression().fit(X_age_train, y_age_train)



yhat_age_test = age_predictor.predict(X_age_test)



from sklearn.metrics import r2_score

print('R2 Score: %.2f' % r2_score(y_age_test, yhat_age_test))

print('Variance score: %.2f' % age_predictor.score(X_age_train, y_age_train))
# Let's train the predictor model now with the full set

age_predictor = LinearRegression().fit(X_age.values, y_age)

age_predictor
# First let's add the Title dummies

df1 = pd.concat([df1, pd.get_dummies(df1[['Title']])], axis=1)

df2 = pd.concat([df2, pd.get_dummies(df2[['Title']])], axis=1)

dfs = [df1, df2]



for df in dfs:

    # Let's delete the last dummy

    df.drop(columns=['Title_Other'], inplace=True)

    

    # Now let's make the prediction column

    df['PredictedAge'] = age_predictor.predict(df[['Pclass', 'FamilySize', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']].values)

    

    # And replace the missing values with the predicted ones

    df['Age'].fillna(df['PredictedAge'].round(0), inplace=True)

    

    # Finally, we don't need the predicted column any longer

    df.drop(columns=['PredictedAge'], inplace=True)

    

df1.head(6)
# Checking missing values

print(df1.isnull().sum())

print('-' * 20)

print(df2.isnull().sum())
# First let's finish cleaning our data

for df in dfs:

    # Let's get rid of the name column

    df.drop(columns=['Name'], inplace=True)



df1.sample(5)
# Now let's see how some features influentes on survival

fig, ax0 = plt.subplots(2, 3,figsize=(20,10))



sns.barplot(x = 'Embarked', y = 'Survived', data=df1, ax = ax0[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=df1, ax = ax0[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=df1, ax = ax0[0,2])



sns.barplot(x = 'Title', y = 'Survived', data=df1, ax = ax0[1,0])

sns.barplot(x = 'Sex', y = 'Survived', data=df1, ax = ax0[1,1])

sns.barplot(x = 'FamilySize', y = 'Survived', data=df1, ax = ax0[1,2])
# Let's check Age and Fare

fig1, ax1 = plt.subplots(1, 2,figsize=(10,5))



sns.boxplot(x = 'Survived', y = 'Fare', data=df1, ax = ax1[0])

sns.boxplot(x = 'Survived', y = 'Age', data=df1, ax = ax1[1])
# Perhaps for Age and Fare we might have a bettr understanding if we split into bins

df1_bins = df1[['Age', 'Fare', 'Survived']]

df1_bins['AgeBin'] = pd.cut(df1_bins['Age'].astype('int'), 5)

df1_bins['FareBin'] = pd.cut(df1_bins['Fare'].astype('int'), 5)



fig2, ax2 = plt.subplots(1, 2,figsize=(10,5))



sns.barplot(x = 'FareBin', y = 'Survived', data=df1_bins, ax = ax2[0])

sns.barplot(x = 'AgeBin', y = 'Survived', data=df1_bins, ax = ax2[1])
df1.head()
# Now let's set the matrix of features of df1

X = df1[['Pclass', 'Sex', 'Age', 'Embarked', 'IsAlone', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]

X_dummies = pd.get_dummies(X[['Sex', 'Embarked']])

X = pd.concat([X, X_dummies], axis=1)

X.drop(columns=['Sex', 'Embarked', 'Sex_male', 'Embarked_S'], inplace=True)



# Now let's set the matrix of features of df2

X_pred = df2[['Pclass', 'Sex', 'Age', 'Embarked', 'IsAlone', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]

X_pred_dummies = pd.get_dummies(X_pred[['Sex', 'Embarked']])

X_pred = pd.concat([X_pred, X_pred_dummies], axis=1)

X_pred.drop(columns=['Sex', 'Embarked', 'Sex_male', 'Embarked_S'], inplace=True)



X.head()
# And the dependent vector

y = df1['Survived']
# Let's scale the X matrix

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_pred = scaler.transform(X_pred)

X[0:5]
# Finally before actually building the prediction model, let's split df1 into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn import metrics

from sklearn.linear_model import LogisticRegression



# Let's find the parameters for the model

num_cs = 100

params = {'solvers': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

          'Cs' : np.linspace(0.01, 1.0, num_cs).tolist()}

evals = np.zeros((num_cs, 5))



for solv in params['solvers']:

    for c in params['Cs']:

        log = LogisticRegression(C=c, solver=solv).fit(X_train,y_train)

        log_yhat=log.predict(X_test)

        evals[params['Cs'].index(c), params['solvers'].index(solv)] = metrics.accuracy_score(y_test, log_yhat)
# Getting the optimal parameters

max_eval_index = np.unravel_index(evals.argmax(), evals.shape)

solv_opt = params['solvers'][max_eval_index[1]]

c_opt = params['Cs'][max_eval_index[0]]



# Bulding model now with optimal parameters

log = LogisticRegression(C=c_opt, solver=solv_opt).fit(X_train, y_train)

log
log_yhat = log.predict(X_test)

log_yhat_prob = log.predict_proba(X_test)
from sklearn.metrics import jaccard_similarity_score

from sklearn.metrics import classification_report

from sklearn.metrics import log_loss



print('Log Loss: %.2f' % log_loss(y_test, log_yhat_prob))

print('Jaccard: %.2f' % jaccard_similarity_score(y_test, log_yhat))

print (classification_report(y_test, log_yhat))
from sklearn.neighbors import KNeighborsClassifier



# Let's find the best k for the model

K_range = 11

accuracies = np.zeros((K_range-1))

for n in range(1,K_range):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    neigh_yhat=neigh.predict(X_test)

    accuracies[n-1] = metrics.accuracy_score(y_test, neigh_yhat)

accuracies
# Building the model for the optmal K and getting its scores

# Finding best K

K_opt = accuracies.tolist().index(accuracies.max()) + 1

# Bulding the optimal model

neigh = KNeighborsClassifier(n_neighbors = K_opt).fit(X_train, y_train)

neigh
neigh_yhat = neigh.predict(X_test)
print('Jaccard: %.2f' % jaccard_similarity_score(y_test, neigh_yhat))

print (classification_report(y_test, neigh_yhat))
from sklearn import svm



# Let's find the parameters for the model

num_cs = 100

params = {'kernels': ['linear', 'poly', 'rbf', 'sigmoid'],

          'Cs' : np.linspace(0.01, 1.0, num_cs).tolist()}

evals = np.zeros((num_cs, 4))



for kern in params['kernels']:

    for c in params['Cs']:

        sv = svm.SVC(C=c, kernel=kern).fit(X_train,y_train)

        sv_yhat=log.predict(X_test)

        evals[params['Cs'].index(c), params['kernels'].index(kern)] = metrics.accuracy_score(y_test, sv_yhat)
# Getting the optimal parameters

max_eval_index = np.unravel_index(evals.argmax(), evals.shape)

kern_opt = params['kernels'][max_eval_index[1]]

c_opt = params['Cs'][max_eval_index[0]]



# Bulding model now with optimal parameters

sv = svm.SVC(C=c_opt, kernel=kern_opt).fit(X_train, y_train)

sv
sv_yhat = sv.predict(X_test)
print('Jaccard: %.2f' % jaccard_similarity_score(y_test, sv_yhat))

print (classification_report(y_test, sv_yhat))
from sklearn.tree import DecisionTreeClassifier



# Let's find the best accuracy evaluation for the model

crits = ['entropy', 'gini']

evals = [0, 0]



for i, criterion in enumerate(crits):

    tree = DecisionTreeClassifier(criterion=criterion).fit(X_train, y_train)

    tree_yhat=tree.predict(X_test)

    evals[i] = metrics.accuracy_score(y_test, tree_yhat)

evals
crit_opt = crits[evals.index(max(evals))]

# Bulding model with optimal parameters

tree = DecisionTreeClassifier(criterion=crit_opt).fit(X_train, y_train)

tree
tree_yhat = tree.predict(X_test)
print('Jaccard: %.2f' % jaccard_similarity_score(y_test, tree_yhat))

print (classification_report(y_test, tree_yhat))
from sklearn import ensemble



# Let's find the parameters for the model

num_est = 20

params = {'crits': ['gini', 'entropy'],

          'est' : np.linspace(10, 200, num_est).tolist()}

evals = np.zeros((num_est, 2))



for crit in params['crits']:

    for est in params['est']:

        forest = ensemble.RandomForestClassifier(n_estimators=int(est), criterion=crit).fit(X_train,y_train)

        forest_yhat=forest.predict(X_test)

        evals[params['est'].index(est), params['crits'].index(crit)] = metrics.accuracy_score(y_test, forest_yhat)
# Getting the optimal parameters

max_eval_index = np.unravel_index(evals.argmax(), evals.shape)

crit_opt = params['crits'][max_eval_index[1]]

est_opt = int(params['est'][max_eval_index[0]])



# Bulding model now with optimal parameters

forest = ensemble.RandomForestClassifier(n_estimators=est_opt, criterion=crit_opt).fit(X_train,y_train)

forest
forest_yhat = forest.predict(X_test)
print('Jaccard: %.2f' % jaccard_similarity_score(y_test, forest_yhat))

print (classification_report(y_test, forest_yhat))
from sklearn.metrics import f1_score



all_metrics = {'LogisticRegression': [jaccard_similarity_score(y_test, log_yhat), f1_score(y_test, log_yhat), log_loss(y_test, log_yhat_prob)],

               'KNN': [jaccard_similarity_score(y_test, neigh_yhat), f1_score(y_test, neigh_yhat), np.nan],

               'SVM': [jaccard_similarity_score(y_test, sv_yhat), f1_score(y_test, sv_yhat), np.nan],

               'DecisionTree': [jaccard_similarity_score(y_test, tree_yhat), f1_score(y_test, tree_yhat), np.nan],

               'RandomForest': [jaccard_similarity_score(y_test, forest_yhat), f1_score(y_test, forest_yhat), np.nan]}



df_metrics = pd.DataFrame(all_metrics, index=['Jaccard', 'F1', 'LogLoss'])

df_metrics
# Bulding the optimal model with full dataset

best_model = KNeighborsClassifier(n_neighbors = K_opt).fit(X, y)

best_model
yhat = best_model.predict(X_pred)

yhat[0:5]
output['Survived'] = yhat

print(output.shape)

output.head()
output.to_csv('titanic-output.csv', index=None)