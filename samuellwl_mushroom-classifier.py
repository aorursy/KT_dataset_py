# =============================================================================

# Import libraries and dataset

# =============================================================================

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix



data = pd.read_csv("../input/mushrooms.csv")
np.shape(data)
data.describe()
data.head()
# Remove NA values

data = data.dropna()



# Recode response variable

data.loc[data.iloc[:,0]=='e','class'] = 'edible'

data.loc[data.iloc[:,0]=='p','class'] = 'poisonous'
sns.set(style="darkgrid")

names = data.columns



# Plot each variable's proportion by level, according to their class (poisonous or edible)

for k in range(4):

    fig, axe = plt.subplots(2, 3, figsize=(20, 25))

    for i in range(1+k*(6),7+k*(6)): 

        if i == 23:

            break

        prop_df = (data.iloc[:,i].groupby(data.iloc[:,0]).value_counts(normalize=True).rename('proportion').reset_index())

        if i-k*(6)<4:

            sns.barplot(hue=prop_df.iloc[:,1], x=prop_df.iloc[:,0], y=prop_df.iloc[:,2], data=prop_df, ax=axe[0][i-k*(6)-1]).set_title(names[i])

        else:

            sns.barplot(hue=prop_df.iloc[:,1], x=prop_df.iloc[:,0], y=prop_df.iloc[:,2], data=prop_df, ax=axe[1][i-k*(6)-3-1]).set_title(names[i])
# =============================================================================

# Data Pre-processing

# =============================================================================

# Separate into predictor variables and response variable

x = (data.iloc[:,1:])

y = (data.iloc[:,0])



# Obtain train and test sets, set seed to 10

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)



# Checking for variables' levels

train_level = x_train.describe().iloc[1,:]

test_level = x_test.describe().iloc[1,:]



# Negate since we want the columns that are different

truth = ~(train_level == test_level)



col = x_train.columns[truth][0]

print(x_train.loc[:,col].value_counts())

print(x_test.loc[:,col].value_counts())



# Convert categorical variables into dummy variables of type int

x_traindummy = pd.get_dummies(x_train, drop_first=True, dtype=int)

x_testdummy = pd.get_dummies(x_test, drop_first=True, dtype=int)



# Encode response variable to 0 and 1

Encoder_y = LabelEncoder()

y_trainencoded = Encoder_y.fit_transform(y_train)

y_testencoded = Encoder_y.fit_transform(y_test)
# Align both train and test sets to ensure columns are the same (Since test-set's cap-surface variable has only 3 levels while train-set has 4)

x_finaltrain, x_finaltest = x_traindummy.align(x_testdummy, fill_value=np.int32(0), axis=1)
data.iloc[:,0].value_counts()
# Convert data type from int to float otherwise there would be DataConversionWarning from StandardScaler

x_finaltrain = x_finaltrain.astype(float)

x_finaltest = x_finaltest.astype(float)



# Cross-validation to determine parameter for regularisation

# Split into 10 folds

kf = KFold(n_splits=10, shuffle=True, random_state=10)
# =============================================================================

# Logistic Regression with Ridge

# =============================================================================

# Range of parameters to test

c_logreg = np.linspace(0,2,21)

c_logreg = c_logreg[1:] # We don't want 0 since 1/0 will produce an error



# Does a grid search over all parameter values and refits entire dataset using best parameters 

parameterslogreg = {'clf__C':c_logreg}

pipelogreg = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(random_state=10, penalty='l2', solver='liblinear'))])

logreg = GridSearchCV(pipelogreg, parameterslogreg, cv=kf)

logreg.fit(x_finaltrain, y_trainencoded)



print("The confusion matrix is")

print(confusion_matrix(y_testencoded, logreg.predict(x_finaltest)))

print("Logistic regression with L2 norm accuracy is", logreg.score(x_finaltest, y_testencoded))
# =============================================================================

# Adaboost classification tree

# =============================================================================

# Range of parameters to test

depth = [1,3,5]

num_est = [50, 100, 150]

rate = [0.0001, 0.01, 1]



# Does a grid search over all parameter values and refits entire dataset using best parameters 

parameterstree = {'clf__base_estimator__max_depth':depth, 'clf__n_estimators':num_est, 'clf__learning_rate':rate}

DTC = DecisionTreeClassifier(random_state=10)

ABC = AdaBoostClassifier(base_estimator = DTC, random_state=10)

pipeada = Pipeline([('scale', StandardScaler()), ('clf', ABC)]) 

adatree = GridSearchCV(pipeada, parameterstree, cv=kf)

adatree.fit(x_finaltrain, y_trainencoded) 



print("The confusion matrix is")

print(confusion_matrix(y_testencoded, adatree.predict(x_finaltest)))

print("Adaboosted classification tree accuracy is", adatree.score(x_finaltest, y_testencoded))
# =============================================================================

# Random Forest

# =============================================================================

# Range of parameters to test

depth = np.linspace(1,10,4)

trees = np.linspace(100,400,4)

trees = trees.astype(np.int64)

m = ['sqrt','log2']



# Does a grid search over all parameter values and refits entire dataset using best parameters 

parametersforest = {'clf__n_estimators':trees, 'clf__max_depth':depth, 'clf__max_features':m}

pipeforest = Pipeline([('scale', StandardScaler()), ('clf', RandomForestClassifier(random_state=10))]) 

rforest = GridSearchCV(pipeforest, parametersforest, cv=kf)

rforest.fit(x_finaltrain, y_trainencoded)



print("The confusion matrix is")

print(confusion_matrix(y_testencoded, rforest.predict(x_finaltest)))

print("Random forest accuracy is", rforest.score(x_finaltest, y_testencoded))

# =============================================================================

# SVM

# =============================================================================

# Range of parameters to test

c_svc = np.linspace(-5, 5, 11)

c_svc = [10**i for i in c_svc]



# Does a grid search over all parameter values and refits entire dataset using best parameters 

parameterssvc = {'clf__C':c_svc}

pipesvc = Pipeline([('scale', StandardScaler()), ('clf', SVC(random_state=10, kernel='linear'))]) 

svc = GridSearchCV(pipesvc, parameterssvc, cv=kf)

svc.fit(x_finaltrain, y_trainencoded)



print("The confusion matrix is")

print(confusion_matrix(y_testencoded, svc.predict(x_finaltest)))

print("Support vector classifier with linear kernel accuracy is", svc.score(x_finaltest, y_testencoded))
