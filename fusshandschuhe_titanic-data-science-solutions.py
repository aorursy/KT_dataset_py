# ! pip install --upgrade pip

# ! pip install -U scikit-learn

# # data analysis and wrangling

# import pandas as pd

# import numpy as np

# import random as rnd



# # visualization

# import seaborn as sns

# import matplotlib.pyplot as plt

# %matplotlib inline



# # machine learning

# from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC, LinearSVC

# from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.naive_bayes import GaussianNB

# from sklearn.linear_model import Perceptron

# from sklearn.linear_model import SGDClassifier

# from sklearn.tree import DecisionTreeClassifier

! pip install --upgrade pip

! pip install -U scikit-learn

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
print(train_df.columns.values)
# preview the data

train_df.head()
train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()

# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.

# Review Parch distribution using `percentiles=[.75, .8]`

# SibSp distribution `[.68, .69]`

# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
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
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
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
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
test_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
## XGBoost ---------------------- (added by JT)

import xgboost as xgb

#booster [default=gbtree] change to gblinear to see. gbtree almost always outperforms though

# xgboost = xgb.XGBClassifier(max_depth=14, n_estimators=1000, learning_rate=0.05,colsample_bytree=1)  #hyperparams

xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

xgboost.fit(X_train, Y_train)

Y_pred= xgboost.predict(X_test)

xgboost.score(X_train, Y_train)

acc_xgboost = round(xgboost.score(X_train, Y_train) * 100, 2)

acc_xgboost
## Extra Tree Regressor--------------------(added by JT)

from sklearn.tree import ExtraTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run

etr = ExtraTreeRegressor()

# Fit model

etr.fit(X_train, Y_train)

Y_pred = etr.predict(X_test)

acc_etr =etr.score(X_train, Y_train)

acc_etr
#based off https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/

from keras.models import Sequential

from keras.layers import Dense



#I'm also messing with the hyperparms as I go. Here's some notes so not to try things many times

#sequential layer, relu, relu, sigmoid works best

#softmax->relu

model = Sequential([ Dense(40, activation='relu', input_shape=(8,)),  Dense(32, activation='relu'), Dense(1, activation='sigmoid'),])

# model.compile(optimizer='sgd',   loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam',   loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_train, Y_train),verbose=0)

# model.evaluate(X_test, Y_test)[1]

Y_pred = model.predict(X_test)

# acc_etr = model.score(X_test, Y_pred)

acc_NN = model.evaluate(X_train, Y_train)[1]

print("Neural Network accuracy: " + str(round(acc_NN * 100, 2)))

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()



# somethings wrong with the neural net predictions and it's not giving binary predictions. not worth fixing

# print(Y_pred)

# #inverse one hot encoding

# pred = list()

# for i in range(len(Y_pred)):

#     pred.append(np.argmax(Y_pred[i]))

# print(pred)

#Converting one hot encoded test label to label

# test = list()

# for i in range(len(y_test)):

#     test.append(np.argmax(y_test[i]))



## MPL Classifier ----------------------------------------------

from sklearn.neural_network import MLPClassifier

mpl = MLPClassifier(solver='lbfgs', alpha=1e-5,

hidden_layer_sizes=(5, 2), random_state=1)

mpl.fit(X_train, Y_train)

MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

Y_pred = mpl.predict(X_test)

acc_mpl =mpl.score(X_train, Y_train)

## MPL Classifier ----------------------------------------------

from sklearn.neural_network import MLPClassifier

mpl = MLPClassifier(solver='lbfgs', alpha=1e-5,

hidden_layer_sizes=(5, 2), random_state=1)

mpl.fit(X_train, Y_train)

MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

Y_pred = mpl.predict(X_test)

acc_mpl =mpl.score(X_train, Y_train)

print(acc_mpl)
from sklearn.model_selection import train_test_split

num_test = 0.3

X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = train_test_split(X_train, Y_train, test_size=num_test, random_state=100)
# Logistic Regression --------------------

logreg = LogisticRegression()

logreg.fit(X_train_sub, Y_train_sub)

Y_pred = logreg.predict(X_test_sub)

acc_log = round(logreg.score(X_test_sub, Y_test_sub) * 100, 2)

print("Logistic Regression: " + str(acc_log))



#SVC -----------------------------------

svc = SVC()

svc.fit(X_train_sub, Y_train_sub)

Y_pred = svc.predict(X_test_sub)

acc_svc = round(svc.score(X_test_sub, Y_test_sub) * 100, 2)

print("SVC: " + str(acc_svc))



#KNN ------------------------------

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train_sub, Y_train_sub)

Y_pred = knn.predict(X_test_sub)

acc_knn = round(knn.score(X_test_sub, Y_test_sub) * 100, 2)

print("KNN: " + str(acc_knn))



# Gaussian Naive Bayes ---------------------------------

gaussian = GaussianNB()

gaussian.fit(X_train_sub, Y_train_sub)

Y_pred = gaussian.predict(X_test_sub)

acc_gaussian = round(gaussian.score(X_test_sub, Y_test_sub) * 100, 2)

print("Gaussian Naive Bayes: " + str(acc_gaussian))



# Perceptron -------------------------------

perceptron = Perceptron()

perceptron.fit(X_train_sub, Y_train_sub)

Y_pred = perceptron.predict(X_test_sub)

acc_perceptron = round(perceptron.score(X_test_sub, Y_test_sub) * 100, 2)

print("Perceptron: " + str(acc_perceptron))



# Linear SVC ------------------------------------

linear_svc = LinearSVC()

linear_svc.fit(X_train_sub, Y_train_sub)

Y_pred = linear_svc.predict(X_test_sub)

acc_linear_svc = round(linear_svc.score(X_test_sub, Y_test_sub) * 100, 2)

print("Linear SVC: " + str(acc_linear_svc))



# Stochastic Gradient Descent -------------------------------

sgd = SGDClassifier()

sgd.fit(X_train_sub, Y_train_sub)

Y_pred = sgd.predict(X_test_sub)

acc_sgd = round(sgd.score(X_test_sub, Y_test_sub) * 100, 2)

print("Stochastic Grad Descent: " + str(acc_sgd))



# Decision Tree ---------------------------------

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train_sub, Y_train_sub)

Y_pred = decision_tree.predict(X_test_sub)

acc_decision_tree = round(decision_tree.score(X_test_sub, Y_test_sub) * 100, 2)

print("Decision Tree: " + str(acc_decision_tree))



#Random Forest --------------------------

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train_sub, Y_train_sub)

Y_pred = random_forest.predict(X_test_sub)

random_forest.score(X_test_sub, Y_test_sub)

acc_random_forest = round(random_forest.score(X_test_sub, Y_test_sub) * 100, 2)

print("Random Forest: " + str(acc_random_forest))



#XG Boost --------------------------------

xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

xgboost.fit(X_train_sub, Y_train_sub)

Y_pred= xgboost.predict(X_test_sub)

xgboost.score(X_test_sub, Y_test_sub)

acc_xgboost = round(xgboost.score(X_train, Y_train) * 100, 2)

print("XGBoost: " + str(acc_xgboost))



xgboost = xgb.XGBClassifier(max_depth=14, n_estimators=1000, learning_rate=0.05,colsample_bytree=1)

xgboost.fit(X_train_sub, Y_train_sub)

Y_pred= xgboost.predict(X_test_sub)

xgboost.score(X_test_sub, Y_test_sub)

acc_xgboost_hyp = round(xgboost.score(X_train, Y_train) * 100, 2)

print("XGBoost with tuning: " + str(acc_xgboost_hyp))



#Extra Tree Regressor ---------------------------------

etr = ExtraTreeRegressor()

etr.fit(X_train_sub, Y_train_sub)

Y_pred = etr.predict(X_test_sub)

acc_etr = round(etr.score(X_test_sub, Y_test_sub) * 100, 2)

print("Extra Tree Reg: " + str(acc_etr))





## MPL Classifier ----------------------------------------------

from sklearn.neural_network import MLPClassifier

mpl = MLPClassifier(solver='lbfgs', alpha=1e-5,

hidden_layer_sizes=(5, 2), random_state=1)

mpl.fit(X_train_sub, Y_train_sub)

MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

Y_pred = mpl.predict(X_test)

acc_mpl = round(mpl.score(X_test_sub, Y_test_sub)*100,2)

print("Multi Layer Perceptron: " + str(acc_mpl))



#Neural Network -------------------------------------

model = Sequential([ Dense(32, activation='relu', input_shape=(8,)),  Dense(32, activation='relu'), Dense(1, activation='sigmoid'),])

model.compile(optimizer='adam',   loss='binary_crossentropy', metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train_sub, Y_train_sub, batch_size=32, epochs=100, validation_data=(X_train_sub, Y_train_sub),verbose=0)

Y_pred = model.predict(X_test_sub)

acc_NN = round(100* model.evaluate(X_test_sub, Y_test_sub)[1],2)

print("Neural Network accuracy: " + str(acc_NN))

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()



plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
## Feature engineering

#Interactions. Consider the interaction between gender and class. Clearly most rich women survived.

from sklearn.preprocessing import LabelEncoder

interactions = train_df['Fare'] + train_df['Sex']

label_enc = LabelEncoder()

data_interaction = train_df.assign(fare_sex=label_enc.fit_transform(interactions))



#Other idea, consider if they work for the ship or not. Crew members are probably more likely to die. Is this info available?

x_all = data_interaction.drop(['Survived'], axis=1)

y_all = data_interaction['Survived']

X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = train_test_split(x_all, y_all, test_size=num_test, random_state=100)



xgboost = xgb.XGBClassifier(max_depth=14, n_estimators=1000, learning_rate=0.05,colsample_bytree=.8)

## Predictions ----------------

xgboost.fit(X_train_sub, Y_train_sub)

Y_pred= xgboost.predict(X_test_sub)

xgboost.score(X_test_sub, Y_test_sub)

acc_xgboost_tuned = round(xgboost.score(X_train_sub, Y_train_sub) * 100, 2)

print("Age+Fare Interactions w/ hyperparm optimized XGB: " + str(acc_xgboost_tuned))





#Random Forest --------------------------

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train_sub, Y_train_sub)

Y_pred = random_forest.predict(X_test_sub)

random_forest.score(X_test_sub, Y_test_sub)

acc_random_forest = round(random_forest.score(X_test_sub, Y_test_sub) * 100, 2)

print("Random Forest: " + str(acc_random_forest))



##Neural Network ---------------------

#how about a new neural net with this?

#add L2 regularization. 

# regularizer: include the squared values of those parameters in our overall loss function, and weight them by 0.01 in the loss function.

# dropout: neurons in the previous layer has a probability of 0.3 in dropping out during training.

from keras.layers import Dropout

from keras import regularizers

model = Sequential([ Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(9,)),  Dropout(0.3),  Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  Dropout(0.3), Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))])

model.compile(optimizer='sgd',   loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train_sub, Y_train_sub, batch_size=32, epochs=100, validation_data=(X_train_sub, Y_train_sub),verbose=0)

Y_pred = model.predict(X_test_sub)

acc_NN_interactions = model.evaluate(X_test_sub, Y_test_sub)[1]

print("Neural Network accuracy: " + str(round(acc_NN_interactions * 100, 2)))

##It actually does wose than with the interaction. Interesting. Though based on how neural nets work it makes sense that they're similar

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()



plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
##based on code from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74



from sklearn.model_selection import RandomizedSearchCV

# from sklearn.ensemble import RandomForestRegressor



# Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

n_estimators = list(range(50,500))

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth = list(range(5,11))

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train_sub, Y_train_sub)

print(rf_random.best_params_)





def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

#     print('Model Performance')

#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

#     print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy

base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)

base_model.fit(X_train_sub, Y_train_sub)

base_accuracy = evaluate(base_model, X_train_sub, Y_train_sub)



best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, X_train_sub, Y_train_sub)



# print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
from pprint import pprint

pprint(rf_random.best_params_)

print(rf_random.best_score_)
##based on code from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

interactions = train_df['Fare'] + train_df['Sex']

label_enc = LabelEncoder()

data_interaction = train_df.assign(fare_sex=label_enc.fit_transform(interactions))



#Other idea, consider if they work for the ship or not. Crew members are probably more likely to die. Is this info available?

x_all = data_interaction.drop(['Survived'], axis=1)

y_all = data_interaction['Survived']

X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = train_test_split(x_all, y_all, test_size=num_test, random_state=100)



# Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)

n_estimators = list(range(50,500))

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth = list(range(5,11))

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train_sub, Y_train_sub)

print(rf_random.best_params_)





base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)

base_model.fit(X_train_sub, Y_train_sub)

base_accuracy = evaluate(base_model, X_test_sub, Y_test_sub)



best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, X_test_sub, Y_test_sub)

rf_random.best_estimator_

acc_rf_tuned = round(rf_random.best_score_*100,2)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
# from pprint import pprint

# pprint(rf_random.best_params_)

# print(rf_random.best_score_)
#import plotly.graph_objs as go



#print(models['Score'].astype(float).corr(models['Score']))

#which models correlate best with one another

#heatmapdata = [

#    go.Heatmap(

#        z= models['Score'].astype(float).corr(models['Score']) ,

#        x=models['Score'],

#        y= models['Score'],

#          colorscale='Viridis',

#            showscale=True,

#            reversescale = True

#    )

#]





#trying to stack

base_learners = [

        #('base_learners_1', SVC()),

        #('base_learners_2', KNeighborsClassifier(n_neighbors = 3)),

        #('base_learners_3', GaussianNB()),

        #('base_learners_4', Perceptron()),

        #('base_learners_5', LinearSVC()),

        #('base_learners_6', SGDClassifier()),

        ('base_learners_7', DecisionTreeClassifier()),

        ('base_learners_8', RandomForestClassifier(n_estimators=1600, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', max_depth= 10, bootstrap= True)),

        ('base_learners_9', xgb.XGBClassifier(max_depth=14, n_estimators=1000, learning_rate=0.05,colsample_bytree=.8))

        #('base_learners_10', ExtraTreeClassifier())

        #('base_learners_11', Sequential([ Dense(40, activation='relu', input_shape=(8,)),  Dense(32, activation='relu'), Dense(1, activation='sigmoid'),]),model.compile(optimizer='sgd',   loss='binary_crossentropy', metrics=['accuracy']))

    ]



clf = StackingClassifier(estimators = base_learners, final_estimator=LogisticRegression(), cv=10)

clf.fit(X_train_sub, Y_train_sub).score(X_test_sub, Y_test_sub)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree','XGBoost', 'XGBoost + Hyperparams', 'Keras Neural Net', 'XGBoost + Hyp + Interactions','RF with hyperparams','MPL'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree,acc_xgboost, acc_xgboost_hyp, acc_NN, acc_xgboost_tuned,acc_rf_tuned,acc_mpl]})

models.sort_values(by='Score', ascending=False)
## Best Predictions ----------------



## Option 1: xgboost with interaction + hyperparameter tuning: ----------------------

# interactions = train_df['Fare'] + train_df['Sex']

# label_enc = LabelEncoder()

# data_interaction = train_df.assign(class_sex=label_enc.fit_transform(interactions))

# interactions_test = test_df['Fare'] + test_df['Sex']

# test_w_interaction = test_df.copy().assign(class_sex=label_enc.fit_transform(interactions_test))

# x_all = data_interaction.drop(['Survived'], axis=1)

# y_all = data_interaction['Survived']

# # X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = train_test_split(x_all, y_all, test_size=num_test, random_state=100)

# X_test  = test_w_interaction.drop("PassengerId", axis=1).copy()

# xgboost = xgb.XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.05,colsample_bytree=.8)



## Predictions ----------------

# xgboost.fit(x_all, y_all)

# Y_pred= xgboost.predict(X_test)



##option 2: Random Forest with interaction -------------------------

# interactions = train_df['Fare'] + train_df['Sex']

# label_enc = LabelEncoder()

# data_interaction = train_df.assign(class_sex=label_enc.fit_transform(interactions))

# interactions_test = test_df['Fare'] + test_df['Sex']

# test_w_interaction = test_df.copy().assign(class_sex=label_enc.fit_transform(interactions_test))

# x_all = data_interaction.drop(['Survived'], axis=1)

# y_all = data_interaction['Survived']

# # X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = train_test_split(x_all, y_all, test_size=num_test, random_state=100)

# X_test  = test_w_interaction.drop("PassengerId", axis=1).copy()

# random_forest = RandomForestClassifier(max_depth = 10, n_estimators=100)

# random_forest.fit(x_all, y_all)

# Y_pred = random_forest.predict(X_test)





##the default submission: the untouched random forest ------------------------

# X_train = train_df.drop("Survived", axis=1)

# Y_train = train_df["Survived"]

# X_test  = test_df.drop("PassengerId", axis=1).copy()

# random_forest = RandomForestClassifier(n_estimators=100)

# random_forest.fit(X_train, Y_train)

# Y_pred = random_forest.predict(X_test)





## random forest classifier with auto hyperparam tuning --------------------------------

# X_train = train_df.drop("Survived", axis=1)

# Y_train = train_df["Survived"]

# X_test  = test_df.drop("PassengerId", axis=1).copy()

# # random_forest = RandomForestClassifier(n_estimators=1600, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', max_depth= 10, bootstrap= True)

# random_forest = RandomForestClassifier(n_estimators=369, min_samples_split=5, min_samples_leaf=1, max_features='auto', max_depth= 5, bootstrap= True)

# random_forest.fit(X_train, Y_train)

# Y_pred = random_forest.predict(X_test)



## random forest classifier with interaction and auto hyperparm tuning --------------------BEST

interactions = train_df['Fare'] + train_df['Sex']

label_enc = LabelEncoder()

data_interaction = train_df.assign(class_sex=label_enc.fit_transform(interactions))

interactions_test = test_df['Fare'] + test_df['Sex']

test_w_interaction = test_df.copy().assign(class_sex=label_enc.fit_transform(interactions_test))

x_all = data_interaction.drop(['Survived'], axis=1)

y_all = data_interaction['Survived']

X_test  = test_w_interaction.drop("PassengerId", axis=1).copy()

random_forest = RandomForestClassifier(n_estimators=369, min_samples_split=5, min_samples_leaf=1, max_features='auto', max_depth= 5, bootstrap= True)

random_forest.fit(x_all, y_all)

Y_pred = random_forest.predict(X_test)





## MPL ---------------------------------------------------------------

# mpl = MLPClassifier(solver='lbfgs', alpha=1e-5,

# hidden_layer_sizes=(5, 2), random_state=1)

# mpl.fit(X_train, Y_train)

# MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

# Y_pred = mpl.predict(X_test)





## random forest regressor with auto hyperparam tuning

# X_train = train_df.drop("Survived", axis=1)

# Y_train = train_df["Survived"]

# X_test  = test_df.drop("PassengerId", axis=1).copy()

# clf.fit(X_train, Y_train)

# Y_pred = clf.predict(X_test)





submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)