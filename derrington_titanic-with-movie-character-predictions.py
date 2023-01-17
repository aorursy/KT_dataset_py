# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the dataset

train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
test_data_PassengerId = pd.read_csv("../input/titanic/test.csv")
movie_data = pd.read_csv("../input/movie-test-with-dummies/movie_test.csv")
# Look at the first few rows to get an idea of what data we are working with
train_data.head(10)
train_data.tail(10)
# Let's get some basic statistical properties for the training set (for numerical columns)
train_data.describe()
print("Number of passengers with unknown age: {}".format(train_data["Age"].isnull().sum()))
# Deal with age first. Idea is to find mean and std and fill Nan values with random numbers in range [mean-std, mean+std]

# Get mean and std of age as well as number of Nan values for train set
train_age_mean = train_data["Age"].mean()
train_age_std = train_data["Age"].std()
train_age_nan = train_data["Age"].isnull().sum()

# Get mean and std of age as well as number of Nan values for train set
test_age_mean = test_data["Age"].mean()
test_age_std = test_data["Age"].std()
test_age_nan = test_data["Age"].isnull().sum()

# Generate enough random numbers in range [mean-0.5*std, mean+0.5*std] for train set
train_age_rand = np.random.randint(train_age_mean - 0.5*train_age_std, train_age_mean + 0.5*train_age_std, size = train_age_nan)

# Generate enough random numbers in range [mean-0.5*std, mean+0.5*std] for test set
test_age_rand = np.random.randint(test_age_mean - 0.5*test_age_std, test_age_mean + 0.5*test_age_std, size = test_age_nan)

# Create a figure for plotting original and preprocessed age data


# Original age data (simply don't plot Nan values)
plt.figure(figsize=(15,7))
#train_data['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
plt.subplot(1,2,1)
plt.style.use('bmh')
plt.xlabel('Age (original data)')
plt.ylabel('Survived')
plt.title('Original Age vs Survival')
plt.hist(train_data.Age[(np.isnan(train_data.Age) == False)], bins= 15, alpha = 0.4, color = 'r', label = 'Before')
plt.hist(train_data.Age[(np.isnan(train_data.Age) == False) & (train_data.Survived == 1)], bins= 15, alpha = 0.4, color = 'b', label = 'After')
#plt.hist(data.Age[data.Age != np.NaN])
plt.legend(loc = 'upper right')


# Preprocessed age data (first fill the Nan values with the random numbers and then plot them)
train_data.loc[train_data.Age.isnull(), 'Age'] = train_age_rand
test_data.loc[test_data.Age.isnull(), 'Age'] = test_age_rand


#train_data["Age"].hist(bins=70, ax = axis2)
plt.subplot(1,2,2)
plt.style.use('bmh')
plt.xlabel('Age (preprocessed data)')
plt.ylabel('Survived')
plt.title('Preprocessed Age vs Survival')
plt.hist(train_data.Age, bins= 15, alpha = 0.4, color = 'r', label = 'Before')
plt.hist(train_data.Age[(train_data.Survived == 1)], bins= 15, alpha = 0.4, color = 'b', label = 'After')
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()


# Using only the preprocessed data, we can further investigate the relationship betweena ge and survival
# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()

# number of survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(40,4))
survival_number_by_age = train_data[["Age", "Survived"]].groupby(['Age'],as_index=False).sum()
sns.barplot(x='Age', y='Survived', data=survival_number_by_age)
# Fill missing Embarked data in training set to avoid entire rows being dropped
train_data.loc[train_data.Embarked.isnull(), 'Embarked'] = 'S'

# Drop PassengerId, Name and Ticket Number from training set 
train_data = train_data.drop(["PassengerId", "Cabin", "Ticket"], axis = 1)

# Drop remaining Nan values
train_data = train_data.dropna() 
# There is one passenger in test set with missing Fare data. Below, we identify him.
test_data.loc[test_data.Fare.isnull()]
# As this Passenger is travelling in 3rd class, it will be sensible to fill his Fare data with th emean fare paid by other third class passengers. 
test_3class = test_data.loc[test_data['Pclass'] == 1]
mean_fare = test_3class[["Fare"]].mean()
test_data.ix[152, 'Fare'] = mean_fare[0]
test_data.ix[152]

# Drop PassengerId, Name and Ticket Number from test set 
test_data = test_data.drop(["PassengerId", "Cabin", "Ticket"], axis = 1)
train_data.head(20)

# First compare class to age
fig = sns.FacetGrid(train_data,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade='True')
oldest = train_data['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# Then a simple bar graph of survived passengers by class
fig, axis1 = plt.subplots(1,1,figsize=(3,4))
survival_number_by_class = train_data[["Pclass", "Survived"]].groupby(['Pclass'],as_index=False).sum()
sns.barplot(x='Pclass', y='Survived', data=survival_number_by_class)

sns.factorplot('Pclass','Survived',data=train_data)
# First compare gender to age
fig = sns.FacetGrid(train_data,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade='True')
oldest = train_data['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# Then a simple bar graph of survived passengers by class
fig, axis1 = plt.subplots(1,1,figsize=(3,4))
survival_number_by_class = train_data[["Sex", "Survived"]].groupby(['Sex'],as_index=False).sum()
sns.barplot(x='Sex', y='Survived', data=survival_number_by_class)

fig = plt.figure(figsize=(30,4))

#create a plot of two subsets, male and female, of the survived variable.
#After we do that we call value_counts() so it can be easily plotted as a bar graph. 
#'barh' is just a horizontal bar graph
df_male = train_data.Survived[train_data.Sex == 'male'].value_counts().sort_index()
df_female = train_data.Survived[train_data.Sex == 'female'].value_counts().sort_index()


ax1 = fig.add_subplot(141)
df_male.plot(kind='barh',label='Male', color = 'blue', alpha=0.9)
plt.title("Male Survival (raw) "); plt.legend(loc='best')
 

#adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(142)
(df_male/float(df_male.sum())).plot(kind='barh',label='Male', color = 'blue', alpha=0.9)  
plt.title("Male survival (proportional)"); plt.legend(loc='best')

ax3 = fig.add_subplot(143)
df_female.plot(kind='barh', color='#FA2379',label='Female', alpha=0.9)
plt.title("Female surivival (raw)"); plt.legend(loc='best')

ax4 = fig.add_subplot(144)
(df_female/float(df_female.sum())).plot(kind='barh',color='#FA2379',label='Female', alpha=0.9)
plt.title("Female survival (proportional)"); plt.legend(loc='best')

plt.tight_layout()
family_data = train_data[['SibSp', 'Parch', 'Survived']].copy()
family_data['FamScore'] = family_data['SibSp'] + family_data['Parch']
family_data.drop(['SibSp','Parch'], axis=1)
columnsTitles=["FamScore","Survived"]
family_data=family_data.reindex(columns=columnsTitles)

# Get distribution of family sizes
family_data.hist('FamScore')

#Transform a nonzero FamScore to "With Family" and a zero FamScore to "Alone"
family_data['FamScore'].loc[family_data['FamScore']>0] = "With Family"
family_data['FamScore'].loc[family_data['FamScore']==0] = "Alone"
sns.factorplot('FamScore',data=family_data,kind='count',palette='Blues')

fig = plt.figure(figsize=(30,4))
#create a plot of two subsets, with family and alone, of the survived variable.
#After we do that we call value_counts() so it can be easily plotted as a bar graph. 
#'barh' is just a horizontal bar graph
df_fam = family_data.Survived[family_data.FamScore == 'With Family'].value_counts().sort_index()
df_alone = family_data.Survived[family_data.FamScore == 'Alone'].value_counts().sort_index()


ax1 = fig.add_subplot(141)
df_fam.plot(kind='barh',label='With Family', color = 'orange', alpha=0.5)
plt.title("With Family Survival (raw) "); plt.legend(loc='best')
 

#adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(142)
(df_fam/float(df_fam.sum())).plot(kind='barh',label='With Family', color = 'orange', alpha=0.5)  
plt.title("With Family survival (proportional)"); plt.legend(loc='best')

ax3 = fig.add_subplot(143)
df_alone.plot(kind='barh', color='green',label='Alone', alpha=0.5)
plt.title("Alone surivival (raw)"); plt.legend(loc='best')

ax4 = fig.add_subplot(144)
(df_alone/float(df_alone.sum())).plot(kind='barh',color='green',label='Alone', alpha=0.5)
plt.title("Alone survival (proportional)"); plt.legend(loc='best')

plt.tight_layout()
fig = sns.FacetGrid(train_data,hue='Survived',aspect=4,size=5)
fig.map(sns.kdeplot,'Fare',shade='True')
oldest = train_data['Fare'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

train_data.Embarked.value_counts().plot(kind='bar', figsize=(5,5))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")

fig = plt.figure(figsize=(30,4))
#create a plot of two subsets, with family and alone, of the survived variable.
#After we do that we call value_counts() so it can be easily plotted as a bar graph. 
#'barh' is just a horizontal bar graph
df_C = train_data.Survived[train_data.Embarked == 'C'].value_counts().sort_index()
df_Q = train_data.Survived[train_data.Embarked == 'Q'].value_counts().sort_index()
df_S = train_data.Survived[train_data.Embarked == 'S'].value_counts().sort_index()

ax1 = fig.add_subplot(161)
df_C.plot(kind='barh',label='Cherbourg', color = '#377eb8', alpha=0.6)
plt.title("Cherbourg (raw) "); plt.legend(loc='best')
 

#adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(162)
(df_C/float(df_C.sum())).plot(kind='barh',label='Cherbourg', color = '#377eb8', alpha=0.6)  
plt.title("Cherbourg (proportional)"); plt.legend(loc='best')

ax3 = fig.add_subplot(163)
df_Q.plot(kind='barh', color='#4daf4a',label='Queenstown', alpha=0.6)
plt.title("Queenstown (raw)"); plt.legend(loc='best')

ax4 = fig.add_subplot(164)
(df_Q/float(df_Q.sum())).plot(kind='barh',color='#4daf4a',label='Queenstown', alpha=0.6)
plt.title("Queenstown (proportional)"); plt.legend(loc='best')

ax5 = fig.add_subplot(165)
df_S.plot(kind='barh', color='#e41a1c',label='Southampton', alpha=0.6)
plt.title("Southampton (raw)"); plt.legend(loc='best')

ax6 = fig.add_subplot(166)
(df_S/float(df_S.sum())).plot(kind='barh',color='#e41a1c',label='Southampton', alpha=0.6)
plt.title("Southampton (proportional)"); plt.legend(loc='best')

plt.tight_layout()
# First split data into independent and dependent variables
y = train_data['Survived'].copy()
X = train_data.drop(["Survived"], axis = 1)

# Deal with categorical features
# Note the dependent variable (Survived) is already binary and therefore the only things that need to be encoded are Sex and Embarked.
X_hot = pd.get_dummies(X, prefix=['Sex', 'Embarked'], columns=['Sex', 'Embarked'])
X_hot = X_hot.drop('Name',axis=1)
X_hot[:10]

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_hot = sc_X.fit_transform(X_hot)

# Split training data to allow us to perform hold-out validation and cross-validation
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_hot, y, test_size = 0.1, random_state = 0)

# Do the same for test data as we'll need it for making final predictions (we don't need to split the test data though)
test_hot = pd.get_dummies(test_data, prefix=['Sex', 'Embarked'], columns=['Sex', 'Embarked'])
test_hot = test_hot.drop('Name',axis=1)
test_hot = sc_X.fit_transform(test_hot)

# Generic code to run on all of our different models
from sklearn.model_selection import GridSearchCV, cross_val_score
def train_test_model(model, hyperparameters, X_train, X_test, y_train, y_test, folds = 5):
    """
    Given a [model] and a set of possible [hyperparameters], an exhaustive search is performed across all possible hyperparameter values. The optimum model is returned.
    We then print out some useful info.
    """
    optimized_model = GridSearchCV(model, hyperparameters, cv = folds, n_jobs = -1)
    optimized_model.fit(X_train, y_train)
    y_pred = optimized_model.predict(X_valid)
    print('Optimized parameters: {}'.format(optimized_model.best_params_))
    print('Model accuracy (hold-out validation): {:.2f}%'.format(optimized_model.score(X_test, y_test)*100))
    # Take our best model and run it on different train/valid splits and take the mean accuracy score. n_jobs=-1 allows all CPU corees to be used.
    kfold_score = np.mean(cross_val_score(
            optimized_model.best_estimator_, np.append(X_train, X_test, axis = 0), 
            np.append(y_train, y_test), cv = folds, n_jobs = -1))
    print('Model accuracy ({}-fold cross validation): {:.2f}%'.format(folds, kfold_score*100))
    return optimized_model
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(X.corr(),cmap="YlGnBu")
%%time
from sklearn import linear_model
lr_model = train_test_model(linear_model.LogisticRegression(random_state = 0), {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight': [None, 'balanced']}, X_train, X_valid, y_train, y_valid)
%%time
from sklearn.tree import DecisionTreeClassifier
# Use our function train_test_model with a variety of values for the hyperparameter C (inverse regularization strength) and class_weight (different weights can be given to different features).
dt_model = train_test_model(DecisionTreeClassifier(random_state = 0), {'min_samples_split': [2, 4, 8, 16], 'min_samples_leaf': [1, 3, 5, 10], 'max_depth': [2,3,4,5, None], 'class_weight': [None, 'balanced']}, X_train, X_valid, y_train, y_valid)
%%time
from sklearn.ensemble import RandomForestClassifier
# Use our function train_test_model with a variety of values for the hyperparameter C (inverse regularization strength) and class_weight (different weights can be given to different features).
rf_model = train_test_model(RandomForestClassifier(random_state = 0), {'min_samples_split': [2, 4, 8, 16], 'min_samples_leaf': [1, 3, 5, 10], 'max_depth': [3, None], 'class_weight': [None, 'balanced']}, X_train, X_valid, y_train, y_valid)
%%time
from sklearn.svm import SVC
# Use our function train_test_model with a variety of values for the hyperparameter C (inverse regularization strength) and class_weight (different weights can be given to different features).
svc_model = train_test_model(SVC(random_state = 0), {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': np.logspace(-9, 3, 13), 'kernel': ['rbf','linear']}, X_train, X_valid, y_train, y_valid)
%%time
from sklearn.neighbors import KNeighborsClassifier
# Use our function train_test_model with a variety of values for the hyperparameter C (inverse regularization strength) and class_weight (different weights can be given to different features).
knn_model = train_test_model(KNeighborsClassifier(), {'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21,23,25]}, X_train, X_valid, y_train, y_valid)
%%time
import keras
# This is required to initialise our ANN
from keras.models import Sequential 
# This is required to build the layers of our ANN
from keras.layers import Dense 
# We initialise the ANN by building an object of the sequential class and then add layes below.
classifier = Sequential() 
# Add the hidden layers
classifier.add(Dense(input_dim = 10, activation = 'relu', units = 6, kernel_initializer = 'uniform')) 
classifier.add(Dense(activation = 'tanh', units = 6, kernel_initializer = 'uniform'))
classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))
classifier.add(Dense(activation = 'tanh', units = 6, kernel_initializer = 'uniform'))
classifier.add(Dense(activation = 'sigmoid', units = 6, kernel_initializer = 'uniform'))
# Add the output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
# Tell the ANN which loss function to use when applying stochastic gradient descent to find optimal weights
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fit classifier to training data
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Make prediction on validation set
y_pred = classifier.predict(X_valid)
y_pred = (y_pred > 0.5) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, y_pred)
correct = cm[0][0]+cm[1][1]
wrong = cm[1][0]+cm[0][1]
accuracy=(correct/(correct+wrong))*100
print("Model accuracy: {:.2f}".format(accuracy))
test_hot.shape
# Call dt_model which loads the Decision Tree Classifier (with optimal hyperparameters) and use it to make predictions on the test set.
best_model = classifier
test_pred = best_model.predict(test_hot)
#test_pred = (test_pred > 0.5)
for i in range(len(test_pred)):
    if test_pred[i] < 0.5:
        test_pred[i] = int(0)
    else:
        test_pred[i] = int(1)
test_pred = test_pred.astype(np.int64)
test_pred = test_pred.T
test_pred = test_pred.reshape(1,418)
test_pred = test_pred[0]
print(test_pred)
submission = pd.DataFrame({
        "PassengerId": test_data_PassengerId["PassengerId"],
        "Survived": test_pred
    })
submission.to_csv('titanic.csv', index=False)
submission.head()
# Explore the movie dataset
movie_data
movie_data = movie_data.drop(["PassengerId", "Cabin", "Ticket"], axis = 1)
movie_hot = pd.get_dummies(movie_data, prefix=['Sex', 'Embarked'], columns=['Sex', 'Embarked'])
movie_hot = movie_hot.drop('Name',axis=1)
movie_hot = sc_X.fit_transform(movie_hot)
movie_hot
movie_hot.shape
best_model = dt_model
movie_pred = best_model.predict(movie_hot)
Names = ['Jack','Rose','Calvin']
Dead = ['die', 'survive to tell the tale to Paramount Pictures']
for i in range(3):
    print("{} will {}".format(Names[i], Dead[movie_pred[i]]))
