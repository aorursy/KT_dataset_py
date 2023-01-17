# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# Load the data



passengers = pd.read_csv('../input/train.csv')

passengers.head()
passengers_test = pd.read_csv('../input/test.csv')

passengers_test.head()
gender_submission = pd.read_csv('../input/gender_submission.csv')

gender_submission.head()
passengers.shape
passengers.info()
passengers.head()
passengers.describe()
passengers.corr()
def correlation_matrix(df):

   from matplotlib import pyplot as plt

   from matplotlib import cm as cm



   fig = plt.figure(figsize=(25,12))

   ax1 = fig.add_subplot(111)

   cmap = cm.get_cmap('binary', 30)

   cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)

   ax1.grid(True)

   plt.title('Feature Correlation')

   labels=list(df.columns)

   #ax1.set_xticklabels(labels,fontsize=5)

   #ax1.set_yticklabels(labels,fontsize=5)

   # Add colorbar, make sure to specify tick locations to match desired ticklabels

   fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])

   plt.show()



correlation_matrix(passengers) # Good to see the current numeric features are not too correlated
# Fill the nan values in the age column

print(passengers['Age'].values)



#Calculate the percentage of entries with nan age in the entries with ground truth label of "not-survived".



number_nan_age = passengers['Age'].isna().sum()

print(number_nan_age/len(passengers['Age']))



print(passengers.isnull().sum())

# Fill the nan values with the average of that column for some other features



def preprocessing(passengers):

    

    passengers['Age_old'] = passengers['Age']

    

    passengers['Age'].fillna(value = passengers.Age.mean(), inplace = True) #

    

    #passengers['Sex'] = passengers['Sex'].apply(lambda x: 1 if x=='male' else 0)

    passengers['Sex'] = passengers['Sex'].map({"male":1, "female":0})

    

    passengers['Cabin_old'] = passengers['Cabin']



    #mean_SibSp = passengers.SibSp.mean()

    

    #passengers['SibSp'].fillna(value = mean_SibSp, inplace = True)



    #mean_Parch = passengers.Parch.mean()

    #passengers['Parch'].fillna(value = mean_Parch, inplace = True)



    mean_Fare = passengers.Fare.mean()

    passengers['Fare'].fillna(value = mean_Fare, inplace = True)



    # Create a Upper column

    passengers['Upper'] = passengers['Pclass'].apply(lambda x: 1 if x==1 else 0)



    # Create a Middle column

    passengers['Middle'] = passengers['Pclass'].apply(lambda x: 1 if x==2 else 0)



    # Create a Lower column

    passengers['Lower'] = passengers['Pclass'].apply(lambda x: 1 if x==3 else 0)



    # Create a Have_Age column

    passengers['Have_Age'] = passengers['Age_old'].isnull().apply(lambda x: 1 if not x else 0)



    # Create a Have_Cabin column

    passengers['Have_Cabin'] = passengers['Cabin'].isnull().apply(lambda x: 1 if not x else 0)

    

    return passengers



passengers = preprocessing(passengers)

passengers.head()
passengers.describe()
correlation_matrix(passengers)
print(passengers.isna().sum())
print(passengers['Have_Cabin'].sum())
# Select the desired features

selected_features = ['Have_Age',  'Have_Cabin', 'Sex', 'Parch', 'Fare', 'Middle',  'Upper', 'Lower', 'SibSp',  'Age'] #,





# Select the desired features

features = passengers[selected_features]

survival = passengers['Survived']
print("{percentage}% of total passengers in test data don't survive!!!".format(percentage=(len(survival)-survival.sum())/len(survival)))
cnt=0

for i in range(len(passengers['Survived'])):

    if passengers['Age_old'].isnull()[i] and passengers['Survived'][i]==0:

        cnt +=1

print("{percentage}% of total passengers in test data BOTH don't have age information AND don't survive!!!".format(percentage=100*cnt/len(passengers['Survived']))) #14%

print("{percentage}% of non-survived passengers don't have age information!!!".format(percentage=100*cnt/(len(survival)-survival.sum())))
cnt=0

for i in range(len(passengers['Survived'])):

    if passengers['Cabin_old'].isnull()[i] and passengers['Survived'][i]==0:

        cnt +=1

print("{percentage}% of total passengers in test data BOTH don't have cabin information AND don't survive!!!".format(percentage=100*cnt/len(passengers['Survived']))) #14%

print("{percentage}% of non-survived passengers don't have cabin information!!!".format(percentage=100*cnt/(len(survival)-survival.sum())))
# Perform train, test, split



train_features, test_features, train_labels, test_labels = train_test_split(features, survival, train_size = 0.25)



# Scale the feature data so it has mean = 0 and standard deviation = 1

normalize = StandardScaler()

train_features = normalize.fit_transform(train_features)

test_features = normalize.transform(test_features)



# Create and train the model

model = LogisticRegression(random_state=200, solver='saga', max_iter=1000)

model.fit(train_features, train_labels)



# Score the model 

print("Score on the training data is {score_train} and score on the test data is {score_test} ".

      format(score_train=model.score(train_features, train_labels), score_test=model.score(test_features, test_labels)))



# Analyze the coefficients



print(model.coef_)

print(list(zip(selected_features, model.coef_[0])))



plt.figure(figsize=(15,12))

plt.bar(selected_features,  model.coef_[0])

plt.xlabel("feature")

plt.ylabel("model coefficient")

plt.title("Impacts of different features in our model")

plt.show()

# Process test data



# Update sex column to numerical



passengers_test
passengers_test=preprocessing(passengers_test)



# Select the desired features

features_test = passengers_test[selected_features]





passengers_test.isnull().sum()
features_test = normalize.transform(np.array(features_test)) 

print(features_test)
my_predict_survival = model.predict(features_test)



print(my_predict_survival)



print(model.predict_proba(features_test))
import xgboost as xgb



X=train_features

y=train_labels



#Decision trees as base learners

# Create the training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)



# Instantiate the XGBRegressor: xg_reg

xg_reg = xgb.XGBClassifier(n_estimators=1000, seed=123, objective="binary:logistic",booster="gbtree", n_jobs=10)



# Fit the regressor to the training set

xg_reg.fit(X_train, y_train)



# Predict the labels of the test set: preds

preds = xg_reg.predict(X_test)





#Evaluating model quality



# Create the DMatrix

dmatrix = xgb.DMatrix(data=X_test, label=y_test)



# Create the parameter dictionary: params

params = {"objective":"reg:logistic", "max_depth":4}



# Perform cross-validation: cv_results

cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=5, num_boost_round=50, metrics="error", as_pandas=True, seed=123)



# Print cv_results

print(cv_results)



# Extract and print final boosting round metric

print((cv_results["test-error-mean"]).tail(1))

print("accuracy:", 1-(cv_results["test-error-mean"]).tail(1)) 





# Plot the tree

xgb.plot_tree(xg_reg)

plt.show()

# Plot the feature importances

xgb.plot_importance(xg_reg)

plt.show()
# Predict the labels of the unknown set: preds_unknown

preds_unknown = xg_reg.predict(features_test)



np.array(preds_unknown)
my_predict_survival
my_submission = [('PassengerId', list(gender_submission.PassengerId)),

         ('Survived', list(my_predict_survival)),

         ]

my_submission = pd.DataFrame.from_items(my_submission)





my_submission_xgb = [('PassengerId', list(gender_submission.PassengerId)),

         ('Survived', list(np.array(preds_unknown))),

         ]

my_submission_xgb = pd.DataFrame.from_items(my_submission_xgb)



my_submission
my_submission_xgb
my_submission.to_csv('my_submission.csv', index= False, header=True)

my_submission_xgb.to_csv('my_submission_xgb.csv', index= False, header=True)
#Future work: Include other features such as embark location.