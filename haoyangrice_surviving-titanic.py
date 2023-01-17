import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



#read in the training and test data into pandas DataFrames

ttrain = pd.read_csv('../input/train.csv')

ttest = pd.read_csv('../input/test.csv')



# a brief preview of the training data.

ttrain.head(8)
#Information of the training and test DataFrames

ttrain.info()

ttest.info()
# For the training data, use the median of available values to fill the Null cells in the Age column.

# Do the same for the test data, using the same value.

ttrain['Age'] = ttrain['Age'].fillna(ttrain['Age'].median())

ttest['Age'] = ttest['Age'].fillna(ttrain['Age'].median())



# For the training data, fill the couple of 'nan' values of the Embarked column with 'S'

ttrain['Embarked'] = ttrain['Embarked'].fillna('S')



# Convert the Sex column into numeric values

ttrain.loc[ ttrain['Sex'] == 'male', 'Sex'] = 0

ttrain.loc[ ttrain['Sex'] == 'female', 'Sex'] =  1



ttest.loc[ ttest['Sex'] == 'male', 'Sex'] = 0

ttest.loc[ ttest['Sex'] == 'female', 'Sex'] =  1



# The Fare column of the test data has a Nan value. Fill it with the median fare,

# using a different way of assigning the values in place.

ttest['Fare'].fillna(ttest['Fare'].median(), inplace=True) 



# Convert the Embarked column to numeric values for linear regression

ttrain.loc[ ttrain['Embarked'] == 'S', 'Embarked'] = 0 

ttrain.loc[ ttrain['Embarked'] == 'C', 'Embarked'] = 1 

ttrain.loc[ ttrain['Embarked'] == 'Q', 'Embarked'] = 2 



ttest.loc[ ttest['Embarked'] == 'S', 'Embarked'] = 0 

ttest.loc[ ttest['Embarked'] == 'C', 'Embarked'] = 1 

ttest.loc[ ttest['Embarked'] == 'Q', 'Embarked'] = 2
from sklearn.linear_model import LinearRegression



lmpredictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']



lm = LinearRegression()



# Fit the linear regression model to the training data

# and make predictions on the test data.

lm.fit(ttrain[lmpredictors], ttrain['Survived'])



predictions = lm.predict(ttest[lmpredictors])



# Convert the predictions from linear regression into survived ('1') or not ('0')

predictions[predictions >= 0.5] = 1

predictions[predictions < 0.5] = 0



# Convert the predictions into integer type.

predictions = predictions.astype(int)



#Create a submission dataframe and write to a csv file.

submission  = pd.DataFrame({'PassengerID': ttest['PassengerId'],

                           'Survived': predictions})



submission.to_csv('kaggle.titanic.lm.1.csv', index=False)



### The linear regression submission has an accuracy of 0.76555, which ranks at 4026.
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(random_state = 1)



# Fit the logistic regression model to the training data

# and make predictions on the test data.

lr.fit(ttrain[lmpredictors], ttrain['Survived'])



predictions = lr.predict(ttest[lmpredictors])



#Create a submission dataframe for the results from logistic regression and write to a csv file.

submission  = pd.DataFrame({'PassengerID': ttest['PassengerId'],

                           'Survived': predictions})



submission.to_csv('kaggle.titanic.logistic.csv', index=False)



### The logistic regression submission has an accuracy of 0.74163, which is not an improvement

### over the linear regression result.
from sklearn.ensemble import RandomForestClassifier



# Use the following features to train a random forest classifier.

rfpredictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Initialize the algorithm with 50 trees, min. number of rows to make a split, 

# and the min. number of leaves at the end of a branch.

rf = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)



# Fit the model and make predicitions. Then create another submission file.

rf.fit(ttrain[rfpredictors], ttrain['Survived'])



predictions = rf.predict(ttest[rfpredictors])



submission  = pd.DataFrame({'PassengerID': ttest['PassengerId'],

                           'Survived': predictions})



submission.to_csv('kaggle.titanic.random.forest.csv', index=False)



### My naive random forest submission has an accuracy of 0.733, which is not an improvement

### over the linear regression result. Fine tuning parameters might get better results.
import re

import operator



# FamilySize = # of siblings + # of Parent/Child

ttrain['FamilySize'] = ttrain['SibSp'] + ttrain['Parch']

ttest['FamilySize'] = ttest['SibSp'] + ttest['Parch']



# Length of a name

ttrain['NameLength'] = ttrain['Name'].apply(lambda x: len(x))

ttest['NameLength'] = ttest['Name'].apply(lambda x: len(x))



# Getting the title of each passenger and map it to a numeric value.



# define a function to search for a title using regular expression.

def get_title(name):

    search = re.search('([A-Za-z]+)\.', name)

    if search:

        return search.group(1)

    return ""



# Apply the get_title function on passenger names.

titles = ttrain['Name'].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Dona": 9, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



ttrain['Title'] = titles



# Do the same for the test data.

titles2 = ttest['Name'].apply(get_title)

for k,n in title_mapping.items():

    titles2[titles2 == k] = v

    

ttest['Title'] = titles2



# Generate another feature showing which family a person is in.

# To do this, we concatenate someone's last name with 'FamilySize'

# to get a unique 'FamilyID'

family_id_mapping = {}



# define a function to get the id given a row/name

def get_family_id(row):

    

    last_name = row["Name"].split(',')[0]

    # create the family id

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    # Look up the id in the mapping

    if family_id not in family_id_mapping:

        if len(family_id_mapping) == 0:

            current_id = 1

        else:

            # Get the maximum id from the mapping and add one to it if we don't have an id

            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)

        family_id_mapping[family_id] = current_id

    return family_id_mapping[family_id]



# Get the family ids with the apply method

family_ids = ttrain.apply(get_family_id, axis=1)



family_ids[ttrain['FamilySize'] < 3] = -1



ttrain['FamilyID'] = family_ids



# Do the same for the test data



family_ids2 = ttest.apply(get_family_id, axis=1)



family_ids2[ttest['FamilySize'] < 3] = -1



ttest['FamilyID'] = family_ids2



#######

# Finding the best features.

from sklearn.feature_selection import SelectKBest, f_classif



predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title', 'FamilyID']



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(ttrain[predictors], ttrain['Survived'])



# convert the raw p-values into scores

scores = -np.log10(selector.pvalues_)



# Plot the scores.

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.xlabel('Feature'); plt.ylabel('Score'); 

plt.title('Evaluating the Features')

plt.show()
bestpredictors = ['Pclass', 'Sex', 'Fare', 'Title']



# Run random forest again with the best predictors.

rf2 = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)



# Fit the model and make predicitions. Then create another submission file.

rf2.fit(ttrain[bestpredictors], ttrain['Survived'])



predictions2 = rf2.predict(ttest[bestpredictors])



submission2  = pd.DataFrame({'PassengerID': ttest['PassengerId'],

                           'Survived': predictions2})



submission.to_csv('kaggle.titanic.random.forest.2.csv', index=False)



#Welp, using the best features, the score is 0.73684, not really an improvement.
# Next, we will try emsembling the gradient boosting classifier 

# and logistic regression for better accuracy.



from sklearn.ensemble import GradientBoostingClassifier



predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyID"]



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



full_predictions = []

for alg, predictors in algorithms:

    # Fit the algorithm using the full training data.

    alg.fit(ttrain[predictors], ttrain["Survived"])

    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.

    predictions = alg.predict_proba(ttest[predictors].astype(float))[:,1]

    full_predictions.append(predictions)



# The gradient boosting classifier generates better predictions, so we weight it higher.

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4



predictions[ predictions > 0.5 ] = 1

predictions[ predictions <= 0.5 ] = 0



# convert predictions to int type

submission = pd.DataFrame({'PassengerId': ttest['PassengerId'], 'Survived': predictions.astype(int) })



submission.to_csv('kaggle.titanic.ensemble.csv', index=False)