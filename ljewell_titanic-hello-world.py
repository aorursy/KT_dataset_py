import matplotlib

matplotlib.style.use('ggplot')

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Pandas likes to index the dataframe, lets use the passenger id as an index as this makes the most sense

# This will output a dataframe into a variable which we can use later.

train = pd.read_csv("../input/train.csv", header=0, index_col='PassengerId')

test = pd.read_csv("../input/test.csv", header=0, index_col='PassengerId')



# Take a look at the training dataset first 10 lines to ensure its read in ok.

train.head(n=5)
# Grab the indexes and store them so we can recover the original data at the end.

train_index = train.index

test_index = test.index

# Combine the datasets together using pandas concat function

data = pd.concat([train, test])

data.head(n=5)
# Let's check the bottom looks the same as well

data.tail(n=5)

# You will notice that 'Survived' column is NaN as the test data doesn't have this data.

# This is what we need the model to do for us.
# Look at the outcome variable (Survived) to see how balanced the dataset is.

# If very few people survided we might need to do things differently.

data.Survived.value_counts()/len(train_index)
data.dtypes
# Pandas has a super easy way to change a column into categorical.

# Just set the type as category and are done.

# Once we are ready to convert we use the pd.get_dummies()

data['Sex'] = data['Sex'].astype('category')

data['Embarked'] = data['Embarked'].astype('category')
# Extract out the different types of ticket categories and drop out any numerical numbering

# Need to remove any formatting from the tickets so we can group them together

data['Ticket_Cat'] = data['Ticket'].str.replace(r'[!@#$./0123456789]', '')

# Some tickets are just numbers so we need to assign a number to them so we can recognise them

data.loc[data.Ticket_Cat == '', 'Ticket_Cat'] = 0

# Convert to a categorical

data['Ticket_Cat'] = data['Ticket_Cat'].astype('category')
# For names the surname may contain some meaniful data, let's group them and convert to categorical

data['Surname'] = data.Name.str.rpartition(',')[0]

data['Surname'] = data['Surname'].fillna('Noname')

data['Surname'] = data['Surname'].str.lower()

data['Surname'] = data['Surname'].astype('category')



# Lets see how many families we have

surname = data['Surname'].value_counts()

surname[1:10]
len(surname)
# Extract out the deck from the cabin number

data['Deck'] = data.Cabin.str.split(' ', expand=True)[0]

data['Deck'] = data['Deck'].str.replace(r'[!@#$./0123456789]', '')

data['Deck'] = data['Deck'].astype('category')

data['Cabin_Num'] = data.Cabin.str.split(' ', expand=True)[0]

data['Cabin_Num'] = data.Cabin_Num.str.replace(r'[A-Z]', '')

data['Cabin_Num'] = pd.to_numeric(data.Cabin_Num, errors='coerce')



# Check the results

data['Deck'].value_counts()
# Let's see what we have got now

data.head(n=5)
# Get a list of the columns so we can easily copy them and remove the columns we don't want

data.columns
# Create a new dataframe providing a list of columns that we want

# We will also drop out survived as it will cause some problems when we try to 

# fix the missing data in the next section. Don't worry we will keep the original data variable

# so we can recover it when we need it.

data_small = pd.DataFrame(data = data, columns =['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex',

       'SibSp', 'Surname', 'Deck', 'Cabin_Num','Ticket_Cat'])
# Check the dataframe looks right.

data_small.head(n=5)
# How many records do we have ? Let's look at the index length

num_records = len(data_small.index)

# Easy way to just sum up the number of nulls and divide by number of records to get a percentage

data_small.isnull().sum()/num_records
# Example of what get_dummies does. Lets just use the sex column and see what it does

pd.get_dummies(data_small.loc[1:10,:].Sex)
# Convert to dummy variables and a numpy array for Train data

data_small_dummies = pd.get_dummies(data_small)

data_small_dummies.head(n=5)
len(data_small_dummies.index)
from sklearn.preprocessing import Imputer



imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer_data = imputer.fit(data_small_dummies)

data_small_dummies_imputed = imputer_data.transform(data_small_dummies)

# We get a numpy array so we just need to convert back to a pandas DataFrame

data_small_dummies_imputed = pd.DataFrame(data=data_small_dummies_imputed, index=data_small_dummies.index, columns=data_small_dummies.columns)
# Count the number of Nan to ensure they have been removed.

# We sum twice so we can just get a grand total of any nan, we hope it will be zero.

data_small_dummies_imputed.isnull().sum().sum()
data_small_dummies_imputed.head(n=5)
data_small_dummies_imputed.shape
from sklearn.preprocessing import MinMaxScaler



minmax_scale_train = MinMaxScaler().fit(data_small_dummies_imputed)

data_small_dummies_normal = minmax_scale_train.transform(data_small_dummies_imputed)

data_small_dummies_normal = pd.DataFrame(data=data_small_dummies_normal, index=data_small_dummies_imputed.index, columns=data_small_dummies_imputed.columns)
data_small_dummies_normal.head(n=5)
from sklearn.model_selection import train_test_split



# Let's recover the train and test data using the original index that we stored

train_data = pd.DataFrame(data=data_small_dummies_normal, index=train_index)

test_data = pd.DataFrame(data=data_small_dummies_normal, index=test_index)



# The response variable is in the first column, all others are features we can use

y = data.Survived[train_index]

x = train_data



# Split the training data into test 30% and train 70%, stratify to ensure we can a even number of the response variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0, stratify=y)
from sklearn.linear_model import LogisticRegression



# Lets use a simple logistic regression

# Pick some parameters that should work ok

model = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True)

# Train the model on the training data

clf = model.fit(x_train, y_train)

# See what score we get the test data (draw from larger train set)

clf.score(x_test, y_test)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



logit = LogisticRegression()

# Run the model on our held out data from our training set

scores = cross_val_score(logit, x, y, cv=10)

print('Logistics Regression average score:',sum(scores)/len(scores))
from sklearn.ensemble import RandomForestClassifier



rf_classifier = RandomForestClassifier()

scores = cross_val_score(rf_classifier, x, y, cv=10)

print('Random Forest average score:',sum(scores)/len(scores))
from sklearn import svm



svm_classifier = svm.LinearSVC()

scores = cross_val_score(svm_classifier, x, y, cv=10)

print('SVM average score:',sum(scores)/len(scores))
from sklearn.model_selection import GridSearchCV



# Just need to put in a range of parameters for our model. 

param_grid = {'n_estimators':[5,10,25,50,100], 'criterion':['gini','entropy'], 

              'max_features':['auto', 'sqrt', 'log2']}



# Lets use a simple logistic regression

model = RandomForestClassifier()

grid_search = GridSearchCV(model, param_grid, cv=10)

grid_search.fit(x_train, y_train)

# Get the best score for the best parameters

grid_search.best_score_
# Get the parameter which got the best results.

grid_search.best_params_
model = RandomForestClassifier(criterion='entropy', max_features='auto', n_estimators=25)

scores = cross_val_score(svm_classifier, x_test, y_test, cv=10)

print('Random Forest average score:',sum(scores)/len(scores))
trained_model = model.fit(train_data, y)

prediction = model.predict(test_data)

results = pd.DataFrame(data=prediction, index=test_index, columns=['Survived'])

results.head(n=5)
results.Survived.value_counts()/len(results.index)
features = pd.DataFrame(data=trained_model.feature_importances_, index=test_data.columns, columns=['Importance'])

features.sort_values('Importance', ascending=False, inplace=True)

features.iloc[:15].plot(kind='bar')
# Create a submission CSV file with the passenger id and survival. Removing the header row

results.to_csv('submission.csv', index=True)