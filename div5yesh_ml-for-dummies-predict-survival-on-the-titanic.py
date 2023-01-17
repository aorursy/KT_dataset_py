# import libraries
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# read the input data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# peek at data
print(len(train))
train.head()

print(len(test))
test.tail()
# concat data for visualization and cleanup
full = train.append(test, sort = True)
titanic = full[:891]

full.tail()
# visualizing the data, patterns and variable values
titanic.describe()
# plot correlation map
# plot distribution of features and output - for numerical values
# plot categories - for categorical values
# data cleanup
full = full.drop(['Survived', 'PassengerId'], axis = 1)

# convert categories to numeric values
full = pd.get_dummies(full)

# fill in missing data
my_imputer = SimpleImputer()
full = my_imputer.fit_transform(full)

# create new variables
# feature engineering - extracting features from data
# assemble datasets

# create datasets
train_X = full[:891]
train_y = titanic.Survived
train_X , test_X , train_y , test_y = train_test_split( train_X , train_y , train_size = .7 )

# assess feature importance
# model selection
model = LogisticRegression()
# train model
model.fit(train_X, train_y)
# evaluate performance
print (model.score( train_X , train_y ) , model.score( test_X , test_y ))
# predict
predictX = full[891:]
predictY = model.predict( predictX )
# export results
passenger_id = test.PassengerId
output = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': predictY.astype(int) } )
output.to_csv( 'titanic_pred.csv' , index = False )