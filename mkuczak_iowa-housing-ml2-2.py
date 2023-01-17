# The first step is to utilize one-hot encoding in order to allow for us to use
# categorical data to make predictions with out Iowa House dataset
import pandas as pd
data = pd.read_csv('../input/train.csv')
# Let's take a look at the columns and choose some categorical values that
# sound useful for determining the selling price of a house
list(data.columns.values)
# Oh, I left SalePrice in there.  Let's save it separately and remove it from our data before we move on.
y_train = data['SalePrice']
# The above list was a lot less helpful than I had anticipated.  Let me exclude all numerical
# column headers and just show the categorical ones.
list(data.select_dtypes(include=['object']))
# This still is not halpful.  Perhaps I need to just look at a few rows and see what sticks out.
cat_data = data.select_dtypes(include=['object'])
cat_data.head(10)
# So, what looks important up there?  I imagine that the following are important:
# Utilities
# Neighborhood
# SaleCondition
# Even if there are more, let us just start with that and see how much our model
# improves vs the other Kernal "Iowa Housing ML2: 1", which ranged from about 19500 to 22000.
# It seems that the easiest way to do this is one-hot encode everything first.
# It would be useful to learm how to not waste time encoding unused columns.
#iowa_data = data.select_dtypes(exclude=["object"]) + data['Utilities']
one_hot_encoded_data = pd.get_dummies(data)
data = data.drop(['SalePrice'], axis=1)
data.head()
ohe_data = pd.get_dummies(data)
ohe_data.head()
# I'm honestly worried because some of the columns seem to be missing now.
# It'll waste a ton of space, but let's look at col names for bthe new data.
list(ohe_data.columns.values)
list(data.columns.values)
# Nothin is missing.  It just seems that when you OHE something, it puts those new columns
# in the back rather than squeezing them in where the original was.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# Let's do this again and see what score we get
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
X = ohe_data
y = y_train
# oops
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
# Before the error, the score was just under 20,000.  Cool.
# This is a little disappointing, but I am rather certain that I did it right.