# First import all necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
# Import training/testing data
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
# only 1460 observations
print("Shape of training set: {}\nShape of testing set: {}".format(df_train.shape,df_test.shape))
# We can see there are columns with lots of nulls
# I would like to fill all the columns with the most occuring
# If NaNs are the most occuring value in the column then
df_train.info()
df_train.head()
df_test.head()
# Remove SalePrice column and save it for cross validation
# Remove the Id field since it does not contribute to SalePrice
labels = df_train["SalePrice"]
# Join training and testing data together
# exclude the SalePrice column from the join
joined = pd.concat([df_train.iloc[:,:-1], df_test], sort=False)
joined.head()
# I see that after the get dummies, there are some NaN values in the columns
dummies = pd.get_dummies(joined)
dummies.head()
# I would like to fill the NaN values with the most occuring value in each column (mode)
dummies.fillna(dummies.mode().iloc[0], inplace=True)
dummies.head()
# Split dummies back into training and test sets
# From the shape method, we can see that the training data is the first 1460 rows
# Also want to remove the ID column since we do not want that to affect the RF Regression
dummies_train = dummies.iloc[:1460,1:]
dummies_test = dummies.iloc[1460:,1:]
dummies_train.head()
# Split up training set for cross validation
x_train, x_test, y_train, y_test = train_test_split(dummies_train, labels, test_size=0.4)
# Show dimensions of test and training data
for item in [x_train, y_train, x_test, y_test]:
    print(item.shape)
# Create random forest regression predictor

# Create a range of 
parameters = {
    "n_estimators": [5, 10, 25, 50, 70, 100, 110, 150], # Test out various amounts of trees in the forest
    "max_features": [50, 70, 100, 120, 150, 200] # Test amount of features
}
regr = RandomForestRegressor()
grid = GridSearchCV(regr, parameters)
# Fit and test to see how accurate the algorithm is
grid.fit(x_train, y_train)
grid.score(x_test, y_test)
# Show the best parameters chosen 
grid.best_params_
# Get predictions from the best parameters chosen by GridSearchCV
predictions = grid.predict(dummies_test)
predictions = pd.DataFrame(predictions, columns=["SalePrice"])
predictions.head()
submission = dummies.iloc[1460:,:1].join(predictions)
submission.to_csv("submission.csv", index=False)
