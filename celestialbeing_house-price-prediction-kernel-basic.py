# Import needed libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



# Read the data, convert to data frame

iowa_file_path  = '../input/home-data-for-ml-course/train.csv'

home_data       = pd.read_csv(iowa_file_path)

feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



# Define the equation y = function(x)

y = home_data.SalePrice

X = home_data[feature_columns]



# Define and fit model

iowa_model = DecisionTreeRegressor()

iowa_model.fit(X, y)



print("First in-sample predictions:", iowa_model.predict(X.head()))

print("Actual target values for those homes:", y.head().tolist())
# Now that everything is working perfectly on the in-sample prediction,

# time to split the data to train and test data for proper and more accurate validation

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
train_X.describe()
test_X.describe()
train_y.describe()
test_y.describe()
# Re-train the model using the train data for both target and feature

iowa_model.fit(train_X, train_y)
# Make a prediction using test features

test_predictions = iowa_model.predict(test_X)
# Get the mean absolute error by comparing test predicted results to actual "should be" test results

from sklearn.metrics import mean_absolute_error

print('The mean absolute error is:')

print(mean_absolute_error(test_y, test_predictions))
# We need to regulate the number of leaves (regularization) since we would like to 

# find the best value of leaves (k). Best to just create a function that I could always reuse



def measure_mae(k, train_X, test_X, train_y, test_y):

    model = DecisionTreeRegressor(max_leaf_nodes=k, random_state=0)

    model.fit(train_X, train_y)

    predicted_values = model.predict(test_X)

    mae = mean_absolute_error(test_y, predicted_values)

    return(mae)
# Now let's put this function to use

for k in [5, 50, 500, 5000]:

    current_mae = measure_mae(k, train_X, test_X, train_y, test_y)

    print("When K = %d, mean absolute error = %d, the smaller the better!"%(k, current_mae))
# Seems like for the given dataset, the best number of leaves (as far as decision tree is concerned) is 50

# Might need to actually result into random forest for better accuracy

# Time to use ensemble model!

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

improved_predicted_values = forest_model.predict(test_X)

print(mean_absolute_error(test_y, improved_predicted_values))
# You can use seaborn to visualize!

# sometimes being able to see something gives you

# a more better understanding of what it is and its patterns

# will plot the values of our train target result just to test seaborn

# will use line plot (you can also make heatmaps and bar charts!)



plt.figure(figsize=(14,6))

plt.title('Iowa Housing Dataset')

sns.lineplot(data=train_y)
# You can also visualize scatter plots

# You just need to define the x(horizontal) and y(vertical) axis

train_X.head()
# I would like to see the how the lot area and bathrooms are related

# at least on the training dataset

sns.scatterplot(x=train_X['LotArea'],y=train_X['FullBath'])
# will use regplot for this purpose (regression + scatterplot)

# using also the training dataset

sns.regplot(x=train_X['LotArea'], y=train_X['FullBath'])
# Seaborn is also capable of creating density plots

# and also histograms by using distplot! :)

# For ths one, I will use kde plot since I would like 

# to know, on my training data set, how the number of bathrooms vary

sns.kdeplot(data=train_X['FullBath'], shade=True)
# Remember the regplot earlier? you can actually present it also

# as histograms of two different entities combined in one graph

sns.distplot(a=train_X['LotArea'], label="Lot Area", kde=False)

sns.distplot(a=train_X['FullBath'], label="Full Bath", kde=False)

plt.title("Relationship between Lot Area and Full Bath")

plt.legend()