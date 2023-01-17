# load the library for manipulating the data

import pandas as pd

# load different models that can be used

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

# load the function to obtain train data and validation data

from sklearn.model_selection import train_test_split

# load the error function

from sklearn.metrics import mean_absolute_error





# we load the data

Xy_file_path = '../input/epf-3a/train_data_reduced.csv'

test_file_path = '../input/epf-3a/test_data_reduced.csv'



Xy_data = pd.read_csv(Xy_file_path)

test_data = pd.read_csv(test_file_path)



print("Setup Complete")



# we select all the features of the houses/flats except the Price feature

features = ['Build Year','Living Area','Number of Bedrooms','Number of Photos','Total Number of Rooms','Zip Code']

X = Xy_data[features]



# we select the target variable: Price column

y = Xy_data['Price']



# we separate the data X between two sets : train_X and val_X standing for respectively training data and validation data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)





# we define the model 

model = LinearRegression()

# uncomment the line below to define a DecisionTreeRegressor model

#model = DecisionTreeRegressor()

# uncomment the line below to define a RandomForestRegressor model

#model = RandomForestRegressor()



# Fit the model: we compute the parameters of the model with the training data : train_X, train_y

model.fit(train_X,train_y)



# we predict the target variable for the val_X data

predictions = model.predict(val_X)

# we compute the error between our predictions and the values expected `val_y`

print("The mean absolute error between val_y and predictions is: ",mean_absolute_error(val_y,predictions))



## Compute our solution

# we make predictions on the test data

my_test_predictions = model.predict(test_data)



# we save our solution to the file "submission.csv" that you can upload and see your rank for this competition (see instructions below)

output = pd.DataFrame({'Id': range(test_data.shape[0]),

                       'Predicted': my_test_predictions})

output.to_csv('submission.csv', index=False)

print("Submission file saved")