#import required libraries and functions
import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 

# read the input file
main_file_path = '../input/housetrain.csv' 
data = pd.read_csv(main_file_path)

# run the code below for some insight into the data
print('Some output from running this cell')
print (data.describe())
print(data.columns)

#take the target variable/field to make prediction
y=data.SalePrice

#select the input fields
predictor_cols = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[predictor_cols]

#split the input file into two sets - one with training data and another with test/validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


###1st model- Decisiion Tree
#define the DecisionTree model
iowa_model= DecisionTreeRegressor();

#fit the model on the training data
iowa_model.fit(train_X, train_y)

#predict the model on the validation data
val_predictions = iowa_model.predict(val_X)

#check the difference between target value of the validation data and the actual prediction output
print('The MAE using DecisionTree is: ' +  str(mean_absolute_error(val_y, val_predictions)))

#write the prediction to the output file
write_output_dt = pd.DataFrame({'SalesPrice': val_predictions})
write_output_dt.to_csv('DT_submission.csv') 



###2nd model- Random Forest
#using the Randomforest modelling
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
forest_pred_iowa = forest_model.predict(val_X)

#note the randomforest modelling MAE. You'd compare that with earlier MAE, producted by decision tree 
print('The MAE using RandomForest is: ' + str(mean_absolute_error(val_y, forest_pred_iowa)))

#write the prediction to the output file
write_output_rf = pd.DataFrame({'SalesPrice': forest_pred_iowa})
write_output_rf.to_csv('RF_submission.csv')



###Output validation for the models

#create a function to find the mean absolute error based on the tree depth
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes to find the optimum depth of the tree 
for max_leaf_nodes in [40, 45, 50, 55]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d   Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
