# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
home_data.head(10)
home_data.describe()
# what features did I use?



features
home_data[features + ["SalePrice"]].head(10)
home_data['MSSubClass'].value_counts()
home_data['MSZoning'].value_counts()
# Street: Type of road access



home_data['Street'].value_counts()

# Alley: Type of alley access



home_data['Alley'].value_counts()

# LotShape: General shape of property

home_data['LotShape'].value_counts()



# LandContour: Flatness of the property

home_data['LandContour'].value_counts()



# Utilities: Type of utilities available

home_data['Utilities'].value_counts()



pd.DataFrame([[1,2,3]], columns=['name', 'max_leaf_nodes', 'MAE'])
pd.concat([pd.DataFrame(columns=['name', 'max_leaf_nodes', 'MAE']), pd.DataFrame([[1,2,3]], columns=['name', 'max_leaf_nodes', 'MAE'])])
result = pd.DataFrame(columns=['name', 'max_leaf_nodes', 'MAE'])

depth = [5, 25, 50, 100, 250, 500]



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

d = home_data.drop(['SalePrice'], axis='columns').select_dtypes(include=numerics)

train_X, val_X, train_y, val_y = train_test_split(d, y, random_state=1)



for column in d.columns:

    depth_result = []

    for max_leaf_nodes in depth:

        model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)

        model.fit(train_X[column].to_frame().fillna(0), train_y)

        predict = model.predict(val_X[column].to_frame().fillna(0))

        depth_result.append((max_leaf_nodes, mean_absolute_error(predict, val_y)))

        

    smallest = {

        "max_leaf_nodes":None,

        "mae": float('infinity')

    }

    for r in depth_result:

        if smallest['mae'] > r[1]:

            smallest['mae'] = r[1]

            smallest['max_leaf_nodes'] = r[0]

            

    result = pd.concat([result, pd.DataFrame([[column, smallest['max_leaf_nodes'], smallest['mae']]], columns=['name', 'max_leaf_nodes', 'MAE'])])

            

result.head(5)
result
import matplotlib.pyplot as plt

import seaborn as sns





df = result.copy()



plt.figure(figsize=(15,10))

plt.title("MAE",fontsize=15)

sns.distplot(df['MAE'], rug=True)

plt.show()



print("Let's set threshold to 56000")
df = result[result['MAE'] < 56000]

features = list(df['name'])



features
train_X, val_X, train_y, val_y = train_test_split(home_data[features].fillna(0), y, random_state=1)

result = []



for max_leaf_nodes in depth:

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)

    model.fit(train_X, train_y)

    predict = model.predict(val_X)

    result.append((max_leaf_nodes, mean_absolute_error(predict, val_y)))



result

pd.DataFrame([[1,2,4], [4,7,8]])
result = pd.DataFrame(columns=['max_leaf_nodes', 'MAE'])





for max_leaf_nodes in range(100, 500):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)

    model.fit(train_X, train_y)

    predict = model.predict(val_X)

    result = pd.concat([result, pd.DataFrame([[max_leaf_nodes, mean_absolute_error(predict, val_y)]], columns=['max_leaf_nodes', 'MAE'])], ignore_index=True)

    

result



result.head(10)
plt.title("MAE by max_leaf_nodes",fontsize=15)

sns.scatterplot(x=result['max_leaf_nodes'], y=result['MAE'])
result.loc[result['MAE'].idxmin()]
model = RandomForestRegressor(max_leaf_nodes=234, random_state=1)

model.fit(home_data[features].fillna(0), y)



predict = model.predict(test_data[features].fillna(0))



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': predict})



output.head(20)
output.to_csv('submission.csv', index=False)