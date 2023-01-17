import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize']=15,10

import os, time, sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Read data 
train = pd.read_csv("../input/Kaggle_Training_Dataset_v2.csv")
test = pd.read_csv("../input/Kaggle_Test_Dataset_v2.csv")
print(train.shape)
print(test.shape)

train.head()
# From the above we can see a need to define the Categorical names

categorical_columns_names = [ 'potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk','stop_auto_buy', 'rev_stop', 'went_on_backorder']
train.went_on_backorder.describe()
# Working with Numeric 

numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
# Check for Null values

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
# Define Functions

# Delete the columns
def DeleteColumn(x,drop_column):
    x.drop(drop_column, axis=1, inplace = True)
    
# Replace Categorical with Numerical    
def categoricalToNumerical(dataset,categorical_columns_names,replace_value_map) :
    for categorical_column_name in categorical_columns_names:
        dataset[categorical_column_name] = dataset[categorical_column_name].map(replace_value_map).astype(int)
    return 
train['lead_time'].fillna(0,inplace = True)
# Check the data 
train.head()
# Fill Correlation Matrix
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# Balancing the Data 

replace_value = {'Yes': 1, 'No': 0}
data_clean = [train,test]

 
for dataset in data_clean:   
    ### Replace NA with Mean value in "lean_time" column with help of Imputer module in SKlearn  
    dataset['lead_time'] = Imputer(strategy="mean").fit_transform(dataset['lead_time'].values.reshape(-1, 1))
    dataset.dropna(axis=0,how='any',inplace =True)
    
### Categorical Columns to Numerical
    categoricalToNumerical(dataset,categorical_columns_names,replace_value)

import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
h2o.init(nthreads = -1)
total_train = h2o.H2OFrame(dataset)
test_hf = h2o.H2OFrame(test)
test_hf['went_on_backorder'] = test_hf['went_on_backorder'].asfactor()
total_train['went_on_backorder'] = total_train['went_on_backorder'].asfactor()
# Now lets split the response of the dataset as train and test data accordingly.
x_test  = total_train[total_train['went_on_backorder'] == '1']
x_train = total_train[total_train['went_on_backorder'] == '0']

X= list(range(0,22))
autoencoder_model = H2OAutoEncoderEstimator(  activation="Tanh",
                                          hidden=[50,20,5,20,50],
                                          ignore_const_cols = False,
                                           stopping_metric='MSE', 
                                            stopping_tolerance=0.00001,
                                             epochs=200)



autoencoder_model.train(x =X, training_frame = x_train)


print("MSE = ",autoencoder_model.mse())
# Reporting Performance on the model
autoencoder_model.model_performance(train=True)

autoencoder_model.train(x=X,training_frame=x_test)
# Get Reconstruction error
rec_error = autoencoder_model.anomaly(total_train)
rec_error.columns

# Combining error column with train dataset
train_err = total_train.cbind(rec_error)

# Convert H20Frame to pandas dataFrame.
train_err = train_err.as_data_frame()

# Sort the dataframe according to 'went_on_backorder' response
train_err = train_err.sort_values('went_on_backorder',ascending = True )

# Add 'id' column to the dataset representing row count of the error train dataset
train_err['id'] = range(1, len(train_err) + 1)
print(train_err.head(5))
# Get Reconstruction error
rec_error1 = autoencoder_model.anomaly(test_hf)
rec_error.columns

# Combining error column with train dataset
test_err = test_hf.cbind(rec_error)

# Convert H20Frame to pandas dataFrame.
test_err = test_err.as_data_frame()

# Sort the dataframe according to 'went_on_backorder' response
test_err = test_err.sort_values('went_on_backorder',ascending = True )

# Add 'id' column to the dataset representing row count of the error train dataset
test_err['id'] = range(1, len(test_err) + 1)
print(train_err.head(5))
# Plot the scatter plot

plt.figure(figsize=(18,18))
sns.FacetGrid(test_err, hue="went_on_backorder",size=8).map(plt.scatter,"id", "Reconstruction.MSE").add_legend()
plt.show()
# Predict on the Test 

pred = autoencoder_model.predict(test_hf)

# Predict Results
pred.head()