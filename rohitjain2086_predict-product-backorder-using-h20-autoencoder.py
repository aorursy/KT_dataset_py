import pandas as pd
import numpy as np
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from imblearn.over_sampling import SMOTE
import os, time, sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
train = pd.read_csv("../input/Kaggle_Training_Dataset_v2.csv")
test = pd.read_csv("../input/Kaggle_Test_Dataset_v2.csv")
# Obserevd that name and number of columns remain the same in both train and test data. 
print(train.shape)
print(test.shape)
print(train.columns)
print(test.columns)
# Identify the column name and count containing NA values in total dataset
train.isnull().sum().sort_values(ascending = False)

# Lets update the missing values in lead_time column as 0
train['lead_time'].fillna(0,inplace = True)

# As size of NA rows are reduced a lot so we can now afford to drop Missing Observations by removing these rows.
train_withoutNa = train.dropna()

# Recheck if there are any NA's available in the train_withoutNa dataset
train_withoutNa.isnull().sum()

# Check if the target data is balanced or not
Counter(train_withoutNa['went_on_backorder'])

# As we can see number of backorders are very less. So we need to apply balancing method to make them balanced
# Converting the levels(Yes, No) of target columns into numeric values of 1 and 0 
y = pd.Series(np.where(train_withoutNa.went_on_backorder.values=='Yes',1,0),train_withoutNa.index)
y = pd.DataFrame(y,columns=['went_on_backorder'])
y_col = list(y)

# Lets encode the categorical features as binary variables 
X = train_withoutNa.drop(['sku','went_on_backorder'], axis = 1)
print(X.columns)
X_encoded = pd.get_dummies(X)
X_col = list(X_encoded.columns.values)
X_col
X_encoded.shape
# Generate the new dataset by balancing the data by SMOTE method.
smote = SMOTE()
smox, smoy = smote.fit_sample(X_encoded, y)
print('Resampled dataset shape {}'.format(Counter(smoy)))
# Converting balanced dataset into dataframe
smox = pd.DataFrame(smox)
smoy = pd.DataFrame(smoy)
# Adding column names to the balanced dataset
smox.columns = X_col
smoy.columns = y_col
print(smox.head(1))
print(smoy.head(1))
# Concatenate X and Y data after balancing it.
Balanced_train = pd.concat([smox, smoy], axis = 1)
h2o.init(ip="localhost", port=54321)

# Convert the pandas dataframe into H20frame
total_train = h2o.H2OFrame(Balanced_train)

# For binary classification, response of the dataset should be a factor
total_train['went_on_backorder'] = total_train['went_on_backorder'].asfactor()

# Now lets split the response of the dataset as train and test data accordingly.
x_test  = total_train[total_train['went_on_backorder'] == '1']
x_train = total_train[total_train['went_on_backorder'] == '0']

x_num = list(range(0,28))
# Create Autoencoder model and train it
anomaly_model = H2OAutoEncoderEstimator(
        activation="Tanh",
        hidden=[25,10,5,10,25],
        ignore_const_cols = False,
        stopping_metric='MSE', 
        stopping_tolerance=0.00005,
        epochs=200)

anomaly_model.train(
            x=x_num,
            training_frame=x_train)
# Jason View result
anomaly_model._model_json['output']
# Display the MSE of the model
print("MSE =",anomaly_model.mse())
# Get Reconstruction error
rec_error = anomaly_model.anomaly(total_train)
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
# Plot the scatter combined graph for product back order response with varying reconstructed error accordingly.

plt.figure(figsize=(18,18))
sns.FacetGrid(train_err, hue="went_on_backorder",size=8).map(plt.scatter,"id", "Reconstruction.MSE").add_legend()
plt.show()
# Plot the scatter individual graph of the product back order responses according to reconstructed error.

train_err[train_err['went_on_backorder']==1].plot(kind='scatter', x='id', y='Reconstruction.MSE',c='red', label='WithBackOrder')

train_err[train_err['went_on_backorder']==0].plot(kind='scatter', x='id', y='Reconstruction.MSE',c='yellow', label='WithoutBackOrder')
plt.show()
h2o.cluster().shutdown(prompt=False)