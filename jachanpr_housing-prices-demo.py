from sklearn import linear_model

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Load Data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
# Show first 5 entries of Train Data

train_data.head()
# Show first 5 entries of Test Data

test_data.head()
# Store Ids of test data for later in the csv creation to kaggle submition

test_ids = test_data["Id"]
# Load Targets 

y = train_data["SalePrice"]

# Convert to numpy array

y = np.array(y)



print(y.shape)
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder





def prepare_data(dataframe, feature_list, fimputer, fscaler, fhot_enc):



    if not feature_list:

        feature_list = list(dataframe)

        

    dataframe = dataframe[feature_list].copy()

    

    # Split dataframe in numerical and categorial data

    num_data = dataframe.select_dtypes(include=[np.number])

    cat_data = dataframe.select_dtypes(include=[object])



    if not num_data.empty:

        

        # Replace all NaN with the median in numerial data

        X_num = fimputer.transform(num_data)

        

        # Scale between -1, 1

        X_num = fscaler.transform(X_num)



        # Check if have categorical data 

        if cat_data.empty:  

            return X_num

    

    if not cat_data.empty:



        # Replace all NaN with "None" as other category in categorical data 

        cat_data.fillna('None', inplace=True)



        facto_cat_data = pd.DataFrame()

        # Factorize each categorical column (string -> int)

        for feature in list(cat_data):

            facto_cat_data[feature], _ = pd.factorize(cat_data[feature])

            

        # Hot encode

        X_cat_1hot = fhot_enc.transform(facto_cat_data.values).todense()



        # Check if have numerical data 

        if num_data.empty:

            return X_cat_1hot



    # Merge Numerical Data with One Hot encoded categorical data

    X = np.append(X_num, X_cat_1hot, axis=1)

    

    return X



def get_train_and_test(train, test, feature_list=[]):

    

    if not feature_list:

        feature_list = list(test)

    

    X = train[feature_list].copy()

    T = test[feature_list].copy()

    

    imputer = Imputer(strategy="median")

    scaler = StandardScaler()

    hot_enc = OneHotEncoder()

    

    all_data = pd.concat([X, T])

    all_num_data = all_data.select_dtypes(include=[np.number])

    all_cat_data = all_data.select_dtypes(include=[object])

    

    all_num_data = imputer.fit_transform(all_num_data)

    scaler.fit(all_num_data)

    

    all_cat_data.fillna('None', inplace=True)

    

    facto_cat_data = pd.DataFrame()

    # Factorize each categorical column (string -> int)

    for feature in list(all_cat_data):

        facto_cat_data[feature], _ = pd.factorize(all_cat_data[feature])

        

    hot_enc.fit(facto_cat_data.values)

    

    X = prepare_data(X, feature_list, imputer, scaler, hot_enc)

    T = prepare_data(T, feature_list, imputer, scaler, hot_enc)

    

    return X, T



#list(train_data) # Print all Features  !Useful More Info in data_description.txt
# Load Selected features

# X = np.array([  # TODO: Select better features

#                 train_data["YearBuilt"], 

#                 train_data["YrSold"], 

#                 train_data["LotArea"]

#             ])



# T = np.array([  # TODO: Select better features

#                 test_data["YearBuilt"], 

#                 test_data["YrSold"], 

#                 test_data["LotArea"]

#             ])



training_features = ["YearBuilt", "YrSold", "LotArea", "Street"] # <- TODO



# Extract training features in X and T from train_data and test_data

X, T = get_train_and_test(train_data, test_data, training_features)



X.shape, T.shape
#Training Features

#             #TamaNo    #Cuartos

# X = np.array([[1000,   2],

#               [2000,   2], 

#               [3000,   3], 

#               [10000,  5]])

# # Training Tagets

# y = np.array([2500, 5800, 7800, 18000])

# Test Data

#T = [[4000, 3], [5000, 5], [6000, 2], [7000, 8]]



# Initialize Regression Object

reg = linear_model.LinearRegression()



# Training

reg.fit(X, y)

# Predictions



# Predict

pred = reg.predict(T)
# For this problem Kaggle do not accept negatives values

# NOTE: We know negatives value are wrong



# TODO Select one



# Option 1: Saturate

if False:

    pred[pred < 0] = 0



# Option 2: Absolute Value

if True:

    pred = np.abs(pred)
# Create a "table" each index name is column name

# Kaggle Format

df_dict = {"SalePrice" : pred,

           "Id" : test_ids }



# Convert to Pandas DataFrame

df = pd.DataFrame(df_dict)



# Show Some data

df.head()
# Save in to csv file

pred_filename = "predictions-with-" + "-".join(training_features) + ".csv"  # Formatting

df.to_csv(pred_filename, index=False)

print("Output file: " + pred_filename)