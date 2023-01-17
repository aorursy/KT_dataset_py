import pandas as pd

# This will control the display options of pandas within this notebook only

pd.options.display.max_columns = None

pd.options.display.max_rows = 20
# Our master table containing main information

df = pd.read_csv("../input/ai-trainee/churn_data.csv")

print("Input data-frame shape: {}".format(df.shape))

df.head()
# The cutomer-specific data

customer = pd.read_csv("../input/ai-trainee/customer_data.csv")

customer.head()
# Contract data

contract = pd.read_csv("../input/ai-trainee/internet_data.csv")

contract.head()
# Read metadata. 

# If you get an error, just inspect the file manually and remove empty spaces or tabs after each column (' ,'>'')

# This actually happens very often that the provided data set contains some strange symbols leading to errors. Get custom to it :)

meta = pd.read_csv("../input/ai-trainee/Telecom Churn Data Dictionary.csv")

# the columns in input data and meta do not match (some are in lowercase, some contain empty spaces, ...)

# Will create a new column that will unify all. Same will be done in following section for input dataframes

meta = meta.assign(name_id=meta["Variable Name"].replace({" ":"","\t":""},regex=True).str.lower()).set_index("name_id")

meta.head()
# Set customer ID (=unique ID) as index for further joining

# This is clearly a repeating task. Imagine doing this ten times or so! => use a for loop

# In addition, rename the columns to be identical with "meta"

for i in [df, customer, contract]:

    i.set_index("customerID",inplace=True)

    i.rename(columns={j:j.lower() for j in i.columns},inplace=True)

    

df.head()
# Join all three data-frames (one after another)

df = df.join(customer).join(contract)



# Make sure no 1:N relation = no duplicates, print shape again (compare number of rows with input above)

print("Joined dataframe shape: {}".format(df.shape))

df.head()
def check_stats(df):

    """

    This function will return a table (dataframe) showing main statistics and additional indicators

    """

    # We will store the data types in a separate dataframe

    dfinfo = pd.DataFrame(df.dtypes,columns=["dtypes"])

    

    # We are interested if we have any missing data (sum all). 

    # Again, join the result with the dfinfo (append new column). Consider '' or ' ' also as missing

    dfinfo = dfinfo.join((df.replace({'':None,' ':None}) if "('O')" in str(df.dtypes.values) else df).isna().sum().rename("isna"))

        



    # In the last step, add statistics (will be computed for numerical columns only)

    # We need to "T"ranspose the dataframe to same shape as df.describe() output

    return dfinfo.T.append(df.describe(),sort=False)
check_stats(df).T.query("isna != 0")
# The missing data are in the input file marked with ' '

# drop them = overwrite the dataframe

df = df[df.totalcharges!=' ']
# for visualization purposes, we will store the results in pandas dataframe and print the result in the next jupyter notebook cell

# nr of unique: will count the number of unique entries

# first 5 unique: will show first 5 unique entries (if less, only those)

dfsummary = pd.DataFrame({"nr of unique":[],"first 5 unique":[]})



# Run loop over all columns computing length (len) of unique entries in each column and converting first 5 enties to a string

for i in df.columns:

    dfsummary = pd.concat([dfsummary,pd.DataFrame({"nr of unique":[len(df[i].unique())],

                                                   "first 5 unique":[str(df[i].unique()[0:5])]},index=[i])],sort=False)
# join the result with metadata-column description

# Will join on column name. However, the provided metadata column names are not identical (e.g, contain empty spaces)

meta = meta.assign(join_name=meta["Variable Name"].replace({" ":"","\t":""},regex=True).str.lower()).set_index("join_name")



# need to do tha same for dfsummary

dfsummary = dfsummary.assign(join_name=dfsummary.index.astype(str).str.lower()).set_index("join_name")



# now having identical indices, join tables

dfsummary = dfsummary.join(meta[["Meaning"]])

dfsummary
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Tenure, seniorcitizen, and monthlycharges do not need to be converted

dfe = df[["tenure","seniorcitizen","monthlycharges"]].copy()



# Convert to float (str because of missing data that was, however, dropped before)

dfe["totalcharges"] = df.totalcharges.astype("float64")
# We want to ensure that 'No' is 0 and 'Yes' is 1. 

# To do that "fit" the encoder first, and apply (=transform) afterwards

# "fit" means that the object/variable "le_no_yes" will "remember" that no=0, and yes=1

le_no_yes = LabelEncoder().fit(['No','Yes'])



# now, apply to all columns where 'Yes', 'No' (or inverse order) occurs

for i in dfsummary[(dfsummary["first 5 unique"]=="['Yes' 'No']") | (dfsummary["first 5 unique"]=="['No' 'Yes']")].index:

    dfe[i] = le_no_yes.transform(df[i])
# to ensure the values are "ordered" = month-to-month = 0, year=1, 2 years = 2, fit first

le_contract = LabelEncoder().fit(['Month-to-month','One year','Two year'])

dfe["contract"] = le_contract.transform(df["contract"])
# Here is a for loop that will convert all remaining columns applying OneHotEncoder

# we will "declare" the encoder just to use its 'categories_' attribute 

ohe = OneHotEncoder() 

for i in df.columns:

    if i not in dfe.columns:

        print(i)

        # fit = get new mapping for each column

        ohe = OneHotEncoder().fit(df[i].unique().reshape(-1,1))

        # OneHotEncoder (just like ML models) expects/require a numpy matrix/array as input

        temp = ohe.transform(df[i].to_numpy().reshape(-1,1)).toarray()

        # following loop would not be necessary when using category-encoders library

        # it just uses the names for new columns

        for cat_i,cat_name in enumerate(ohe.categories_[0]):

            dfe["{}:{}".format(i,cat_name.lower().replace(" ","_"))] = temp[:,cat_i]
dfe.head()
check_stats(dfe)
# see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

from sklearn.preprocessing import StandardScaler
# It makes, of course, only sense to apply scaling only to numerical values

scaler = StandardScaler()

for i in [["tenure","monthlycharges","totalcharges"]]:

    dfe[i] = scaler.fit_transform(dfe[i])
check_stats(dfe)
import matplotlib.pyplot as plt
# Our target is "churn"

# The imbalance is rather small-to-modes, i.e. around 1:3

dfe.churn.value_counts().plot(kind="bar");
# Nice example of multiclass imbalance is "multiplelines" 

# If we would want to predict if customer has multiple lines, 

# then we would have to apply some of the methods to combat the imbalance.

df.multiplelines.value_counts().plot(kind="bar");
from sklearn.model_selection import train_test_split
# select features & labels

X = dfe.drop(columns=["churn"]).to_numpy()

y = dfe["churn"].to_numpy()



# Split in 70:30% ration

# Apply stratification = keep the class imbalance also in train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.3, 

                                                    stratify=y)
X_train.shape
X_test.shape
# Classification

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.metrics import classification_report
# Generate some artificial labels

y_true = [2,2,2,2,1,1,1,1,1,0,0,0]

y_pred = [2,1,0,1,1,1,0,1,1,1,1,0]



# Run for all scoring metrics

for (n,s) in zip(["Accuracy score","Balanced accuracy","Classification report"],

                 [accuracy_score,balanced_accuracy_score,classification_report]):

    print(n," : \n",s(y_true,y_pred))
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from sklearn.metrics import explained_variance_score, r2_score 

import numpy as np
def rmse(y1,y2):

    return np.sqrt(mean_squared_error(y1,y2))



def std(y1,y2):

    return np.std(y1-y2)
def evalregmetric(outlier=True,subt_mean=True):

    # create a copy of prediction and modify it according to settings

    y_use = y_pred.copy();

    if outlier == True:

        y_use[-1] = y_use[-1]+4;

    if subt_mean == True:

        y_use = y_use + np.mean(y_true-y_use);

    # run loop over all metrics

    out = np.array([]);

    for metric in [mean_squared_error,rmse,std,mean_absolute_error,median_absolute_error,

                         explained_variance_score,r2_score]:

        out = np.append(out,metric(y_true,y_use));

    return out
# Declare true and predicted values

y_true = np.array([2,2,2,2,1,1,1,1,1,0,0,0]);

y_pred = np.array([2,1,0,1,1,1,0,1,1,1,1,0]);



# Store the result in a dataframe for better visualisation (table)

result = pd.DataFrame(np.array(["MSE","RMSE","STD","MAE","MedAE","EV","R2"]),columns=["Metric"])

result = pd.DataFrame(result).set_index("Metric")



# Run for all combinations: Outlier/Mean subtraction

for (o,m) in zip([False,True,False,True],[False,False,True,True]):

    temp = pd.DataFrame({"Metric": np.array(["MSE","RMSE","STD","MAE","MedAE","EV","R2"]),

     "outlier:"+str(o)+" mean:"+str(m): evalregmetric(outlier=o,subt_mean=m)})

    result = result.merge(pd.DataFrame(temp).set_index("Metric"),left_index=True,right_index=True)

# Show result

result.round(3)