# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
train_df = pd.read_csv("../input/bigmart-sales-data/Train.csv")

pred_df = pd.read_csv("../input/bigmart-sales-data/Test.csv")
def ohe(df, col):

    label_encoder = LabelEncoder()

    label_encoded = label_encoder.fit_transform(df[col]).reshape(-1, 1)

    one_hot_encoder = OneHotEncoder(sparse = False)

    column_names = [col + "_" + str(i) for i in label_encoder.classes_]

    one_hot_encoded = one_hot_encoder.fit_transform(label_encoded)

    return pd.DataFrame(one_hot_encoded, columns = column_names)
def standarize(df):

    standard_scaler = StandardScaler()

    return standard_scaler.fit_transform(df)
def process_data(df):

    # Delete this feature because it has a big percentage of missing values

    # del df["Outlet_Size"]

    df["Outlet_Identifier"] = df["Outlet_Identifier"].astype("category")

    df["Item_Fat_Content"] = df["Item_Fat_Content"].astype("category")

    df["Item_Type"] = df["Item_Type"].astype("category")

    df["Outlet_Location_Type"] = df["Outlet_Location_Type"].astype("category")

    df["Outlet_Type"] = df["Outlet_Type"].astype("category")

    df["Outlet_Establishment_Year"] = df["Outlet_Establishment_Year"].astype("category")

    ###################################################

    os = df[["Outlet_Type", "Outlet_Size"]].groupby("Outlet_Type").apply(lambda x:x.mode())

    miss_bool = df['Outlet_Size'].isnull()

    df.loc[miss_bool,'Outlet_Size'] = df.loc[miss_bool,'Outlet_Type'].apply(lambda x: os.loc[x]["Outlet_Size"][0])

    df["Outlet_Size"] = df["Outlet_Size"].astype("category")

    #################################################

    # Replace the missing values with the mean each item identifier

    im = df[["Item_Identifier", "Item_Weight"]].groupby("Item_Identifier").mean()

    miss_bool = df['Item_Weight'].isnull() 

    df.loc[miss_bool,'Item_Weight'] = df.loc[miss_bool,'Item_Identifier'].apply(lambda x: im.loc[x])

    df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace = True)

    ###############################################

    # Replace 0 item visibility with the mean of the item visibility of each item

    iv = df[["Item_Identifier", "Item_Visibility"]].groupby("Item_Identifier").mean()

    miss_bool = df['Item_Visibility'] == 0

    df.loc[miss_bool, "Item_Visibility"] = df.loc[miss_bool, "Item_Identifier"].apply(lambda x: iv.loc[x])

    ######################################################

    df["Item_Visibility_MeanRatio"] = df.apply(lambda x: x["Item_Visibility"]/ iv.loc[x["Item_Identifier"]],axis = 1)

    ##################################################

    #df["Item_Identifier"] = df["Item_Identifier"].astype("category")

    del df["Item_Identifier"]

    categorical_df = df.select_dtypes(include = "category")

    for column in categorical_df.columns:

        temp = ohe(df, column)

        del df[column]

        df = pd.concat([df, temp], axis = 1)

    # Standardization

    # df = standarize(df)

    return df
def get_model(df):

    model = Sequential()

    #model.add(Dense(1024,input_dim = df.shape[1], activation = "relu"))

    model.add(Dense(512, input_dim = df.shape[1], activation = "relu"))

    model.add(Dropout(0.05))

    model.add(Dense(256, activation = "relu"))

    model.add(Dropout(0.05))

    model.add(Dense(128, activation = "relu"))

    model.add(Dropout(0.05))

    model.add(Dense(64, activation = "relu"))

    model.add(Dense(32, activation = "relu"))

    model.add(Dense(16, activation = "relu"))

    model.add(Dense(8, activation = "relu"))

    model.add(Dense(1, activation = "linear"))

    model.compile(loss = "mean_squared_error" , optimizer = "adam", metrics = ["mean_squared_error"])

    return model
train_target = train_df["Item_Outlet_Sales"]

del train_df["Item_Outlet_Sales"]
train_df = process_data(train_df)
my_model = get_model(train_df)
estimator = KerasRegressor(build_fn= lambda : get_model(train_df), epochs = 30, batch_size = 32, verbose = 0)
result = cross_val_score(estimator, train_df, train_target, scoring="neg_mean_squared_error", cv = 10)
result.mean()
my_model.fit(train_df, train_target, epochs = 30, batch_size = 32)
identifiers = pred_df[["Item_Identifier", "Outlet_Identifier"]]
pred_df = process_data(pred_df)
pred = my_model.predict(pred_df)
final = pd.concat([identifiers, pd.DataFrame(pred, columns=["Item_Outlet_Sales"])], axis = 1)
final.to_csv("submission.csv", index=False)
target = df["Item_Outlet_Sales"]
del df["Item_Outlet_Sales"]
del df["Item_Identifier"]
df["Item_Fat_Content"] = df["Item_Fat_Content"].astype("category")
df["Item_Type"] = df["Item_Type"].astype("category")
df["Outlet_Identifier"] = df["Outlet_Identifier"].astype("category")
# Delete this feature because it has a big percentage of missing values

del df["Outlet_Size"]
df["Outlet_Location_Type"] = df["Outlet_Location_Type"].astype("category")
df["Outlet_Type"] = df["Outlet_Type"].astype("category")
# Replace the missing values with the mode of the feature

df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mode()[0])
df["Outlet_Establishment_Year"] = df["Outlet_Establishment_Year"].astype("category")
categorical_df = df.select_dtypes(include = "category")
def ohe(df, col):

    label_encoder = LabelEncoder()

    label_encoded = label_encoder.fit_transform(df[col]).reshape(-1, 1)

    one_hot_encoder = OneHotEncoder(sparse = False)

    column_names = [col + "_" + str(i) for i in label_encoder.classes_]

    one_hot_encoded = one_hot_encoder.fit_transform(label_encoded)

    return pd.DataFrame(one_hot_encoded, columns = column_names)
for column in categorical_df.columns:

    temp = ohe(df, column)

    del df[column]

    df = pd.concat([df, temp], axis = 1)
def get_model():

    model = Sequential()

    model.add(Dense(512,input_dim = df.shape[1], activation = "relu"))

    model.add(Dense(256,  activation = "relu"))

    model.add(Dense(128, activation = "relu"))

    model.add(Dense(64, activation = "relu"))

    model.add(Dense(32, activation = "relu"))

    model.add(Dense(16, activation = "relu"))

    model.add(Dense(8, activation = "relu"))

    model.add(Dense(1, activation = "linear"))

    model.compile(loss = "mean_squared_error" , optimizer = "adam", metrics = ["mean_squared_error"])

    return model
estimator = KerasRegressor(build_fn=get_model, epochs = 30, batch_size = 32, verbose = 0)
results = cross_val_score(estimator, df.values, target.values, scoring = "neg_mean_squared_error", cv = 10)
my_model = get_model()
my_model.fit(df.values, target.values, epochs = 30, batch_size = 32)
x = to_predict.groupby(by = ["Item_Identifier", "Outlet_Identifier"])
len(to_predict["Item_Identifier"].unique())
x.count()
x.head()
to_predict.shape[0] - x.shape[0]
np.sum(x["Item_Identifier"] == "FDW58")