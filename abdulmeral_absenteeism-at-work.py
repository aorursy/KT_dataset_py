import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_selection import VarianceThreshold

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/employee-absenteeism-prediction/Absenteeism-data.csv")

data.head()
data.describe()
data.isnull().sum()
# if you want to see all columns and raws:

# pd.options.display.max_columns = None

# pd.options.display.max_rows = None

# display(data)
data.info()
data.shape
# here for simplicity I will use only numerical variables

# select numerical columns:



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data_numerical = data[numerical_vars]

data_numerical.shape
# remove constant features

constant_features = [

    feat for feat in data_numerical.columns if data_numerical[feat].std() == 0

]



len(constant_features)
# remove quasi-constant features

sel = VarianceThreshold(threshold=0.01) # 0.1 indicates 99% of observations approximately



sel.fit(data_numerical)  # fit finds the features with low variance



sum(sel.get_support()) # how many not quasi-constant?
data.duplicated().sum()
data[data.duplicated()]
data = data.drop_duplicates(keep="first").reset_index()
data.shape
data.head()
def correlation(dataset, threshold):

    col_corr = set()  # Set of all the names of correlated columns

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value

                colname = corr_matrix.columns[i]  # getting the name of column

                col_corr.add(colname)

    return col_corr
corr_features = correlation(data, 0.8)

len(set(corr_features))
data.drop(["ID","index"],axis=1,inplace=True)

data.head()
# Reason for Absence

data["Reason for Absence"].unique()
reason_columns = pd.get_dummies(data["Reason for Absence"],drop_first=True)

reason_columns.head()
# Make Groups For Reasons

reason_type_1 = reason_columns.loc[:,:14].max(axis=1)

reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)

reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)

reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
# concat:

df = pd.concat([data.drop("Reason for Absence",axis=1), reason_type_1, reason_type_2, reason_type_3, reason_type_4],axis=1)

df.head()
df.columns.values
columns_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours', "Reason_1", "Reason_2", "Reason_3", "Reason_4"]

df.columns = columns_names

df.head()
df.columns.values
# reorder columns

column_names_reordered = ['Reason_1','Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',

                           'Daily Work Load Average', 'Body Mass Index', 'Education',

                           'Children', 'Pets', 'Absenteeism Time in Hours']

df = df[column_names_reordered]

df.head()
# create a chechpoint: to reduce the risk of losing data

df_reason_mod = df.copy()
# date:

# df_reason_mod["Date"].apply(lambda x: x.split("/"))

df_reason_mod["Date"] = pd.to_datetime(df_reason_mod["Date"], format="%d/%m/%Y")

df_reason_mod["Date"][0:5]
# extract month

list_months = []

for i in range(len(df_reason_mod["Date"])):

    list_months.append(df_reason_mod["Date"][i].month)
df_reason_mod["Month Value"] = list_months

df_reason_mod.head()
# extract the day of the week:0,1,2,3,4,5,6

df_reason_mod["Date"][0].weekday()
def day_to_weekday(date_value):

    return date_value.weekday()
df_reason_mod["Day of the Week"] = df_reason_mod["Date"].apply(day_to_weekday)

df_reason_mod.head()
# Education:

df_reason_mod["Education"].unique()
df_reason_mod["Education"].value_counts()
# 1 => 0

# 2,3,4 => 1

df_reason_mod["Education"] = df_reason_mod["Education"].map({1:0,2:1,3:1,4:1})
df_reason_mod["Education"].unique()
df_reason_mod["Education"].value_counts()
# create a checkpoint

df_model = df_reason_mod.copy()
# use median cut-off hours and making targets 

df_model["Absenteeism Time in Hours"].median()
targets = np.where(df_model["Absenteeism Time in Hours"] > 3, 1, 0)

targets[0:10]
# add to df

df_model["Excessive Absenteeism"] = targets

df_model.head()
targets = pd.Series(targets)

targets.value_counts()
# drop Absenteeism Time in Hours

df_model.drop(["Absenteeism Time in Hours","Date"],axis=1,inplace=True)
df_model is df_reason_mod
unscaled_inputs = df_model.iloc[:,:-1]
# this class is just for selecting features to standardization:



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



class CustomScaler(BaseEstimator,TransformerMixin): 

    

    def __init__(self,columns,copy=True,with_mean=True,with_std=True):

        self.scaler = StandardScaler(copy,with_mean,with_std)

        self.columns = columns

        self.mean_ = None

        self.var_ = None



    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.var_ = np.var(X[self.columns])

        return self



    def transform(self, X, y=None, copy=None):

        init_col_order = X.columns

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

unscaled_inputs.columns.values
columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 'Distance to Work',

                    'Age', 'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pets']



# Because these features are dummy variable

# columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education'] 
sc = CustomScaler(columns_to_scale)
scaled_inputs = sc.fit_transform(unscaled_inputs)

scaled_inputs.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs,targets,train_size=0.8,shuffle=True,random_state=7)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
reg = LogisticRegression()
reg.fit(x_train,y_train)
reg.score(x_train,y_train)
model_outputs = reg.predict(x_train)
model_outputs == y_train
np.sum(model_outputs == y_train)
model_outputs.shape
np.sum(model_outputs == y_train) / model_outputs.shape[0]
np.sum(model_outputs == y_train) / model_outputs.shape[0] == reg.score(x_train,y_train)
reg.intercept_
reg.coef_.T
# Make df to show better:

feature_name = unscaled_inputs.columns

summary_table = pd.DataFrame(data=feature_name,columns=["Feature Name"])

summary_table["Coefficient"] = reg.coef_.T

summary_table
# add intercept:

summary_table.index = summary_table.index + 1

summary_table.loc[0] = ["Intercept", reg.intercept_[0]]

summary_table = summary_table.sort_index()

summary_table
summary_table["Odds_ratio"] = np.exp(summary_table.Coefficient)

summary_table
summary_table.sort_values("Odds_ratio", ascending=False)
reg.score(x_test, y_test)
# predict_proba(x) : returns the probability estimates for all possible outputs

predict_proba = reg.predict_proba(x_test)

predict_proba
# see sum is 1

0.72033435 + 0.27966565, 0.87854892 + 0.12145108
# 1. column: probality of being 0

# 2. column: probality of being 1

predict_proba.shape
predict_proba[:,1]
import pickle
# model: file name / wb: write bites / dump:save

with open("model", "wb") as file:

    pickle.dump(reg,file)
with open("model_2", "wb") as file_2:

    pickle.dump(sc,file_2)
from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server