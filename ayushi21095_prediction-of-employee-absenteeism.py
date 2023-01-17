# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Below you can find the code that builds up to the ‘df_reason_mod’ checkpoint.

# Additionally, in the comments you can see the code that we ran in the lectures to check 

# the current state of a specific object while explaining various programming or data analytics concepts. 



raw_csv_data = pd.read_csv("../input/Absenteeism-data.csv")



# type(raw_csv_data)

# raw_csv_data



df = raw_csv_data.copy()

# df



pd.options.display.max_columns = None

pd.options.display.max_rows = None

display(df)

# df.info()







########## Drop 'ID': ############################

##################################################





# df.drop(['ID'])

# df.drop(['ID'], axis = 1)

df = df.drop(['ID'], axis = 1)



# df

# raw_csv_data







########## 'Reason for Absence' ##################

##################################################





# df['Reason for Absence'].min()

# df['Reason for Absence'].max()

# pd.unique(df['Reason for Absence'])

# df['Reason for Absence'].unique()

# len(df['Reason for Absence'].unique())

# sorted(df['Reason for Absence'].unique())







########## '.get_dummies()' an dropping the Reason 0 ######################

##################################################





reason_columns = pd.get_dummies(df['Reason for Absence'])

reason_columns



reason_columns['check'] = reason_columns.sum(axis=1)

reason_columns



reason_columns['check'].sum(axis=0)

# reason_columns['check'].unique()



reason_columns = reason_columns.drop(['check'], axis = 1)

# reason_columns



reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)

# reason_columns







########## Group the Reasons for Absence##########

##################################################





# df.columns.values

# reason_columns.columns.values

df = df.drop(['Reason for Absence'], axis = 1)    #coz we have seperated this column as 'reason_column'

# df



# reason_columns.loc[:, 1:14].max(axis=1)

reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)

reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)

reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)

reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)



#reason_type_1

# reason_type_2

# reason_type_3

# reason_type_4







########## Concatenate Column Values #############

##################################################





# df



df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)

#df



# df.columns.values

column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']



df.columns = column_names

df.head()







########## Reorder Columns #######################

##################################################





column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 

                          'Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours']



df = df[column_names_reordered]

df.head()







########## Create a Checkpoint 1 ###################

##################################################





df_reason_mod = df.copy()

# df_reason_mod







########## 'Date' ################################

##################################################





# df_reason_mod['Date']         # day/month/year

#df_reason_mod['Date'][0]

#type(df_reason_mod['Date'][0])   #Type: string





# df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'])

# df_reason_mod['Date']



df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format = '%d/%m/%Y')

#df_reason_mod['Date']

#type(df_reason_mod['Date'][0])       #pandas._libs.tslibs.timestamps.Timestamp

# df_reason_mod.info()







########## Extract the Month Value ############### (Jan-Dec: 1-12)

##################################################





#df_reason_mod['Date'][0]               #Timestamp('2015-07-07 00:00:00')

#df_reason_mod['Date'][0].month         # 7



list_months = []

# list_months



# df_reason_mod.shape



for i in range(df_reason_mod.shape[0]):

    list_months.append(df_reason_mod['Date'][i].month)

    

# list_months

#len(list_months)     # 700

df_reason_mod['Month Value'] = list_months

#df_reason_mod.head(20)







########## Extract the Day of the week ###############  (Mon-Tues: 0-6)

##################################################





#df_reason_mod['Date'][699].weekday()   #3 i.e. Thursday

#df_reason_mod['Date'][699]             #Timestamp('2018-05-31 00:00:00')



def date_to_weekday(date_value):

    return date_value.weekday()



df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)



df_reason_mod.head()







########## Remove the Date column & Reorder Month Value and Day of the Week at same place where date column was##############################

###################





df_reason_mod = df_reason_mod.drop(['Date'], axis = 1)

# df_reason_mod.head()

# df_reason_mod.columns.values

column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',

       'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',

       'Pets', 'Absenteeism Time in Hours']

df_reason_mod = df_reason_mod[column_names_upd]

df_reason_mod.head()







########## Create a Checkpoint 2 ###################

##################################################





df_reason_date_mod = df_reason_mod.copy()

df_reason_date_mod



########## Analyzing other columns ###################

##################################################



type(df_reason_date_mod['Transportation Expense'][0])

type(df_reason_date_mod['Distance to Work'][0])

type(df_reason_date_mod['Age'][0])

type(df_reason_date_mod['Daily Work Load Average'][0])

type(df_reason_date_mod['Body Mass Index'][0])    

     



########## Working on "Education", "Children", "Pets" ###################

##################################################



df_reason_date_mod['Education'].unique()

df_reason_date_mod['Education'].value_counts()

df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})

df_reason_date_mod['Education'].unique()

df_reason_date_mod['Education'].value_counts()

     

     

########## Final Checkpoint ###################

##################################################  

     

df_preprocessed = df_reason_date_mod.copy()

df_preprocessed.head(10)

data_preprocessed = df_preprocessed

data_preprocessed.head()
data_preprocessed['Absenteeism Time in Hours'].median()
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 

                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)

targets
data_preprocessed['Excessive Absenteeism'] = targets

data_preprocessed.head()
targets.sum() / targets.shape[0]
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours','Day of the Week',

                                            'Daily Work Load Average','Distance to Work'],axis=1)
data_with_targets is data_preprocessed

data_with_targets.head()
data_with_targets.shape
data_with_targets.iloc[:,:14]
data_with_targets.iloc[:,:-1]
unscaled_inputs = data_with_targets.iloc[:,:-1]
from sklearn.preprocessing import StandardScaler



absenteeism_scaler = StandardScaler()
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
#columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 'Distance to Work',

       #'Age', 'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pet']



columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

scaled_inputs
scaled_inputs.shape
from sklearn.model_selection import train_test_split
train_test_split(scaled_inputs, targets)

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, #train_size = 0.8, 

                                                                            test_size = 0.2, random_state = 20)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
reg = LogisticRegression()

reg.fit(x_train,y_train)
reg.score(x_train,y_train)
model_outputs = reg.predict(x_train)

model_outputs
y_train
model_outputs == y_train
np.sum((model_outputs==y_train))
model_outputs.shape[0]
np.sum((model_outputs==y_train)) / model_outputs.shape[0]
reg.intercept_
reg.coef_
unscaled_inputs.columns.values
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)



summary_table['Coefficient'] = np.transpose(reg.coef_)



summary_table
summary_table.index = summary_table.index + 1

summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

summary_table = summary_table.sort_index()

summary_table
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)

summary_table
summary_table.sort_values('Odds_ratio', ascending=False)
reg.score(x_test,y_test)
predicted_proba = reg.predict_proba(x_test)

predicted_proba
predicted_proba.shape
predicted_proba[:,1]
import pickle

with open('model', 'wb') as file:

    pickle.dump(reg, file)
with open('scaler','wb') as file:

    pickle.dump(absenteeism_scaler, file)