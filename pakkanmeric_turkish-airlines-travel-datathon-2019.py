import pandas as pd

train_df = pd.read_csv("../input/datathon/assessment/Assessment Data/Assessment Train Data.csv")

result_df = pd.read_csv("../input/datathon/assessment/Assessment Data/Assessment Result File.csv")

print(train_df.shape)

print(result_df.shape)

train_df.head()
train_df.dtypes
train_df['Departure_YMD_LMT'] = pd.to_datetime(train_df['Departure_YMD_LMT'], format='%Y%m%d')

train_df['Operation_YMD_LMT'] = pd.to_datetime(train_df['Operation_YMD_LMT'], format='%Y%m%d')

result_df['Departure_YMD_LMT'] = pd.to_datetime(result_df['Departure_YMD_LMT'], format='%Y%m%d')

result_df['Operation_YMD_LMT'] = pd.to_datetime(result_df['Operation_YMD_LMT'], format='%Y%m%d')
train_df.dtypes
train_df.describe()
for col_name in train_df.columns:

    if train_df[col_name].dtype.name == 'object':

        train_df[col_name] = train_df[col_name].astype('category')

        result_df[col_name] = result_df[col_name].astype('category')
train_df.dtypes
for col_name in train_df.columns:

    if train_df[col_name].dtype.name == 'category':

        print(col_name, ":", train_df[col_name].unique())
dict = {"JW": 'Online',

        "TW": 'Online',

        "TS": 'Mobile',

        "JM": 'Mobile',

        "TY":"Counter",

        "QC":"Counter",

        "SC":"Kiosks",

        "IR":"Other",

        "?":"Other",

        "IA":"Other",

        "BD":"Other",

        "CC":"Other",

        "QR":"Other",

        "QP":"Other",

        "QA":"Other"

        }

train_df['Operation_Channel_Group'] = train_df['Operation_Channel'].map(dict)

train_df['Operation_Channel_Group'].unique()
result_df['Operation_Channel_Group'] = result_df['Operation_Channel'].map(dict)

result_df['Operation_Channel_Group'].unique()
train_df["Operation_Channel_Group"] = train_df["Operation_Channel_Group"].astype('category')

result_df["Operation_Channel_Group"] = result_df["Operation_Channel_Group"].astype('category')
(train_df.isnull().mean()*100).round(4)
(result_df.isnull().mean()*100).round(4)
import numpy as np

def unknown_perc(df):

  print("Column Name\t Percentage")

  for col_name in df.columns:

        if df[col_name].dtype.name == 'category' and (df[col_name] == "?").any():

          count = df[col_name].value_counts(dropna=False)['?']

          percentage = (count/len(df)*100).round(3)

          print(col_name,"\t", percentage)

  return

        

unknown_perc(train_df)
unknown_perc(result_df)
train_df['Operation_Sonic_Code_Flag'] = np.where(train_df['Operation_Sonic_Code']=='?', '0', '1')

train_df['Operation_Sonic_Code_Flag'] = train_df['Operation_Sonic_Code_Flag'].astype(int)

#train_df['Terminal_Number_Flag'] = np.where(train_df['Terminal_Number']=='?', '0', '1')

#train_df['Terminal_Number_Flag'] = train_df['Terminal_Number_Flag'].astype(int)

result_df['Operation_Sonic_Code_Flag'] = np.where(result_df['Operation_Sonic_Code']=='?', '0', '1')

result_df['Operation_Sonic_Code_Flag'] = result_df['Operation_Sonic_Code_Flag'].astype(int)

#result_df['Terminal_Number_Flag'] = np.where(result_df['Terminal_Number']=='?', '0', '1')

#result_df['Terminal_Number_Flag'] = result_df['Terminal_Number_Flag'].astype(int)
import numpy as np

#train_df['Terminal_Number'] = train_df['Terminal_Number'].replace('?', np.nan)

#train_df['Operation_Channel'] = train_df['Operation_Channel'].replace('?', np.nan)

train_df['Passenger_Title'] = train_df['Passenger_Title'].replace('?', np.nan)

train_df['Passenger_Gender'] = train_df['Passenger_Gender'].replace('?', np.nan)

train_df['Inbound_Departure_Airport'] = train_df['Inbound_Departure_Airport'].replace('?', "Unknown")

train_df['Outbound_Arrival_Airport'] = train_df['Outbound_Arrival_Airport'].replace('?', "Unknown")

train_df['Cabin_Class'] = train_df['Cabin_Class'].replace('?', np.nan)

train_df["Operation_Initials"] = train_df["Operation_Initials"].replace("?",np.nan)

train_df["Operation_Sonic_Code"] = train_df["Operation_Sonic_Code"].replace("?",np.nan)



#result_df['Terminal_Number'] = result_df['Terminal_Number'].replace('?', np.nan)

#result_df['Operation_Channel'] = result_df['Operation_Channel'].replace('?', np.nan)

result_df['Passenger_Title'] = result_df['Passenger_Title'].replace('?', np.nan)

result_df['Passenger_Gender'] = result_df['Passenger_Gender'].replace('?', np.nan)

result_df['Inbound_Departure_Airport'] = result_df['Inbound_Departure_Airport'].replace('?', "Unknown")

result_df['Outbound_Arrival_Airport'] = result_df['Outbound_Arrival_Airport'].replace('?', "Unknown")

result_df['Cabin_Class'] = result_df['Cabin_Class'].replace('?', np.nan)

result_df["Operation_Initials"] = train_df["Operation_Initials"].replace("?",np.nan)

result_df["Operation_Sonic_Code"] = result_df["Operation_Sonic_Code"].replace("?",np.nan)
(train_df.isnull().mean()*100).round(4)
(result_df.isnull().mean()*100).round(4)
#train_df2 = train_df.copy()

#result_df2 = result_df.copy()

train_df = train_df.drop(columns = [ "Departure_Airport", "Operation_Sonic_Code"]) #"Terminal_Number", 

result_df = result_df.drop(columns = ["Departure_Airport", "Operation_Sonic_Code"]) #"Terminal_Number", 
# Replace missing values whose titles are MISTER with M

train_df.loc[(train_df.Passenger_Gender.isna() ) & (train_df.Passenger_Title=='MISTER'),"Passenger_Gender"] = "M"

result_df.loc[(result_df.Passenger_Gender.isna() ) & (result_df.Passenger_Title=='MISTER'),"Passenger_Gender"] = "M"



# Replace missing values whose titles are MISS or MISSES with F

train_df.loc[(train_df.Passenger_Gender.isna() ) & ((train_df.Passenger_Title=='MISS') | (train_df.Passenger_Title=='MISSES')) ,"Passenger_Gender"] = "F"

result_df.loc[(result_df.Passenger_Gender.isna() ) & ((result_df.Passenger_Title=='MISS') | (result_df.Passenger_Title=='MISSES')) ,"Passenger_Gender"] = "F"



sum(train_df["Passenger_Gender"].isnull())
train_df.groupby("Operation_Channel_Group")['Passenger_Gender'].apply(lambda x: x.value_counts().index[0])#.reset_index()
train_df.groupby("Operation_Channel_Group")['Passenger_Gender'].apply(lambda x: x.value_counts())
result_df.groupby("Operation_Channel_Group")['Passenger_Gender'].apply(lambda x: x.value_counts())
train_df['Passenger_Gender'] = train_df['Passenger_Gender'].replace(np.nan, "M")

train_df['Passenger_Gender'].unique()
result_df['Passenger_Gender'] = train_df['Passenger_Gender'].replace(np.nan, "M")

result_df['Passenger_Gender'].unique()
train_df.groupby("Operation_Channel_Group")['Passenger_Title'].apply(lambda x: x.value_counts())
result_df.groupby("Operation_Channel_Group")['Passenger_Title'].apply(lambda x: x.value_counts())
train_df['Passenger_Title'] = train_df['Passenger_Title'].replace(np.nan, "MISTER")

train_df['Passenger_Title'].unique()
result_df['Passenger_Title'] = result_df['Passenger_Title'].replace(np.nan, "MISTER")

result_df['Passenger_Title'].unique()
train_df.groupby("Operation_Channel_Group")['Cabin_Class'].apply(lambda x: x.value_counts())
train_df["Cabin_Class"].unique()
result_df.groupby("Operation_Channel_Group")['Cabin_Class'].apply(lambda x: x.value_counts())
train_df['Cabin_Class'] = train_df['Cabin_Class'].replace(np.nan, "Y")

train_df['Cabin_Class'].unique()
result_df['Cabin_Class'] = result_df['Cabin_Class'].replace(np.nan, "Y")

result_df['Cabin_Class'].unique()
train_df.groupby("Operation_Channel_Group")['Operation_Initials'].apply(lambda x: x.value_counts().index[0])
result_df.groupby("Operation_Channel_Group")['Operation_Initials'].apply(lambda x: x.value_counts().index[0])
result_df['Operation_Initials'] = result_df['Operation_Initials'].replace(np.nan, "KS")

result_df['Operation_Initials'].unique()
train_df.loc[(train_df.Operation_Channel_Group == "Counter") & (train_df.Operation_Initials.isna()),"Operation_Initials"] = "KS"

train_df.loc[(train_df.Operation_Channel_Group == "Kiosks") & (train_df.Operation_Initials.isna()),"Operation_Initials"] = "SC"

train_df.loc[(train_df.Operation_Channel_Group != "Kiosks") & (train_df.Operation_Channel_Group != "Counter") & (train_df.Operation_Initials.isna()),"Operation_Initials"] = "MK"
train_df['Early_Check_In'] = (train_df.Departure_YMD_LMT - train_df.Operation_YMD_LMT)

train_df['Early_Check_In'] = (train_df['Early_Check_In']/86400000000000).astype(int)

train_df['Early_Check_In'].unique()
train_df[train_df['Early_Check_In']>100].sort_values('Operation_YMD_LMT')
train_df.loc[train_df.Early_Check_In > 100, 'Early_Check_In_Status'] = 'Peculiar'

train_df.loc[(train_df.Early_Check_In == 0) | (train_df.Early_Check_In == -1), 'Early_Check_In_Status'] = 'On-time'

train_df.loc[(train_df.Early_Check_In == 1) | (train_df.Early_Check_In == 2) | (train_df.Early_Check_In == 3), 'Early_Check_In_Status'] = 'Early'
result_df['Early_Check_In'] = (result_df.Departure_YMD_LMT - result_df.Operation_YMD_LMT)

result_df['Early_Check_In'] = (result_df['Early_Check_In']/86400000000000).astype(int)

result_df.loc[result_df.Early_Check_In > 100, 'Early_Check_In_Status'] = 'Peculiar'

result_df.loc[(result_df.Early_Check_In == 0) | (result_df.Early_Check_In == -1), 'Early_Check_In_Status'] = 'On-time'

result_df.loc[(result_df.Early_Check_In == 1) | (result_df.Early_Check_In == 2) | (result_df.Early_Check_In == 3), 'Early_Check_In_Status'] = 'Early'
train_df['Direct_Flight'] = np.where((train_df.Inbound_Departure_Airport == 'Unknown') & (train_df.Outbound_Arrival_Airport == 'Unknown'), 1, 0)

result_df['Direct_Flight'] = np.where((result_df.Inbound_Departure_Airport == 'Unknown') & (result_df.Outbound_Arrival_Airport == 'Unknown'), 1, 0)
train_df.loc[(train_df.Operation_Airport == train_df.Inbound_Departure_Airport), 'Checkin_Inbound'] = 1

train_df['Checkin_Inbound'] = train_df['Checkin_Inbound'].replace(np.nan, 0)



result_df.loc[(result_df.Operation_Airport == result_df.Inbound_Departure_Airport), 'Checkin_Inbound'] = 1

result_df['Checkin_Inbound'] = result_df['Checkin_Inbound'].replace(np.nan, 0)







train_df.loc[(train_df.Operation_Airport == train_df.Outbound_Arrival_Airport), 'Checkin_Outbound'] = 1

train_df['Checkin_Outbound'] = train_df['Checkin_Outbound'].replace(np.nan, 0)



result_df.loc[(result_df.Operation_Airport == result_df.Outbound_Arrival_Airport), 'Checkin_Outbound'] = 1

result_df['Checkin_Outbound'] = result_df['Checkin_Outbound'].replace(np.nan, 0)

train_df.groupby('Operation_Airport').count().sort_values('Operation_Initials', ascending=False).head(10)

result_df.groupby('Operation_Airport').count().sort_values('Operation_Initials', ascending=False).head(10)

train_df['Operation_Airport_Reduced'] = np.where((train_df.Operation_Airport == 'KDT') | (train_df.Operation_Airport == 'IST') | (train_df.Operation_Airport == 'SKW') | (train_df.Operation_Airport == 'EST'), train_df.Operation_Airport, 'OTHERS')

result_df['Operation_Airport_Reduced'] = np.where((result_df.Operation_Airport == 'KDT') | (result_df.Operation_Airport == 'IST') | (result_df.Operation_Airport == 'SKW') | (result_df.Operation_Airport == 'EST'), result_df.Operation_Airport, 'OTHERS')
train_df.groupby('Operation_Initials').count().sort_values('Operation_Airport', ascending=False).head(10)
result_df.groupby('Operation_Initials').count().sort_values('Operation_Airport', ascending=False).head(10)

train_df['Operation_Initials_Reduced'] = np.where((train_df.Operation_Initials == 'KS') | (train_df.Operation_Initials == 'MK') | (train_df.Operation_Initials == 'SC') | (train_df.Operation_Initials == 'Q7') | (train_df.Operation_Initials == 'EY') | (train_df.Operation_Initials == 'LK'), train_df.Operation_Initials, 'OTHERS')

result_df['Operation_Initials_Reduced'] = np.where((result_df.Operation_Initials == 'KS') | (result_df.Operation_Initials == 'MK') | (result_df.Operation_Initials == 'SC') | (result_df.Operation_Initials == 'Q7') | (result_df.Operation_Initials == 'EY') | (result_df.Operation_Initials == 'LK'), result_df.Operation_Initials, 'OTHERS')

train_df.groupby('Operation_Initials_Reduced').count().sort_values('Operation_Airport', ascending=False).head(10)
train_df.groupby('Inbound_Departure_Airport').count().sort_values('Operation_Airport', ascending=False).head(10)
result_df.groupby('Inbound_Departure_Airport').count().sort_values('Operation_Airport', ascending=False).head(10)
train_df['Inbound_Departure_Airport_Reduced'] = np.where((train_df.Inbound_Departure_Airport == 'Unknown') | (train_df.Inbound_Departure_Airport == 'IST') | (train_df.Inbound_Departure_Airport == 'SKW') | (train_df.Inbound_Departure_Airport == 'EST'), train_df.Inbound_Departure_Airport, 'OTHERS')

result_df['Inbound_Departure_Airport_Reduced'] = np.where((result_df.Inbound_Departure_Airport == 'Unknown') | (result_df.Inbound_Departure_Airport == 'IST') | (result_df.Inbound_Departure_Airport == 'SKW') | (result_df.Inbound_Departure_Airport == 'EST'), result_df.Inbound_Departure_Airport, 'OTHERS')
train_df.groupby('Outbound_Arrival_Airport').count().sort_values('Operation_Airport', ascending=False).head(10)
result_df.groupby('Outbound_Arrival_Airport').count().sort_values('Operation_Airport', ascending=False).head(10)
train_df['Outbound_Arrival_Airport_Reduced'] = np.where((train_df.Outbound_Arrival_Airport == 'Unknown') | (train_df.Outbound_Arrival_Airport == 'KDT'), train_df.Outbound_Arrival_Airport, 'OTHERS')

result_df['Outbound_Arrival_Airport_Reduced'] = np.where((result_df.Outbound_Arrival_Airport == 'Unknown') | (result_df.Outbound_Arrival_Airport == 'KDT'), result_df.Outbound_Arrival_Airport, 'OTHERS')
import datetime

train_df['Weekend'] = [x in [5,6] for x in train_df.Departure_YMD_LMT.dt.weekday]

train_df['Weekend'] = train_df['Weekend'].replace(True, int(1))

train_df['Weekend'] = train_df['Weekend'].replace(False, int(0))

result_df['Weekend'] = [x in [5,6] for x in result_df.Departure_YMD_LMT.dt.weekday]

result_df['Weekend'] = result_df['Weekend'].replace(True, int(1))

result_df['Weekend'] = result_df['Weekend'].replace(False, int(0))
train_df.Weekend.head()
train_df['Departure Day'] = train_df['Departure_YMD_LMT'].dt.weekday_name



result_df['Departure Day'] = result_df['Departure_YMD_LMT'].dt.weekday_name



train_df['Departure Day'].head()
train_df.insert(1,'Day_of_Month','foo')

train_df['Day_of_Month'] = train_df['Departure_YMD_LMT'].dt.day



result_df.insert(1,'Day_of_Month','foo')

result_df['Day_of_Month'] = result_df['Departure_YMD_LMT'].dt.day



train_df['Day_of_Month'].head()
dict = {"Y": 1,

        "C": 0

        }

train_df['Economy_Class'] = train_df['Cabin_Class'].map(dict)

train_df = train_df.drop("Cabin_Class", axis = 1)

result_df['Economy_Class'] = result_df['Cabin_Class'].map(dict)

result_df = result_df.drop("Cabin_Class", axis = 1)

train_df['Economy_Class'].unique()
train_df['Fly_Light'] = np.where(train_df['Passenger_Baggage_Count']==0, 1, 0)

result_df['Fly_Light'] = np.where(result_df['Passenger_Baggage_Count']==0, 1, 0)
train_df = train_df.drop(columns = ["Departure_YMD_LMT", 

                                    "Operation_YMD_LMT", 

                                    "Operation_Initials", 

                                    "Operation_Airport",

                                    "Inbound_Departure_Airport",

                                    "Outbound_Arrival_Airport",

                                    "Terminal_Name",

                                    "Early_Check_In"], axis =1) 
for col_name in train_df.columns:

    if train_df[col_name].dtype.name == 'object':

        train_df[col_name] = train_df[col_name].astype('category')

        result_df[col_name] = result_df[col_name].astype('category')
train_df.dtypes
train_onehot = train_df.copy()

#train_onehot.drop(columns = ["Operation_Initials", "Terminal_Name"], axis =1)

result_onehot = result_df.copy()

for cols in train_df.columns: #leave as traidf!!!

  if train_onehot[cols].dtype.name == 'category':

    print(cols)

    one_hot = pd.get_dummies(train_df[cols], prefix = cols)

    train_onehot = train_onehot.drop(cols,axis = 1)

    train_onehot = train_onehot.join(one_hot)

  
train_onehot.columns
import seaborn as sns

corr = train_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1,figsize=(10, 8))

train_df["Operation_Count"].hist(bins=500, color="blue", ax=ax)

import seaborn as sns

sns.barplot(x='Departure Day',y='Operation_Count',data=train_df)
sns.barplot(x='Weekend',y='Operation_Count',data=train_df)
sns.barplot(x='Direct_Flight',y='Operation_Count',data=train_df)
sns.barplot(x='Passenger_Baggage_Count',y='Operation_Count',data=train_df)
sns.barplot(x='Fly_Light',y='Operation_Count',data=train_df)
sns.distplot(train_df['Passenger_Baggage_Weight'])
import lightgbm as lgb



#train_onehot = train_onehot.drop(columns = ["Departure_YMD_LMT", "Operation_YMD_LMT", "Operation_Initials", "Operation_Airport"], axis =1)

target = train_onehot["Operation_Count"]

train = train_onehot.drop(["Operation_Count"], axis = 1)

#lightGBM model fit

gbm = lgb.LGBMRegressor()

gbm.fit(train, target)

gbm.booster_.feature_importance()

""

# importance of each attribute

fea_imp_ = pd.DataFrame({'cols':train.columns, 'fea_imp':gbm.feature_importances_})

fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)
from sklearn.model_selection import train_test_split

# define target

y = train_onehot.Operation_Count

# define features

X = train_onehot.drop(columns = ["Operation_Count"])

# stratified sampling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # stratify=X_train.Operation_Channel_Group, 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Validation Features Shape:', X_val.shape)

print('Validation Labels Shape:', y_val.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
y_train_log = np.where(y_train == 1, 1, 0)

y_val_log = np.where(y_val == 1, 1, 0)

y_test_log = np.where(y_test == 1, 1, 0)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y_train_log)
X_val['Prediction1'] = logreg.predict(X_val)

y_val_log_pred=X_val['Prediction1']
X_train_multi = X_train[y_train_log == 0]

y_train_multi = y_train[y_train_log == 0]



X_val_multi = X_val[y_val_log_pred == 0]

y_val_multi = y_val[y_val_log_pred == 0]
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso



reg = LassoCV()

reg.fit(X_train_multi, y_train_multi)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)





#regressor = LinearRegression()  

#regressor.fit(X_train_multi, y_train_multi) #training the algorithm
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
cols = ['Fly_Light', 'Operation_Initials_Reduced_SC', 'Direct_Flight', 'Operation_Initials_Reduced_EY', 'Economy_Class', 'Passenger_Gender_M', 

        'Inbound_Departure_Airport_Reduced_Unknown', 'Early_Check_In_Status_Early', 'Operation_Channel_TS', 'SWC_FLY']
X_train_multi = X_train_multi[cols]

X_val_multi = X_val[cols]
regressor = LinearRegression()  

regressor.fit(X_train_multi, y_train_multi) #training the algorithm



X_val['Prediction2'] = regressor.predict(X_val_multi)
X_val['Prediction2'] = X_val['Prediction2'].round()

y_val_multi_pred= X_val['Prediction2']
X_val['Prediction2'] = X_val['Prediction2'].astype('int64')
X_val['Prediction_fin'] = np.where((X_val.Prediction1 == 1), 1, X_val.Prediction2)
X_val['Prediction_fin'] = X_val['Prediction_fin'].astype('int64')
from sklearn.metrics import accuracy_score



score = accuracy_score(X_val['Prediction_fin'], y_val)

score

# Calculate the absolute errors

errors = abs(X_val['Prediction_fin'] - y_val)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_val)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
X_test['Prediction1'] = logreg.predict(X_test)

y_test_log_pred=X_test['Prediction1']
X_test_multi = X_test[cols]
X_test['Prediction2'] = regressor.predict(X_test_multi)
X_test['Prediction2'] = X_test['Prediction2'].round()

y_test_multi_pred= X_test['Prediction2']
X_test['Prediction2'] = X_test['Prediction2'].astype('int64')
X_test['Prediction_finito'] = np.where((X_test.Prediction1 == 1), 1, X_test.Prediction2)
X_test['Prediction_finito'] = X_test['Prediction_finito'].astype('int64')
# Calculate the absolute errors

errors = abs(X_test['Prediction_finito'] - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
result_df = result_df.drop(columns = ["Departure_YMD_LMT", 

                                    "Operation_YMD_LMT", 

                                    "Operation_Initials", 

                                    "Operation_Airport",

                                    "Inbound_Departure_Airport",

                                    "Outbound_Arrival_Airport",

                                    "Terminal_Name",

                                    "Early_Check_In",

                                       "Operation_Count"], axis =1)
result_onehot = result_df.copy()

for cols in result_df.columns: #leave as train_df!!!

  if result_onehot[cols].dtype.name == 'category':

    print(cols)

    one_hot = pd.get_dummies(train_df[cols], prefix = cols)

    result_onehot = result_onehot.drop(cols,axis = 1)

    result_onehot = result_onehot.join(one_hot)

result_onehot['Prediction1'] = logreg.predict(result_onehot)

result_onehot_multi = result_onehot[cols]
result_onehot['Prediction2'] = regressor.predict(result_onehot_multi)