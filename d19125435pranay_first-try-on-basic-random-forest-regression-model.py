# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

import seaborn as sn

from matplotlib import cm

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.compose import ColumnTransformer

# Only necessary in Jupyter notebook

%matplotlib inline




# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

sales_train_validation = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
calendar["event_type_1_snap"] = pd.notna(calendar["event_type_1"]) 

calendar["event_type_2_snap"] = pd.notna(calendar["event_type_2"]) 

calendar["date"] =  pd.to_datetime(calendar["date"])

calendar["d_month"] = calendar["date"].dt.day

calendar["year"] = pd.to_numeric(calendar["year"])

calendar["wday"] = pd.to_numeric(calendar["wday"])

print(calendar.shape)

calendar.head()
print(sell_prices.shape)

sell_prices.head()
print(sales_train_validation.shape)

sales_train_validation.head()
calendar_snap_byWday = calendar.groupby(['year','wday','weekday'])[("snap_CA","snap_TX","snap_WI")].sum().sort_index(1)

fig, ax = plt.subplots()

calendar_snap_byEvent = calendar.groupby(["year","event_type_1_snap"])[("snap_CA","snap_TX","snap_WI")].sum().unstack().plot(ax=ax)
sales_data = pd.merge(sell_prices, calendar[["year","month","d","wday","weekday","event_type_1_snap","event_type_2_snap","wm_yr_wk"]], left_on='wm_yr_wk', right_on='wm_yr_wk')

ax = plt.gca()

ax.yaxis.set_major_formatter(FormatStrFormatter('$%.2f M'))

sales_data[["year","sell_price"]].groupby(["year"]).mean().unstack().plot(kind='bar',stacked=True,ax=ax)

plt.show()
clus20 = sales_train_validation.iloc[:,2:]

data = clus20.groupby("state_id").sum()

data

x = sn.heatmap(data)
clus20 = sales_train_validation.iloc[:,2:]

data = clus20.groupby("cat_id").sum()

data

x = sn.heatmap(data)
#Data Prepartions & removing tempoprary objects from memory 



column_index = [1,2,3,4,5]

for i in range(6 , len(sales_train_validation.columns)):

    column_index.append(i)



clus_hobbies = sales_train_validation.iloc[:,column_index].query("cat_id == 'HOBBIES'")

clus_household = sales_train_validation.iloc[:,column_index].query("cat_id == 'HOUSEHOLD'")

clus_foods = sales_train_validation.iloc[:,column_index].query("cat_id == 'FOODS'")

clus_ca = sales_train_validation.iloc[:,column_index].query("state_id == 'CA'")

clus_tx = sales_train_validation.iloc[:,column_index].query("state_id == 'TX'")

clus_wi = sales_train_validation.iloc[:,column_index].query("state_id == 'WI'")

clus = sales_train_validation.iloc[:,column_index]

#Bucket columns by calander days of month

from datetime import datetime

columnsets = []

for i in range(1,32):      

    d = calendar[:1913].query("d_month == "+ str(i))["d"]

    columnsets.append([d.values])
# Label encoding for catagorical data

def label_encoding(data_preap,cat_features):

    categorical_names = {}

    data = []

    encoders = []

    

    data = data_preap[:]

    for feature in cat_features:

        le = sklearn.preprocessing.LabelEncoder()

        le.fit(data.iloc[:,feature])

        data.iloc[:, feature] = le.transform(data.iloc[:, feature])

        categorical_names[feature] = le.classes_

        encoders.append(le)

    X_data = data.astype(float)

    return X_data, encoders



# Training random forest model

def train_model(X_train, X_test, Y_train, Y_test):

    # Random forest regressor model with Training dataset

    start_time = datetime.today()

    regressor = RandomForestRegressor(n_estimators = 350, random_state = 50)

    regressor.fit(X_train,Y_train)



    print("Time taken to Train Model: " + str(datetime.today() - start_time))



    # Running Regession model score check

    Y_score = regressor.score(X_test,Y_test)

    return regressor,Y_score
# Predict function from model

def model_prediect(regressor,X_data):

    # Predicting model model result

    Y_pred = regressor.predict(X_data)

    return Y_pred
# Validating model with last year data & generating rmse value for the model predection

def validate_model(regressor,X_validation, Y_validation):

   

    Y_validation_pred = model_prediect(regressor, X_validation)

    mse = mean_squared_error(Y_validation, Y_validation_pred)

    rmse = np.sqrt(mse)

    return rmse, Y_validation_pred
 # Basic function for geting data from pandas based on range

def get_data_range(Inital_Range,start_index,end_index):

    result = []

    [result.append(a) for a in Inital_Range]

    for i in range(max(Inital_Range) +1 + start_index, end_index):

        result.append(i)

    return result
 # main function to run predictions

def run_predictions(orig_data):

    process_data = orig_data[:]

    results = pd.DataFrame()

    for s in range(1,29):

        categorical_features = [0,1]

        data = []

        data_range = []

        for i in range(0,s):

            [data_range.append(a) for a in columnsets[i]]

        data_list = [process_data[a] for a in data_range]

        data  = pd.concat(data_list,axis = 1)





        data.insert(loc=0, column='item_id', value=process_data["item_id"])

        data.insert(loc=1, column='store_id', value=process_data["store_id"])

        X_data_preap = data[:]



        d = get_data_range(categorical_features,0,len(X_data_preap.columns)-1)   

        X,label_encoders = label_encoding(X_data_preap.iloc[:,d],categorical_features)

        Y = X.iloc[:,-1]



        d_validation = get_data_range(categorical_features,1,len(X_data_preap.columns))   

        X_validation,label_encoders_validation = label_encoding(X_data_preap.iloc[:,d_validation],categorical_features)

        Y_validation = X_validation.iloc[:,-1]



        print("Running Model for Day " + str(s))

        # Sampling data for train & split

        X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:,0:len(X.columns)-1],Y,test_size = 0.2, random_state = 0)

        model, score = train_model(X_train, X_test, Y_train, Y_test)

        print("Model Score: " + str(score))

        

       # Uncomment for inital model

        rmse,validation_predictions = validate_model(model,X_validation.iloc[:,0:len(X_validation.columns)-1], Y_validation)

        print("RMSE Result: " + str(rmse))

        

        if (len(results.columns) == 0):

            for feature in categorical_features:

                results[feature] = label_encoders_validation[feature].inverse_transform(X_validation.iloc[:,feature].astype(int))



        results["d_" + str(s)] = validation_predictions.astype(int)

        print(results)

        results.to_csv('pd_predictions_' + str(s) +'.csv')

    return results
# Calling Predic function for 28 days of month

# Uncomment to run predictions 

#pd_predictions = run_predictions(clus)
#Display first few recods of the predictions

#pd_predictions.head()