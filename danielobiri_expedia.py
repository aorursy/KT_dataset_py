# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

 

    

import pandas as pd



raw_data= pd.read_csv('../input/expedia-hotel-ranking-exercise/train.csv')

test_data=pd.read_csv("../input/expedia-hotel-ranking-exercise/test.csv")


#View data

raw_data.head(5)
#check to if class is balanced

raw_data['prop_booking_bool'].value_counts()
#view columns to see remove id's

raw_data.columns

len(raw_data.columns)
#group columns into search_level and property_level
search_level=[

    'srch_id', 

    #'srch_date_time', 

    #'srch_visitor_id',

       'srch_visitor_visit_nbr', 

    #'srch_visitor_loc_country',

      # 'srch_visitor_loc_region', 'srch_visitor_loc_city',

       #'srch_visitor_wr_member', 

    #'srch_posa_continent', 'srch_posa_country',

      # 'srch_hcom_destination_id', 

    'srch_dest_longitude', 

    'srch_dest_latitude',

       #'srch_ci', 'srch_co', 'srch_ci_day', 'srch_co_day', 

    'srch_los',

       'srch_bw', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',

       'srch_mobile_bool',

    #'srch_mobile_app',

    #'srch_device', 

    #'srch_currency' #'srch_local_date'

]
len(search_level)
property_level=[#'prop_key',

    'prop_travelad_bool', 'prop_dotd_bool',

       'prop_price_without_discount_local', 

    'prop_price_without_discount_usd',

       'prop_price_with_discount_local', 'prop_price_with_discount_usd',

       'prop_imp_drr', 'prop_brand_bool',

       'prop_starrating',

    #'prop_super_region', 

    #'prop_continent',

      # 'prop_country', 

    #'prop_market_id', 

    #'prop_submarket_id',

       'prop_room_capacity', 'prop_review_score', 'prop_review_count',

       'prop_hostel_bool']
len(property_level)


#find the dataypes of features

raw_data[search_level+property_level].dtypes
raw_data.loc[:,search_level]
raw_data.loc[:,property_level]
raw_data.loc[:,search_level+property_level]
#assign class label

target=["prop_booking_bool"]

#target=raw_data.loc[:,target]
refined_data=raw_data.loc[:,search_level+property_level +target]


#check to see if data has duplicates

refined_data[refined_data.duplicated()]

#remove duplicated rows from datasets



refined_data.drop_duplicates(inplace=True)
refined_data
#Remove the the columns of the prices, since the usd equivalance is the dataset

refined_data.drop('prop_price_with_discount_local', axis=1,inplace=True)

refined_data.drop('prop_price_without_discount_local', axis=1,inplace=True)
#import datetime as dt 

#refined_data['srch_date_time'] = pd.to_datetime(refined_data['srch_date_time']) 

#refined_data['srch_date_time']=refined_data['srch_date_time'].map(dt.datetime.toordinal)
refined_data.columns
#Imbalance data

refined_data.prop_booking_bool.value_counts()
#check for columns that have missing values in the dataset

refined_data[refined_data.values==np.NAN].columns
#find number of columns in columns

refined_data.isnull().sum(axis = 0)
#drop rows containing missing values

refined_data.dropna(inplace=True)



refined_data
#determine important features using correlation matrix, see how they correlerate with the class,prop_booking_bool and 

#multicollinearity aming predictors

import matplotlib.pyplot as plt

import seaborn as sns 



refined_data.describe

plt.figure(figsize=(20,20))

cor=refined_data.corr()

sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)

#remove column prop_price_without_discount_usd due to high correlation between this column and prop_price_with_discount_usd

refined_data.drop('prop_price_without_discount_usd', axis=1,inplace=True)



#drop srch_bw column due to low correlation with target

refined_data.drop("srch_bw",axis=1,inplace=True)
refined_data.columns
#shuffle dataset to avoid bias when used in training

from sklearn.utils import shuffle



refined_data=shuffle(refined_data)

refined_data.value_counts()
from sklearn.utils import shuffle



#deal with imbalnace in data class by doing undersampling



class_count_0, class_count_1 =refined_data['prop_booking_bool'].value_counts()



class_0=refined_data[refined_data['prop_booking_bool']==0]

class_1=refined_data[refined_data['prop_booking_bool']==1]



class_0_under=class_0.sample(class_count_1)



refined_data=pd.concat([class_0_under,class_1])
refined_data
#select predictors

X=refined_data.iloc[:,1:19]

print(X)



from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

# Create the Scaler object

#standardize data due to different units



scaler = preprocessing.StandardScaler()# Fit data on the scaler object

X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

import sklearn as skl

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import make_scorer

#from sklearn.metrics import mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression

import pickle

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

import collections

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score







from sklearn.model_selection import cross_val_score



#select the dependent variable 

y=refined_data.prop_booking_bool.values



#split data into training and testing data



X_train, X_test, y_train, y_test = train_test_split(B, y,test_size=0.2,random_state=0)



print(X_train.shape)


#use SMV model for prediction 

from xgboost import XGBRegressor



regressor = SVC()







regressor.fit(X_train, y_train) 





y_pred=regressor.predict(X_test)











from sklearn.metrics import ndcg_score

from sklearn.metrics import average_precision_score





from scipy.stats import spearmanr



#calculate the model performance using MAP score

print(average_precision_score(y_test,y_pred))
Pred_train_data=refined_data.iloc[:,1:19]

Pred_train_data
#Preprocess the test data as the train data

test_data.dropna(inplace=True)

#test_data.drop('srch_bw', axis=1,inplace=True)






Pred_test_data=pd.DataFrame(columns=Pred_train_data.columns)

for column in Pred_test_data.columns:

    if column in test_data.columns:

        Pred_test_data[column]=test_data[column]

    else:

        Pred_test_data[column]=None

Pred_test_data


#Pred_test_data.drop("srch_id",axis=1,inplace=True)

Pred_test_data
#Use model to predict using standardized dataset



test_result=regressor.predict(scaler.fit_transform(Pred_test_data.iloc[:,0:]))





newdataframe=pd.DataFrame({'srch_id':test_data['srch_id'],'prop_key':test_data['prop_key'], 'prop_booking_bool':test_result})

newdataframe



newdataframe.to_csv('pred_booking_bool.csv', index=False)



newdataframe