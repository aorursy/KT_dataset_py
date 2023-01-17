# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns                       #visualisation

import matplotlib.pyplot as plt             #visualisation

%matplotlib inline     

sns.set(color_codes=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_flight = pd.read_csv("../input/penyisihan-datavidia/flight.csv")

data_hotel = pd.read_csv("../input/penyisihan-datavidia/hotel.csv")

data_test = pd.read_csv("../input/penyisihan-datavidia/test.csv")
#Step 1

print(data_flight.isnull().sum()) #tidak ada missing value pada setiap atribut data_flight

print(data_hotel.isnull().sum()) #tidak ada missing value pada setiap atribut data_hotel
#Step 1

data_flight.info #untuk melihat info data
#step 1

data_flight.shape #untuk melihat dimensi data
#Step 1

data_flight.columns #untuk melihat atribut apa saja yg digunakan
#Step 1

data_flight.dtypes
#Step 2

plt.figure(figsize=(10,5)) 

Maps= data_flight.corr()

sns.heatmap(Maps,cmap="BrBG",annot=True)

Maps
#Step 3

i = data_flight[np.logical_or(data_flight['airlines_name'] == '9855a1d3de1c46526dde37c5d6fb758c',data_flight['airlines_name'] == '6872b49542519aea7ae146e23fab5c08')].index

data_flight.drop(i,inplace=True)

data_flight[['member_duration_days','no_of_seats']] = data_flight[['member_duration_days','no_of_seats']].astype('int32') #Ubah tipe data ke integer

data_flight.dtypes

data_flight.head()
#Step 4



data_hotel = data_hotel.drop('city',axis=1) #Drop kolom 'city' berisi sama

data_hotel.head()
#Step 1

merged = data_flight.merge(data_hotel,left_on='hotel_id',right_on='hotel_id',how='left').fillna("None") #Merge data_flight dengan data_hotel

merged.head(5)
#Step 2

def is_cross_sell(data_hotel):

    if data_hotel == "None":

        return "No"

    else:

        return "Yes"

    

data_flight['is_cross_sell'] = data_flight['hotel_id'].apply(is_cross_sell)

data_flight.drop('hotel_id',axis=1,inplace=True)

data_flight
#Step 3

le = LabelEncoder()

columns_to_label = ['gender','trip','service_class','is_tx_promo','is_cross_sell']

def encoding(df):

        for feature in columns_to_label:

            try:

                df[feature] = le.fit_transform(df[feature])

            except:

                print('Error encoding '+feature)

        

        columnsToOHE = ['airlines_name']

        for feature in columnsToOHE:

            ohe = pd.get_dummies(df[feature])

            df = df.join(ohe)

            df.drop(feature,axis=1,inplace=True)

        return df



preprocess = encoding(data_flight)

preprocess.head()
#Step 4

temp = preprocess.drop(['account_id','log_transaction','visited_city','trip','gender','route'],axis=1).set_index('order_id')

y = temp['is_cross_sell']

X = temp.drop('is_cross_sell',axis=1)

column_names = X.columns.tolist()

column_names
#Step 1 

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=1,test_size=0.33)

ytest.dtypes
#Step 2

def AccuracyTracker(Xtrain,Xtest,ytrain,ytest,n):

    model = DecisionTreeClassifier(max_leaf_nodes=n,random_state=1)

    model.fit(Xtrain,ytrain)

    print(n,accuracy_score(ytest,model.predict(Xtest)))

for i in range(2,50):

    AccuracyTracker(Xtrain,Xtest,ytrain,ytest,i)
#Step 3



scaler = MinMaxScaler(feature_range=(0, 1))



def encoding_test(data_flight):

        columns_to_label = ['service_class','is_tx_promo']

        for feature in columns_to_label:

            #try:

            data_flight[feature] = le.fit_transform(data_flight[feature])

           # except:

             #   print('Error encoding '+feature)

        

        columns_to_OHE = ['airlines_name','gender','service_class']

        for feature in columns_to_OHE:

            ohe = pd.get_dummies(data_flight[feature])

            data_flight.drop(feature,axis=1,inplace=True)

            

            if feature == "airlines_name":

                data_flight = data_flight.join(ohe)

                return data_flight

    

def preprocess(data_flight):

    data_flight = data_flight.drop('route',axis=1)

    data_flight = data_flight.drop('gender',axis=1)

    data_flight = data_flight.drop('trip',axis=1)

    data_flight[['member_duration_days','no_of_seats']] = data_flight[['member_duration_days','no_of_seats']].astype('int32')

    data_flight = encoding_test(data_flight)

    data_flight = data_flight.drop(['account_id','log_transaction','visited_city'],axis=1).set_index('order_id')

#     flight = scaler.transform(flight)

    return data_flight

    

testing_x = preprocess(data_test)
pd.DataFrame(testing_x)
#Step 4

model = DecisionTreeClassifier(max_leaf_nodes=30,random_state=1)

model.fit(Xtrain,ytrain)
prediction_result = model.predict(testing_x)

def is_cross_sell_prediction(df):

    if df>0.3:

        return "Yes"

    else:

        return "No"

    

hokyaaa_prediction = pd.DataFrame(pd.DataFrame(prediction_result)[0].apply(is_cross_sell_prediction)).join(data_test['order_id'])

columnsTitles=["order_id",0]

hokyaaa_prediction = hokyaaa_prediction.reindex(columns=columnsTitles)

hokyaaa_prediction.columns = ['order_id','is_cross_sell']

hokyaaa_prediction.to_csv('hokyaaa_prediction.csv',index = False)
import pandas as pd

flight = pd.read_csv("../input/penyisihan-datavidia/flight.csv")

hotel = pd.read_csv("../input/penyisihan-datavidia/hotel.csv")

sample_submission = pd.read_csv("../input/penyisihan-datavidia/sample_submission.csv")

test = pd.read_csv("../input/penyisihan-datavidia/test.csv")