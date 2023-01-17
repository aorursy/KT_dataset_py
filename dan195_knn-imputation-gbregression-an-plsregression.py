import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

pd.set_option("display.max_columns",len(test))

train.shape,test.shape

# Any results you write to the current directory are saved as output.
missing_vals = pd.concat([train.isnull().sum()/len(train),test.isnull().sum()/len(test)],axis=1, keys=['Train','Test'])

missing_vals.sort_values(ascending=False,by="Train")
import datetime



def time_stamp(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["year"] = df["timestamp"].dt.year

    df["month"] =  df["timestamp"].dt.month

    df["day"] = df["timestamp"].dt.day 

    del df["timestamp"]



time_stamp(train)

time_stamp(test)







train_cont = [x for x in train.columns if train.dtypes[x] != 'object' and (x not in ['day','month','year'])]

test_cont = [x for x in test.columns if test.dtypes[x] != 'object' and (x not in ['day','month','year'])]
"""Examining Correlations"""

pd.DataFrame(train.corr()["price_doc"].sort_values(ascending=False))
del_low_corr = ["trc_sqm_5000","prom_part_500","build_count_1971-1995",

               "school_quota","ID_railroad_station_walk","cemetery_km","water_km",

               "big_church_count_500","cafe_sum_3000_max_price_avg",

               "cafe_avg_price_3000","cafe_sum_3000_min_price_avg",

               "build_count_1921-1945","16_29_male",

               "female_f","full_all","ID_bus_terminal","ID_railroad_station_avto",

               "ID_big_road1","ID_big_road2","trc_count_500","trc_count_1000",

               "trc_sqm_1500","cafe_count_500_price_4000","cafe_count_500_price_2500",

               "market_count_5000","hospital_beds_raion",

               "cafe_avg_price_500","cafe_sum_500_max_price_avg",

               "cafe_sum_500_min_price_avg","preschool_quota","cafe_count_1500"]

for i in del_low_corr:

    del train[i]

    del test[i]

    
for i in train.skew().keys():

    if abs(train.skew()[i]) > .5 and (i!= 'price_doc'):

        train[i] = np.log1p(train[i])

        test[i] = np.log1p(test[i])

        print("Just finished {} with skew {}".format(str(i), str(train.skew()[i])))

        #Distribution plots if needed

        #sns.distplot(train[i].dropna())

        #plt.show()
train = pd.get_dummies(train)

test = pd.get_dummies(test)



sns.distplot(train.price_doc)

plt.title("price")

plt.show()





train.price_doc = np.log1p(train.price_doc)



sns.distplot(train.price_doc)

plt.title("log transformed price")

plt.show()
"""from fancyimpute import  KNN



train.material = train.material.fillna(1) #As none are missing in test.



train_columns = list(train) #fancyimpute removes var names

test_columns = list(test)



train = pd.DataFrame(KNN(k=3).complete(train))

test = pd.DataFrame(KNN(k=3).complete(test))



train.columns = train_columns 

test.columns = test_columns



#Make material an object again. 

train.material = train.material.astype("object")

test.material = test.material.astype("object")

train = pd.get_dummies(train)

test = pd.get_dummies(test)"""
"""my_ids = train['id']

test_id = test['id']

from sklearn.cross_decomposition import PLSRegression

from sklearn.preprocessing import scale

X = np.array(train.drop(["price_doc","id"],axis=1))

y = np.array(train["price_doc"])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .25,random_state=1)



pls = PLSRegression(n_components=20,scale=False)

pls.fit(X_train,y_train)

y_pred = pls.predict(X_test)

r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred)"""
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(loss='huber',learning_rate=.05,n_estimators=500,

                              max_features='sqrt',min_samples_leaf=10)

                                #max_features = 'sqrt' for multicollinearity.

"""gb.fit(X_train,y_train)

gb_pred = gb.predict(X_test)

r2_score(y_test,gb_pred),mean_squared_error(y_test,gb_pred)"""