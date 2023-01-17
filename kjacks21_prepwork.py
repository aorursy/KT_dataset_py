# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

dist = pd.read_csv("../input/destinations.csv")

destinations = pd.read_csv("../input/destinations.csv")

train = pd.read_csv('../input/train.csv',
                            usecols=["date_time", "user_location_country", "user_location_region", "user_location_city",
                                    "user_id", "is_booking", "orig_destination_distance",
                                     "hotel_cluster", "srch_ci", "srch_co", "srch_destination_id", 
                                     "hotel_continent", "hotel_country", "hotel_market"],
                            dtype={'date_time':np.str_, 'user_location_country':np.int8, 
                                   'user_location_region':np.int8, 'user_location_city':np.int8, 
                                   'user_id':np.int32, 'is_booking':np.int8,
                                   "orig_destination_distance":np.float64,
                                   "hotel_cluster":np.int8,
                                   'srch_ci':np.str_, 'srch_co':np.str_,
                                   "srch_destination_id":np.int32,
                                   "hotel_continent":np.int8,
                                   "hotel_country":np.int8,
                                   "hotel_market":np.int8}                        
                           )
                           
test = pd.read_csv('../input/test.csv',
                           usecols=["id", "date_time", "user_location_country", "user_location_region", "user_location_city",
                                "user_id", "orig_destination_distance",
                                   "srch_ci", "srch_co", "srch_destination_id",
                                   "hotel_continent", "hotel_country", "hotel_market"],
                            dtype={'id':np.int32, 'date_time':np.str_, 'user_location_country':np.int8, 
                            'user_location_region':np.int8, 'user_location_city':np.int8, 
                            'user_id':np.int32, 
                            "orig_destination_distance":np.float64, 'srch_ci':np.str_, 'srch_co':np.str_,
                                   "srch_destination_id":np.int32,
                                   "hotel_continent":np.int8,
                                   "hotel_country":np.int8,
                                   "hotel_market":np.int8})	
train.shape
test.shape
train.head(5)
test.head(5)

train.head()
# assesing feature importance for random forest
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_hotel.columns[1:]
forest = RandomForestClassifier(
    n_estimators=10000,
    random_state=0
)
forest.fit(x_train, y_train)
importances=forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f],importances[indices[f]]))    
    
    
