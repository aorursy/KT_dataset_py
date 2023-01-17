

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

HOUSING_PATH = "../input"



# Any results you write to the current directory are saved as output.

def load_housing_data(housing_path=HOUSING_PATH):    

    properties_file = os.path.join(housing_path, "properties_2016.csv")

    train_file = os.path.join(housing_path, "train_2016_v2.csv")

    return pd.read_csv(properties_file), pd.read_csv(train_file)

    

housing_prop, housing_train = load_housing_data()



housing = pd.merge(housing_prop, housing_train, how='inner',on=['parcelid'])

#housing.info()



#import matplotlib.pyplot as plt

#housing.hist(bins=50, figsize=(20,15))

#plt.show()



#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1);



housing_train = housing.drop('logerror', axis=1)

housing_label = housing['logerror'].copy()

housing_train_num = housing_train.drop(['airconditioningtypeid','architecturalstyletypeid',

                                        'basementsqft','buildingclasstypeid','buildingqualitytypeid',

                                        'decktypeid','poolcnt',

                                        'pooltypeid10','pooltypeid2','pooltypeid7',

                                        'storytypeid','assessmentyear','hashottuborspa',

                                        'propertycountylandusecode','propertyzoningdesc',

                                        'fireplaceflag','taxdelinquencyflag','transactiondate'],axis=1);



#housing_train_num.describe()



housing_train_num.head()

#Create Pipeline here....

from sklearn.preprocessing import LabelBinarizer,LabelEncoder,Imputer

imputer = Imputer(strategy="median")

imputer.fit(housing_train_num)

imputer.statistics_

housing_train_num

#encoder = LabelBinarizer()

#encoder = LabelEncoder()

#taxdelinquencyflag_encoded = encoder.fit_transform(housing_train['taxdelinquencyflag'])

#taxdelinquencyflag_encoded

#taxdelinquencyflag_1hot = encoder.fit_transform(housing_train['taxdelinquencyflag'])

#taxdelinquencyflag_1hot

#housing_train_num.shape
#run model

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_train_num, housing_label)