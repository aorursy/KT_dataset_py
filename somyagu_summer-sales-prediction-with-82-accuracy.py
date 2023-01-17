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
import numpy as np
data=pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
data.head()
data.columns
data.drop(columns=['title','title_orig','product_color'],inplace=True)
data.drop(columns=['product_variation_size_id','product_variation_inventory','shipping_option_name'],inplace=True)
data.head()
#Display Maximum columns in the data
pd.set_option('display.max_columns', 37)
data.head()
data.columns
data.drop(columns=['currency_buyer','tags','urgency_text'],inplace=True)
data.drop(columns=[ 'origin_country','merchant_title', 'merchant_name','merchant_info_subtitle'],inplace=True)
data.drop(columns=['merchant_id','product_url'],inplace=True) 
data.drop(columns=['product_picture','product_id'],inplace=True)
data
data.info()
data.drop(columns=['crawl_month','merchant_profile_picture'],inplace=True)
#Using Label Encoder to categorical features into numeric values like 0,1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['theme']=le.fit_transform(data['theme'])
data.info()
data.duplicated().sum()
data.drop_duplicates(inplace=True)
data.isnull().sum()
data['has_urgency_banner']=data['has_urgency_banner'].fillna(0)
data.isnull().sum()
#Calculating mean
mean_5=data['rating_five_count'].mean()
mean_4=data['rating_four_count'].mean()
mean_3=data['rating_three_count'].mean()
mean_2=data['rating_two_count'].mean()
mean_1=data['rating_one_count'].mean()
data['rating_five_count']=data['rating_five_count'].fillna(mean_5)
data['rating_four_count']=data['rating_four_count'].fillna(mean_4)
data['rating_three_count']=data['rating_three_count'].fillna(mean_3)
data['rating_two_count']=data['rating_two_count'].fillna(mean_2)
data['rating_one_count']=data['rating_one_count'].fillna(mean_1)
data.isnull().sum()
data
y=data['units_sold']
x=data.drop('units_sold',axis=1)
x
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
model = LinearRegression()

k=model.fit(x_train,y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
#Accuracy
r2_score(y_test,y_pred)