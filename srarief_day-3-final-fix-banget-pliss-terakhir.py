# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import numpy as np

from matplotlib import pyplot as plt

import datetime as dt



pd.set_option("display.max_columns", 500) 

pd.set_option("display.max_rows", 200)

plt.rcParams['figure.figsize'] = [15, 6]

sns.set_style("darkgrid")
!pip install pandas-profiling
# Load data to pandas



online_sales = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv', sep=',')

online_sales.head()
# Menampilkan beberapa kolom

online_sales [['Date', 'Product SKU', 'Quantity', 'Revenue', 'Tax']]
# Menyimpan data ke variabel baru

test = online_sales [['Date', 'Product SKU', 'Quantity', 'Revenue', 'Tax']]



# Tidak menampilkan output

# Jika setelah disimpan divariabel lalu ingin ditampilkan 

test.head(10)

# head bagian atas, tail bagian bawah data (sesuai kolom di csv)
# Melakukan filter berdasarkan nilai suatu kolom

# Misal ambil data dengan Product SKU bernilai GGOENEBQ079099

# Cara 1

condition1= online_sales['Product SKU'] == 'GGOENEBQ079099'



online_sales[ (condition1) ]

# online_sales[online_sales['Product SKU'] == 'GGOENEBQ079099']





condition2= online_sales['Quantity'] > 2

online_sales[ (condition1) & (condition2)]
# Cara 2

# Ingin mengambil data 

online_sales.query('Quantity > 10').head(5)
test = online_sales [['Date', 'Product SKU', 'Quantity', 'Revenue', 'Tax', 'Delivery']]

## Kolom baru

test['Net_Income'] = test['Revenue'] - test['Tax'] - test['Delivery']

test.head()
## Update kolom yg ada berdasar kondisi



kondisi = test['Tax'].isnull() 

# bisa juga menggunakan >, <, == nilai tertentu 

# kondisi= test['Quantity'] > 2



test.loc[kondisi, ['Tax']] = 1
# Group by untuk aggregasi



test = online_sales.groupby(['Date'])['Quantity'].sum().reset_index()

test.head()
#Group by multiple kolom

test = online_sales.groupby(['Date', 'Product SKU'])['Quantity'].sum().reset_index()

test.head()
online_sales.head(3)
# Agregasi beberapa tipe

test = online_sales.groupby(['Date']).agg({'Quantity': 'sum',

                                                      'Revenue': 'sum',

                                                      'Tax': 'sum',

                                                      'Product SKU': 'count',

                                                      'Transaction ID': 'count',

                                                     }).reset_index()

test.head()
#Sort data berdasar quantity descending

online_sales.sort_values(by=['Quantity'], ascending = False).head(15)
test.head(3)
test.rename(index=str, columns={"Quantity": "Total Quantity", "Revenue": "Total Revenue"}, inplace = True)



test.drop(columns=['Product SKU', 'Transaction ID'], inplace = True)

test.head()
## Tahap 1 Olah data hingga mendapat total revenue per hari

online_sales = online_sales = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')

backup = online_sales.copy() ##untuk debugging pengajar, tak usah dihiraukan



daily_online_revenue = online_sales.groupby(['Date'])['Revenue'].sum().reset_index()

daily_online_revenue.tail()
## Plot



g = sns.lineplot(data = daily_online_revenue, x= 'Date', y = 'Revenue')     
daily_online_revenue.info()
daily_online_revenue['Date'] = daily_online_revenue['Date'].astype(str)

daily_online_revenue['Date'] = pd.to_datetime(daily_online_revenue['Date'])

g = sns.lineplot(data = daily_online_revenue, x= 'Date', y = 'Revenue')     
## Tahap 2 Tambahkan data yang akan diprediksi

add_data = [['2017-12-01', 0], ['2017-12-02', 0], ['2017-12-03', 0],

            ['2017-12-04', 0], ['2017-12-05', 0], ['2017-12-06', 0],

            ['2017-12-07', 0], ['2017-12-08', 0], ['2017-12-09', 0],

            ['2017-12-10', 0], ['2017-12-11', 0], ['2017-12-10', 0],

            ['2017-12-13', 0], ['2017-12-14', 0]

           ] 

  

# Create the pandas DataFrame 

add_data_df = pd.DataFrame(add_data, columns = ['Date', 'Revenue']) 

add_data_df['Date'] = add_data_df['Date'].astype(str)

add_data_df['Date'] = pd.to_datetime(add_data_df['Date'])





daily_online_revenue = daily_online_revenue.append(add_data_df)

df_rev = daily_online_revenue.copy()  ##untuk evaluasi pemateri



daily_online_revenue.tail(20)
## 3 Persiapan data prediksi dan training



## Add feature 

daily_online_revenue = 'Edit this part'


# If you want to submit, please change the value to "2017-11-30"

end_of_training_date = "2017-11-16" 



train_data = 'Edit this part'

test_data = 'Edit this part'
# Pisahkan kolom yang ingin diprediksi (biasa disebut label menjadi y), dan variabel lain menjadi x

x_train = 'Edit this part'

y_train = 'Edit this part'



x_test = 'Edit this part'

y_test = 'Edit this part'
from sklearn.ensemble import RandomForestRegressor



def fit(x_train, y_train):

    model = RandomForestRegressor(random_state=1)  #14045

    model.fit(x_train, y_train)

    return model



def predict(model, x_test):

    y_pred = model.predict(x_test)

    return y_pred



model = fit(x_train, y_train)



# Predict the model

df_rev2 = df_rev.copy()

n_iteration = len(x_test)

result = []

for i in range(n_iteration):

    y_pred = predict(model, pd.DataFrame(x_test.iloc[i]).transpose())

    result.append(y_pred[0])

    df_rev2.loc[df_rev2["Date"]==x_test.index[i],"Revenue"] = y_pred

    

    # Repeat the whole process, except for model fitting

    daily_online_revenue = preprocess(df_rev2).set_index('Date')

    _, testing_data = split_train_test(daily_online_revenue,end_of_training_date)

    x_test, _ = split_label_and_predictor(testing_data)

result
y_test
# THIS IS FOR VALIDATION PURPOSE

# Prediction vs Actual

from sklearn.metrics import mean_squared_error

from math import sqrt



comparison = pd.DataFrame({"Prediction":result,"Actual":y_test['Revenue']})

comparison.index = y_test.index

error = sqrt(mean_squared_error(comparison["Actual"], comparison["Prediction"]))

print("Error Score (RMSE) = {}".format(round(error,2)))



historical = pd.DataFrame(y_train).rename(columns={"Revenue":"Actual"}).tail(14)



pd.concat([historical,comparison],sort=True).plot();
# Save the result to CSV for submission

formatted_result = pd.DataFrame(result).reset_index().rename(columns={"index":"Id",0:"Revenue"})

display(formatted_result)



# Uncomment the code below if you want to save the result

formatted_result[['Id', 'Revenue']].to_csv("result.csv",index=False)