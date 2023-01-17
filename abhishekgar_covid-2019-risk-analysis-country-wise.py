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

import numpy as np

import os

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns



from matplotlib.pyplot import figure
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")  #(source: kaggle forecasting challenge..)

train["Date"] = pd.to_datetime(train["Date"])





## recovered cases: 

recover = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")  ## source: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset

recover = pd.melt(recover, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date')

recover["Date"] = pd.to_datetime(recover["Date"], format= "%m/%d/%y")

recover = recover.rename(columns= {"value": "recovered"})

recover = recover[["Lat", "Long", "Country/Region", "Date", "recovered"]].drop_duplicates()



train = pd.merge(train, recover, on= ["Lat", "Long", "Country/Region", "Date"], how="left")
## data at country level

final = train.groupby(["Country/Region", "Date"])["ConfirmedCases","Fatalities", "recovered"].sum().reset_index()



## consider only when case>0

final = final[final["ConfirmedCases"]>0]



## since recovered dataset is having data till 2020-03-23

final = final[final["Date"]<= "2020-03-23"]



print(final.Date.min())

print(final.Date.max())
## check if there is any bug while merging  dataset coming from 2 different source:

final[final["ConfirmedCases"]< final["recovered"]]
## adding active cases

final["active_cases"] = final["ConfirmedCases"] - final["recovered"]
# finction to get confirmed cases, active cases, velocity and acceleration of a given country: 



'''

country : name of the country. 

dt: day span you want to take aggregation on. let's say if select 7 days: we will calculate velocity, acceleration etc wrt 7th day lag value

cutoff: cases cutoff below which you want to ignore your analysis. because graph is very noisy/unstable for very low cases making difficult for any interpretation. 



return: dataframe with all required metrics like cases (confirmed/active), velocity and acceleration..

'''



def gen_growth (country, dt=1, cutoff = 5): 

    temp = final[(final["Country/Region"] == country)& (final["ConfirmedCases"]>= cutoff)].groupby("Date")["ConfirmedCases", "active_cases"].sum().reset_index()

    

    # cases velocity

    train_lag =temp.shift(periods=dt)

    temp['lag_confirmedcases']=train_lag['ConfirmedCases']

    temp["velocity"] = temp['ConfirmedCases'] - temp['lag_confirmedcases']



    # Acceleration

    train_lag =temp.shift(periods=dt)

    temp['lag_velocity']=train_lag['velocity']

    temp["acceleration"] = temp['velocity'] - temp['lag_velocity']

    

    temp["confirm_scale"] = temp["ConfirmedCases"]*10/temp["ConfirmedCases"].max()

    temp["active_scale"] = temp["active_cases"]*10/temp["active_cases"].max()

    temp["velocity_scale"] = temp["velocity"]*10/temp["velocity"].max()

    temp["acc_scale"] = temp["acceleration"]*10/temp["acceleration"].max()

    

    return temp[["Date","ConfirmedCases",  "confirm_scale", "active_scale", "velocity_scale", "acc_scale"]]


cutoff = 10

dt = 7  ## graphical representation will be smoothned for higher dt 

num = 30  ## get graphs of top 20 countries with higher corona cases.. 

temp_countries = final.groupby(['Country/Region']).ConfirmedCases.max().reset_index().sort_values("ConfirmedCases", ascending=False).head(num)

countries = temp_countries['Country/Region'].unique()



for cont in countries:

    figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

    temp = gen_growth(cont, dt=dt, cutoff = cutoff)

    max_case = temp.ConfirmedCases.max()

    

    plt.plot(list(temp.confirm_scale))

    plt.plot(list(temp.active_scale))

    plt.plot(list(temp.velocity_scale))

    plt.plot(list(temp.acc_scale))

    plt.legend(['confirmed cases', "active cases", "velocity", "acceleration"], loc='upper left')

    plt.title(cont+"_"+str(max_case))



    plt.show()   
temp_countries = final.groupby(['Country/Region']).ConfirmedCases.max().reset_index().sort_values("ConfirmedCases", ascending=False).head(30)

countries = temp_countries['Country/Region'].unique()



temp_final = pd.DataFrame()



for cont in countries:

    temp = gen_growth(cont, dt=7, cutoff = 10)

    temp = temp[temp["Date"]== temp.Date.max()]

    temp["country"] = cont

    temp_final = temp_final.append(temp)

    

bins = [-10, 0, 4,  8, 10]  

labels = [-1, 0, 1, 2]

temp_final['velocity_binned'] = pd.cut(temp_final['velocity_scale'],  bins=bins, labels=labels)



bins = [-10, 0, 3,  8, 10]  

labels = [-1, 0, 1, 2]

temp_final['acc_binned'] = pd.cut(temp_final['acc_scale'],  bins=bins, labels=labels)



temp_final = temp_final[~(temp_final["acc_scale"].isna())].reset_index()

temp_final
### Creating heat map:



df1 = pd.DataFrame(list(range(-1,3)))

df2 = pd.DataFrame(list(range(0,3)))



df1['key'] = 0

df2['key'] = 0



matrix = pd.merge(df1, df2, on="key")

matrix = matrix[["0_x", "0_y"]].rename(columns= {"0_x": "acceleration", "0_y": "velocity"})

matrix["risk"] = [0, 1, 10,        ## High value indicating high risk and lower values as lower risk .. 

                  1, 10, 20, 

                  10, 20, 30,

                  20, 30, 40]



matrix = matrix.pivot("acceleration", "velocity", "risk")

heat_map = sns.heatmap(matrix)
## countries according to risk:

lowest = temp_final[(temp_final["velocity_binned"] == 0)& (temp_final["acc_binned"] <= 0) ].country.unique()

lower = temp_final[(temp_final["velocity_binned"] <= 1)& (temp_final["acc_binned"] <= 0) & (~(temp_final["country"].isin(lowest)))].country.unique()

moderate = temp_final[(temp_final["velocity_binned"] <= 1)& (temp_final["acc_binned"] <= 1) & (~(temp_final["country"].isin(lowest)))& (~(temp_final["country"].isin(lower)))].country.unique()





print("lowest risk:", lowest)

print("lower risk:", lower)

print("moderate risk:", moderate)

print("high risk: almost rest of the countries.. ")