# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

sample_submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
df_train = df_train.replace(np.nan, '', regex=True)

df_test = df_test.replace(np.nan, '', regex=True)



groups_train = df_train.groupby(['Country_Region', 'Province_State'])

min_date = groups_train['Date'].min()

min_date_sorted = min_date.sort_values()

index = 149

key_ = min_date_sorted.index[index]

print(index, key_)

record = df_train[(df_train['Country_Region'] == key_[0]) & (df_train['Province_State'] == key_[1])]



# x axis values 

x = record['Date']

# corresponding y axis values 

y1 = record['ConfirmedCases']

# y2 = record['Fatalities']



# plotting the points  

plt.plot(x, y1, label = "Confirmed Cases") 

# plt.plot(x, y2, label = "Fatalities") 

# naming the x axis 

plt.xlabel('Date') 

# naming the y axis 

plt.ylabel('Label') 



# giving a title to my graph 

plt.title(str(key_[0]) + " " + str(key_[1]) + ' - Confirmed Cases') 



# function to show the plot 

plt.show() 




indices = [83, 109, 63, 248]



for index in indices:

    key_ = min_date_sorted.index[index]

    print(index, key_)

    record = df_train[(df_train['Country_Region'] == key_[0]) & (df_train['Province_State'] == key_[1])]



    # x axis values 

    x = record['Date']

    # corresponding y axis values 

    y1 = record['ConfirmedCases']

    y2 = record['Fatalities']



    # plotting the points  

    plt.plot(x, y1, label = "Confirmed Cases") 

    # plt.plot(x, y2, label = "Fatalities") 

    # naming the x axis 

    plt.xlabel('Date') 

    # naming the y axis 

    plt.ylabel('Label') 



    # giving a title to my graph 

    plt.title(str(key_[0]) + " " + str(key_[1]) + ' - Confirmed Cases') 



    # function to show the plot 

    plt.show() 


indices = [63, 7, 33, 221, 149]

stages = ['Spread', 'Lockdown', 'Controlled', 'Re-Emerge', 'Re-Controlled']

count = 0

for index, stage in zip(indices, stages):

    key_ = min_date_sorted.index[index]

    print(index, key_)

    record = df_train[(df_train['Country_Region'] == key_[0]) & (df_train['Province_State'] == key_[1])]



    # x axis values 

    x = record['Date']

    # corresponding y axis values 

    y1 = record['ConfirmedCases']

    # y2 = record['Fatalities']



    # plotting the points  

    plt.plot(x, y1, label = "Confirmed Cases") 

    # plt.plot(x, y2, label = "Fatalities") 

    # naming the x axis 

    plt.xlabel('Date') 

    # naming the y axis 

    plt.ylabel('Label') 



    # giving a title to my graph 

    plt.title("STAGE " + str(count + 1) + "     " + stage) 



    # function to show the plot 

    plt.show()

    count = count + 1


# index = 0

# for key_,_ in zip(min_date_sorted.index, min_date_sorted):

#     print(index, key_)

#     record = df_train[(df_train['Country_Region'] == key_[0]) & (df_train['Province_State'] == key_[1])]



#     # x axis values 

#     x = record['Date']

#     # corresponding y axis values 

#     y1 = record['ConfirmedCases']

#     y2 = record['Fatalities']



#     # plotting the points  

#     plt.plot(x, y1, label = "Confirmed Cases") 

#     # plt.plot(x, y2, label = "Fatalities") 

#     # naming the x axis 

#     plt.xlabel('Date') 

#     # naming the y axis 

#     plt.ylabel('Label') 



#     # giving a title to my graph 

#     plt.title(str(key_[0]) + " " + str(key_[1]) + ' - Confirmed Cases') 



#     # function to show the plot 

#     plt.show() 

#     index = index + 1
