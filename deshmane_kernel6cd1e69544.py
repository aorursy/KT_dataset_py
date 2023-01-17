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
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

sample_submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
df_train
df_test
sample_submission
train_countries = list(df_train['Country_Region'].unique())

test_countries = list(df_test['Country_Region'].unique())
print(len(train_countries))

print(len(test_countries))

print(len(set(train_countries).intersection(set(test_countries))))
df_train_original = df_train

df_test_original = df_test


df_train = df_train_original.replace(np.nan, '', regex=True)

df_test = df_test_original.replace(np.nan, '', regex=True)
df_train[['Country_Region', 'Province_State']]
df_train['location'] = df_train[['Country_Region', 'Province_State']].apply(lambda x: '-'.join(x), axis=1)

df_test['location'] = df_test[['Country_Region', 'Province_State']].apply(lambda x: '-'.join(x), axis=1)
locations_train = df_train['location'].unique()

locations_test = df_train['location'].unique()
print(len(locations_train))

print(len(locations_test))
df_train[df_train['ConfirmedCases'] > 0]
groups_train = df_train.groupby(['Country_Region', 'Province_State'])

print(len(groups_train))
# groups_train = df_train[df_train['ConfirmedCases'] > 0].groupby(['Country/Region', 'Province/State'])

# print(len(groups_train))

# groups_test = df_test[df_test['ConfirmedCases'] > 0].groupby('location')
# groups.get_group('China-Hebei')

min_date = groups_train['Date'].min()
min_date
min_date_sorted = min_date.sort_values()
for x,y in zip(min_date_sorted.index, min_date_sorted):

    print(x,y)
list(df_train[df_train['Country_Region'] == 'China']['Province_State'])
df_train[(df_train['Country_Region'] == 'Pakistan')]
df_train[(df_train['Country_Region'] == 'US') & (df_train['Province_State'] == 'Washington')]
index = 0

for x,y in zip(min_date_sorted.index, min_date_sorted):

    print(index, x, y)

    index = index + 1
min_date_sorted.shape
import matplotlib.pyplot as plt 
index = 34
record = df_train[(df_train['Country_Region'] == min_date_sorted.index[index][0]) & (df_train['Province_State'] == min_date_sorted.index[index][1])]
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

plt.title(str(min_date_sorted.index[index][0]) + " " + str(min_date_sorted.index[index][1]) + ' - Confirmed Cases') 

  

# function to show the plot 

plt.show() 
# x axis values 

x = record['Date']

# corresponding y axis values 

y1 = record['ConfirmedCases']

y2 = record['Fatalities']

  

# plotting the points  

# plt.plot(x, y1, label = "Confirmed Cases") 

plt.plot(x, y2, label = "Fatalities") 

# naming the x axis 

plt.xlabel('Date') 

# naming the y axis 

plt.ylabel('Label') 

  

# giving a title to my graph 

plt.title(str(min_date_sorted.index[index][0]) + " " + str(min_date_sorted.index[index][1]) + ' - Fatalities') 

  

# function to show the plot 

plt.show() 
import numpy as np 

  

# curve-fit() function imported from scipy 

from scipy.optimize import curve_fit 

  

from matplotlib import pyplot as plt 

  

# numpy.linspace with the given arguments 

# produce an array of 40 numbers between 0 

# and 10, both inclusive 

x = np.linspace(0, 10, num = 40) 

  

  

# y is another array which stores 3.45 times 

# the sine of (values in x) * 1.334.  

# The random.normal() draws random sample  

# from normal (Gaussian) distribution to make 

# them scatter across the base line 

y = 3.45 * np.sin(1.334 * x) + np.random.normal(size = 40) 

  

# Test function with coefficients as parameters 

def test(x, a, b): 

    return a * np.sin(b * x) 

  

# curve_fit() function takes the test-function 

# x-data and y-data as argument and returns  

# the coefficients a and b in param and 

# the estimated covariance of param in param_cov 

param, param_cov = curve_fit(test, x, y)
from numpy import exp, linspace, random, log

import math

t = np.arange(0., 90., 1)



# red dashes, blue squares and green triangles

# f = [70000*1/(1+math.pow(math.e,-(x-30)/2)) for x in t]

# f = [700*1/(1+math.pow(math.e,-(x-20)/2)) * (700*1/(1+math.pow(math.e,-(x-40)/2)) + 700) for x in t]



f = 9000*1/(1+exp(-1*(t-42)/2.7)) + 50*log(1+exp((t-50))) - 0

print(f)

plt.plot(t, f, 'r--')

plt.show()
record = df_train[(df_train['Country_Region'] == min_date_sorted.index[index][0]) & (df_train['Province_State'] == min_date_sorted.index[index][1])]

record = record[record['ConfirmedCases'] > 0]



from datetime import datetime

# datetime.date.today()

base_date_object = datetime.strptime('2020-01-22', "%Y-%m-%d").date()

record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in record['Date']]



record2 = record[record['Fatalities'] > 0]

# x axis values 

x = record['days'].values

x2 = record2['days'].values

# corresponding y axis values 

y1 = record['ConfirmedCases'].values

y2 = record2['Fatalities'].values
# print(record['Date'][3600])



x
# record['days']

y2
from scipy.optimize import curve_fit

from numpy import exp, linspace, random, log



def gaussian(x, amp, cen, wid):

    return amp * exp(-(x-cen)**2 / wid)



def test(x, a, b, c): 

    return a*1/(1+exp(-b*(x-c)))



def test_linear(x, a, b, c, d, e, f): 

    return a*1/(1+exp(-b*(x-c))) + d*log(1+exp(x-e)) - f



def custom(x, a, b , c, d, e, f, g):

    return a*1/(1+exp(-(x-b)/c)) * (d*1/(1+exp(-(x-e)/f)) + g)
y_max_ = y1[-1]

y1_prime = np.diff(y1)

y1_prime2 = np.diff(y1_prime)

if len(y1)>0 and len(y1_prime) > 0 and len(y1_prime2) > 0:

    max_slope_index = len(y1_prime) - 1 - list(y1_prime)[::-1].index(max(y1_prime))

    max_slope_range = (max_slope_index+1)/len(y1_prime)

    y_max_ = y1[-1]

    if max_slope_range < 0.75:

        if y1_prime[max_slope_index] > 0 and max_slope_range < 0.75 and (((y1_prime[max_slope_index] - max(y1_prime[-2:]))/y1_prime[max_slope_index]) < 0.5):

            y_max_ = y1[-1]

            pass

        else:

            y_max_ = y1[max_slope_index + 1]

            pass

    else:

        y_max_ = y1[-1]
y1_prime
y1
y_max_
# param, param_cov = curve_fit(test, np.array(x), np.array(y1), [700, 0.5, 19], bounds=([1,0.1,-30],[800,2,150]))

# param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y1[-1]/2, 0.5, (x[-1] - x[0])/2 + x[0]], bounds=([y1[-1]/2, 0.1, -30],[y1[-1] + 1000, 2, 150]))

# param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_/2, 0.1, 0],[y_max_*2 + 1500, 1, 150]))

param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*4, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*3, 0.1, 0],[y_max_*8 + 1500, 1, 150]))

# param, param_cov = curve_fit(test_linear, np.array(x), np.array(y1), [y_max_, 0.5, (x[-1] - x[0])/2 + x[0], 50, 45, 0], maxfev = 100000, bounds=([y_max_/2, 0.1, 0, 1, 30, -100],[y_max_*2 + 1500, 1, 150, 100, 100, 1000]))

# param, param_cov = curve_fit(custom, np.array(x), np.array(y1), maxfev = 100000, bounds=([1, 0, 1, 1, 30, 1, 1], [2, 60, 8, 200, 90, 8, 1400]))

# np.array([1,2,3,4])
# gmodel = Model(custom)

# print('parameter names: {}'.format(gmodel.param_names))

# print('independent variables: {}'.format(gmodel.independent_vars))

# params = gmodel.make_params()

# result = gmodel.fit(y1, params, x=x)
param
index
y1_pred = test(x,param[0], param[1], param[2])

# y1_pred = test_linear(x,param[0], param[1], param[2], param[3], param[4], param[5])

# y1_pred = custom(x,param[0], param[1], param[2], param[3], param[4], param[5], param[6])

base_x = range(61,100,1)

# y1_pred_test = custom(base_x, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

y1_pred_test = test(base_x, param[0], param[1], param[2])

# y1_pred_test = test_linear(base_x, param[0], param[1], param[2], param[3], param[4], param[5])
# x axis values 

# x = record['Date']

# corresponding y axis values 

# y1 = record['ConfirmedCases']

# y2 = record['Fatalities']

# plotting the points  

plt.plot(x, y1, label = "Confirmed Cases") 

plt.plot(x, y1_pred, label = "Predicted") 

plt.plot(base_x, y1_pred_test, label = "Predicted") 

# naming the x axis 

plt.xlabel('Date') 

# naming the y axis 

plt.ylabel('Label') 

  

# giving a title to my graph 

plt.title(str(min_date_sorted.index[index][0]) + " " + str(min_date_sorted.index[index][1]) + ' - Confirmed Cases') 

  

# function to show the plot 

plt.show() 
test([-30],param[0], param[1], param[2])
np.diff(y1)
y2 = record2['Fatalities'].values

print(y2)

print(len(y2))
x2
(x2[-1] - x2[0])/2 + x2[0]
y2[-1]


# param2, param_cov2 = curve_fit(test, np.array(x), np.array(y2), [6, 0.5, 10], bounds=([6,0.4,0],[100,0.8,150]))

param2, param_cov2 = curve_fit(test, np.array(x2), np.array(y2), [y2[-1]/2, 0.5, (x2[-1] - x2[0])/2 + x2[0] -3], bounds=([y2[-1]/2, 0.2, 0],[y2[-1] + 1, 0.8, 150]))
param2
y2_pred = test(x2,param2[0], param2[1], param2[2])
# x axis values 

# x = record['Date']

# corresponding y axis values 

# y1 = record['ConfirmedCases']

# y2 = record2['Fatalities']

  

# plotting the points  

plt.plot(x2, y2, label = "Confirmed Cases") 

plt.plot(x2, y2_pred, label = "Fatalities") 

# naming the x axis 

plt.xlabel('Date') 

# naming the y axis 

plt.ylabel('Label') 

  

# giving a title to my graph 

plt.title(str(min_date_sorted.index[index][0]) + " " + str(min_date_sorted.index[index][1]) + ' - Fatalities') 

  

# function to show the plot 

plt.show() 
print(x2)

print(y2_pred)

print(y2)
record
base_x = range(61,100,1)

print(len(base_x))

base_y1 = test(base_x,param[0], param[1], param[2])

base_y2 = test(base_x,param2[0], param2[1], param2[2])
base_y1
base_y2
x
day_index_pred = 0

diff1_list = []

diff2_list = []

for day in base_x:

    if day in x:

        day_index = np.where(x == day)

        diff1 = y1[day_index] - base_y1[day_index_pred]

        diff1_list.append(diff1)

    if day in x2:

        day_index = np.where(x2 == day)

        diff2 = y2[day_index] - base_y2[day_index_pred]

        diff2_list.append(diff2)

    day_index_pred = day_index_pred + 1



diff1_mean = np.max(diff1_list)

diff2_mean = np.max(diff2_list)



#     print('diff1_mean', diff1_mean)

#     print('diff2_mean', diff2_mean)

if np.isnan(diff1_mean):

    pass

else:

    base_y1_mod = list(np.array(base_y1) + diff1_mean)

if np.isnan(diff2_mean):

    pass

else:

    base_y2_mod = list(np.array(base_y2) + diff2_mean)



base_y1_pred = [int(n) for n in base_y1_mod]

base_y2_pred = [int(m) for m in base_y2_mod]
print(list(base_x))

print(base_y1)

print(base_y1_mod)

print(base_y1_pred)

print(diff1_list)

print(diff1_mean)
print(list(base_x))

print(base_y2)

print(base_y2_mod)

print(base_y2_pred)

print(diff2_list)

print(diff2_mean)
# index = 0

# for key_,_ in zip(min_date_sorted.index, min_date_sorted):

    

#     record = df_train[(df_train['Country_Region'] == key_[0]) & (df_train['Province_State'] == key_[1])]

#     record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in record['Date']]

#     # x axis values 

#     x = record['days']

#     # corresponding y axis values 

#     y1 = record['ConfirmedCases']

#     y2 = record['Fatalities']

#     y1_prime = np.diff(y1)

#     print(index, key_)

#     print(y1_prime)

#     print('**************************************************')

#     plt.plot(x[1:], y1_prime, label = "Daily Increment") 

#     # plt.plot(x, y2, label = "Fatalities") 

#     # naming the x axis 

#     plt.xlabel('Date') 

#     # naming the y axis 

#     plt.ylabel('Label') 



#     # giving a title to my graph 

#     plt.title(str(key_[0]) + " " + str(key_[1]) + ' - Daily Increment') 



#     # function to show the plot 

#     plt.show()

#     index = index + 1

    
df_test
test_groups = df_test.groupby(['Country_Region', 'Province_State'])
group = test_groups.get_group(("Afghanistan",""))

print(group['Date'].values)

print(group['ForecastId'].values)
index = 0

for key_,_ in zip(min_date_sorted.index, min_date_sorted):

    

    record = df_train[(df_train['Country_Region'] == key_[0]) & (df_train['Province_State'] == key_[1])]

    record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in record['Date']]

    # x axis values 

    x = record['days']

    # corresponding y axis values 

    y1 = record['ConfirmedCases']

    y2 = record['Fatalities']

    y1_prime = np.diff(y1)

    

#     print(y1_prime)

    stage0 = False

    stage1 = False

    stage2 = False

    stage3 = False

    count1 = 0

    count2 = 0

    for start in range(len(y1_prime)-3):

        if sum(y1_prime[start:start+3]) <=12:

            count1 = count1 + 1

            count2 = 0

        else:

            count2 = count2 + 1

            count1 = 0

        if not stage0 and count2 == 0 and count1 > 2:

            stage0 = True

            count1 = 0

        if not stage1 and count1 == 0 and count2 > 5:

            stage0 = True

            stage1 = True

            count2 = 0

        if stage1 and count2 == 0 and count1 > 3:

            stage2 = True

            count1 = 0

        if stage2 and count1 == 0 and count2 > 2:

            stage3 = True

            count2 = 0

    if stage3:

        print(index, key_)

        print(y1_prime)

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

    index = index + 1
# Get average fatality rate with respect to confirmed cases

total_confirmed = 0

total_fatalities = 0

rate = []

max_y1 = []

max_y2 = []

details = []

for index, start_date in zip(min_date_sorted.index, min_date_sorted):

    print(index, start_date)

#     print(list(min_date_sorted.index).index(index))

    record = df_train[(df_train['Country_Region'] == index[0]) & (df_train['Province_State'] == index[1])]

    if len(record[record['ConfirmedCases'] > 0]) != 0:

        record = record[record['ConfirmedCases'] > 0]

    record2 = record

    if len(record[record['Fatalities'] > 0]) != 0:

        record2 = record[record['Fatalities'] > 0]

    # corresponding y axis values 

    

    y1 = record['ConfirmedCases'].values

    y2 = record2['Fatalities'].values

    

    b = -1

    bad_index = 0

    mod_count = 0

    y1_copy = list(y1)

    for a in y1:

        if a < b:

            y1[bad_index] = b

            mod_count = mod_count + 1

        else:

            b = a

        bad_index = bad_index + 1

#     if mod_count > 0:

#         print("*****************")

#         print(list(min_date_sorted.index).index(index), index)

#         print(mod_count)

#         print(y1)

#         print(y1_copy)

#         print("*****************")

    b = -1

    bad_index = 0

    mod_count = 0

    y2_copy = list(y2)

    for a in y2:

        if a < b:

            y2[bad_index] = b

            mod_count = mod_count + 1

        else:

            b = a

        bad_index = bad_index + 1

#     if mod_count > 0:

#         print("*****************")

#         print(list(min_date_sorted.index).index(index), index)

#         print(mod_count)

#         print(y2)

#         print(y2_copy)

#         print("*****************")

    

    

    y1_prime = np.diff(y1)

    y1_prime2 = np.diff(y1_prime)

#     print(y1)

#     print("-------------------------------")

#     print(y1_prime)

#     print("-------------------------------")

#     print(y1_prime2)

#     print("*******************************")

    y_max_ = y1[-1]*2 + 1500

    

    if len(y1)>0 and len(y1_prime) > 0 and len(y1_prime2) > 0:

        max_slope_index = len(y1_prime) - 1 - list(y1_prime)[::-1].index(max(y1_prime))

        max_slope_range = (max_slope_index+1)/len(y1_prime)

        if max_slope_range < 0.75:

            if y1_prime[max_slope_index] > 0 and max_slope_range < 0.5 and (((y1_prime[max_slope_index] - max(y1_prime[-2:]))/y1_prime[max_slope_index]) < 0.5):

                y_max_ = y1[-1]*2 + 1500

    #             print("*******************************")

    #             print(list(min_date_sorted.index).index(index), index)

    #             print(max_slope_index + 1, len(y1_prime))

    #             print(max_slope_range, "Max slope range")

    #             print(y1)

    #             print("-------------------------------")

    #             print(y1_prime)

    #             print("-------------------------------")

    #             print(y1_prime2)

    #             print("*******************************")

                pass

            else:

                y_max_ = y1[max_slope_index + 1]*2 + 1500

    #             print("*******************************")

    #             print(index)

    #             print(max_slope_index + 1, len(y1_prime))

    #             print(max_slope_range, "Max slope range")

    #             print(y1)

    #             print("-------------------------------")

    #             print(y1_prime)

    #             print("-------------------------------")

    #             print(y1_prime2)

    #             print("*******************************")

                pass

        else:

            y_max_ = y1[-1]*2 + 1500



    #     if(index[1] == 'California'):

    #         print("*******************************")

    #         print(index)

    #         print(max_slope_index + 1, len(y1_prime))

    #         print(max_slope_range, "Max slope range")

    #         print(y1)

    #         print("-------------------------------")

    #         print(y1_prime)

    #         print("-------------------------------")

    #         print(y1_prime2)

    #         print("*******************************")

    ratio = 0

    if y2[-1] > 0:

        ratio = y1[-1]/y2[-1]

    else:

        ratio = y1[-1]

    max_y1.append(y1[-1])

    max_y2.append(y2[-1])

    rate.append(ratio)

    details.append(" ".join([str(x) for x in [y1[-1], " ------- ", y2[-1], " ---- ", ratio, " --------------- ", record['Date'].values[-1], " ---- ", index, "----", list(min_date_sorted.index).index(index)]]))

#     print(y1[-1], " ------- ", y2[-1], " ---- ", ratio, " --------------- ", record['Date'].values[-1], " ---- ", index, "----", list(min_date_sorted.index).index(index))

    total_confirmed = total_confirmed + y1[-1]

    total_fatalities = total_fatalities + y2[-1]



print(total_confirmed/total_fatalities)
for a1, a2, b, c in zip(max_y1, max_y2, rate, details):

    print(c)
print(pd.DataFrame(rate).describe())

print('Median ', np.median(rate))

print('Mean ', np.mean(rate))

print('Average', np.average(rate, weights = max_y2))

avg = np.average(rate, weights = max_y2)
for a1, a2, b, c in zip(max_y1, max_y2, rate, details):

    if(a1 < 100 and a2 < 4 and b < avg):

        # correct the y2_pred as per average rate = 22

        print(c)

        pass

    else:

        ## correct the y2_pred as per the rate

#         print(c)

        pass
list(min_date_sorted.index)
# mapping = "0 ## (\'Afghanistan\', \'\') ## 2$$1 ## (\'Saudi Arabia\', \'\') ## 2$$2 ## (\'San Marino\', \'\') ## 1.1$$3 ## (\'Saint Vincent and the Grenadines\', \'\') ## 1$$4 ## (\'Saint Lucia\', \'\') ## 1$$5 ## (\'Saint Kitts and Nevis\', \'\') ## 1$$6 ## (\'Rwanda\', \'\') ## 1.2$$7 ## (\'Russia\', \'\') ## 4$$8 ## (\'Romania\', \'\') ## 2$$9 ## (\'Qatar\', \'\') ## 3$$10 ## (\'Portugal\', \'\') ## 2$$11 ## (\'Poland\', \'\') ## 4$$12 ## (\'Philippines\', \'\') ## 2$$13 ## (\'Peru\', \'\') ## 4$$14 ## (\'Paraguay\', \'\') ## 2$$15 ## (\'Papua New Guinea\', \'\') ## 1$$16 ## (\'Senegal\', \'\') ## 2$$17 ## (\'Panama\', \'\') ## 2$$18 ## (\'Serbia\', \'\') ## 4$$19 ## (\'Sierra Leone\', \'\') ## 2$$20 ## (\'Thailand\', \'\') ## 1.2$$21 ## (\'Tanzania\', \'\') ## 1$$22 ## (\'Taiwan*\', \'\') ## 1$$23 ## (\'Syria\', \'\') ## 1$$24 ## (\'Switzerland\', \'\') ## 1.5$$25 ## (\'Sweden\', \'\') ## 3$$26 ## (\'Suriname\', \'\') ## 1$$27 ## (\'Sudan\', \'\') ## 1$$28 ## (\'Sri Lanka\', \'\') ## 1$$29 ## (\'Spain\', \'\') ## 1.2$$30 ## (\'South Africa\', \'\') ## 1$$31 ## (\'Somalia\', \'\') ## 1$$32 ## (\'Slovenia\', \'\') ## 1$$33 ## (\'Slovakia\', \'\') ## 2$$34 ## (\'Singapore\', \'\') ## 3$$35 ## (\'Seychelles\', \'\') ## 1$$36 ## (\'Timor-Leste\', \'\') ## 1$$37 ## (\'Pakistan\', \'\') ## 4$$38 ## (\'Norway\', \'\') ## 2$$39 ## (\'Mauritania\', \'\') ## 1$$40 ## (\'Malta\', \'\') ## 4$$41 ## (\'Mali\', \'\') ## 4$$42 ## (\'Maldives\', \'\') ## 1$$43 ## (\'Malaysia\', \'\') ## 2$$44 ## (\'Madagascar\', \'\') ## 2$$45 ## (\'MS Zaandam\', \'\') ## 1$$46 ## (\'Luxembourg\', \'\') ## 1.5$$47 ## (\'Lithuania\', \'\') ## 1$$48 ## (\'Liechtenstein\', \'\') ## 1$$49 ## (\'Libya\', \'\') ## 1$$50 ## (\'Liberia\', \'\') ## 2$$51 ## (\'Lebanon\', \'\') ## 1.1$$52 ## (\'Latvia\', \'\') ## 1.1$$53 ## (\'Laos\', \'\') ## 1$$54 ## (\'Mauritius\', \'\') ## 1.5$$55 ## (\'Oman\', \'\') ## 4$$56 ## (\'Mexico\', \'\') ## 4$$57 ## (\'Monaco\', \'\') ## 1.5$$58 ## (\'North Macedonia\', \'\') ## 2$$59 ## (\'Nigeria\', \'\') ## 1.5$$60 ## (\'Niger\', \'\') ## 2$$61 ## (\'Nicaragua\', \'\') ## 1$$62 ## (\'New Zealand\', \'\') ## 1.5$$63 ## (\'Netherlands\', \'Sint Maarten\') ## 3$$64 ## (\'Netherlands\', \'Curacao\') ## 3$$65 ## (\'Netherlands\', \'Aruba\') ## 2$$66 ## (\'Netherlands\', \'\') ## 1.5$$67 ## (\'Nepal\', \'\') ## 1$$68 ## (\'Namibia\', \'\') ## 1$$69 ## (\'Mozambique\', \'\') ## 1$$70 ## (\'Morocco\', \'\') ## 1.2$$71 ## (\'Montenegro\', \'\') ## 1.2$$72 ## (\'Mongolia\', \'\') ## 1$$73 ## (\'Moldova\', \'\') ## 2$$74 ## (\'Togo\', \'\') ## 3$$75 ## (\'Trinidad and Tobago\', \'\') ## 1$$76 ## (\'Tunisia\', \'\') ## 1.1$$77 ## (\'US\', \'Wisconsin\') ## 3$$78 ## (\'US\', \'West Virginia\') ## 3$$79 ## (\'US\', \'Washington\') ## 2$$80 ## (\'US\', \'Virginia\') ## 2$$81 ## (\'US\', \'Virgin Islands\') ## 1$$82 ## (\'US\', \'Vermont\') ## 2$$83 ## (\'US\', \'Utah\') ## 2$$84 ## (\'US\', \'Texas\') ## 2$$85 ## (\'US\', \'Tennessee\') ## 2$$86 ## (\'US\', \'South Dakota\') ## 2$$87 ## (\'US\', \'South Carolina\') ## 2$$88 ## (\'US\', \'Rhode Island\') ## 2$$89 ## (\'US\', \'Puerto Rico\') ## 2$$90 ## (\'US\', \'Pennsylvania\') ## 2$$91 ## (\'US\', \'Oregon\') ## 2$$92 ## (\'US\', \'Wyoming\') ## 2$$93 ## (\'US\', \'Oklahoma\') ## 2$$94 ## (\'Uganda\', \'\') ## 1$$95 ## (\'United Arab Emirates\', \'\') ## 2$$96 ## (\'West Bank and Gaza\', \'\') ## 2$$97 ## (\'Vietnam\', \'\') ## 1.2$$98 ## (\'Venezuela\', \'\') ## 1.2$$99 ## (\'Uzbekistan\', \'\') ## 3$$100 ## (\'Uruguay\', \'\') ## 1.5$$101 ## (\'United Kingdom\', \'Turks and Caicos Islands\') ## 1$$102 ## (\'United Kingdom\', \'Montserrat\') ## 1$$103 ## (\'United Kingdom\', \'Isle of Man\') ## 1.5$$104 ## (\'United Kingdom\', \'Gibraltar\') ## 1.2$$105 ## (\'United Kingdom\', \'Channel Islands\') ## 1.5$$106 ## (\'United Kingdom\', \'Cayman Islands\') ## 2$$107 ## (\'United Kingdom\', \'British Virgin Islands\') ## 1$$108 ## (\'United Kingdom\', \'Bermuda\') ## 1$$109 ## (\'United Kingdom\', \'Anguilla\') ## 1$$110 ## (\'United Kingdom\', \'\') ## 2$$111 ## (\'Ukraine\', \'\') ## 2$$112 ## (\'US\', \'Ohio\') ## 2$$113 ## (\'US\', \'North Dakota\') ## 2$$114 ## (\'US\', \'North Carolina\') ## 2$$115 ## (\'US\', \'Idaho\') ## 2$$116 ## (\'US\', \'Hawaii\') ## 2$$117 ## (\'US\', \'Guam\') ## 2$$118 ## (\'US\', \'Georgia\') ## 2$$119 ## (\'US\', \'Florida\') ## 2$$120 ## (\'US\', \'District of Columbia\') ## 2$$121 ## (\'US\', \'Delaware\') ## 2$$122 ## (\'US\', \'Connecticut\') ## 2$$123 ## (\'US\', \'Colorado\') ## 2$$124 ## (\'US\', \'California\') ## 2$$125 ## (\'US\', \'Arkansas\') ## 2$$126 ## (\'US\', \'Arizona\') ## 2$$127 ## (\'US\', \'Alaska\') ## 2$$128 ## (\'US\', \'Alabama\') ## 2$$129 ## (\'Turkey\', \'\') ## 4$$130 ## (\'US\', \'Illinois\') ## 2$$131 ## (\'US\', \'Indiana\') ## 2$$132 ## (\'US\', \'Iowa\') ## 2$$133 ## (\'US\', \'Kansas\') ## 2$$134 ## (\'US\', \'New York\') ## 2$$135 ## (\'US\', \'New Mexico\') ## 2$$136 ## (\'US\', \'New Jersey\') ## 2$$137 ## (\'US\', \'New Hampshire\') ## 2$$138 ## (\'US\', \'Nevada\') ## 2$$139 ## (\'US\', \'Nebraska\') ## 2$$140 ## (\'US\', \'Montana\') ## 2$$141 ## (\'Kyrgyzstan\', \'\') ## 2$$142 ## (\'US\', \'Missouri\') ## 2$$143 ## (\'US\', \'Minnesota\') ## 2$$144 ## (\'US\', \'Michigan\') ## 2$$145 ## (\'US\', \'Massachusetts\') ## 2$$146 ## (\'US\', \'Maryland\') ## 2$$147 ## (\'US\', \'Maine\') ## 2$$148 ## (\'US\', \'Louisiana\') ## 2$$149 ## (\'US\', \'Kentucky\') ## 2$$150 ## (\'US\', \'Mississippi\') ## 2$$151 ## (\'Kuwait\', \'\') ## 2$$152 ## (\'Kosovo\', \'\') ## 2$$153 ## (\'Korea, South\', \'\') ## 1.1$$154 ## (\'China\', \'Anhui\') ## 1$$155 ## (\'Chile\', \'\') ## 2$$156 ## (\'Chad\', \'\') ## 1$$157 ## (\'Central African Republic\', \'\') ## 1$$158 ## (\'Canada\', \'Yukon\') ## 1$$159 ## (\'Canada\', \'Saskatchewan\') ## 1.5$$160 ## (\'Canada\', \'Quebec\') ## 2$$161 ## (\'Canada\', \'Prince Edward Island\') ## 2$$162 ## (\'Canada\', \'Ontario\') ## 2$$163 ## (\'Canada\', \'Nova Scotia\') ## 2$$164 ## (\'Canada\', \'Northwest Territories\') ## 1$$165 ## (\'Canada\', \'Newfoundland and Labrador\') ## 1$$166 ## (\'Canada\', \'New Brunswick\') ## 1$$167 ## (\'Canada\', \'Manitoba\') ## 1.5$$168 ## (\'Canada\', \'British Columbia\') ## 1.5$$169 ## (\'China\', \'Beijing\') ## 1.1$$170 ## (\'Canada\', \'Alberta\') ## 2$$171 ## (\'China\', \'Chongqing\') ## 1$$172 ## (\'China\', \'Gansu\') ## 1.1$$173 ## (\'China\', \'Liaoning\') ## 1.1$$174 ## (\'China\', \'Jilin\') ## 1$$175 ## (\'China\', \'Jiangxi\') ## 1$$176 ## (\'China\', \'Jiangsu\') ## 1$$177 ## (\'China\', \'Inner Mongolia\') ## 1.1$$178 ## (\'China\', \'Hunan\') ## 1$$179 ## (\'China\', \'Hubei\') ## 1$$180 ## (\'China\', \'Hong Kong\') ## 1.2$$181 ## (\'China\', \'Henan\') ## 1$$182 ## (\'China\', \'Heilongjiang\') ## 1.1$$183 ## (\'China\', \'Hebei\') ## 1$$184 ## (\'China\', \'Hainan\') ## 1$$185 ## (\'China\', \'Guizhou\') ## 1$$186 ## (\'China\', \'Guangxi\') ## 1$$187 ## (\'China\', \'Guangdong\') ## 1$$188 ## (\'China\', \'Fujian\') ## 1.1$$189 ## (\'Cameroon\', \'\') ## 1$$190 ## (\'Cambodia\', \'\') ## 1$$191 ## (\'Cabo Verde\', \'\') ## 1$$192 ## (\'Australia\', \'Western Australia\') ## 1$$193 ## (\'Australia\', \'Victoria\') ## 1.1$$194 ## (\'Australia\', \'Tasmania\') ## 1.1$$195 ## (\'Australia\', \'South Australia\') ## 1.1$$196 ## (\'Australia\', \'Queensland\') ## 1.1$$197 ## (\'Australia\', \'Northern Territory\') ## 1.1$$198 ## (\'Australia\', \'New South Wales\') ## 1.1$$199 ## (\'Australia\', \'Australian Capital Territory\') ## 1.1$$200 ## (\'Armenia\', \'\') ## 2$$201 ## (\'Argentina\', \'\') ## 1.5$$202 ## (\'Antigua and Barbuda\', \'\') ## 2$$203 ## (\'Angola\', \'\') ## 2$$204 ## (\'Andorra\', \'\') ## 1.5$$205 ## (\'Algeria\', \'\') ## 1.5$$206 ## (\'Albania\', \'\') ## 1.5$$207 ## (\'Austria\', \'\') ## 1.5$$208 ## (\'Azerbaijan\', \'\') ## 3$$209 ## (\'Bahamas\', \'\') ## 2$$210 ## (\'Bahrain\', \'\') ## 2$$211 ## (\'Burundi\', \'\') ## 1$$212 ## (\'Burma\', \'\') ## 1$$213 ## (\'Burkina Faso\', \'\') ## 1.5$$214 ## (\'Bulgaria\', \'\') ## 1.5$$215 ## (\'Brunei\', \'\') ## 1$$216 ## (\'Brazil\', \'\') ## 4$$217 ## (\'Botswana\', \'\') ## 1$$218 ## (\'China\', \'Macau\') ## 1.1$$219 ## (\'Bosnia and Herzegovina\', \'\') ## 3$$220 ## (\'Bhutan\', \'\') ## 1$$221 ## (\'Benin\', \'\') ## 1$$222 ## (\'Belize\', \'\') ## 1$$223 ## (\'Belgium\', \'\') ## 1.5$$224 ## (\'Belarus\', \'\') ## 2$$225 ## (\'Barbados\', \'\') ## 1.5$$226 ## (\'Bangladesh\', \'\') ## 4$$227 ## (\'Bolivia\', \'\') ## 2$$228 ## (\'Zambia\', \'\') ## 1$$229 ## (\'China\', \'Ningxia\') ## 1$$230 ## (\'China\', \'Shaanxi\') ## 1$$231 ## (\'Guinea\', \'\') ## 1.5$$232 ## (\'Guatemala\', \'\') ## 2$$233 ## (\'Grenada\', \'\') ## 1$$234 ## (\'Greece\', \'\') ## 1.5$$235 ## (\'Ghana\', \'\') ## 4$$236 ## (\'Germany\', \'\') ## 1.5$$237 ## (\'Georgia\', \'\') ## 1.5$$238 ## (\'Gambia\', \'\') ## 2$$239 ## (\'Gabon\', \'\') ## 4$$240 ## (\'France\', \'St Martin\') ## 1$$241 ## (\'France\', \'Saint Barthelemy\') ## 1$$242 ## (\'France\', \'Reunion\') ## 1.1$$243 ## (\'France\', \'New Caledonia\') ## 1.1$$244 ## (\'France\', \'Mayotte\') ## 1.5$$245 ## (\'France\', \'Martinique\') ## 1.1$$246 ## (\'Guinea-Bissau\', \'\') ## 4$$247 ## (\'France\', \'Guadeloupe\') ## 1.1$$248 ## (\'Guyana\', \'\') ## 1$$249 ## (\'Holy See\', \'\') ## 1$$250 ## (\'Kenya\', \'\') ## 1.5$$251 ## (\'Kazakhstan\', \'\') ## 2$$252 ## (\'Jordan\', \'\') ## 1.2$$253 ## (\'Japan\', \'\') ## 2$$254 ## (\'Jamaica\', \'\') ## 2$$255 ## (\'Italy\', \'\') ## 1.2$$256 ## (\'Israel\', \'\') ## 2$$257 ## (\'Ireland\', \'\') ## 2$$258 ## (\'Iraq\', \'\') ## 3$$259 ## (\'Iran\', \'\') ## 1.5$$260 ## (\'Indonesia\', \'\') ## 3$$261 ## (\'India\', \'\') ## 3$$262 ## (\'Iceland\', \'\') ## 1.2$$263 ## (\'Hungary\', \'\') ## 2$$264 ## (\'Honduras\', \'\') ## 1$$265 ## (\'Haiti\', \'\') ## 1$$266 ## (\'France\', \'French Polynesia\') ## 1.2$$267 ## (\'France\', \'French Guiana\') ## 1.5$$268 ## (\'France\', \'\') ## 1.5$$269 ## (\'Croatia\', \'\') ## 1.5$$270 ## (\"Cote d\'Ivoire\", \'\') ## 1.5$$271 ## (\'Costa Rica\', \'\') ## 1.5$$272 ## (\'Congo (Kinshasa)\', \'\') ## 1.5$$273 ## (\'Congo (Brazzaville)\', \'\') ## 1.5$$274 ## (\'Colombia\', \'\') ## 4$$275 ## (\'China\', \'Zhejiang\') ## 1$$276 ## (\'China\', \'Yunnan\') ## 1$$277 ## (\'China\', \'Xinjiang\') ## 1$$278 ## (\'China\', \'Tibet\') ## 1$$279 ## (\'China\', \'Tianjin\') ## 1.1$$280 ## (\'China\', \'Sichuan\') ## 1$$281 ## (\'China\', \'Shanxi\') ## 1$$282 ## (\'China\', \'Shanghai\') ## 1.1$$283 ## (\'China\', \'Shandong\') ## 1$$284 ## (\'Cuba\', \'\') ## 3$$285 ## (\'Cyprus\', \'\') ## 1.5$$286 ## (\'Czechia\', \'\') ## 1.5$$287 ## (\'Denmark\', \'\') ## 2$$288 ## (\'Finland\', \'\') ## 2$$289 ## (\'Fiji\', \'\') ## 2$$290 ## (\'Ethiopia\', \'\') ## 2$$291 ## (\'Eswatini\', \'\') ## 1$$292 ## (\'Estonia\', \'\') ## 1.1$$293 ## (\'Eritrea\', \'\') ## 1$$294 ## (\'Equatorial Guinea\', \'\') ## 1$$295 ## (\'China\', \'Qinghai\') ## 1$$296 ## (\'El Salvador\', \'\') ## 3$$297 ## (\'Ecuador\', \'\') ## 1.1$$298 ## (\'Dominican Republic\', \'\') ## 1.5$$299 ## (\'Dominica\', \'\') ## 1$$300 ## (\'Djibouti\', \'\') ## 3$$301 ## (\'Diamond Princess\', \'\') ## 1$$302 ## (\'Denmark\', \'Greenland\') ## 1$$303 ## (\'Denmark\', \'Faroe Islands\') ## 1$$304 ## (\'Egypt\', \'\') ## 2$$305 ## (\'Zimbabwe\', \'\') ## 1"

# mapped_list = mapping.split("$$")

# mapped_list = [mapped_row.split("##") for mapped_row in mapped_list]

# mapped_list = [(a[1].strip(), float(a[2].strip())) for a in mapped_list]

# mapped_dict = {}

# for pair in mapped_list:

#     fields = pair[0][1:-1].split(', \'')

#     mapped_dict.update({(fields[0].strip()[1:-1], fields[1].strip()[:-1]): pair[1]})



# rate_1 = []

# rate_1_1 = []

# rate_1_2 = []

# rate_1_5 = []

# rate_2 = []

# rate_3 = []

# rate_4 =[]

# rates = [1,1.1,1.2,1.5,2,3,4]

# indices = [rate_1, rate_1_1, rate_1_2, rate_1_5, rate_2, rate_3, rate_4]



# hr_id = list(min_date_sorted.index)

# for index, pair in enumerate(min_date_sorted.index):

#     if pair in mapped_dict.keys():

#         indices[rates.index(mapped_dict[pair])].append(index)

#     else:

#         print(index, pair)

#         indices[rates.index(2)].append(index)



# print(rate_1)

# print(rate_1_1)

# print(rate_1_2)

# print(rate_1_5)

# print(rate_2)

# print(rate_3)

# print(rate_4)
# mapping = "0 ## (\'Afghanistan\', \'\') ## 2$$1 ## (\'Saudi Arabia\', \'\') ## 2$$2 ## (\'San Marino\', \'\') ## 1.1$$3 ## (\'Saint Vincent and the Grenadines\', \'\') ## 1$$4 ## (\'Saint Lucia\', \'\') ## 1$$5 ## (\'Saint Kitts and Nevis\', \'\') ## 1$$6 ## (\'Rwanda\', \'\') ## 1.2$$7 ## (\'Russia\', \'\') ## 4$$8 ## (\'Romania\', \'\') ## 2$$9 ## (\'Qatar\', \'\') ## 3$$10 ## (\'Portugal\', \'\') ## 2$$11 ## (\'Poland\', \'\') ## 4$$12 ## (\'Philippines\', \'\') ## 2$$13 ## (\'Peru\', \'\') ## 4$$14 ## (\'Paraguay\', \'\') ## 2$$15 ## (\'Papua New Guinea\', \'\') ## 1$$16 ## (\'Senegal\', \'\') ## 2$$17 ## (\'Panama\', \'\') ## 2$$18 ## (\'Serbia\', \'\') ## 4$$19 ## (\'Sierra Leone\', \'\') ## 2$$20 ## (\'Thailand\', \'\') ## 1.2$$21 ## (\'Tanzania\', \'\') ## 1$$22 ## (\'Taiwan*\', \'\') ## 1$$23 ## (\'Syria\', \'\') ## 1$$24 ## (\'Switzerland\', \'\') ## 1.5$$25 ## (\'Sweden\', \'\') ## 3$$26 ## (\'Suriname\', \'\') ## 1$$27 ## (\'Sudan\', \'\') ## 1$$28 ## (\'Sri Lanka\', \'\') ## 1$$29 ## (\'Spain\', \'\') ## 1.2$$30 ## (\'South Africa\', \'\') ## 1$$31 ## (\'Somalia\', \'\') ## 1$$32 ## (\'Slovenia\', \'\') ## 1$$33 ## (\'Slovakia\', \'\') ## 2$$34 ## (\'Singapore\', \'\') ## 3$$35 ## (\'Seychelles\', \'\') ## 1$$36 ## (\'Timor-Leste\', \'\') ## 1$$37 ## (\'Pakistan\', \'\') ## 4$$38 ## (\'Norway\', \'\') ## 2$$39 ## (\'Mauritania\', \'\') ## 1$$40 ## (\'Malta\', \'\') ## 4$$41 ## (\'Mali\', \'\') ## 4$$42 ## (\'Maldives\', \'\') ## 1$$43 ## (\'Malaysia\', \'\') ## 2$$44 ## (\'Madagascar\', \'\') ## 2$$45 ## (\'MS Zaandam\', \'\') ## 1$$46 ## (\'Luxembourg\', \'\') ## 1.5$$47 ## (\'Lithuania\', \'\') ## 1$$48 ## (\'Liechtenstein\', \'\') ## 1$$49 ## (\'Libya\', \'\') ## 1$$50 ## (\'Liberia\', \'\') ## 2$$51 ## (\'Lebanon\', \'\') ## 1.1$$52 ## (\'Latvia\', \'\') ## 1.1$$53 ## (\'Laos\', \'\') ## 1$$54 ## (\'Mauritius\', \'\') ## 1.5$$55 ## (\'Oman\', \'\') ## 4$$56 ## (\'Mexico\', \'\') ## 4$$57 ## (\'Monaco\', \'\') ## 1.5$$58 ## (\'North Macedonia\', \'\') ## 2$$59 ## (\'Nigeria\', \'\') ## 1.5$$60 ## (\'Niger\', \'\') ## 2$$61 ## (\'Nicaragua\', \'\') ## 1$$62 ## (\'New Zealand\', \'\') ## 1.5$$63 ## (\'Netherlands\', \'Sint Maarten\') ## 3$$64 ## (\'Netherlands\', \'Curacao\') ## 3$$65 ## (\'Netherlands\', \'Aruba\') ## 2$$66 ## (\'Netherlands\', \'\') ## 1.5$$67 ## (\'Nepal\', \'\') ## 1$$68 ## (\'Namibia\', \'\') ## 1$$69 ## (\'Mozambique\', \'\') ## 1$$70 ## (\'Morocco\', \'\') ## 1.2$$71 ## (\'Montenegro\', \'\') ## 1.2$$72 ## (\'Mongolia\', \'\') ## 1$$73 ## (\'Moldova\', \'\') ## 2$$74 ## (\'Togo\', \'\') ## 3$$75 ## (\'Trinidad and Tobago\', \'\') ## 1$$76 ## (\'Tunisia\', \'\') ## 1.1$$77 ## (\'US\', \'Wisconsin\') ## 3$$78 ## (\'US\', \'West Virginia\') ## 3$$79 ## (\'US\', \'Washington\') ## 2$$80 ## (\'US\', \'Virginia\') ## 2$$81 ## (\'US\', \'Virgin Islands\') ## 1$$82 ## (\'US\', \'Vermont\') ## 2$$83 ## (\'US\', \'Utah\') ## 2$$84 ## (\'US\', \'Texas\') ## 2$$85 ## (\'US\', \'Tennessee\') ## 2$$86 ## (\'US\', \'South Dakota\') ## 2$$87 ## (\'US\', \'South Carolina\') ## 2$$88 ## (\'US\', \'Rhode Island\') ## 2$$89 ## (\'US\', \'Puerto Rico\') ## 2$$90 ## (\'US\', \'Pennsylvania\') ## 2$$91 ## (\'US\', \'Oregon\') ## 2$$92 ## (\'US\', \'Wyoming\') ## 2$$93 ## (\'US\', \'Oklahoma\') ## 2$$94 ## (\'Uganda\', \'\') ## 1$$95 ## (\'United Arab Emirates\', \'\') ## 2$$96 ## (\'West Bank and Gaza\', \'\') ## 2$$97 ## (\'Vietnam\', \'\') ## 1.2$$98 ## (\'Venezuela\', \'\') ## 1.2$$99 ## (\'Uzbekistan\', \'\') ## 3$$100 ## (\'Uruguay\', \'\') ## 1.5$$101 ## (\'United Kingdom\', \'Turks and Caicos Islands\') ## 1$$102 ## (\'United Kingdom\', \'Montserrat\') ## 1$$103 ## (\'United Kingdom\', \'Isle of Man\') ## 1.5$$104 ## (\'United Kingdom\', \'Gibraltar\') ## 1.2$$105 ## (\'United Kingdom\', \'Channel Islands\') ## 1.5$$106 ## (\'United Kingdom\', \'Cayman Islands\') ## 2$$107 ## (\'United Kingdom\', \'British Virgin Islands\') ## 1$$108 ## (\'United Kingdom\', \'Bermuda\') ## 1$$109 ## (\'United Kingdom\', \'Anguilla\') ## 1$$110 ## (\'United Kingdom\', \'\') ## 2$$111 ## (\'Ukraine\', \'\') ## 2$$112 ## (\'US\', \'Ohio\') ## 2$$113 ## (\'US\', \'North Dakota\') ## 2$$114 ## (\'US\', \'North Carolina\') ## 2$$115 ## (\'US\', \'Idaho\') ## 2$$116 ## (\'US\', \'Hawaii\') ## 2$$117 ## (\'US\', \'Guam\') ## 2$$118 ## (\'US\', \'Georgia\') ## 2$$119 ## (\'US\', \'Florida\') ## 2$$120 ## (\'US\', \'District of Columbia\') ## 2$$121 ## (\'US\', \'Delaware\') ## 2$$122 ## (\'US\', \'Connecticut\') ## 2$$123 ## (\'US\', \'Colorado\') ## 2$$124 ## (\'US\', \'California\') ## 2$$125 ## (\'US\', \'Arkansas\') ## 2$$126 ## (\'US\', \'Arizona\') ## 2$$127 ## (\'US\', \'Alaska\') ## 2$$128 ## (\'US\', \'Alabama\') ## 2$$129 ## (\'Turkey\', \'\') ## 4$$130 ## (\'US\', \'Illinois\') ## 2$$131 ## (\'US\', \'Indiana\') ## 2$$132 ## (\'US\', \'Iowa\') ## 2$$133 ## (\'US\', \'Kansas\') ## 2$$134 ## (\'US\', \'New York\') ## 2$$135 ## (\'US\', \'New Mexico\') ## 2$$136 ## (\'US\', \'New Jersey\') ## 2$$137 ## (\'US\', \'New Hampshire\') ## 2$$138 ## (\'US\', \'Nevada\') ## 2$$139 ## (\'US\', \'Nebraska\') ## 2$$140 ## (\'US\', \'Montana\') ## 2$$141 ## (\'Kyrgyzstan\', \'\') ## 2$$142 ## (\'US\', \'Missouri\') ## 2$$143 ## (\'US\', \'Minnesota\') ## 2$$144 ## (\'US\', \'Michigan\') ## 2$$145 ## (\'US\', \'Massachusetts\') ## 2$$146 ## (\'US\', \'Maryland\') ## 2$$147 ## (\'US\', \'Maine\') ## 2$$148 ## (\'US\', \'Louisiana\') ## 2$$149 ## (\'US\', \'Kentucky\') ## 2$$150 ## (\'US\', \'Mississippi\') ## 2$$151 ## (\'Kuwait\', \'\') ## 2$$152 ## (\'Kosovo\', \'\') ## 2$$153 ## (\'Korea, South\', \'\') ## 1.1$$154 ## (\'China\', \'Anhui\') ## 1$$155 ## (\'Chile\', \'\') ## 2$$156 ## (\'Chad\', \'\') ## 1$$157 ## (\'Central African Republic\', \'\') ## 1$$158 ## (\'Canada\', \'Yukon\') ## 1$$159 ## (\'Canada\', \'Saskatchewan\') ## 1.5$$160 ## (\'Canada\', \'Quebec\') ## 2$$161 ## (\'Canada\', \'Prince Edward Island\') ## 2$$162 ## (\'Canada\', \'Ontario\') ## 2$$163 ## (\'Canada\', \'Nova Scotia\') ## 2$$164 ## (\'Canada\', \'Northwest Territories\') ## 1$$165 ## (\'Canada\', \'Newfoundland and Labrador\') ## 1$$166 ## (\'Canada\', \'New Brunswick\') ## 1$$167 ## (\'Canada\', \'Manitoba\') ## 1.5$$168 ## (\'Canada\', \'British Columbia\') ## 1.5$$169 ## (\'China\', \'Beijing\') ## 1.1$$170 ## (\'Canada\', \'Alberta\') ## 2$$171 ## (\'China\', \'Chongqing\') ## 1$$172 ## (\'China\', \'Gansu\') ## 1.1$$173 ## (\'China\', \'Liaoning\') ## 1.1$$174 ## (\'China\', \'Jilin\') ## 1$$175 ## (\'China\', \'Jiangxi\') ## 1$$176 ## (\'China\', \'Jiangsu\') ## 1$$177 ## (\'China\', \'Inner Mongolia\') ## 1.1$$178 ## (\'China\', \'Hunan\') ## 1$$179 ## (\'China\', \'Hubei\') ## 1$$180 ## (\'China\', \'Hong Kong\') ## 1.2$$181 ## (\'China\', \'Henan\') ## 1$$182 ## (\'China\', \'Heilongjiang\') ## 1.1$$183 ## (\'China\', \'Hebei\') ## 1$$184 ## (\'China\', \'Hainan\') ## 1$$185 ## (\'China\', \'Guizhou\') ## 1$$186 ## (\'China\', \'Guangxi\') ## 1$$187 ## (\'China\', \'Guangdong\') ## 1$$188 ## (\'China\', \'Fujian\') ## 1.1$$189 ## (\'Cameroon\', \'\') ## 1$$190 ## (\'Cambodia\', \'\') ## 1$$191 ## (\'Cabo Verde\', \'\') ## 1$$192 ## (\'Australia\', \'Western Australia\') ## 1$$193 ## (\'Australia\', \'Victoria\') ## 1.1$$194 ## (\'Australia\', \'Tasmania\') ## 1.1$$195 ## (\'Australia\', \'South Australia\') ## 1.1$$196 ## (\'Australia\', \'Queensland\') ## 1.1$$197 ## (\'Australia\', \'Northern Territory\') ## 1.1$$198 ## (\'Australia\', \'New South Wales\') ## 1.1$$199 ## (\'Australia\', \'Australian Capital Territory\') ## 1.1$$200 ## (\'Armenia\', \'\') ## 2$$201 ## (\'Argentina\', \'\') ## 1.5$$202 ## (\'Antigua and Barbuda\', \'\') ## 2$$203 ## (\'Angola\', \'\') ## 2$$204 ## (\'Andorra\', \'\') ## 1.5$$205 ## (\'Algeria\', \'\') ## 1.5$$206 ## (\'Albania\', \'\') ## 1.5$$207 ## (\'Austria\', \'\') ## 1.5$$208 ## (\'Azerbaijan\', \'\') ## 3$$209 ## (\'Bahamas\', \'\') ## 2$$210 ## (\'Bahrain\', \'\') ## 2$$211 ## (\'Burundi\', \'\') ## 1$$212 ## (\'Burma\', \'\') ## 1$$213 ## (\'Burkina Faso\', \'\') ## 1.5$$214 ## (\'Bulgaria\', \'\') ## 1.5$$215 ## (\'Brunei\', \'\') ## 1$$216 ## (\'Brazil\', \'\') ## 4$$217 ## (\'Botswana\', \'\') ## 1$$218 ## (\'China\', \'Macau\') ## 1.1$$219 ## (\'Bosnia and Herzegovina\', \'\') ## 3$$220 ## (\'Bhutan\', \'\') ## 1$$221 ## (\'Benin\', \'\') ## 1$$222 ## (\'Belize\', \'\') ## 1$$223 ## (\'Belgium\', \'\') ## 1.5$$224 ## (\'Belarus\', \'\') ## 2$$225 ## (\'Barbados\', \'\') ## 1.5$$226 ## (\'Bangladesh\', \'\') ## 4$$227 ## (\'Bolivia\', \'\') ## 2$$228 ## (\'Zambia\', \'\') ## 1$$229 ## (\'China\', \'Ningxia\') ## 1$$230 ## (\'China\', \'Shaanxi\') ## 1$$231 ## (\'Guinea\', \'\') ## 1.5$$232 ## (\'Guatemala\', \'\') ## 2$$233 ## (\'Grenada\', \'\') ## 1$$234 ## (\'Greece\', \'\') ## 1.5$$235 ## (\'Ghana\', \'\') ## 4$$236 ## (\'Germany\', \'\') ## 1.5$$237 ## (\'Georgia\', \'\') ## 1.5$$238 ## (\'Gambia\', \'\') ## 2$$239 ## (\'Gabon\', \'\') ## 4$$240 ## (\'France\', \'St Martin\') ## 1$$241 ## (\'France\', \'Saint Barthelemy\') ## 1$$242 ## (\'France\', \'Reunion\') ## 1.1$$243 ## (\'France\', \'New Caledonia\') ## 1.1$$244 ## (\'France\', \'Mayotte\') ## 1.5$$245 ## (\'France\', \'Martinique\') ## 1.1$$246 ## (\'Guinea-Bissau\', \'\') ## 4$$247 ## (\'France\', \'Guadeloupe\') ## 1.1$$248 ## (\'Guyana\', \'\') ## 1$$249 ## (\'Holy See\', \'\') ## 1$$250 ## (\'Kenya\', \'\') ## 1.5$$251 ## (\'Kazakhstan\', \'\') ## 2$$252 ## (\'Jordan\', \'\') ## 1.2$$253 ## (\'Japan\', \'\') ## 2$$254 ## (\'Jamaica\', \'\') ## 2$$255 ## (\'Italy\', \'\') ## 1.2$$256 ## (\'Israel\', \'\') ## 2$$257 ## (\'Ireland\', \'\') ## 2$$258 ## (\'Iraq\', \'\') ## 3$$259 ## (\'Iran\', \'\') ## 1.5$$260 ## (\'Indonesia\', \'\') ## 3$$261 ## (\'India\', \'\') ## 3$$262 ## (\'Iceland\', \'\') ## 1.2$$263 ## (\'Hungary\', \'\') ## 2$$264 ## (\'Honduras\', \'\') ## 1$$265 ## (\'Haiti\', \'\') ## 1$$266 ## (\'France\', \'French Polynesia\') ## 1.2$$267 ## (\'France\', \'French Guiana\') ## 1.5$$268 ## (\'France\', \'\') ## 1.5$$269 ## (\'Croatia\', \'\') ## 1.5$$270 ## (\"Cote d\'Ivoire\", \'\') ## 1.5$$271 ## (\'Costa Rica\', \'\') ## 1.5$$272 ## (\'Congo (Kinshasa)\', \'\') ## 1.5$$273 ## (\'Congo (Brazzaville)\', \'\') ## 1.5$$274 ## (\'Colombia\', \'\') ## 4$$275 ## (\'China\', \'Zhejiang\') ## 1$$276 ## (\'China\', \'Yunnan\') ## 1$$277 ## (\'China\', \'Xinjiang\') ## 1$$278 ## (\'China\', \'Tibet\') ## 1$$279 ## (\'China\', \'Tianjin\') ## 1.1$$280 ## (\'China\', \'Sichuan\') ## 1$$281 ## (\'China\', \'Shanxi\') ## 1$$282 ## (\'China\', \'Shanghai\') ## 1.1$$283 ## (\'China\', \'Shandong\') ## 1$$284 ## (\'Cuba\', \'\') ## 3$$285 ## (\'Cyprus\', \'\') ## 1.5$$286 ## (\'Czechia\', \'\') ## 1.5$$287 ## (\'Denmark\', \'\') ## 2$$288 ## (\'Finland\', \'\') ## 2$$289 ## (\'Fiji\', \'\') ## 2$$290 ## (\'Ethiopia\', \'\') ## 2$$291 ## (\'Eswatini\', \'\') ## 1$$292 ## (\'Estonia\', \'\') ## 1.1$$293 ## (\'Eritrea\', \'\') ## 1$$294 ## (\'Equatorial Guinea\', \'\') ## 1$$295 ## (\'China\', \'Qinghai\') ## 1$$296 ## (\'El Salvador\', \'\') ## 3$$297 ## (\'Ecuador\', \'\') ## 1.1$$298 ## (\'Dominican Republic\', \'\') ## 1.5$$299 ## (\'Dominica\', \'\') ## 1$$300 ## (\'Djibouti\', \'\') ## 3$$301 ## (\'Diamond Princess\', \'\') ## 1$$302 ## (\'Denmark\', \'Greenland\') ## 1$$303 ## (\'Denmark\', \'Faroe Islands\') ## 1$$304 ## (\'Egypt\', \'\') ## 2$$305 ## (\'Zimbabwe\', \'\') ## 1"

# mapped_list = mapping.split("$$")

# mapped_list = [mapped_row.split("##") for mapped_row in mapped_list]

# mapped_list = [(a[1].strip(), float(a[2].strip())) for a in mapped_list]

# mapped_dict = {}

# for pair in mapped_list:

#     fields = pair[0][1:-1].split(', \'')

#     mapped_dict.update({(fields[0].strip()[1:-1], fields[1].strip()[:-1]): pair[1]})



# rate_1 = []

# rate_1_1 = []

# rate_1_2 = []

# rate_1_5 = []

# rate_2 = []

# rate_3 = []

# rate_4 =[]

# rates = [1,1.1,1.2,1.5,2,3,4]

# indices = [rate_1, rate_1_1, rate_1_2, rate_1_5, rate_2, rate_3, rate_4]



# hr_id = list(min_date_sorted.index)

# for index, pair in enumerate(min_date_sorted.index):

#     if pair in mapped_dict.keys():

#         indices[rates.index(mapped_dict[pair])].append(index)

#     else:

#         print(index, pair)

#         indices[rates.index(2)].append(index)



# print(rate_1)

# print(rate_1_1)

# print(rate_1_2)

# print(rate_1_5)

# print(rate_2)

# print(rate_3)

# print(rate_4)



# rate_1 = [3,4,5,15,21,22,23,26,27,28,30,31,32,35,36,39,42,45,47,48,49,53,61,67,68,69,72,75,81,94,101,102,107,108,109,154,156,157,158,164,165,166, 171,174,175,176,178,179,181,183, 184,185,186,187,189,190,191,192,211,212,215,217,220,221,222,228,229,230,233,240,241,248,249,264,265,275,276,277,278,280,281,283,291,293,294,295,299,301,302,303,305]

# rate_1_1 = [2,51,52,76,153,193,194,195,196,197,198,199,242,243,245,247,292,297]

# rate_1_2 = [6,20,29,70,71,97,98,104,180,252,255,262,266]

# rate_1_5 = [24,46,54,57,59,62,66,100,103,105,159,167,168,201,204,205,206,207,213,214,223,225,231,234,236,237,244,250,259,267,268,269,270,271,272,273,285,286,298]

# rate_2 = []

# rate_3 = [9,25,34,63,64,74,77,78,99,208,219,258,260,261,284,296,300]

# rate_4 = [7,11,13,18,37,40,41,55,56,129,216,226,235,239,246,274]

# rate_custom = [169,172,173,177,182,188,218,279,282]



# rate_1 = [4, 5, 6, 18, 21, 22, 23, 26, 27, 28, 31, 32, 33, 36, 37, 41, 44, 48, 50, 51, 52, 58, 63, 70, 71, 72, 77, 84, 94, 104, 106, 111, 112, 113, 155, 156, 158, 160, 161, 162, 168, 169, 170, 174, 179, 180, 181, 183, 185, 186, 188, 189, 190, 191, 194, 195, 196, 197, 216, 217, 220, 222, 223, 225, 226, 227, 233, 234, 235, 239, 246, 248, 251, 252, 253, 254, 284, 285, 286, 288, 291, 294, 300, 302, 303, 308, 310, 311, 312]

# rate_1_1 = [3, 54, 78, 135, 176, 177, 178, 182, 198, 199, 200, 201, 202, 203, 204, 249, 250, 256, 272, 273, 301, 306]

# rate_1_2 = [7, 20, 29, 73, 74, 100, 101, 107, 193, 259, 263, 269, 274]

# rate_1_5 = [24, 40, 49, 57, 61, 64, 69, 103, 109, 114, 163, 171, 172, 206, 209, 210, 211, 212, 218, 219, 228, 230, 237, 240, 242, 243, 257, 262, 266, 275, 277, 278, 279, 280, 281, 282, 295, 296, 307]

# rate_2 = [0, 1, 2, 9, 11, 13, 15, 16, 19, 30, 34, 38, 45, 46, 47, 53, 56, 59, 60, 62, 68, 75, 79, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 105, 108, 110, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 157, 159, 164, 165, 166, 167, 184, 205, 207, 208, 214, 215, 229, 232, 238, 244, 247, 255, 258, 260, 261, 264, 270, 271, 292, 293, 297, 298, 299, 305]

# rate_3 = [10, 25, 35, 65, 67, 76, 80, 81, 102, 213, 224, 265, 267, 268, 276, 304, 309]

# rate_4 = [8, 12, 14, 17, 39, 42, 43, 55, 66, 133, 221, 231, 236, 241, 245, 283]







rate_1 = [2, 3, 4, 5, 6, 18, 26, 27, 30, 41, 44, 48, 67, 71, 111, 113, 158, 168, 174, 176, 180, 183, 185, 186, 189, 190, 191, 194, 202, 225, 233, 234, 235, 246, 247, 248, 250, 284, 285, 286, 294, 302, 303, 310]

rate_1_1 = [7, 19, 23, 40, 42, 49, 50, 51, 57, 63, 68, 75, 76, 77, 94, 98, 100, 104, 107, 121, 135, 155, 156, 169, 170, 171, 173, 175, 177, 178, 179, 181, 188, 193, 200, 201, 203, 204, 207, 208, 220, 223, 236, 239, 251, 253, 256, 281, 287, 288, 289, 291, 295, 298]

rate_1_2 = [20, 21, 22, 24, 29, 31, 33, 54, 59, 64, 65, 70, 72, 74, 78, 84, 99, 101, 109, 112, 114, 119, 120, 125, 131, 142, 146, 151, 153, 163, 165, 172, 182, 192, 195, 197, 198, 212, 230, 240, 242, 249, 254, 259, 263, 265, 266, 269, 270, 272, 273, 274, 279, 280, 282, 290, 296, 308]

rate_1_5 = [9, 11, 12, 13, 15, 16, 25, 34, 47, 53, 60, 61, 62, 69, 79, 80, 81, 82, 85, 86, 87, 88, 92, 93, 97, 103, 115, 116, 117, 118, 123, 124, 127, 128, 130, 132, 134, 137, 139, 140, 141, 143, 147, 149, 150, 152, 154, 157, 162, 164, 205, 206, 209, 210, 211, 213, 214, 216, 219, 224, 226, 227, 228, 238, 257, 260, 261, 262, 264, 271, 275, 276, 277, 278, 283, 292, 293, 297, 301, 307]

rate_2 = [0, 10, 28, 32, 36, 37, 38, 45, 46, 52, 55, 73, 83, 89, 90, 91, 95, 105, 106, 108, 110, 122, 126, 129, 133, 138, 144, 145, 148, 159, 160, 161, 166, 167, 184, 187, 196, 215, 217, 218, 221, 222, 229, 232, 237, 241, 244, 245, 252, 267, 299, 300, 304, 306, 309, 312]

rate_3 = [1, 14, 35, 39, 43, 56, 58, 66, 96, 102, 136, 199, 231, 243, 255, 258, 268, 305, 311]

rate_4 = [8, 17]



rate_custom = [173, 175, 187, 192, 287, 289, 290]



rates = [1,1.1,1.2,1.5,2,3,4]

indices = [rate_1, rate_1_1, rate_1_2, rate_1_5, rate_2, rate_3, rate_4]

mult = [2]*len(min_date_sorted.index)

i = 0

for index_list in indices:

    for j in index_list:

        mult[j] = rates[i]

    i = i + 1

# for a,b in enumerate(mult):

#     print(a,b)

# for index, hr_index in enumerate(min_date_sorted.index):

rate = rate_1 + rate_1_1 + rate_1_2 + rate_1_5 + rate_2 + rate_3 + rate_4

print(len(rate))
df = pd.DataFrame(columns = ['ForecastId','ConfirmedCases','Fatalities'])

df_hr = pd.DataFrame(columns = ['ForecastId', 'Country_Region', 'Province_State', 'Days', 'ConfirmedCases','Fatalities','Date'])

# record = df_train[(df_train['Country_Region'] == 'Italy') & (df_train['Province_State'] == '')]

# record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in record['Date']]

# record = record[record['days'] > public_start_day]

# record
public_start_date = '2020-04-02'

public_end_date = '2020-04-15'



count = 0

for index, start_date in zip(min_date_sorted.index, min_date_sorted):

    print(list(min_date_sorted.index).index(index), index, start_date)

    record = df_train[(df_train['Country_Region'] == index[0]) & (df_train['Province_State'] == index[1])]

    if len(record[record['ConfirmedCases'] > 0]) == 0:

        pass

    else:

        record = record[record['ConfirmedCases'] > 0]

    base_date_object = datetime.strptime(start_date, "%Y-%m-%d").date()

    public_start_date_object = datetime.strptime(public_start_date, "%Y-%m-%d").date()

    public_end_date_object = datetime.strptime(public_end_date, "%Y-%m-%d").date()

    record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in record['Date']]

    public_start_day = (public_start_date_object - base_date_object).days + 1

    public_end_day = (public_end_date_object - base_date_object).days + 1

    

    ## limit the training data to date before public start date

    if len(record[record['days'] < public_start_day]) > 0:

        record = record[record['days'] < public_start_day]

    # x axis values 

    

    record2 = record

    if len(record[record['Fatalities'] > 0]) != 0:

        record2 = record[record['Fatalities'] > 0]

    x = record['days'].values

    x2 = record2['days'].values

    # corresponding y axis values 

    y1 = record['ConfirmedCases'].values

    y2 = record2['Fatalities'].values

    

    

    ####  data correction ####

    b = -1

    bad_index = 0

    mod_count = 0

#     y1_copy = list(y1)

    for a in y1:

        if a < b:

            y1[bad_index] = b

            mod_count = mod_count + 1

        else:

            b = a

        bad_index = bad_index + 1

    if mod_count > 0:

        print("*****************")

        print(list(min_date_sorted.index).index(index), index)

        print(mod_count)

        print(y1)

#         print(y1_copy)

        print("*****************")

    b = -1

    bad_index = 0

    mod_count = 0

#     y2_copy = list(y2)

    for a in y2:

        if a < b:

            y2[bad_index] = b

            mod_count = mod_count + 1

        else:

            b = a

        bad_index = bad_index + 1

    if mod_count > 0:

        print("*****************")

        print(list(min_date_sorted.index).index(index), index)

        print(mod_count)

        print(y2)

#         print(y2_copy)

        print("*****************")    

    ####  data correction ####



    if len(y1) > 0:

        y_max_ = y1[-1]

#         y1_prime = np.diff(y1)

#         y1_prime2 = np.diff(y1_prime)

#         if len(y1)>0 and len(y1_prime) > 0 and len(y1_prime2) > 0:

#             max_slope_index = len(y1_prime) - 1 - list(y1_prime)[::-1].index(max(y1_prime))

#             max_slope_range = (max_slope_index+1)/len(y1_prime)

#             y_max_ = y1[-1]

#             if max_slope_range < 0.75:

#                 if y1_prime[max_slope_index] > 0 and max_slope_range < 0.75 and (((y1_prime[max_slope_index] - max(y1_prime[-2:]))/y1_prime[max_slope_index]) < 0.5):

#                     y_max_ = y1[-1]

#                     pass

#                 else:

#                     y_max_ = y1[max_slope_index + 1]

#                     pass

#             else:

#                 y_max_ = y1[-1]

    else:

        y_max_ = 0

    

    stage0 = False

    stage1 = False

    stage2 = False

    stage3 = False

    count1 = 0

    count2 = 0

    for start in range(len(y1_prime)-3):

        if sum(y1_prime[start:start+3]) <=12:

            count1 = count1 + 1

            count2 = 0

        else:

            count2 = count2 + 1

            count1 = 0

        if not stage0 and count2 == 0 and count1 > 2:

            stage0 = True

            count1 = 0

        if not stage1 and count1 == 0 and count2 > 5:

            stage0 = True

            stage1 = True

            count2 = 0

        if stage1 and count2 == 0 and count1 > 3:

            stage2 = True

            count1 = 0

        if stage2 and count1 == 0 and count2 > 2:

            stage3 = True

            count2 = 0

#     if stage3:

    print('Rate ', mult[list(min_date_sorted.index).index(index)])

    if list(min_date_sorted.index).index(index) in rate_custom:

        param, param_cov = curve_fit(custom, np.array(x), np.array(y1), maxfev = 100000, bounds=([1, 0, 1, 1, 30, 1, 1], [1.1, 60, 8, 200, 100, 8, 1400]))

        y1_pred = custom(x, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

#     elif index[0] == 'Korea, South':

#         param, param_cov = curve_fit(test_linear, np.array(x), np.array(y1), [y_max_, 0.5, (x[-1] - x[0])/2 + x[0], 50, 45, 0], maxfev = 100000, bounds=([y_max_/2, 0.1, 0, 1, 30, -100],[y_max_*5 + 1500, 1, 150, 100, 100, 1000]))

#         y1_pred = test_linear(x, param[0], param[1], param[2], param[3], param[4], param[5])

#     elif index[0] in ['US', 'Spain', 'Germany', 'France', 'Iran', 'United Kingdom']:

#         param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*7, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*5, 0.1, 0],[y_max_*10 + 1500, 1, 150]))

#         y1_pred = test(x, param[0], param[1], param[2])

#     elif index[0] == 'China':

#         param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_/2, 0.1, 0],[y_max_*5 + 1500, 1, 150]))

#         y1_pred = test(x, param[0], param[1], param[2])

    elif mult[list(min_date_sorted.index).index(index)] == 1:

        param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*mult[list(min_date_sorted.index).index(index)] + 1, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*mult[list(min_date_sorted.index).index(index)], 0.1, 0],[y_max_*mult[list(min_date_sorted.index).index(index)]*1.5 + 30, 1, 150]))

        y1_pred = test(x, param[0], param[1], param[2])

    elif mult[list(min_date_sorted.index).index(index)] == 1.1:

        param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*mult[list(min_date_sorted.index).index(index)] + 1, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*mult[list(min_date_sorted.index).index(index)], 0.1, 0],[y_max_*mult[list(min_date_sorted.index).index(index)]*1.5 + 200, 1, 150]))

        y1_pred = test(x, param[0], param[1], param[2])

    elif mult[list(min_date_sorted.index).index(index)] == 1.2:

        param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*mult[list(min_date_sorted.index).index(index)] + 1, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*mult[list(min_date_sorted.index).index(index)], 0.1, 0],[y_max_*mult[list(min_date_sorted.index).index(index)]*1.5 + 200, 1, 150]))

        y1_pred = test(x, param[0], param[1], param[2])

    else:

        param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*mult[list(min_date_sorted.index).index(index)] + 1, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*mult[list(min_date_sorted.index).index(index)], 0.1, 0],[y_max_*4 + 1500, 1, 150]))

        y1_pred = test(x, param[0], param[1], param[2])

    param2, param_cov2 = curve_fit(test, np.array(x2), np.array(y2), [y2[-1]/2, 0.5, (x2[-1] - x2[0])/2 + x2[0] - 3], maxfev = 100000, bounds=([y2[-1]/2, 0.1, 0],[y2[-1]*5 + 1, 0.8, 150]))

    y2_pred = test(x2,param2[0], param2[1], param2[2])

    

#     print(y1)

#     print(y1_pred)

#     print("----------------------------------")

#     print(y2)

#     print(y2_pred)

#     print("----------------------------------")

    

    group = test_groups.get_group(index)

#     print(group['Date'].values)

    group['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in group['Date'].values]

    

    group = group[group['days'] <= public_end_day]

    

    ids = group['ForecastId'].values

    days = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in group['Date'].values]

    

    prev_days = range(public_start_day - 6, public_start_day - 1, 1)

    

#     if stage3:

    if list(min_date_sorted.index).index(index) in rate_custom:

        test_y1_pred_raw = custom(days, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

        prev_y1_pred_raw = custom(prev_days, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

#     elif index[0] == 'Korea, South':

#         test_y1_pred_raw = test_linear(days, param[0], param[1], param[2], param[3], param[4], param[5])

#         prev_y1_pred_raw = test_linear(prev_days, param[0], param[1], param[2], param[3], param[4], param[5])

#     elif index[0] in ['US', 'Spain', 'Germany', 'France', 'Iran', 'United Kingdom']:

#         test_y1_pred_raw = test(days, param[0], param[1], param[2])

#         prev_y1_pred_raw = test(prev_days, param[0], param[1], param[2])

    else:

        test_y1_pred_raw = test(days, param[0], param[1], param[2])

        prev_y1_pred_raw = test(prev_days, param[0], param[1], param[2])

    test_y2_pred_raw = test(days, param2[0], param2[1], param2[2])

    prev_y2_pred_raw = test(prev_days, param2[0], param2[1], param2[2])

    

    day_index_pred = 0

    diff1_list = []

    diff2_list = []

    for day in prev_days:

        if day in x:

            day_index = np.where(x == day)

            diff1 = y1[day_index] - prev_y1_pred_raw[day_index_pred]

            diff1_list.append(diff1)

        if day in x2:

            day_index = np.where(x2 == day)

            diff2 = y2[day_index] - prev_y2_pred_raw[day_index_pred]

            diff2_list.append(diff2)

        day_index_pred = day_index_pred + 1

    

    if len(diff1_list) > 0:

        diff1_mean = np.max(diff1_list)

    else:

        diff1_mean = 0

    if len(diff2_list) > 0:

        diff2_mean = np.max(diff2_list)

    else:

        diff2_mean = 0

#     print('diff1_mean', diff1_mean)

#     print('diff2_mean', diff2_mean)

    if np.isnan(diff1_mean):

        pass

    else:

        test_y1_pred_raw = list(np.array(test_y1_pred_raw) + diff1_mean)

    if np.isnan(diff2_mean):

        pass

    else:

        test_y2_pred_raw = list(np.array(test_y2_pred_raw) + diff2_mean)

    

#     test_y1_pred = [int(n) for n in test_y1_pred_raw]

#     test_y2_pred = [int(m) for m in test_y2_pred_raw]



    test_y1_pred = test_y1_pred_raw

    test_y2_pred = test_y2_pred_raw



    

    ratio = 0

    if y2[-1] > 0:

        ratio = y1[-1]/y2[-1]

    else:

        ratio = y1[-1]

    

    train_day_index = days.index(public_start_day) - 1

    

    if(y1[-1] < 100 and y2[-1] < 4 and ratio < avg):

        # correct the y2_pred as per average rate = 22

        for pred_index in range(len(test_y2_pred)):

            if pred_index > train_day_index:

                if test_y2_pred[pred_index] < test_y1_pred[pred_index]/avg:

                    test_y2_pred[pred_index] = test_y1_pred[pred_index]/avg

    else:

        ## correct the y2_pred as per the rate

        for pred_index in range(len(test_y2_pred)):

            if pred_index > train_day_index:

                if test_y2_pred[pred_index] < test_y1_pred[pred_index]/ratio:

                    test_y2_pred[pred_index] = test_y1_pred[pred_index]/ratio

    

#     test_y1_pred = [int(n) for n in test_y1_pred]

#     test_y2_pred = [int(m) for m in test_y2_pred]

    

    local_df_hr = pd.DataFrame(ids, columns=['ForecastId'])

    print()

    local_df_hr.insert(1, 'Country_Region', [index[0]]*len(days))

    local_df_hr.insert(2, 'Province_State', [index[1]]*len(days))

    local_df_hr.insert(3, 'Days', days)

    local_df_hr.insert(4, 'ConfirmedCases', test_y1_pred)

    local_df_hr.insert(5, 'Fatalities', test_y2_pred)

    local_df_hr.insert(6, 'Date', group['Date'].values)

    

    local_df = pd.DataFrame(ids, columns=['ForecastId'])

    local_df.insert(1, 'ConfirmedCases', test_y1_pred)

    local_df.insert(2, 'Fatalities', test_y2_pred)

    df = df.append(local_df)

    df_hr = df_hr.append(local_df_hr)

    

    # x axis values 

#     if stage3:

#     if not stage3 and index[0] not in ['US', 'Spain', 'Germany', 'France', 'Iran', 'United Kingdom']:

#     if y1[-1] > test_y1_pred[0]:

#     if index[0] in ['US', 'Spain', 'Germany', 'France', 'Iran', 'United Kingdom']:

#     if list(min_date_sorted.index).index(index) in rate_1_1:

#     if index[0] == 'Russia':

#         print(mult[list(min_date_sorted.index).index(index)])

#         print(y_max_)

    actual_record = df_train[(df_train['Country_Region'] == index[0]) & (df_train['Province_State'] == index[1])]

    actual_record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in actual_record['Date']]

    actual_record = actual_record[actual_record['days'] > public_start_day]

    actual_days = actual_record['days'].values

    actual_y = actual_record['ConfirmedCases'].values

    actual_y2 = actual_record['Fatalities'].values



#     x = record['days'].values

#     plt.plot(x, y1, label = "ConfirmedCases") 

#     plt.plot(x, y1_pred, label = "Predicted") 

#     plt.plot(days, test_y1_pred, label = "Forecast")

#     plt.plot(actual_days, actual_y, label = "Actual")

#     plt.xlabel('Date') 

#     plt.ylabel('Label') 

#     plt.title(str(index[0]) + " " + str(index[1]) + ' - ConfirmedCases') 

#     plt.show() 



#     plt.plot(x2, y2, label = "Fatalities") 

#     plt.plot(x2, y2_pred, label = "Predicted")

#     plt.plot(days, test_y2_pred, label = "Forecast")

#     plt.plot(actual_days, actual_y2, label = "Actual")

#     plt.xlabel('Date') 

#     plt.ylabel('Label') 

#     plt.title(str(index[0]) + " " + str(index[1]) + ' - Fatalities') 

#     plt.show() 

    count = count + 1

#     break

print(df.shape)

print(df_hr.shape)
private_start_date = '2020-04-16'

private_end_date = '2020-05-14'



count = 0

for index, start_date in zip(min_date_sorted.index, min_date_sorted):

    print(list(min_date_sorted.index).index(index), index, start_date)

    record = df_train[(df_train['Country_Region'] == index[0]) & (df_train['Province_State'] == index[1])]

    if len(record[record['ConfirmedCases'] > 0]) == 0:

        pass

    else:

        record = record[record['ConfirmedCases'] > 0]

    base_date_object = datetime.strptime(start_date, "%Y-%m-%d").date()

    private_start_date_object = datetime.strptime(private_start_date, "%Y-%m-%d").date()

    private_end_date_object = datetime.strptime(private_end_date, "%Y-%m-%d").date()

    record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in record['Date']]

    private_start_day = (private_start_date_object - base_date_object).days + 1

    private_end_day = (private_end_date_object - base_date_object).days + 1

    

    ## limit the training data to date before public start date

    if len(record[record['days'] < private_start_day]) > 0:

        record = record[record['days'] < private_start_day]

    # x axis values 

    

    record2 = record

    if len(record[record['Fatalities'] > 0]) != 0:

        record2 = record[record['Fatalities'] > 0]

    x = record['days'].values

    x2 = record2['days'].values

    # corresponding y axis values 

    y1 = record['ConfirmedCases'].values

    y2 = record2['Fatalities'].values

    

    

    ####  data correction ####

    b = -1

    bad_index = 0

    mod_count = 0

#     y1_copy = list(y1)

    for a in y1:

        if a < b:

            y1[bad_index] = b

            mod_count = mod_count + 1

        else:

            b = a

        bad_index = bad_index + 1

    if mod_count > 0:

        print("*****************")

        print(list(min_date_sorted.index).index(index), index)

        print(mod_count)

        print(y1)

#         print(y1_copy)

        print("*****************")

    b = -1

    bad_index = 0

    mod_count = 0

#     y2_copy = list(y2)

    for a in y2:

        if a < b:

            y2[bad_index] = b

            mod_count = mod_count + 1

        else:

            b = a

        bad_index = bad_index + 1

    if mod_count > 0:

        print("*****************")

        print(list(min_date_sorted.index).index(index), index)

        print(mod_count)

        print(y2)

#         print(y2_copy)

        print("*****************")    

    ####  data correction ####



    y_max_ = y1[-1]

#     y1_prime = np.diff(y1)

#     y1_prime2 = np.diff(y1_prime)

#     if len(y1)>0 and len(y1_prime) > 0 and len(y1_prime2) > 0:

#         max_slope_index = len(y1_prime) - 1 - list(y1_prime)[::-1].index(max(y1_prime))

#         max_slope_range = (max_slope_index+1)/len(y1_prime)

#         y_max_ = y1[-1]

#         if max_slope_range < 0.75:

#             if y1_prime[max_slope_index] > 0 and max_slope_range < 0.75 and (((y1_prime[max_slope_index] - max(y1_prime[-2:]))/y1_prime[max_slope_index]) < 0.5):

#                 y_max_ = y1[-1]

#                 pass

#             else:

#                 y_max_ = y1[max_slope_index + 1]

#                 pass

#         else:

#             y_max_ = y1[-1]

    

    

    stage0 = False

    stage1 = False

    stage2 = False

    stage3 = False

    count1 = 0

    count2 = 0

    for start in range(len(y1_prime)-3):

        if sum(y1_prime[start:start+3]) <=12:

            count1 = count1 + 1

            count2 = 0

        else:

            count2 = count2 + 1

            count1 = 0

        if not stage0 and count2 == 0 and count1 > 2:

            stage0 = True

            count1 = 0

        if not stage1 and count1 == 0 and count2 > 5:

            stage0 = True

            stage1 = True

            count2 = 0

        if stage1 and count2 == 0 and count1 > 3:

            stage2 = True

            count1 = 0

        if stage2 and count1 == 0 and count2 > 2:

            stage3 = True

            count2 = 0

#     if stage3:

    if list(min_date_sorted.index).index(index) in rate_custom:

        param, param_cov = curve_fit(custom, np.array(x), np.array(y1), maxfev = 100000, bounds=([1, 0, 1, 1, 30, 1, 1], [1.1, 60, 8, 200, 100, 8, 1400]))

        y1_pred = custom(x, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

#     elif index[0] == 'Korea, South':

#         param, param_cov = curve_fit(test_linear, np.array(x), np.array(y1), [y_max_, 0.5, (x[-1] - x[0])/2 + x[0], 50, 45, 0], maxfev = 100000, bounds=([y_max_/2, 0.1, 0, 1, 30, -100],[y_max_*5 + 1500, 1, 150, 100, 100, 1000]))

#         y1_pred = test_linear(x, param[0], param[1], param[2], param[3], param[4], param[5])

#     elif index[0] in ['US', 'Spain', 'Germany', 'France', 'Iran', 'United Kingdom']:

#         param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*7, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*5, 0.1, 0],[y_max_*10 + 1500, 1, 150]))

#         y1_pred = test(x, param[0], param[1], param[2])

#     elif index[0] == 'China':

#         param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_/2, 0.1, 0],[y_max_*5 + 1500, 1, 150]))

#         y1_pred = test(x, param[0], param[1], param[2])

    elif mult[list(min_date_sorted.index).index(index)] == 1:

        param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*mult[list(min_date_sorted.index).index(index)] + 1, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*mult[list(min_date_sorted.index).index(index)], 0.1, 0],[y_max_*mult[list(min_date_sorted.index).index(index)]*1.5 + 30, 1, 150]))

        y1_pred = test(x, param[0], param[1], param[2])

    elif mult[list(min_date_sorted.index).index(index)] == 1.1:

        param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*mult[list(min_date_sorted.index).index(index)] + 1, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*mult[list(min_date_sorted.index).index(index)], 0.1, 0],[y_max_*mult[list(min_date_sorted.index).index(index)]*1.5 + 200, 1, 150]))

        y1_pred = test(x, param[0], param[1], param[2])

    elif mult[list(min_date_sorted.index).index(index)] == 1.2:

        param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*mult[list(min_date_sorted.index).index(index)] + 1, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*mult[list(min_date_sorted.index).index(index)], 0.1, 0],[y_max_*mult[list(min_date_sorted.index).index(index)]*1.5 + 200, 1, 150]))

        y1_pred = test(x, param[0], param[1], param[2])

    else:

        param, param_cov = curve_fit(test, np.array(x), np.array(y1), [y_max_*mult[list(min_date_sorted.index).index(index)] + 1, 0.5, (x[-1] - x[0])/2 + x[0]], maxfev = 100000, bounds=([y_max_*mult[list(min_date_sorted.index).index(index)], 0.1, 0],[y_max_*4 + 1500, 1, 150]))

        y1_pred = test(x, param[0], param[1], param[2])

    param2, param_cov2 = curve_fit(test, np.array(x2), np.array(y2), [y2[-1]/2, 0.5, (x2[-1] - x2[0])/2 + x2[0] - 3], maxfev = 100000, bounds=([y2[-1]/2, 0.1, 0],[y2[-1]*5 + 1, 0.8, 150]))

    y2_pred = test(x2,param2[0], param2[1], param2[2])

    

#     print(y1)

#     print(y1_pred)

#     print("----------------------------------")

#     print(y2)

#     print(y2_pred)

#     print("----------------------------------")

    

    group = test_groups.get_group(index)

#     print(group['Date'].values)

    group['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in group['Date'].values]

    

    group = group[group['days'] >= private_start_day]

    ids = group['ForecastId'].values

    days = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in group['Date'].values]

    

    

    prev_days = range(private_start_day - 6, private_start_day - 1, 1)

    

#     if stage3:

    if list(min_date_sorted.index).index(index) in rate_custom:

        test_y1_pred_raw = custom(days, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

        prev_y1_pred_raw = custom(prev_days, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

#     elif index[0] == 'Korea, South':

#         test_y1_pred_raw = test_linear(days, param[0], param[1], param[2], param[3], param[4], param[5])

#         prev_y1_pred_raw = test_linear(prev_days, param[0], param[1], param[2], param[3], param[4], param[5])

#     elif index[0] in ['US', 'Spain', 'Germany', 'France', 'Iran', 'United Kingdom']:

#         test_y1_pred_raw = test(days, param[0], param[1], param[2])

#         prev_y1_pred_raw = test(prev_days, param[0], param[1], param[2])

    else:

        test_y1_pred_raw = test(days, param[0], param[1], param[2])

        prev_y1_pred_raw = test(prev_days, param[0], param[1], param[2])

    test_y2_pred_raw = test(days, param2[0], param2[1], param2[2])

    prev_y2_pred_raw = test(prev_days, param2[0], param2[1], param2[2])

    

    

    day_index_pred = 0

    diff1_list = []

    diff2_list = []

    for day in prev_days:

        if day in x:

            day_index = np.where(x == day)

            diff1 = y1[day_index] - prev_y1_pred_raw[day_index_pred]

            diff1_list.append(diff1)

        if day in x2:

            day_index = np.where(x2 == day)

            diff2 = y2[day_index] - prev_y2_pred_raw[day_index_pred]

            diff2_list.append(diff2)

        day_index_pred = day_index_pred + 1

    

    if len(diff1_list) > 0:

        diff1_mean = np.max(diff1_list)

    else:

        diff1_mean = 0

    if len(diff2_list) > 0:

        diff2_mean = np.max(diff2_list)

    else:

        diff2_mean = 0

    

#     print('diff1_mean', diff1_mean)

#     print('diff2_mean', diff2_mean)

    if np.isnan(diff1_mean):

        pass

    else:

        test_y1_pred_raw = list(np.array(test_y1_pred_raw) + diff1_mean)

    if np.isnan(diff2_mean):

        pass

    else:

        test_y2_pred_raw = list(np.array(test_y2_pred_raw) + diff2_mean)

    

#     test_y1_pred = [int(n) for n in test_y1_pred_raw]

#     test_y2_pred = [int(m) for m in test_y2_pred_raw]

    test_y1_pred = test_y1_pred_raw

    test_y2_pred = test_y2_pred_raw

    

    ratio = 0

    if y2[-1] > 0:

        ratio = y1[-1]/y2[-1]

    else:

        ratio = y1[-1]

    

    train_day_index = days.index(private_start_day) - 1

    

    if(y1[-1] < 100 and y2[-1] < 4 and ratio < avg):

        # correct the y2_pred as per average rate = 22

        for pred_index in range(len(test_y2_pred)):

            if pred_index > train_day_index:

                if test_y2_pred[pred_index] < test_y1_pred[pred_index]/avg:

                    test_y2_pred[pred_index] = test_y1_pred[pred_index]/avg

    else:

        ## correct the y2_pred as per the rate

        for pred_index in range(len(test_y2_pred)):

            if pred_index > train_day_index:

                if test_y2_pred[pred_index] < test_y1_pred[pred_index]/ratio:

                    test_y2_pred[pred_index] = test_y1_pred[pred_index]/ratio

    

#     test_y1_pred = [int(n) for n in test_y1_pred]

#     test_y2_pred = [int(m) for m in test_y2_pred]



    local_df_hr = pd.DataFrame(ids, columns=['ForecastId'])

    local_df_hr.insert(1, 'Country_Region', [index[0]]*len(days))

    local_df_hr.insert(2, 'Province_State', [index[1]]*len(days))

    local_df_hr.insert(3, 'Days', days)

    local_df_hr.insert(4, 'ConfirmedCases', test_y1_pred)

    local_df_hr.insert(5, 'Fatalities', test_y2_pred)

    local_df_hr.insert(6, 'Date', group['Date'].values)

    

    local_df = pd.DataFrame(ids, columns=['ForecastId'])

    local_df.insert(1, 'ConfirmedCases', test_y1_pred)

    local_df.insert(2, 'Fatalities', test_y2_pred)

    df = df.append(local_df)

    df_hr = df_hr.append(local_df_hr)

    # x axis values 

#     if stage3:

#     if index[0] in ['US', 'Spain', 'Germany', 'France', 'Iran', 'United Kingdom']:

#     print(y1_prime)

#     if count > 166 and count < 173:

#     x = record['days'].values

#     plt.plot(x, y1, label = "Confirmed Cases") 

#     plt.plot(x, y1_pred, label = "Predicted") 

#     plt.plot(days, test_y1_pred, label = "Forecast")

#     plt.xlabel('Date') 

#     plt.ylabel('Label') 

#     plt.title(str(index[0]) + " " + str(index[1]) + ' - Confirmed Cases') 

#     plt.show() 



#     plt.plot(x2, y2, label = "Fatalities") 

#     plt.plot(x2, y2_pred, label = "Predicted")

#     plt.plot(days, test_y2_pred, label = "Forecast")

#     plt.xlabel('Date') 

#     plt.ylabel('Label') 

#     plt.title(str(index[0]) + " " + str(index[1]) + ' - Fatalities') 

#     plt.show() 

    count = count + 1

#     break

print(df.shape)

print(df_hr.shape)
df = df.sort_values(by=['ForecastId'], ascending=True)

df_hr = df_hr.sort_values(by=['ForecastId'], ascending=True)
df.to_csv('submission.csv', index=False)

df_hr.to_csv('hr_submission.csv', index=False)
df.shape
for row in df_hr.values:

    print(row)
df_test.shape
df.columns
df
df_hr