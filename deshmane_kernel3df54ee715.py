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
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

sample_submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
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
rate_1 = [3,4,5,15,21,22,23,26,27,28,30,31,32,35,36,39,42,45,47,48,49,53,61,67,68,69,72,75,81,94,101,102,107,108,109,154,156,157,158,164,165,166, 171,174,175,176,178,179,181,183, 184,185,186,187,189,190,191,192,211,212,215,217,220,221,222,228,229,230,233,240,241,248,249,264,265,275,276,277,278,280,281,283,291,293,294,295,299,301,302,303,305]

rate_1_1 = [2,51,52,76,153,193,194,195,196,197,198,199,242,243,245,247,292,297]

rate_1_2 = [6,20,29,70,71,97,98,104,180,252,255,262,266]

rate_1_5 = [24,46,54,57,59,62,66,100,103,105,159,167,168,201,204,205,206,207,213,214,223,225,231,234,236,237,244,250,259,267,268,269,270,271,272,273,285,286,298]

rate_2 = []

rate_3 = [9,25,34,63,64,74,77,78,99,208,219,258,260,261,284,296,300]

rate_4 = [7,11,13,18,37,40,41,55,56,129,216,226,235,239,246,274]

rate_custom = [169,172,173,177,182,188,218,279,282]

rates = [1,1.1,1.2,1.5,2,3,4]

indices = [rate_1, rate_1_1+rate_custom, rate_1_2, rate_1_5, rate_2, rate_3, rate_4]

mult = [2]*306

i = 0

for index_list in indices:

    for j in index_list:

        mult[j] = rates[i]

    i = i + 1

for a,b in enumerate(mult):

    print(a,b)
df = pd.DataFrame(columns = ['ForecastId','ConfirmedCases','Fatalities'])

df_hr = pd.DataFrame(columns = ['ForecastId', 'Country_Region', 'Province_State', 'Days', 'ConfirmedCases','Fatalities','Date'])

# record = df_train[(df_train['Country_Region'] == 'Italy') & (df_train['Province_State'] == '')]

# record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in record['Date']]

# record = record[record['days'] > public_start_day]

# record
public_start_date = '2020-03-26'

public_end_date = '2020-04-08'



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

    

    test_y1_pred = [int(n) for n in test_y1_pred]

    test_y2_pred = [int(m) for m in test_y2_pred]

    

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

#     actual_record = df_train[(df_train['Country_Region'] == index[0]) & (df_train['Province_State'] == index[1])]

#     actual_record['days'] = [(datetime.strptime(date, "%Y-%m-%d").date() - base_date_object).days + 1 for date in actual_record['Date']]

#     actual_record = actual_record[actual_record['days'] > public_start_day]

#     actual_days = actual_record['days'].values

#     actual_y = actual_record['ConfirmedCases'].values

#     actual_y2 = actual_record['Fatalities'].values



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
private_start_date = '2020-04-09'

private_end_date = '2020-05-07'



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

    

    test_y1_pred = [int(n) for n in test_y1_pred]

    test_y2_pred = [int(m) for m in test_y2_pred]



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

    x = record['days'].values

    plt.plot(x, y1, label = "Confirmed Cases") 

    plt.plot(x, y1_pred, label = "Predicted") 

    plt.plot(days, test_y1_pred, label = "Forecast")

    plt.xlabel('Date') 

    plt.ylabel('Label') 

    plt.title(str(index[0]) + " " + str(index[1]) + ' - Confirmed Cases') 

    plt.show() 



    plt.plot(x2, y2, label = "Fatalities") 

    plt.plot(x2, y2_pred, label = "Predicted")

    plt.plot(days, test_y2_pred, label = "Forecast")

    plt.xlabel('Date') 

    plt.ylabel('Label') 

    plt.title(str(index[0]) + " " + str(index[1]) + ' - Fatalities') 

    plt.show() 

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