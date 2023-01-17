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
def RMSLE(pred,actual):

        return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))
data = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test_data = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')
data["Date"] = data["Date"].apply(lambda x: x.replace("-",""))

data["Date"]  = data["Date"].astype(int)

test_data["Date"] = test_data["Date"].apply(lambda x: x.replace("-",""))

test_data["Date"]  = test_data["Date"].astype(int)

data['key'] = data['Country/Region'].astype('str') + " " + data['Province/State'].astype('str')

test_data['key'] = test_data['Country/Region'].astype('str') + " " + test_data['Province/State'].astype('str')

data_train = data
test_last_day = int(test_data.shape[0]/284) + int(data_train[data_train.Date<20200312].shape[0]/284)
test_days = np.arange(int(data_train[data_train.Date<20200312].shape[0]/284), test_last_day, 1)
pivot_train = pd.pivot_table(data_train, index='Date', columns = 'key', values = 'ConfirmedCases')

pivot_train_d = pd.pivot_table(data_train, index='Date', columns = 'key', values = 'Fatalities')

np_train = pivot_train.to_numpy()

np_train_d = pivot_train_d.to_numpy()
shift = [0,1,2,3,4,5,6,7]

for s in shift:

    sum = 0

    for i in range(1,20):

        sum += np.abs((np_train_d[-i][:]/(np_train[-i-s][:]+0.0001)-np_train_d[-i-1][:]/(np_train[-i-1-s][:]+0.0001)).mean())

    print(sum, s)
mask_deaths = np.zeros_like(np_train[0])

for i in range(1,21):

    mask_deaths += np_train_d[-i]/(np_train[-i]+0.0001)

mask_deaths = mask_deaths/20    
mask_deaths[(mask_deaths < 0.5) & (mask_deaths!=0)].mean()
mask_deaths[(mask_deaths> 0.5)|(mask_deaths<0.005)] = mask_deaths[(mask_deaths < 0.5) & (mask_deaths!=0)].mean()
mask_mesh = np.meshgrid(mask_deaths, test_days)[0].T.flatten()
assert mask_mesh.shape[0] == data_test.shape[0]
as_exponent = [0, 1, 2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 27

              , 34, 35, 39, 41, 42, 43, 46, 48, 50, 63, 70, 75, 80, 81, 82, 83, 84, 85

              , 92, 93, 94, 95, 96, 100, 102, 105, 106, 108, 110, 112, 113, 114, 115,

              121, 122, 125, 128, 130, 131, 132, 134, 137, 138, 139, 143, 145, 148,

              149, 161, 162, 163, 165, 166, 168, 172

              , 175, 177, 178, 184, 185, 190, 192, 194, 198, 202

              ,203, 204, 205, 207, 212, 213, 214, 216, 218, 220,

              221, 224, 225, 228, 229, 230, 231, 232, 233, 234, 235, 237, 238,

              239, 241, 242, 243, 244, 246, 247, 248, 249, 250, 252, 253, 255, 256, 257, 258, 260,

              261, 264, 265, 266, 269, 272, 273, 274, 282]

as_linear = [4, 7, 9, 17, 19, 22, 25, 26, 29, 30, 32, 33, 36, 37, 38, 40, 45, 47, 49, 51, 53, 54, 55, 56, 57,58,60, 61,62, 64, 65, 66,

             67, 68, 69, 71, 72,73, 74, 76, 77, 78, 79, 86, 88, 89, 91, 97, 98, 99, 101,103,104, 107, 109, 111, 116, 117, 118,

             120,123, 129, 135, 136, 141, 142, 144,

          146,147,  150, 151, 152, 153, 154, 156, 164, 167, 169, 170, 171,174,

             180, 182, 183,186, 187, 188, 189, 191, 195, 196, 197,200, 201, 208,

             209, 210, 211, 217, 219, 222, 226, 236, 240, 245, 251, 254, 259, 262, 263,

             267, 268, 270, 271, 277, 278, 279, 280, 281, 283]



# 101 107 122????

as_sigmoid = [6, 8, 21, 23,28, 31, 44, 52, 59, 87, 90

             , 119, 124, 126, 127, 133, 140, 155, 157, 158, 159, 160, 173, 176, 179, 181,

               193, 199, 206, 215, 223, 227, 276, 275]
def exp(x, a, b, d, p):

    return d * np.exp(a * x - b) + p





def linear(x, a, b, c):

    return a*(x-b)+c



def sigmoid(x, a, b, d, p):

    return d/(1 + np.exp(-(a*x-b))) + p
np_train[-1,101], np_train[-2,101]  #косяк
np_train[-1,101] = np_train[-2,101]*1.1
coefs = []

from scipy.optimize import curve_fit



X = np.arange(45, np_train.shape[0], 1)



for i in range(np_train.shape[1]):

    if i in as_exponent:

        coefs.append(curve_fit(exp,  X, np_train[45:, i],p0 = (0.5, X[0], 2, 0), maxfev=100000)[0])

    if i in as_linear:

        coefs.append(curve_fit(linear,  X[10:], np_train[55:, i], p0 = (1,0,0), maxfev=100000)[0])

    if i in as_sigmoid:

        coefs.append(curve_fit(sigmoid,  X, np_train[45:, i] , p0 = (1, X[0], np_train[-1, i]/2,0), maxfev=100000)[0])

          

        

        
import matplotlib.pyplot as plt

for i in as_linear:

    plt.plot(np_train[45:,i], label = str(i))

    plt.plot(linear(X, *coefs[i]))

    plt.legend()

    plt.show()
import matplotlib.pyplot as plt

for i in as_sigmoid:

    plt.plot(np_train[45:,i], label = str(i))

    plt.plot(sigmoid(X, *coefs[i]))

    plt.legend()

    plt.show()
import matplotlib.pyplot as plt

for i in as_exponent:

    plt.plot(np_train[45:,i], label = str(i))

    plt.plot(exp(X, *coefs[i]))

    plt.legend()

    plt.show()
ConfirmedCases_test = np.zeros((284, test_days.shape[0]))
test_days
def new_linear(x, a, b, c):

    return (a*(x-b)+c)*(a*(x-b)+c>=0)



def new_sigmoid(x, a, b, d, p):

    return sigmoid(x, a, b, d, p)*(sigmoid(x, a, b, d, p)>=0)



def new_epx(x, a, b, d, p):

    return exp(x, a, b, d, p)*(exp(x, a, b, d, p)>=0)
for i in range(np_train.shape[1]):

    if i in as_exponent:

        function = new_epx

    if i in as_linear:

        function = new_linear

    if i in as_sigmoid:

        function = new_sigmoid

    ConfirmedCases_test[i] = function(test_days, *coefs[i])

        
ConfirmedCases_test.flatten().shape
test_data['predict'] = ConfirmedCases_test.flatten()

test_data[test_data['Country/Region']=='Russia']
submission['ConfirmedCases'] = ConfirmedCases_test.flatten()

submission['Fatalities'] = ConfirmedCases_test.flatten()*mask_mesh

submission.to_csv('submission.csv', index=False)