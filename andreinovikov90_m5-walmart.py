import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import sys

import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error, mean_squared_error



import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs

from scipy.optimize import minimize



import matplotlib.pyplot as plt
path = "/kaggle/input/m5-forecasting-accuracy/"

#path = ""

df1_calendar = pd.read_csv(path+'calendar.csv')

df2_sell_prices = pd.read_csv(path+'sell_prices.csv')

df3_sales_train_valid = pd.read_csv(path+'sales_train_validation.csv')

df4_sampl_sub = pd.read_csv(path+'sample_submission.csv')
df1_calendar


cols = df1_calendar.columns

for col in cols:

    print(col)

    print(df1_calendar[col].unique())

    print('-------------------------')
df2_sell_prices
cols = df2_sell_prices.columns

for col in cols:

    print(col)

    print(df2_sell_prices[col].unique())

    print('-------------------------')
df3_sales_train_valid
cols = df3_sales_train_valid.columns

for col in cols:

    print(col)

    print(df3_sales_train_valid[col].unique())

    print('-------------------------')
# df4_sampl_sub
# Вывод продаж за весь период для произвольного товара

days = ["id"]

for i in range(1, 1914):

    days.append("d_"+str(i))



# plt.scatter(range(1, 1914), df3_sales_train_valid[days[1:]].iloc[3])

plt.figure(figsize=(15,4)) 

plt.plot(range(1, 1914), df3_sales_train_valid[days[1:]].iloc[3])

plt.title('товар HOBBIES_1_004_CA_1_validation')

plt.xlabel('период продаж')

plt.ylabel('число продаж за день')

plt.show()
# Гистограмма распределения

plt.figure(figsize=(15,5))

plt.hist(df3_sales_train_valid[days[1:]].iloc[3], bins=100, color='grey');

plt.title('Age distribution');

plt.show();
# def min_day(x):

#     i_min = 0

#     for i in range(len(x)):

#         if x[i]>0:

#             i_min = i

#             break

#     return i_min  



# def max_day(x):

#     i_max = 1914

#     for i in range(len(x)):

#         if x[-1-i]>0:

#             i_max = len(x)-1-i

#             break

#     return i_max    



# days = []

# for i in range(1, 1914):

#     days.append("d_"+str(i))

# df3_sales_train_valid["start_day"] = df3_sales_train_valid[days].apply(min_day, axis=1)

# df3_sales_train_valid["end_day"] = df3_sales_train_valid[days].apply(max_day, axis=1)
df3_sales_train_valid
# del df3_groupStateCat["start_day"]

# del df3_groupStateCat["end_day"]
# df3_sales_train_valid
df3_groupState = df3_sales_train_valid.groupby(['state_id']).mean().reset_index()

df3_groupState
plt.figure(figsize=(15,4))

a = 1

b = 1913

plt.plot(range(a, b+1), df3_groupState[days[a:b+1]].iloc[0])

plt.plot(range(a, b+1), df3_groupState[days[a:b+1]].iloc[1],color="black")

plt.plot(range(a, b+1), df3_groupState[days[a:b+1]].iloc[2],color="red")

plt.title('Интегральная продажа по штатам')

plt.xlabel('days')

plt.ylabel('уровень продаж')

plt.legend(['CA','TX','WI'])

plt.show()
plt.figure(figsize=(15,4))

a = 650

b = 700

plt.plot(range(a, b+1), df3_groupState[days[a:b+1]].iloc[0])

plt.plot(range(a, b+1), df3_groupState[days[a:b+1]].iloc[1],color="black")

plt.plot(range(a, b+1), df3_groupState[days[a:b+1]].iloc[2],color="red")

plt.title('Интегральная продажа по штатам')

plt.xlabel('days')

plt.ylabel('уровень продаж')

plt.legend(['CA','TX','WI'])

plt.show()
# *********** Детальные графики ***********************



states =  df3_groupState["state_id"]



for state in states:

    if (state == 'CA'):

        colorName = 'DodgerBlue'

    if (state == 'TX'):

        colorName = 'black'

    if (state == 'WI'):

        colorName = 'red'

    plt.figure(figsize=(15,4))

    arrayPlot = df3_groupState[(df3_groupState["state_id"]==state)]

    plt.hist(arrayPlot[days[1:]].iloc[0],bins=100,color=colorName) 

    plt.title('Распределение продаж по штату '+state); 

    plt.show();
# *********** Общий график ***********************



plt.figure(figsize=(15,4))

for state in states:

    if (state == 'CA'):

        colorName = 'DodgerBlue'

    if (state == 'TX'):

        colorName = 'black'

    if (state == 'WI'):

        colorName = 'red'

    arrayPlot = df3_groupState[(df3_groupState["state_id"]==state)]

    plt.hist(arrayPlot[days[1:]].iloc[0],bins=100,color=colorName) 



plt.title('Общее распределение продаж по штатам')

plt.legend(['CA','TX','WI'])

plt.show();
df3_groupCategory = df3_sales_train_valid.groupby(['cat_id']).sum().reset_index()

df3_groupCategory
categorys =  df3_groupCategory["cat_id"]



# *********** Детальные графики ***********************



for cat in categorys:

    if(cat=='HOBBIES'):

        colorName='Goldenrod'

    if(cat=='FOODS'):

        colorName='YellowGreen'

    if(cat=='HOUSEHOLD'):

        colorName='OrangeRed'

    plt.figure(figsize=(15,4))

    arrayPlot = df3_groupCategory[(df3_groupCategory["cat_id"]==cat)]

    plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

    plt.title('Интегральная продажа по категории '+cat);

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.show()
# *********** Общий график ***********************



plt.figure(figsize=(15,4))

for cat in categorys:

    if(cat=='HOBBIES'):

        colorName='Goldenrod'

    if(cat=='FOODS'):

        colorName='YellowGreen'

    if(cat=='HOUSEHOLD'):

        colorName='OrangeRed'

    arrayPlot = df3_groupCategory[(df3_groupCategory["cat_id"]==cat)]

    plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

plt.title('Интегральная продажа по категорим ');

plt.xlabel('days')

plt.ylabel('уровень продаж')

plt.legend(['FOODS','HOBBIES','HOUSEHOLD'])

plt.show()
df3_groupStateCat = df3_sales_train_valid.groupby(['state_id','cat_id']).sum()

df3_groupStateCat 

df3_groupStateCat  = df3_groupStateCat.reset_index()
df3_groupStateCat
CA_FOODS = df3_groupStateCat[(df3_groupStateCat["cat_id"]=="FOODS") &

                             (df3_groupStateCat["state_id"]=="CA")]

CA_FOODS
categorys = df3_groupStateCat["cat_id"].unique()

states =  df3_groupStateCat["state_id"].unique()



for cat in categorys:

    plt.figure(figsize=(15,4))

    for state in states:

        if (state == 'CA'):

            colorName = 'DodgerBlue'

        if (state == 'TX'):

            colorName = 'black'

        if (state == 'WI'):

            colorName = 'red'

        arrayPlot = df3_groupStateCat[(df3_groupStateCat["cat_id"]==cat) &

                             (df3_groupStateCat["state_id"]==state)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

        

    plt.title('Интегральная продажа по штатам и категории '+cat)

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.legend(['CA','TX','WI'])

    plt.show()
df3_groupStoreCat = df3_sales_train_valid.groupby(['store_id','cat_id']).sum()

# df3_groupStoreCat

df3_groupStoreCat  = df3_groupStoreCat.reset_index()
df3_groupStoreCat
categorys = df3_groupStoreCat["cat_id"].unique()

store =  df3_groupStoreCat["store_id"].unique()

store_CA = ['CA_1', 'CA_2', 'CA_3', 'CA_4']

store_TX = ['TX_1', 'TX_2', 'TX_3']

store_WI = ['WI_1', 'WI_2', 'WI_3']
# *********** Детальные графики ***********************



for cat in categorys:

    if(cat=='HOBBIES'):

        colorName='Goldenrod'

    if(cat=='FOODS'):

        colorName='YellowGreen'

    if(cat=='HOUSEHOLD'):

        colorName='OrangeRed'

    for store in store_CA:

        plt.figure(figsize=(10,4))

        arrayPlot = df3_groupStoreCat[(df3_groupStoreCat["cat_id"]==cat) &

                             (df3_groupStoreCat["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

        plt.title('Интегральная продажа по магазину '+store+' и категории '+cat)

        plt.xlabel('days')

        plt.ylabel('уровень продаж')

        plt.show()
# ************ Общие графики ***********************



for cat in categorys:

    plt.figure(figsize=(15,4))

    for store in store_CA:

        arrayPlot = df3_groupStoreCat[(df3_groupStoreCat["cat_id"]==cat) &

                             (df3_groupStoreCat["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0])

        

    plt.title('Интегральная продажа по магазинам штата CA и категории '+cat)

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.legend(store_CA)

    plt.show()
# *********** Детальные графики ***********************



for cat in categorys:

    if(cat=='HOBBIES'):

        colorName='Goldenrod'

    if(cat=='FOODS'):

        colorName='YellowGreen'

    if(cat=='HOUSEHOLD'):

        colorName='OrangeRed'

    for store in store_TX:

        plt.figure(figsize=(10,4))

        arrayPlot = df3_groupStoreCat[(df3_groupStoreCat["cat_id"]==cat) &

                             (df3_groupStoreCat["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

        plt.title('Интегральная продажа по магазину '+store+' и категории '+cat)

        plt.xlabel('days')

        plt.ylabel('уровень продаж')

        plt.show()
# ************ Общие графики ***********************



for cat in categorys:

    plt.figure(figsize=(15,4))

    for store in store_TX:

        arrayPlot = df3_groupStoreCat[(df3_groupStoreCat["cat_id"]==cat) &

                             (df3_groupStoreCat["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0])

        

    plt.title('Интегральная продажа по магазинам штата TX и категории '+cat)

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.legend(store_TX)

    plt.show()
# *********** Детальные графики ***********************



for cat in categorys:

    if(cat=='HOBBIES'):

        colorName='Goldenrod'

    if(cat=='FOODS'):

        colorName='YellowGreen'

    if(cat=='HOUSEHOLD'):

        colorName='OrangeRed'

    for store in store_WI:

        plt.figure(figsize=(10,4))

        arrayPlot = df3_groupStoreCat[(df3_groupStoreCat["cat_id"]==cat) &

                             (df3_groupStoreCat["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

        plt.title('Интегральная продажа по магазину '+store+' и категории '+cat)

        plt.xlabel('days')

        plt.ylabel('уровень продаж')

        plt.show()
# ************ Общие графики ***********************



for cat in categorys:

    plt.figure(figsize=(15,4))

    for store in store_WI:

        arrayPlot = df3_groupStoreCat[(df3_groupStoreCat["cat_id"]==cat) &

                             (df3_groupStoreCat["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0])

        

    plt.title('Интегральная продажа по магазинам штата WI и категории '+cat)

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.legend(store_WI)

    plt.show()
# Вопрос: нужно ли строить распределения, метрики ??
# Импорт бибилиотеки для регулярных выражений

import re



df3_sales_train_valid["item_id"] = df3_sales_train_valid["item_id"].apply(lambda x: re.sub(r"_\d_","_", x))
df3_groupeItemState = df3_sales_train_valid.groupby(['state_id','item_id']).sum()

df3_groupeItemState  = df3_groupeItemState.reset_index()

df3_groupeItemState
CA_ITEM = df3_groupeItemState[(df3_groupeItemState["item_id"]=="FOODS_001") & 

                              (df3_groupeItemState["state_id"]=="CA")]

CA_ITEM
items = ['FOODS_001']
# *********** Детальные графики ***********************



for state in states:

    colorName = 'SteelBlue'

    for item in items:

        plt.figure(figsize=(10,4))

        arrayPlot = df3_groupeItemState[(df3_groupeItemState["item_id"]==item) &

                             (df3_groupeItemState["state_id"]==state)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

        plt.title('Интегральная продажа товара '+item+' по штату '+state)

        plt.xlabel('days')

        plt.ylabel('уровень продаж')

        plt.show()
# ************ Общие графики ***********************



for item in items:

    plt.figure(figsize=(15,4))

    for state in states:

        arrayPlot = df3_groupeItemState[(df3_groupeItemState["item_id"]==item) &

                             (df3_groupeItemState["state_id"]==state)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0])

        

    plt.title('Интегральная продажа товара '+item+' по штатам')

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.legend(['CA','TX','WI'])

    plt.show()
df3_groupItemStore = df3_sales_train_valid.groupby(['store_id','item_id']).sum()

df3_groupItemStore  = df3_groupItemStore.reset_index()

df3_groupItemStore
store_CA = ['CA_1', 'CA_2', 'CA_3', 'CA_4']

store_TX = ['TX_1', 'TX_2', 'TX_3']

store_WI = ['WI_1', 'WI_2', 'WI_3']
# *********** Детальные графики ***********************



for store in store_CA:

    colorName = 'SteelBlue'

    for item in items:

        plt.figure(figsize=(15,2))

        arrayPlot = df3_groupItemStore[(df3_groupItemStore["item_id"]==item) &

                             (df3_groupItemStore["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

        plt.title('Интегральная продажа товара '+item+' по магазину '+store)

        plt.xlabel('days')

        plt.ylabel('уровень продаж')

        plt.show()
# ************ Общие графики ***********************



for item in items:

    plt.figure(figsize=(15,4))

    for store in store_CA:

        arrayPlot = df3_groupItemStore[(df3_groupItemStore["item_id"]==item) &

                             (df3_groupItemStore["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0])

        

    plt.title('Интегральная продажа товара '+item+' по магазинам CA')

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.legend(store_CA)

    plt.show()
# *********** Детальные графики ***********************



for store in store_TX:

    colorName = 'Coral'

    for item in items:

        plt.figure(figsize=(15,2))

        arrayPlot = df3_groupItemStore[(df3_groupItemStore["item_id"]==item) &

                             (df3_groupItemStore["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

        plt.title('Интегральная продажа товара '+item+' по магазину '+store)

        plt.xlabel('days')

        plt.ylabel('уровень продаж')

        plt.show()
# ************ Общие графики ***********************



for item in items:

    plt.figure(figsize=(15,4))

    for store in store_TX:

        arrayPlot = df3_groupItemStore[(df3_groupItemStore["item_id"]==item) &

                             (df3_groupItemStore["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0])

        

    plt.title('Интегральная продажа товара '+item+' по магазинам TX')

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.legend(store_TX)

    plt.show()
# *********** Детальные графики ***********************



for store in store_WI:

    colorName = 'LightSeaGreen'

    for item in items:

        plt.figure(figsize=(15,2))

        arrayPlot = df3_groupItemStore[(df3_groupItemStore["item_id"]==item) &

                             (df3_groupItemStore["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

        plt.title('Интегральная продажа товара '+item+' по магазину '+store)

        plt.xlabel('days')

        plt.ylabel('уровень продаж')

        plt.show()
# ************ Общие графики ***********************



for item in items:

    plt.figure(figsize=(15,4))

    for store in store_WI:

        arrayPlot = df3_groupItemStore[(df3_groupItemStore["item_id"]==item) &

                             (df3_groupItemStore["store_id"]==store)]

        plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0])

        

    plt.title('Интегральная продажа товара '+item+' по магазинам WI')

    plt.xlabel('days')

    plt.ylabel('уровень продаж')

    plt.legend(store_WI)

    plt.show()
dataset = df3_sales_train_valid[days[1:]].iloc[3]

print(dataset)
def moving_average(series, n):

    return np.average(series[-n:])



moving_average(dataset, 24)
df3_sales_train_valid[:4]
def plotMovingAverage(series, n):



    """

    series - dataframe with timeseries

    n - rolling window size 



    """



    rolling_mean = series.rolling(window=n).mean()



    # При желании, можно строить и доверительные интервалы для сглаженных значений

    #rolling_std =  series.rolling(window=n).std()

    #upper_bond = rolling_mean+1.96*rolling_std

    #lower_bond = rolling_mean-1.96*rolling_std



    plt.figure(figsize=(15,5))

    plt.title("Moving average\n window size = {}".format(n))

    plt.plot(range(1, 1914),rolling_mean, "g", label="Rolling mean trend", color="red")



    #plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")

    #plt.plot(lower_bond, "r--")

    plt.plot(dataset[n:], label="Actual values")

    plt.legend(loc="upper left")

    plt.grid(True)



plotMovingAverage(dataset, 24*7) # сглаживаем по неделям
plotMovingAverage(dataset, 24) # сглаживаем по дням
# Вывод продаж за весь период для произвольного товара

days = ["id"]

for i in range(1, 1914):

    days.append("d_"+str(i))



# plt.scatter(range(1, 1914), df3_sales_train_valid[days[1:]].iloc[3])

plt.figure(figsize=(15,4)) 

plt.plot(range(1, 1914), df3_sales_train_valid[days[1:]].iloc[3])

plt.title('товар HOBBIES_1_004_CA_1_validation')

plt.xlabel('период продаж')

plt.ylabel('число продаж за день')

plt.show()


# скользящая средняя



rolling = dataset.rolling(24).mean()# сглаживание по дням

plt.figure(figsize=(15,4)) 

plt.plot(range(1, 1914), df3_sales_train_valid[days[1:]].iloc[3],alpha=0.5)

plt.plot(range(1, 1914),rolling,"g",color='red')

plt.xlabel('период продаж')

plt.ylabel('число продаж за день')

plt.show()
rolling = dataset.rolling(24*7).mean()# сглаживание по неделям

plt.figure(figsize=(15,4)) 

plt.plot(range(1, 1914), df3_sales_train_valid[days[1:]].iloc[3],alpha=0.5)

plt.plot(range(1, 1914),rolling,"g",color='red')

plt.xlabel('период продаж')

plt.ylabel('число продаж за день')

plt.show()
# Экспоненциальное сглаживание, модель Хольта-Винтерса



def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series

    for n in range(1, len(series)):

        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return result



with plt.style.context('seaborn-white'):    

    plt.figure(figsize=(20, 8))

    plt.plot(dataset.values, label = "Actual",color='gray',alpha=0.7)

    for alpha in [0.3, 0.05]:

        plt.plot(exponential_smoothing(dataset, alpha), label="Alpha {}".format(alpha))

    plt.legend(loc="best")

    plt.axis('tight')

    plt.title("Exponential Smoothing")

    plt.grid(True)
arrayPlot = df3_groupStoreCat[(df3_groupStoreCat["cat_id"]=='FOODS') & (df3_groupStoreCat["store_id"]=='CA_1')]

plt.figure(figsize=(10,4))

plt.plot(range(1, 1914), arrayPlot[days[1:]].iloc[0],color=colorName)

plt.title('Интегральная продажа по магазину CA_1 и категории FOODS')

plt.xlabel('days')

plt.ylabel('уровень продаж')

plt.show()
del arrayPlot["cat_id"]

del arrayPlot["store_id"]

new_arrayPlot = arrayPlot.transpose()

atcr = []

for i in range(64):

    atcr.append(new_arrayPlot[0].autocorr(lag=i))

for i in range(10):

    plt.plot(7*i+0*np.array(range(2)), -0.25+1.25*np.array(range(2)), color ="black")

plt.plot(range(64), atcr)

plt.show()
#del arrayPlot["cat_id"]

#del arrayPlot["store_id"]

new_arrayPlot = arrayPlot.transpose()

atcr = []

vol=4

z_t = new_arrayPlot.copy()

z_t = z_t.reset_index()

del z_t["index"]

z_t_k = z_t.iloc[k:].copy()

z_t_k = z_t_k.reset_index()

del z_t_k["index"]

for k in range(vol):

    z_prime_t = z_t.copy()

    z_prime_t_k = z_t_k.copy()

    for i in range(1,k):

        divisor = 2**(15+2*i)

        z_t_i = z_t.iloc[i:].copy()

        z_prime_t -= np.dot(z_t.iloc[:-i].transpose()/divisor, z_t_i/divisor)/np.dot(z_t_i.transpose()/divisor, z_t_i/divisor)*z_t_i

        if i==2 and k==3:

            print(len(z_t_i.iloc[:-k+i]), len(z_t_k))

            print(pd.concat([z_t_i.iloc[:-k+i],z_t_k],axis=1))

            print(k, i, np.dot(z_t_k.transpose()/divisor, z_t_i.iloc[:-k+i]/divisor), np.dot(z_t_i.transpose()/divisor, z_t_i/divisor))

        z_prime_t_k -= np.dot(z_t_k.transpose()/divisor, z_t_i.iloc[:-k+i]/divisor)/np.dot(z_t_i.transpose()/divisor, z_t_i/divisor)*z_t_i

    z_prime_t_k["t_k"] = z_prime_t_k

    del z_prime_t_k[0]

    z_prime_t["t"] = z_prime_t

    del z_prime_t[0]

    cor_df = pd.concat([z_prime_t_k, z_prime_t], axis=1)

    cor_df = cor_df.dropna()

    atcr.append(cor_df.corr()["t_k"].loc["t"])

    print(atcr)

plt.plot(range(vol), atcr)

plt.show()
week_arrayPlot = new_arrayPlot.rolling(7).mean()

atcr = []

for i in range(365):

    atcr.append(week_arrayPlot[0].autocorr(lag=i))

plt.plot(range(365), atcr)

for i in range(13):

    plt.plot(30*i+(i+1)%2*(i<=5)+i%2*(i>=5)+0*np.array(range(2)), -0.1+1.1*np.array(range(2)), color ="black")

#plt.grid()

plt.show()
month_arrayPlot = new_arrayPlot.rolling(30).mean()

atcr = []

for i in range(360):

    atcr.append(month_arrayPlot[0].autocorr(lag=i))

plt.plot(range(360), atcr)

plt.grid()

plt.show()
month_arrayPlot = new_arrayPlot.rolling(7).mean().rolling(30).mean()

atcr = []

for i in range(1900):

    atcr.append(month_arrayPlot[0].autocorr(lag=i))

plt.plot(range(1900), atcr)

plt.plot(365+0*np.array(range(2)), -1+2*np.array(range(2)), color ="black")

plt.plot(365*2+0*np.array(range(2)), -1+2*np.array(range(2)), color ="black")

plt.plot(365*3+0*np.array(range(2)), -1+2*np.array(range(2)), color ="black")

plt.plot(365*4+0*np.array(range(2)), -1+2*np.array(range(2)), color ="black")

plt.plot(365*5+0*np.array(range(2)), -1+2*np.array(range(2)), color ="black")

plt.grid()

plt.show()
# from numpy import corcoef, dot



# #провести тест на значимость коэффициента корреяции

# corcoef(df3_sales_train_valid.loc[i,:],df3_sales_train_valid.loc[j,:])

# #|cor/sqrt(1-corr^2)*sqrt(len(days)-2)|>1.96

# # В R вызывется       qt(0.975, df=n-2)



# #провести кластеризацию испотльзуя угловое расстояние

# #acos(np.dot(x,y)/math.sqrt(np.dot(x,x)*np.dot(y,y)))