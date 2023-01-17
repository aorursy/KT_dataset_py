import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/plant-1-generation-data/Plant_1_Generation_Data.csv',
                  delimiter=',',
                  header=0)
data
data = pd.read_csv('../input/plant-1-generation-data/Plant_1_Generation_Data.csv',
                  delimiter=',',
                  header=1)
data
data = pd.read_csv('../input/plant-1-generation-data/Plant_1_Generation_Data.csv',
                      delimiter=',',
                      header=0)
data.head(10)
data.info()
#get sample values
data.sample(n=5)
#handle missing values
data = data.fillna(np.nan)
data
#sort in descending order giving 10 records
data.sort_values(by='DC_POWER', ascending=False)[:10]
data['AC_POWER'].value_counts()
data['AC_POWER'].value_counts()[0]
add = int(2+3)
add
div = int(2/3)
divfloat= float(2/3)
print(div)
print(divfloat)
#grouping data
group_by_SOURCE_KEY = data.groupby(['SOURCE_KEY'])
print('General output',group_by_SOURCE_KEY)
group_by_SOURCE_KEY.size()
data.info()
#format the data
data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'],format='%d-%m-%Y %H:%M')
data.info()
np.dtype('datetime64[ns]') == np.dtype('<M8[ns]')
data.head()
#lambda arguments : expression

#A lambda function that adds 10 to the number passed in as an argument, and print the result:
x = lambda a,c,b: a +b+c + 10
print(x(5,2,3))

#normal function
def add(a,b,c):

     return a + b +c

#when i call this function with arguments it will print the addition of the two numbers

print(add(10,5,6))

#A lambda function that multiplies argument a with argument b and print the result:
x = lambda a, b : a * b
print(x(5, 6))



#The power of lambda is better shown when you use them as an anonymous function inside another function.

#Say you have a function definition that takes one argument, and that argument will be multiplied with an unknown number:
def myfunc(n):
  return lambda a : a * n

multitwo = myfunc(2)

print(multitwo(6))

#write multithree function
#Create a lambda function that takes one parameter (a) and returns it.
x=lambda a:a
x(4)
#lamda function
df=pd.DataFrame({
    'id':[1,2,3,4,5],
    'name':['Jeremy','Frank','Janet','Ryan','Mary'],
    'age':[20,25,15,10,30],
    'income':[4000,7000,200,0,10000]
})
df
#Let’s say we have an error in the age variable. We recorded ages with a difference of 3 years. 
#So, to remove this error from the Pandas dataframe, we have to add three years to every person’s age. 
#We can do this with the apply() function in Pandas.
df['age']=df.apply(lambda x: x['age']+3,axis=1)
df
df['age']=df['age'].apply(lambda x: x+3)
df
list(filter(lambda x: x>21,df['age']))
#lamda function
lambda x:x+3

#extract year, month, date and time using Lambda.
import datetime
now = datetime.datetime.now()
print('The date and time crrently are',now)
year = lambda x: x.year
month = lambda x: x.month
day = lambda x: x.day
time = lambda x: x.time()
hour = lambda x: x.hour
Min = lambda x: x.minute
sec = lambda x: x.second
print(year(now))
print(month(now))
print(day(now))
print(time(now))
print(hour(now))
print(Min(now))
print(sec(now))
#split data and time (e.g. if we want to group all entries for a date.)
data['DATE'] = data['DATE_TIME'].apply(lambda x:x.date())
data['TIME'] = data['DATE_TIME'].apply(lambda x:x.time())

data.info()
data.head()
data['DATE'] = pd.to_datetime(data['DATE'],format='%Y-%m-%d')
data['HOUR'] = pd.to_datetime(data['TIME'], format='%H:%M:%S').dt.hour
data['MINUTES'] = pd.to_datetime(data['TIME'], format='%H:%M:%S').dt.minute
data.info()
data.tail()
#condition
new_df = data[(data['DC_POWER']>40) & (data['AC_POWER']>30)]
new_df
len(data['SOURCE_KEY'].unique())
len(data['DATE'].unique())
#We have data over a month (34 days to be precise) - staring on the 15th of May and going on till the 17th of June.
start_date = data.sort_values(['DATE']).head()
end_date = data.sort_values(['DATE']).tail()
print(start_date)
print(end_date)
#Average readings per invertor
# 22 inverters, 34 days, 24 hours, 4 (every 15 minutes)readings per hour
34 * 24 * 4
#how many entries per day per invertor/source key
df = data.head(100)
dataframe = pd.DataFrame(df, columns = ['SOURCE_KEY','DATE','AC_POWER','TIME']) 
reading_per_day = dataframe[df['SOURCE_KEY'] == 'uHbuxQJl8lW7ozc']
reading_per_day
#reading_per_day = dataframe.groupby(['SOURCE_KEY'])
#reading_per_day.size()
#no of entries per day
#We have about 2000 entries every day.
data['DATE'].value_counts()
# Total no of entries per day including all invertors
#22 total SOURCE KEYS
#data colected every 15 minutes 
#for the duration of 24 hours
22*4*24
#check for missing data
data[data['SOURCE_KEY']=='uHbuxQJl8lW7ozc'][data['HOUR']==7]['DATE'].value_counts()

count = data[data['SOURCE_KEY']=='uHbuxQJl8lW7ozc'][data['HOUR']==7]['DATE'].value_counts()
count.describe()
#no of entries per day
print('no of entries per day',24*4)
#check the entries for 27th
data[data['DATE']=='2020-05-27']['SOURCE_KEY'].value_counts()
data[data['DATE']=='2020-05-27'][data['HOUR']==5]['SOURCE_KEY'].value_counts()
#plotting
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(data.DATE_TIME,
        data.AC_POWER.rolling(window=20).max(),
        label='AC'
       )

ax.plot(data.DATE_TIME,
        data.DC_POWER.rolling(window=20).max(),
        label='DC'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC_POWER and DC_POWER over 34 Days')
#plt.xlabel('AC_POWER')
#plt.ylabel('DC_POWER')
plt.show()

#sorting = data.sort_values(by=['AC_POWER','DC_POWER'],ascending=False)
DATE1 = data['DATE'] == '2020-05-29'

data[DATE1].sort_values(by=['AC_POWER','DC_POWER'],ascending=False)