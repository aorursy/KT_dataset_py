%matplotlib inline

%config InlineBackend.figure_format = 'svg' # svg output 
import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')
ts = pd.date_range('1/1/2020', periods = 366) # generate 2020 all year data, sum up 366 days

ts = pd.Series(ts).apply(str) # convert Datetime into string

new = ts.str.split(" ", n = 1, expand = True) # extract YYYY-MM-DD date 

ts = new[0]

ts
month = ts.apply(lambda x: int(x[5:7])) # slicing month from YYYY-MM-DD format

mday = ts.apply(lambda x: int(x[8:10]))



import datetime

def week_day(date):

    '''generate week days from YYYY-MM-DD format

    '''

    year, month, day = (int(x) for x in date.split('-'))   

    answer = datetime.date(year, month, day).weekday() 

    answer = int(answer) + 1

    return answer



weekdays = ts.apply(week_day)

weekdays
df = pd.DataFrame({'date_time': ts, 'weekdays':weekdays,

              'mday':mday,'month': month,

              'val': np.random.randn(366)}) # val is random numbers



num_weekdays = {1:'Mon',2:'Tue',3: 'Wed',4:'Thu',5 :'Fri',6:'Sat', 7 : 'Sun'} # map between number and weekdays

num_mont = {1:'Jan',2:'Feb',3:'Mar',4:'Apr', 5: 'May',6:'Jun', 7:'Jul', 8:'Aug',

            9:'Sep',10:'Oct',11:'Nov', 12:'Dec'}



df['weekdays'] = df['weekdays'].map(num_weekdays)

df['month'] = df['month'].map(num_mont)



df
wdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] # set factor order level

wdays.reverse() # order reverse 

mont = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sep','Oct','Nov','Dec']



def recode_ordered(array,level):

    '''recode string data to ordered factors

    '''

    cate = pd.api.types.CategoricalDtype(categories=level, ordered= True)

    array = array.astype(cate)

    return array



df['month'] = recode_ordered(df['month'], mont)

df['weekdays'] = recode_ordered(df['weekdays'], wdays)
from math import ceil



def week_of_month(date):

    '''Returns the week of the month for the specified date.

    '''

    year, month, day = (int(x) for x in date.split('-'))   

    dt = datetime.date(year, month, day)

    first_day = dt.replace(day=1)

    adjusted_dom = dt.day + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))



df['monthweek'] =  df['date_time'].apply(week_of_month)

df   
from plotnine import * # use python's ggplot2

(ggplot(df, aes('monthweek','weekdays',fill = 'val')) +

  geom_tile(color ='gray') +

  geom_text(aes(label = 'mday'),size=5,color = 'black')+

  facet_wrap('~month' ,nrow=3) +

  #scale_fill_gradient(low="red", high="yellow") +

  labs(title = 'Record',x='week of month',y = ' ',fill = 'Times') + 

  scale_x_continuous() 

)
df.to_csv('df.csv', index=False, header=True)