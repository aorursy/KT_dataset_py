# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import kagglegym

import random

from sklearn import ensemble, linear_model, metrics

import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA 



import seaborn as sns

%matplotlib inline

plt.style.use('classic')



sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
env = kagglegym.make()

o = env.reset()

train = o.train  #train dataset - the first half of the full dataframe

print(train.shape) #print shape
#Courtesy Jeff Moser https://www.kaggle.com/jeffmoser/two-sigma-financial-modeling/kagglegym-api-overview

with pd.HDFStore("../input/train.h5", "r") as train1:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train1.get("train") #df is the complete dataframe
def findMatchedColumnsUsingPrefix(prefix, df):

    columns = df.columns[df.columns.str.startswith(prefix)]

    return list(columns.values)
derived_columns = findMatchedColumnsUsingPrefix("derived", df)

fundamental_columns = findMatchedColumnsUsingPrefix("fundamental", df)

technical_columns = findMatchedColumnsUsingPrefix("technical", df)



print("There are {} derived columns".format(len(derived_columns)))

print("There are {} fundamental columns".format(len(fundamental_columns)))

print("There are {} technical columns".format(len(technical_columns)))
#Thanks to Chase:

#https://www.kaggle.com/chaseos/two-sigma-financial-modeling/understanding-id-and-timestamp

# id counts w.r.t time

temp = train.groupby('timestamp').apply(lambda x: x['id'].nunique())#Also can use count() 

len(train)

#as we know the

#id is unique for a timstamp

plt.figure(figsize=(8,4))

plt.plot(temp, color="red")

plt.xlabel('timestamp')

plt.ylabel('id count')

plt.title('Number of ids over time')
# lifespan of each id

temp = train.groupby('id').apply(len)

temp = temp.sort_values()

temp = temp.reset_index()

plt.figure(figsize=(8,4))

plt.plot(temp[0], color="blue")

plt.xlabel('index for each id sorted by number of timestamps')

plt.ylabel('number of timestamps')

plt.title('Number of timestamps ("Lifespan") for each id')

print(temp[0].describe())
byTS=train.pivot(index='timestamp', columns='id', values='y')

byTS.fillna(0,inplace=True)

byTS

#Reset index so that 'timestamp' is a column 

byTS.reset_index(level=0,inplace=True)

byTS.timestamp
datestart = '2010-01-01'

dateend = '2016-12-13'

timeperiods = len(byTS)





dayse=pd.date_range(datestart, freq='B', periods=timeperiods)

dayse
byTS.datetime = pd.to_datetime(dayse,unit='B',errors='coerce')

byTS
TSdict={}



#history = [x for x in train]

for pos,col in enumerate(byTS.columns[1:]):

    ps = pd.Series(byTS[col].values,index=byTS.datetime)

    #TS.append(ps)

     

    TSdict[int(col)] = ps

TSdict

TSdict[25]
from statsmodels.tsa import stattools as stt 

def is_stationary(df, maxlag=15, autolag=None, regression='ct'): 

    """Run the Augmented Dickey-Fuller test from Statsmodels 

    and print output. 

    """ 

    outpt = stt.adfuller(df,maxlag=maxlag, autolag=autolag, 

                         regression=regression) 

    print('adf\t\t {0:.3f}'.format(outpt[0])) 

    print('p\t\t {0:.3g}'.format(outpt[1])) 

    print('crit. val.\t 1%: {0:.3f}, 5%: {1:.3f}, 10%: {2:.3f}'.format(outpt[4]["1%"], outpt[4]["5%"], outpt[4]["10%"])) 

    print('stationary?\t {0}'.format(['true', 'false'][outpt[0]>outpt[4]['5%']])) 

    return outpt 
TS=TSdict[25]

diff1=is_stationary(TS.diff(1).dropna())        



diff1
diff12=is_stationary(TS.diff(1).diff(12).dropna());

diff12
import seaborn as sns

TS.diff(1).plot(label='1 period', title='Y Values', 

                      dashes=(15,5)) 

TS.diff(1).diff(12).plot(label='1 and 12 period(s)', 

                               color='Coral') 

plt.legend(loc='best')

ax=plt.gca()

plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

ax.locator_params(axis='y', nbins=5)

sns.despine() 

plt.xlabel('Date') 