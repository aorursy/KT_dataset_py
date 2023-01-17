# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import seaborn as sns

from datetime import datetime, timedelta

from sklearn import linear_model
%matplotlib inline

import matplotlib.pyplot as plt
train = pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])

train.shape
fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(16,5))

ax[0].hist(train['logerror'],bins=100)

ax[0].set_title('Full range or logerror',fontsize=18,fontweight='bold')

ax[1].hist(train['logerror'],bins=100,range=(-.6,.6))

ax[1].set_title('Zoom in',fontsize=18,fontweight='bold')

plt.tight_layout()
print('There is a slight right shift in the data...')

median = np.percentile(train['logerror'],50)

print('* The median value is: {:.3f}'.format(median))

above_zero = (train['logerror']>0).mean()

print('* Values above zero: {:.0%}'.format(above_zero))

range95 = np.percentile(train['logerror'],[2.5,97.5])

print('* 95% of values are between: {:.2f}'.format(range95[0]) + ' and {:.2f}'.format(range95[1]))
def get_zest_price(sales_price,log_error):

    log_sales_price  = np.log10(sales_price)

    log_zest_price = log_sales_price + log_error

    return 10**log_zest_price

    

def get_delta_error(sales_price, log_error):

    zest_price = get_zest_price(sales_price, log_error)

    return zest_price - sales_price



def get_percent_diff(sales_price, log_error):

    delta = get_delta_error(sales_price, log_error)

    return float(delta) / sales_price



def print_stories(logerror):

    sales_points = [150000, 250000, 500000, 750000, 1000000]

    print('an absolute log error of ' + str(logerror) +' is in the {:.0%}'.format((abs(train['logerror']) <=logerror).mean())+ ' percentile of error, which means...')

    for sales_price in sales_points:

        zest_price = get_zest_price(sales_price, logerror)

        delta = zest_price - sales_price

        print('a sales price of ${:,.0f}'.format(sales_price) +  ' means the zestimate was ${:,.0f}'.format(zest_price) + ', with the difference of ${:,.0f}'.format(delta))
plt.figure(figsize=(16,7))

values = np.linspace(-.2,.2,100)

plt.plot(values, [get_percent_diff(100000, le) for le in values])

yticks = np.linspace(-.7,.7,15)

plt.yticks(yticks,['{:.0%}'.format(yt) for yt in yticks])

plt.ylabel('% error (Zestimat - sales price)',fontsize=16)

xticks = np.linspace(-.2,.2,17)

plt.xticks(xticks)

plt.xlabel('logerror',fontsize=16)

plt.tight_layout()
# low error of .005

print('Low error')

print('-----------')

print_stories(.005)

print('#####################')

print(' ')



# median error of .0325

print('median error')

print('-----------')

print_stories(.0325)

print('#####################')

print(' ')





# high error of .2

print('high error')

print('-----------')

print_stories(.2)

print('#####################')

print(' ')
# I want to trend weekday sales because of low activity on weekends

train['WeekDay'] = [d.weekday()<5 for d in train['transactiondate']]



piv = pd.pivot_table(train[train['WeekDay']==True], index='transactiondate',values='logerror',aggfunc=[np.size,np.sum,np.median])
plt.figure(figsize=(16,5))

plt.title('Daily (weekday) homesales volume',fontsize=20,fontweight='bold' )

plt.plot_date(piv.index,piv['size'],alpha=.6)

plt.plot_date(piv.index,piv['size'].rolling(window=30).mean(),'r-',linewidth=5)

plt.tight_layout()
piv_all = pd.pivot_table(train, index='transactiondate',values='logerror',aggfunc=[np.size,np.sum])

piv_restricted = pd.pivot_table(train[abs(train['logerror'])<.2], index='transactiondate',values='logerror',aggfunc=[np.size,np.sum])
plt.figure(figsize=(16,5))

rolling_avg_all = piv_all['sum'].rolling(window=30).sum() / piv_all['size'].rolling(window=30).sum()

rolling_avg_rest = piv_restricted['sum'].rolling(window=30).sum() / piv_restricted['size'].rolling(window=30).sum()

plt.plot_date(piv_all.index,rolling_avg_all,label = '30 Day rolling Avg: All values')

plt.plot_date(piv_restricted.index,rolling_avg_rest,label = '30 Day rolling Avg: No Outliers')

plt.ylim(0,0.03)

plt.legend(loc=0,fontsize=16)

plt.title('Logerror secular trends', fontsize=20,fontweight='bold')

plt.tight_layout()
train_pre_oct_16 = train[(train['transactiondate']<datetime(2016,10,16))]

piv_pre = pd.pivot_table(train_pre_oct_16, index='transactiondate',values='logerror',aggfunc=[np.size,np.sum])

rolling_avg_all_pre = piv_pre['sum'].rolling(window=30).sum() / piv_pre['size'].rolling(window=30).sum()

scatter_df = pd.DataFrame(list(zip(piv_pre['size'].rolling(window=30).mean(),rolling_avg_all_pre)),columns=['Rolling size','Rolling avg'])

scatter_df = scatter_df[scatter_df['Rolling size'].notnull()]


sns.jointplot("Rolling size", "Rolling avg", data=scatter_df, kind='scatter',

                  xlim=(0,600), ylim=(0,.03),color="r", size=7)



plt.ylabel('Logerror')

plt.xlabel('Sales volume ')

plt.tight_layout()
X_input = np.array([[v] for v in scatter_df['Rolling size'].values])

y_input = np.array(scatter_df['Rolling avg'].values)

regr = linear_model.LinearRegression()

regr.fit(X_input, y_input)
plt.figure(figsize=(8,8))

plt.scatter(X_input,y_input)

plt.plot(X_input, regr.predict(X_input), color='blue',

         linewidth=3)

plt.xlabel('Daily Sales volume')

plt.ylabel('Log error')

plt.title('Modeling log error by daily sales volume', fontsize=18, fontweight='bold')

plt.text(225,.025,'change of {:,.3f}'.format(regr.coef_[0]* 100)+' for ever 100 additional daily sales')

plt.tight_layout()