# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.neighbors import LocalOutlierFactor
import scipy

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import time
import datetime

print(os.listdir("../input"));
csv = pd.read_csv('../input/passflow-w.csv',parse_dates=['txdate']);
csv = csv.sort_values(by='txdate');


# Get data
p = csv.loc[(csv.txdate>='2013-07-01')&(csv.txdate<='2013-12-31')];
p_p = csv.loc[(csv.txdate>='2014-01-01')&(csv.txdate<='2014-01-31')];
p.index = np.arange(len(p.index))
m = np.zeros(len(p.txdate));
for i in range(len(p.txdate)):
    r = p.txdate[i];
    m[i]=r.month
    
tmax = int(max(m));
tmin = int(min(m));
ymax = np.max(p.passflow)*1.1;
ymin = np.min(p.passflow)*0.8;

p['month'] = m;
t1 = tmax-tmin+1;
if(t1%2==1):
    t2 = int(t1/2+1);
else:
    t2 = int(t1/2);
# Any results you write to the current directory are saved as output.
print(p.head(10))
print(p.dtypes)
#Find Outlier plot
# get outlier from 1.5 quartile from q2
'''
r = p.passflow.sort_values();
a = np.percentile(r, [25, 50, 75]);
q3 = 1.8*(a[2]-a[1])+a[1];
q1 = a[1]-1.8*(a[1]-a[0]);
p_outlier = p.loc[(p.passflow<=q1)|(p.passflow>=q3)];
'''
# outlier : returns true for all elements more than three standard deviations from the mean.
pstd = 3*np.std(p.passflow);
pmean = np.mean(p.passflow);
p_outlier = p.loc[(p.passflow<=(pmean-pstd))|(p.passflow>=(pmean+pstd))];
r = np.array(p_outlier.index);
outlier = p.loc[r];

fig = plt.figure(figsize=(14,6));
plt.plot(p.index,p.passflow,label='linear');
plt.scatter(outlier.index,outlier.passflow,marker='x',c = 'red');
plt.title("PassngerFlow with outliner")
plt.xlabel('Days');
plt.ylabel('PassngerFlow');
plt.xticks()
plt.grid();
plt.show();
#Smooth and fitting data
psmooth = scipy.ndimage.filters.gaussian_filter(p.passflow,4);
fig = plt.figure(figsize=(14,6));
plt.plot(p.passflow,':',c='gray',alpha=0.5)
plt.plot(psmooth,color='red')
plt.xlabel('Days');
plt.ylabel('PassngerFlow')
plt.title('PassngerFlow from smoothdata');
#The trend of passnger flow for every weekday
def getweekday(num):
    if num == 1:
        days = 'Monday'
    elif num ==2:
        days = 'Tuesday'
    elif num ==3:
        days = 'Wednesday'
    elif num ==4:
        days = 'Thursday'
    elif num ==5:
        days = 'Friday'
    elif num ==6:
        days = 'Saturday'        
    elif num ==7:
        days = 'Sunday'        
    return days;
fig = plt.figure(figsize=(14,14));
ymax = np.max(p.passflow)*1.1;
ymin = np.min(p.passflow)*0.8;
xmax = int(len(p.txdate)/7+2);
for i in range(1,7):
    r = p.loc[p.weekday==i];
    plt.subplot(7,1,i)
    plt.plot(np.array(r.passflow));
    plt.title(getweekday(i));
    plt.ylim(ymin,ymax);
    plt.xlim(0,xmax);
    plt.grid(drawstyle='steps-mid');
    plt.tight_layout(1)
#The trend of passnger flow for every month
def getmonth(num):
    if num == 1:
        month = 'January'
    elif num ==2:
        month = 'February'
    elif num ==3:
        month = 'March'
    elif num ==4:
        month = 'April'
    elif num ==5:
        month = 'May'
    elif num ==6:
        month = 'June'        
    elif num ==7:
        month = 'July' 
    elif num ==8:
        month = 'August'
    elif num ==9:
        month = 'September'
    elif num ==10:
        month = 'October'
    elif num ==11:
        month = 'November'
    elif num ==12:
        month = 'December'
    return month;
fig = plt.figure(figsize=(14,14));

m=1
for i in np.arange(tmin,tmax+1):
    plt.subplot(t2,2,m)
    p2 = p.loc[p.month==i];
    plt.plot(p2.passflow);
    plt.title(getmonth(i));
    plt.ylim(ymin,ymax);
    plt.grid();
    m = m+1;
# Normal Distribution
# X is passnger flow , y is number of days.
fig = plt.figure(figsize=(14,14));
m=1
for i in np.arange(tmin,tmax+1):
    plt.subplot(t2,2,m)
    p2 = p.loc[p.month==i];
    nd = np.array(p2.passflow);
    plt.hist(p2.passflow,bins=6,color='b',alpha = 0.2,histtype = 'barstacked');
    plt.title(getmonth(i));
    plt.xlim(ymin,ymax*0.9);
    plt.grid();
    m = m+1;
plt.show();
fig = plt.figure(figsize=(14,14));
m=1
for i in np.arange(tmin,tmax+1):
    plt.subplot(t2,2,m)
    p2 = p.loc[p.month==i];
    nd = np.array(p2.passflow);
    plt.hist(np.diff(p2.passflow,1),bins=6,color='b',alpha = 0.2,histtype = 'barstacked');
    plt.title(getmonth(i));
    plt.grid();
    m = m+1;
plt.show();
