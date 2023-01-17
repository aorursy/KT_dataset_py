# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/smart-meters-in-london/hhblock_dataset/hhblock_dataset/block_31.csv')
base_dir = '/kaggle/input/smart-meters-in-london/'
house = pd.read_csv(base_dir+'informations_households.csv')
house_ids = house[house['Acorn']=='ACORN-B']['LCLid']
files = house[house['Acorn']=='ACORN-B']['file']
blocks = files.unique()

files = pd.DataFrame(columns=data.columns)
for file in blocks:
    file_a = pd.read_csv(base_dir + 'hhblock_dataset/hhblock_dataset/{}.csv'.format(file))
    files = pd.concat([files,file_a])

files = files[files['LCLid'].isin(house_ids)]
files.reset_index()
date_test = '2012-05-02'

from datetime import datetime

dt = datetime.strptime(date_test, "%Y-%m-%d")
dt.month

from datetime import datetime

files['month'] = [datetime.strptime(i, "%Y-%m-%d").month for i in files['day']]
files['year'] = [datetime.strptime(i, "%Y-%m-%d").year for i in files['day']]

#combining into 1hr removing other data
col_names = ['hour_{}'.format(i) for i in range(0,24)]

for ind, col in enumerate(col_names):
    files[col] = files['hh_{}'.format(ind*2)] + files['hh_{}'.format(ind*2+1)]

to_drop = files.columns[[range(2,50)]]
files = files.drop(columns=to_drop)
files
files[files['month']==12]['hour_12'].notna()
#return best distribution

import warnings
import scipy.stats as st
import statsmodels as sm

def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    
#     DISTRIBUTIONS = [        
#         st.gamma,st.lognorm,st.norm,st.powerlognorm,st.gumbel_l,
#         st.gumbel_r, st.beta, st.rayleigh   
#     ]
    DISTRIBUTIONS = [        
        st.gamma,st.lognorm,st.norm,st.powerlognorm,
        st.beta, st.rayleigh   
    ]
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)
                print(distribution, params)
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass
    best_params = [i for i in best_params]
    return (best_distribution.name, best_params)

import random

house_list = []
num_house  = 20
while len(house_list)<num_house:
    housenow = random.choice(list(house_ids))
    if housenow not in house_list:
        file1 = files[files['LCLid'] == housenow]
        file2 = file1[file1['year']==2013]
        if len(file2) > 360:
            house_list.append(housenow)
            
#files[files['LCLid'].isin(house_list)]
col_hour = files.columns[[range(4, 28)]]

name = []
dist_names = []
dist_params = []

house_data = files[files['LCLid'].isin(house_list)]
house_data = house_data[house_data['year']==2013]
for i in range(1,13):
    #filter data per month
    focused_data = house_data[house_data['month']==i]
    for hour in col_hour:
        data_here = focused_data[focused_data[hour].notna()][hour] #remove na points
        #if sum(data_here.isna()) == 0: ##change to remove na points
        row_name = str(i) + '_' + hour
        name.append(row_name)
        dist_name, param = best_fit_distribution(data_here)
        dist_names.append(dist_name)
        dist_params.append(param)

        print(len(dist_names))
col_hour = files.columns[[range(4, 28)]]


for i in range(1,13):
    #filter data per month
    focused_data = files[files['month']==i]
    for hour in col_hour:
        data_here = focused_data[focused_data[hour].notna()][hour]
        
        #print(sum(data_here.isna()))
dist_params
data_dist = pd.DataFrame({'confid':name,'dist_name':dist_names, 'params':dist_params})
data_dist
data_dist.to_csv('mycsvfile.csv',index=False)
data_dist.iloc[34]
# data_dist = pd.read_csv('/kaggle/input/datadist1/data_dist_20_B.csv')
# data_dist
#
from scipy import stats

data_dict = {}
for i in range(1,13):
    #filter data per month
    focused_data = files[files['month']==i]
    for hour in col_hour:
        row_name = str(i) + '_' + hour
        data_dict[row_name] = focused_data[hour]

dataa = np.array([np.array(data_dict[key]) for key in data_dict.keys()])
#x = stats.kruskal(dataa[26], dataa[28])
for i in range(len(data)-1):
    for j in range(i+1,len(dataa)):
        temp_stat = stats.kruskal(dataa[i], dataa[j]) 
        #print pairs that we do not reject that it share the same mean
        if temp_stat.pvalue >0.005:
            print(i,j)
        
x = ''
for i in range(24):
    if i != 23:
        x+= 'data[{}],'.format(i)
    else :x += 'data[i]'.format(i)
    
x
    
for i in range(0,288,24):
    x = ''
    maxlim = i+24
    for j in range(i, maxlim):
        if j != maxlim-1:
            x+= 'dataa[{}],'.format(j)
        else :x += 'dataa[{}]'.format(j)
    
    temp_val = eval('stats.kruskal' + '('+ x +')')
    print(temp_val)




for month, i in enumerate(range(0,288,24)):
    print('-- month ', month+1,'--')
    print('|hour1|hour2|stats|')
    maxlim = i+24
    for j in range(i, maxlim):
        for k in range(j+1, maxlim):
            temp_val = stats.kruskal(dataa[j], dataa[k])
            if temp_val.pvalue>0.05:
                print(j-month*24,k-month*24,temp_val)

import scipy.stats as st

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """
    params = [float(i) for i in params.strip('[]').split(',')]
    
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = eval('st.' + dist + '.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)')
    end = eval('st.' + dist + '.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)')

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = eval('st.' + dist + '.pdf(x, loc=loc, scale=scale, *arg)')
    pdf = pd.Series(y, x)

    return pdf
data_dist['dist_name'].unique()
data_dist.iloc[169]
data_dist
import matplotlib.pyplot as plt



for i in range(data_dist.shape[0]):
    name, distname, params = data_dist.iloc[i]
    print(i)
    pdf = make_pdf(distname, str(params))
    x = i%24
    #plt.figure(figsize=(12,8))
    plt.figure(x)
    pdf.plot(lw=2, label=name, legend=True)


    
import matplotlib.pyplot as plt



for i in range(data_dist.shape[0]):
    name, distname, params = data_dist.iloc[i]
    
    pdf = make_pdf(distname, str(params))
    x = name.split('_')[0]
    plt.figure(figsize=(12,8))
    plt.figure(x)
    pdf.plot(lw=2, label=name, legend=True)

#by season.
# Spring: March to May.
# Summer: June to August.
# Autumn: September to November.
# Winter: December to Februar
    
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

spring = [3,4,5]
summer = [6,7,8]
autumn = [9,10,11]
winter = [12,1,2]

for i in range(data_dist.shape[0]):
    name, distname, params = data_dist.iloc[i]
    month = int(name.split('_')[0])
#     print(month)
    if month in spring:
        color = 'red'
        label = 'spring'
    elif month in summer:
        color = 'blue'
        label = 'summer'
    elif month in autumn:
        color = 'green'
        label = 'autumn'
    elif month in winter:
        color = 'black'
        label = 'winter'
    
    pdf = make_pdf(distname, str(params))
    plt.figure(figsize=(12,8))

    x = i%24
    plt.figure(x)
    pdf.plot(lw=2, label= label, legend=True, color=color)
    plt.title('Plot at hour {}'.format(name.split('_')[-1]))

_, distribution, params = data_dist.iloc[1]
print(params)
name, distname, params = data_dist.iloc[169]
    # Separate parts of parameters
arg = params[:-2]
loc = params[-2]
scale = params[-1]


x = np.linspace(0,3,100)
y = eval('st.' + distname + '.pdf(x, loc=loc, scale=scale, *arg)')
start = eval('st.' + distname + '.ppf(0.01, *arg, loc=loc, scale=scale) if arg else distname.ppf(0.01, loc=loc, scale=scale)')

plt.plot(x,y)
print(start)
