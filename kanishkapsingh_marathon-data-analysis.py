# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import datetime as dt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
challenge=pd.read_csv('../input/challenge.csv')

run=pd.read_csv('../input/run1.csv')

challenge.head()
print(challenge['10km Time'].dtype)

def get_gender(a):

    return a[1]

def get_class(a):

    return a[2:]

def get_time(x):

    if x=='':

        return None

    elif x=='nan':

        return None

    else:

        return (dt.datetime.strptime(str(x),"%H:%M:%S")-dt.datetime.strptime('00:00:00',"%H:%M:%S")).total_seconds()/60
challenge['gender']=challenge['Category'].apply(get_gender)

run['gender']=run['Category'].apply(get_gender)

challenge['class']=challenge['Category'].apply(get_class)

run['class']=run['Category'].apply(get_class)

challenge['Official Time']=challenge['Official Time'].apply(get_time)

run['Official Time']=run['Official Time'].apply(get_time)

challenge['Net Time']=challenge['Net Time'].apply(get_time)

run['Net Time']=run['Net Time'].apply(get_time)
help(challenge.plot)
help(plt.boxplot)


# Analyzing the finishing times according to the gender.

with sns.axes_style('darkgrid'):

    plt.boxplot([challenge[challenge['gender']=='M']['Official Time'],challenge[challenge['gender']=='M']['Official Time']])
help(plt.boxplot)
challenge.head()
challenge['Official Time'].dtype
grouped=challenge[['gender','Official Time']].groupby('gender',group_keys=True).apply(lambda x:x)