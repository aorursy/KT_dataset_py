# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
regions = pd.read_csv("../input/india-regions/cities.csv")
marketing = pd.read_csv("../input/marketing-groups/marketing_group.csv")
print(regions.head())
print(marketing.head())
re = regions.groupby([ "subdistname"]).size().reset_index()
re.shape
re = pd.DataFrame(re)
train = regions.groupby([ 'StateName', 'subdistname'])['subdistname'].agg({'subdistname':['count']}).reset_index()
train = pd.DataFrame(train)
train.head()
train.columns = ['StateName',  'subdistname','count']
# print(train.loc[train['StateName'] == 'HARYANA'])
# regions.drop(['Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16','Unnamed: 17','Unnamed: 18','Unnamed: 19'], axis=1, inplace=True)
# train.shape

marketing.columns = [['1', 'subdistname', 'City', '3', '229', '0000-00-00 00:00:00']]
marketing.head()
df =  pd.concat( [train,marketing], axis=1)
df.head()
df.shape
df.to_csv('submission.csv', index=False)
