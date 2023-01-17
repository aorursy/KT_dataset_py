# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc # We're gonna be clearing memory a lot

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cl_train=pd.read_csv('../input/clicks_train.csv')

cl_test=pd.read_csv('../input/clicks_test.csv')

##cl_train.head()

temp_test=pd.DataFrame(cl_train.groupby('display_id')['ad_id'].count().value_counts())

temp_test.head()
temp_test2=pd.DataFrame(cl_train.groupby('display_id')['ad_id'].count())

temp_test2
cl_train.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.barplot(temp_test.index, temp_test.values, alpha=0.8, label='train')
ad_usage_train = pd.DataFrame(cl_train.groupby('ad_id')['ad_id'].count())

ad_usage_train.head()

#plt.hist(ad_usage_train, bins=50,log=True)
#documents_categories=pd.read_csv('../input/documents_categories.csv')

#documents_categories.head()
#documents_entities=pd.read_csv('../input/documents_entities.csv')

#documents_entities.head()
##documents_meta=pd.read_csv('../input/documents_meta.csv')

#documents_meta.head()
#documents_topics=pd.read_csv('../input/documents_topics.csv')

#documents_topics.head()
events=pd.read_csv('../input/events.csv')

events.head()
#page_views_sample=pd.read_csv('../input/page_views_sample.csv')

#page_views_sample.head()
promoted_content=pd.read_csv('../input/promoted_content.csv')

promoted_content.head()