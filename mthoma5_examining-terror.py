# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from collections import Counter

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
terror_all =pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1',

                  low_memory=False,na_filter=False,parse_dates = ['iyear'],

                   infer_datetime_format=True)



terror = terror_all[terror_all["success"] == 1]

no_terror = terror_all[terror_all["success"] == 0]
#Assuming the count of a reference is an incident, lets look 

#at the frequency of incidents by geo

country_count = pd.Series(Counter(terror["country_txt"])).to_frame(name="Count")



country_count[country_count['Count'] > country_count["Count"].mean()].sort_values("Count",

                                                              ascending=True).plot(kind='barh',

                                                                                  figsize=(12,15))
#Extend this logic to time

over_years = pd.Series(Counter(terror["iyear"])).to_frame(name="Count")



over_years.plot(kind='area',figsize=(15,6))
region_count = pd.Series(Counter(terror["region_txt"])).to_frame(name="Count")

region_count.sort_values("Count",ascending=True).plot(kind='barh',

                                                                                   figsize=(15,12),

                                                                                   fontsize=12,

                                                                                    legend=True)

plt.tight_layout()
group_count = pd.Series(Counter(terror["gname"])).to_frame(name="Count")

group_count.drop("Unknown",0,inplace=True)

group_count.sort_values("Count",ascending=True)[-25:].plot(kind='barh',figsize=(15,12),

                                                                 fontsize=12,

                                                                 legend=True)

taliban = terror[terror["gname"] == 'Taliban']



taliban_over_time = pd.Series(Counter(taliban["iyear"])).to_frame(name="Count").plot(kind='area',

                                                                           figsize=(15,6))
taliban_x_ge0 = pd.Series(Counter(taliban["country_txt"])).to_frame(name="Count").plot(kind="barh",figsize=(8,6))