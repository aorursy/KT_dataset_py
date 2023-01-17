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
data=pd.read_csv("/kaggle/input/udemy-courses/udemy_courses.csv")

data.head()
only_paid=data[data["is_paid"]=="True"]

only_paid["price"]=only_paid["price"].astype(float)

data.loc[only_paid["price"].idxmax]
only_paid[only_paid["price"]>=200]
for link in only_paid[only_paid["price"]>=200]["url"]:

    print(link)
import matplotlib.pyplot as plt

import seaborn as sns

most_expensive=only_paid[only_paid["price"]>=200]

sns.countplot(data=most_expensive,x="subject")
most_expensive=only_paid[only_paid["price"]>=200]

sns.countplot(data=most_expensive,x="level")
for most_expensive_expert_link in most_expensive[most_expensive["level"]=="Expert Level"]["url"]:

    print(most_expensive_expert_link)
only_paid[only_paid["level"]=="Expert Level"].sort_values(by="price")[0:10][["course_title","url","price"]]
cheapest=only_paid[only_paid["price"]<=20]

for  cheapest_expert_course in cheapest[cheapest["level"]=="Expert Level"]["url"]:

    print(cheapest_expert_course)
median_of_subs=only_paid["num_subscribers"].median()

bigger_than_median=only_paid[only_paid["num_subscribers"]>=median_of_subs]

sns.countplot(data=bigger_than_median,x="subject")
data[["subject","num_subscribers"]].groupby(by="subject").sum().sort_values(by="num_subscribers",ascending=False)