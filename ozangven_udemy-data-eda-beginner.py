# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/finance-accounting-courses-udemy-13k-course/udemy_output_All_Finance__Accounting_p1_p626.csv")
df.shape
df.head()
df.nunique(axis=0)

#It shows unique values for each variable

#We can understand that in "is_wishlisted" column there is no variable except for "False". 

#Then, we will not need to investigate this column.

#Same situation is also valid for "discount_price_currency", and "price_detail_currency".

#So the first thing I will do in data-cleaning is to get rid of those column.

#Also we ve seen that there are 13608 columns from df.shape. And in this list, we can see that there are 13608 different id variables.

#Which means, every course is unique and it was obvious. The second thing I will get rid of will be the "id" column.
# Letting unnecessary columns go

df = df.drop(columns = ["id", "is_wishlisted", "discount_price__currency", "price_detail__currency"])
# percentage of missing values in each column

#Reference: https://www.kaggle.com/gauravduttakiit/creditcard-fraud-detection-by-logistic-regression

round(100 * (df.isnull().sum()/len(df)),2).sort_values(ascending=False)
# percentage of missing values in each row

#Reference: https://www.kaggle.com/gauravduttakiit/creditcard-fraud-detection-by-logistic-regression

round(100 * (df.isnull().sum(axis=1)/len(df)),2).sort_values(ascending=False)
# There are some missing values. Lets find them.

#Reference: https://www.youtube.com/watch?v=eMOA1pPVUc4

new_df = df[df.isna().any(axis=1)]

display(new_df)
# We will have to drop those rows by the method below if we want to make an analysis on a specific column which includes NaN.

df = df.drop(df.index[[33,39,40,80,90]])
df.describe()
df.info()
#Lets investigate the relationship btw the year of publication and number of subscribers

#First, we should scrap the year

df["published_time"] = pd.to_datetime(df["published_time"])
df["publication_year"] = df["published_time"].dt.year
year_df = df.groupby("publication_year").sum()
year_df
years = [year for year, df in df.groupby(['publication_year'])]

plt.bar(years,df.groupby(['publication_year']).sum()['num_subscribers'])

plt.ylabel('Subscribers')

plt.xlabel('Years')

plt.xticks(years, rotation='vertical', size=10)

plt.show()

#Reference: https://www.youtube.com/watch?v=eMOA1pPVUc4
#It can be seen that the number of subscribers skyrocketed btw 2012-2017, and then, it started to decrease.