# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting, https://matplotlib.org/api/pyplot_summary.html

import seaborn as sns # more data vis, https://seaborn.pydata.org/



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
all_degrees = pd.read_csv('../input/degrees-that-pay-back.csv')

all_degrees.head()
all_degrees.columns = ['major','starting','midcareer','delta', 'mid_10', 'mid_25', 'mid_75', 'mid_90']

for x in all_degrees.columns:

    if x != 'major' and x != 'delta':

        salary = all_degrees[x].str.replace("$", "")

        salary = salary.str.replace(",", "")

        all_degrees[x] = pd.to_numeric(salary)

        

all_degrees.head()
degrees = all_degrees.drop(all_degrees.columns[[4,5,6,7]],axis=1,inplace=False)

degrees.head()
degrees.describe()
top_degrees = degrees.nlargest(10, 'midcareer').reset_index()

top_degrees.head(10)
x = top_degrees['midcareer']

y = len(top_degrees.index) - top_degrees.index #swap high and low

labels = top_degrees['major']



plt.scatter(x, y, color='g', label = 'Mid Career Median Salary')

plt.yticks(y, labels)

plt.show()
sns.barplot('midcareer', 'major', data=top_degrees)
# background will be midcareer

sns.barplot(x = "midcareer", y = "major", data=top_degrees, color = "red")



#Plot 2 - overlay - "bottom" series

bottom_plot = sns.barplot(x = "starting", y = "major", data=top_degrees, color = "#0000A3")

bottom_plot.set_xlabel("Salaries (starting->midcareer)")
mid_degrees = all_degrees.drop(['starting','delta'],axis=1,inplace=False)

mid_degrees.head()

plt.figure(figsize=(20,12))

df = mid_degrees.sort_values('mid_90', ascending=False).head(10)

pl_90 = sns.barplot(x = "mid_90", y = "major", data=df, color = "red", label = '90%')

pl_75 = sns.barplot(x = "mid_75", y = "major", data=df, color = "blue", label = '75%')

pl_50 = sns.barplot(x = "midcareer", y = "major", data=df, color = "green", label = '50%')

pl_25 = sns.barplot(x = "mid_25", y = "major", data=df, color = "orange", label = '25%')

pl_10 = sns.barplot(x = "mid_10", y = "major", data=df, color = "teal", label = '10%')

pl_10.set_xlabel("Salaries")

pl_10.legend(loc=4) #move the legend

plt.show()
college_type_degrees = pd.read_csv('../input/salaries-by-college-type.csv')

college_type_degrees.columns = ['school','type','starting','midcareer', 'mid_10', 'mid_25', 'mid_75', 'mid_90']

college_type_degrees.head()

college_region_degrees = pd.read_csv('../input/salaries-by-region.csv')

college_region_degrees.columns = ['school','region','starting','midcareer', 'mid_10', 'mid_25', 'mid_75', 'mid_90']

college_region_degrees.head()

print(len(college_type_degrees.index)-college_type_degrees.count())

len(college_region_degrees.index)-college_region_degrees.count()
# first drop everyone but the school & region from region dataset

# since we are just going to merge those values and use salary info from the type dataset



truncated_colege_regions = college_region_degrees.drop(['starting','midcareer', 'mid_10', 'mid_25', 'mid_75', 'mid_90'], axis=1, inplace=False)

college_salaries = pd.merge(college_type_degrees, truncated_colege_regions, on='school')

college_salaries.head()
college_salaries = college_salaries[['school', 'type', 'region', 'starting', 'midcareer']]

salary_cols = ['starting', 'midcareer']

for x in salary_cols:

    salary = college_salaries[x].str.replace("$", "")

    salary = salary.str.replace(",", "")

    college_salaries[x] = pd.to_numeric(salary)

college_salaries.head()
print(college_salaries.groupby('type')['school'].nunique())



print(college_salaries.groupby('region')['school'].nunique())

sns.barplot(x = "region", y = "midcareer", data=college_salaries)

sns.barplot(x = "type", y = "midcareer", data=college_salaries)

sns.barplot(x = "type", y = "midcareer", hue = "region", data=college_salaries)

college_salaries = college_salaries.query('type != "Ivy League"');



plt.figure(figsize=(12,8))

plot = sns.barplot(x = "type", y = "midcareer", hue = "region", data=college_salaries, palette="muted")

plot.legend(loc=1) #move the legend