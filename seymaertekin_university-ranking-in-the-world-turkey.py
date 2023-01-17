# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_times = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
data_times.head(10)
data_times.info()
data_times.tail(10)

data_times.columns
print('number of NaNs per column:')
data_times.isna().sum()
data_times['world_rank'] = pd.to_numeric(data_times['world_rank'], errors='coerce')
data_times['teaching'] = pd.to_numeric(data_times['teaching'], errors='coerce')
data_times['num_students'] = pd.to_numeric(data_times['num_students'], errors='coerce')
data_times['income'] = pd.to_numeric(data_times['income'], errors='coerce')
data_times['international'] = pd.to_numeric(data_times['international'], errors='coerce')

# pd.to_numeric converts argument to a numeric type.
data_times.info()
data_times.corr()
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(data_times.corr(),annot=True,linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

# drop female male ratio because 233 rows are missing
# drop total score because it's missing and too similar to world rank
# drop num_students because it's missing 
data_times.drop(columns=['female_male_ratio', 'total_score', 'num_students'])
data_times['world_rank'].fillna(201, inplace=True)

data_times.tail()
the_first_200 = data_times.loc[data_times['world_rank'] != 201]
the_first_200_2016 = the_first_200.loc[the_first_200['year'] == 2016]
#the_first_200_2016 gives the top 200 university in 2016
the_first_200.tail()
the_first_200_2016.head()
the_first_200 = the_first_200.drop(columns=['num_students'])
the_first_200_2016 = the_first_200_2016.drop(columns=['num_students'])

f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(the_first_200.corr(),annot=True,linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
the_first_200.plot(kind='scatter',x='teaching',y='world_rank', color='g', label='teaching', linewidth=1, grid=True, linestyle=':')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('teaching')              # label = name of label
plt.ylabel('world_rank')
plt.title = ('Line Plot')
plt.show()
the_first_200.plot(kind='scatter',x='student_staff_ratio',y='world_rank', color='r', label='teaching', grid=True,)
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('teaching')              # label = name of label
plt.ylabel('world_rank')
plt.title = ('Line Plot')
plt.show()
the_first_200.plot(kind='scatter',x='income',y='world_rank', color='b', label='teaching', grid=True,)
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('income')              # label = name of label
plt.ylabel('world_rank')
plt.title = ('Line Plot')
plt.show()


# create dummy variable them group by that
# set the legend to false because we'll fix it later
the_first_200_2016.assign(dummy = 1).groupby(
  ['dummy','country']
).size().to_frame().unstack().plot(kind='bar',stacked=True,legend=False)


# other it'll show up as 'dummy' 
plt.xlabel('country')

# disable ticks in the x axis
plt.xticks([])

# fix the legend
current_handles, _ = plt.gca().get_legend_handles_labels()
reversed_handles = reversed(current_handles)

labels = reversed(the_first_200_2016['country'].unique())

plt.legend(reversed_handles,labels,loc='best')
plt.show()
university_country = data_times["country"].unique()
print(university_country)

data_times_Turkey = data_times[(data_times["country"] == "Turkey" )]
data_times_Turkey.head(10)
data_times_Turkey = data_times_Turkey.drop(columns=['world_rank','total_score', 'num_students','female_male_ratio'])
data_times_Turkey.describe()
data_times_Turkey.plot(kind="scatter", x="year", y="teaching",alpha = 0.5,color = "g")
plt.xlabel("Year")              # label = name of label
plt.ylabel("teaching")
plt.show()
data_times_Turkey.teaching.plot(kind='line', color='g', label='teaching', linewidth=1, grid=True, linestyle=':')
data_times_Turkey.research.plot(kind='line', color='b', label='Research')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title = ('Line Plot')
plt.show()
data_times_Turkey.plot(kind="scatter", x="year", y="student_staff_ratio",alpha = 0.5,color = "b")
plt.xlabel("Year")              # label = name of label
plt.ylabel("student_staff_ratio")
plt.show()
data_times_Turkey.plot(kind="scatter", x="year", y="citations",alpha = 0.5,color = "r")
plt.xlabel("Year")              # label = name of label
plt.ylabel("citations")
plt.show()
data_times_Turkey.year.plot(kind='hist', bins=50)
plt.show()
data_times_Turkey.plot(kind="scatter", x="research", y="citations",alpha = 0.5,color = "b")
plt.xlabel("research")              # label = name of label
plt.ylabel("citations")
plt.show()
data_times_Turkey.plot(kind="scatter", x="teaching", y="citations",alpha = 0.5,color = "g")
plt.xlabel("teaching")              # label = name of label
plt.ylabel("citations")
plt.show()
data_times_Turkey_METU = data_times_Turkey[(data_times_Turkey["university_name"] == "Middle East Technical University")]
data_times_Turkey_METU.head(10)
data_times_Turkey_METU.describe()
data_times_Turkey_METU.plot(kind="scatter", x="year", y="teaching",alpha = 0.5,color = "b")
plt.xlabel("Year")              # label = name of label
plt.ylabel("teaching")
plt.show()
data_times_Turkey_METU.plot(kind="scatter", x="citations", y="international",alpha = 0.5,color = "r")
plt.xlabel("citations")              # label = name of label
plt.ylabel("international")
plt.show()
#plt.plot('citations','international','-ok')
data_times_Turkey_METU.plot(kind="scatter", x="research", y="international",alpha = 0.5,color = "r")
plt.xlabel("research")              # label = name of label
plt.ylabel("international")
plt.show()
data_shanghai= pd.read_csv("/kaggle/input/world-university-rankings/shanghaiData.csv")
data_shanghai.head(10)
data_shanghai.shape
data_shanghai.columns
data_shanghai.info()
data_cwur=pd.read_csv("/kaggle/input/world-university-rankings/cwurData.csv")
data_cwur.head(10)

data_cwur.columns
data_cwur.shape
data_cwur.info()
data_times.head(10)
data_times.tail()
data_times.columns
print(data_times["country"].value_counts(dropna=False))
print(data_shanghai["year"].value_counts(dropna=False))
print(data_cwur["year"].value_counts(dropna=False))
data_cwur.describe()
data_cwur.boxplot(column="citations", by = "year", showfliers=True)
m = data_cwur.head()
m
# id_vars = what we do not wish to melt
# value_vars = what we want to melt

melted=pd.melt(frame=m,id_vars="institution", value_vars=["alumni_employment","score"])
melted
melted.pivot(index="institution", columns="variable", values="value")
data1= data_cwur.head()
data2 = data_cwur.tail()
conc_data = pd.concat([data1,data2], axis=0, ignore_index=True) #axis=0 means adding the dataframes in rows, ignore_index ignores the old index and give new index
conc_data
d1=data_times["country"].head(10)
d2=data_cwur["country"].head(10)
concat_d=pd.concat([d1,d2],axis=1)
concat_d
data_cwur.dtypes
data_cwur["institution"]=data_cwur["institution"].astype("category")
data_cwur.dtypes #you'll see the datatype of institution has changed to category
data_cwur.info()
data_cwur["broad_impact"].value_counts(dropna=False)
data_cwur["broad_impact"].dropna(inplace=True) #inplace=True means changes automatically assigned to data
assert data_cwur["broad_impact"].notnull().all()
data_times["total_score"].fillna(value=0, inplace=True)
data_times.tail()
data_times["num_students"]=[0 for i in data_times["num_students"]]
data_times.tail()