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
df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
df.head()
df.info()
df.isnull().sum()
df.describe()
df['published_timestamp'] =  pd.to_datetime(df['published_timestamp'])

df['month'] = df['published_timestamp'].dt.month_name()





plt.figure(figsize = (18,8))

sns.set_style("darkgrid")

ax = sns.countplot(df['month'],palette = 'Wistia')

ax.set_xlabel('Month', fontsize = 20)

ax.set_ylabel('No. of Courses', fontsize = 20)

ax.set_title('Month With most Courses', fontsize = 30)



plt.show()

df_bf = df.loc[df['subject'] == "Business Finance"]

df_wd = df.loc[df['subject'] == "Web Development"]

df_gd = df.loc[df['subject'] == "Graphic Design"]

df_mi = df.loc[df['subject'] == "Musical Instruments"]
bf_month = df_bf['month'].value_counts()

wd_month = df_wd['month'].value_counts()

gd_month = df_gd['month'].value_counts()

mi_month = df_mi['month'].value_counts()
plt.rcParams['figure.figsize'] = (18, 8)

sns.set_style('whitegrid')

ax = sns.lineplot(x = bf_month.index  , y = bf_month.values, linewidth=2.5,label = 'Business Finance')

ax1 = sns.lineplot(x = wd_month.index  , y = wd_month.values, linewidth=2.5,label = "Web Development")

ax2 = sns.lineplot(x = gd_month.index  , y = gd_month.values, linewidth=2.5,label = "Graphic Design")

ax3 = sns.lineplot(x = mi_month.index  , y = mi_month.values, linewidth=2.5,label = "Musical Instruments")

ax.set_xlabel('Month', fontsize =20)

ax.set_ylabel('No. of Courses',fontsize =20)

ax.set_title('Month with most number of Courses(subject wise) ', fontsize = 30)

plt.show()
df['year'] = df['published_timestamp'].dt.year





plt.figure(figsize = (18,8))

# sns.set_style("darkgrid")

ax = sns.countplot(df['year'],palette = 'bone')

ax.set_xlabel('Year', fontsize = 20)

ax.set_ylabel('No. of Courses', fontsize = 20)

ax.set_title('Year With most Courses', fontsize = 30)



plt.show()
def removing_outlier(column, compare_column, dfname):



    q1 = df[column].quantile(0.25)

    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5*iqr

    upper_limit = q3 + 1.5*iqr

    

    dfname = df[(df[column]>lower_limit)  & (df[column]<upper_limit)]

    

    plt.figure(figsize = (18,8))

    ax = sns.boxplot(x = compare_column , y = column, data = dfname)

    

    ax.set_xlabel('Subject', fontsize = 20)

    ax.set_ylabel('Counts of Reviews', fontsize = 20)

    

    return plt.show()
#Number Reviews Subject Wise



reviews_no_out = ''

removing_outlier('num_reviews','subject',reviews_no_out)
#Number Subscribers Subject Wise



subs_no_out = ''



removing_outlier('num_subscribers','subject',subs_no_out)
#Number of lectures, Subject Wise



lec_no_out = ''



removing_outlier('num_lectures','subject',lec_no_out)
subs_free = df.loc[(df['num_subscribers']) & (df['is_paid'] == False)]

subs_free.sort_values('num_subscribers',ascending = False).head()
gk = df.groupby('subject')

gk_sum = gk['num_subscribers'].sum()

labels = gk_sum.index

sizes = gk_sum.values

colors = plt.cm.rainbow(np.linspace(0,5))



# explode = (0.2,0.1, 0, 0,0,0,0,0,0,0,0,0)

plt.rcParams['figure.figsize'] = (15,9)



plt.pie(sizes, labels=labels,  autopct='%1.1f%%',shadow=True, startangle=90, colors = colors)



plt.axis('equal')  

plt.title("Level with most lectures", fontsize =20)

plt.show()
def free_courses(subject):

    df_free =  df.loc[(df['subject'] == subject) & (df['is_paid'] == False)]



    df_free.drop(['is_paid','price','url','published_timestamp'], axis = 1,inplace = True)



    return df_free.sort_values('num_subscribers',ascending = False).head()
free_courses('Business Finance')
free_courses('Graphic Design')
free_courses('Musical Instruments')
free_courses('Web Development')
df['popularity'] = df['num_subscribers'] + df['num_reviews']

df.sort_values('popularity', ascending=False).head()
plt.figure(figsize = (18,8))

ax = sns.countplot(df['subject'], hue = df['level'],palette ='husl')

ax.set_xlabel('Subject', fontsize = 20)

ax.set_ylabel('Counts of Level', fontsize = 20)

# ax.set_title('Year With most Courses', fontsize = 30)



plt.show()
plt.style.use('dark_background')

plt.figure(figsize = (18,8))

ax = sns.lineplot(x = df['subject'], y = df['content_duration'], color = 'gold')

ax.set_xlabel('Subject', fontsize = 20)

ax.set_ylabel('Counts of Level', fontsize = 20)



plt.show()