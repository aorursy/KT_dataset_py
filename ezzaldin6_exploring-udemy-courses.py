import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
import re
plt.style.use('ggplot')
sns.set(style='darkgrid', context='notebook')
%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
udemy_courses=pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv', parse_dates=['published_timestamp'])
udemy_courses.head(3)
udemy_courses.info()
udemy_courses['price']=udemy_courses['price'].str.replace('Free', '0').str.replace('TRUE', '0')
udemy_courses['price']=udemy_courses['price'].astype('float')
udemy_courses['number_of_contents']=udemy_courses['content_duration'].str.extract('([\d\.]+)\s[\w]+').astype('float')
udemy_courses['content_duration_type']=udemy_courses['content_duration'].str.extract('[\d\.]+\s([\w]+)')
udemy_courses.drop('content_duration', axis=1, inplace=True)
udemy_courses.dropna(inplace=True)
new_dates=[]
for i in udemy_courses['published_timestamp']:
    new_date=dt.datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ')
    new_dates.append(new_date)
udemy_courses['published_timestamp']=new_dates
udemy_courses['is_paid'].value_counts()
udemy_courses['is_paid']=udemy_courses['is_paid'].str.replace('TRUE', 'True')
udemy_courses['is_paid']=udemy_courses['is_paid'].str.replace('FALSE', 'False')
udemy_courses['content_duration_type']=udemy_courses['content_duration_type'].str.replace('hours', 'hour')
udemy_courses['is_paid'].value_counts()
g=sns.catplot(x='subject',
            data=udemy_courses,
            kind='count',
            hue='is_paid')
g.fig.suptitle('free/paid categories comparison', y=1.03)
plt.xticks(rotation=90)
plt.show()
pd.pivot_table(index='is_paid', values='num_subscribers', data=udemy_courses, aggfunc='mean')
g=sns.catplot(x='level',
            data=udemy_courses,
            kind='count',
            hue='is_paid')
g.fig.suptitle('free/paid lavel comparison', y=1.03)
plt.xticks(rotation=90)
plt.show()
# Box-plot, CDF, histogram
def cdf(lst):
    x=np.sort(lst)
    y=np.arange(1, len(x)+1)/len(x)
    return x, y
fig, ax=plt.subplots(1,3, figsize=(15,5))
x_price, y_price=cdf(udemy_courses['price'])
ax[0].plot(x_price, y_price)
ax[0].set_title('CDF of prices')
ax[1].hist(udemy_courses['price'])
ax[1].set_title('histogram distribution of prices')
ax[2].boxplot(udemy_courses['price'])
ax[2].set_title('boxplot of prices')
plt.show()
print('median: ',udemy_courses['price'].median())
print('mean: ',udemy_courses['price'].mean())
udemy_courses[udemy_courses['is_paid']=='True'].groupby('level')['price'].mean().sort_values(ascending=False)
price_category=[]
for i in udemy_courses['price']:
    if i==0:
        price_category.append('Free')
    elif i>0 and i<=45:
        price_category.append('cheap')
    elif i>45 and i<=95:
        price_category.append('expensive')
    else:
        price_category.append('very expensive')
udemy_courses['price_category']=price_category
udemy_courses['price_category'].value_counts()
# relation between number of lectures and price
udemy_courses.groupby('price_category')['num_lectures'].mean().sort_values(ascending=False)
udemy_courses['content_duration_type'].value_counts()
# let's see the courses that thier content duration type = questions.
udemy_courses[udemy_courses['content_duration_type']=='questions']

mins_courses=udemy_courses[udemy_courses['content_duration_type']=='mins']
g=sns.catplot(x='price_category',
            data=mins_courses,
            kind='count',
            hue='level')
g.fig.suptitle("price category (minutes courses) in each level.", y=1.03, x=0.4)
plt.show()
udemy_courses['published_timestamp'].dt.year.value_counts().plot.bar()
plt.xlabel('published year')
plt.ylabel('frequency')
plt.title('number of courses published each year(2011-2017)')
plt.show()
udemy_courses['published_year']=udemy_courses['published_timestamp'].dt.year
pd.pivot_table(index='published_year',
               columns='subject',
               values='course_id',
               data=udemy_courses,
               aggfunc='count',
               fill_value=0)