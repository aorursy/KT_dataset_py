# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#required libraries are imported
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import pandas_profiling
import matplotlib.pyplot as plt

% matplotlib inline
#facebook data csv file is read
fb_data = pd.read_csv('../input/pseudo_facebook.csv')
fb_data.head()
#profiling of data is done to the know type and features of data
report = pandas_profiling.ProfileReport(fb_data)
report.to_file("fb_data.html")
#the data is grouped by the variable gender and total entries under each gender is obtained
gender = fb_data.groupby('gender').count()
gender.head()
#max_gen = gender[gender.userid.max() == gender.userid].index.values
#NA valyes are dropped from the data
fb_data.dropna(inplace=True)
fb_data.head()
#profiling is done again to check if any other NA values are there and information on data
report = pandas_profiling.ProfileReport(fb_data)
report.to_file("fb_data_after.html")
cnt_1900,cnt_1920,cnt_1940,cnt_1960,cnt_1980,cnt_2000,other = 0,0,0,0,0,0,0
#creating separate dataframe based on the variable dob_year and having of people born in that year
for i in fb_data['dob_year']:
    if i>=1900 and i<1920:
        cnt_1900 += 1
    elif i>=1920 and i<1940:
        cnt_1920 +=1
    elif i>=1940 and i<1960:
        cnt_1940 +=1
    elif i>=1960 and i<1980:
        cnt_1960 +=1
    elif i>=1980 and i<2000:
        cnt_1980 +=1
    elif i>=2000:
        cnt_2000 +=1
    else:
        other +=1

year_wise = pd.DataFrame({'1900-1920':[cnt_1900], '1920-1940':[cnt_1920], '1940-1960':[cnt_1940], '1960-1980':[cnt_1960], 
                         '1980-2000':[cnt_1980], '2000-more':[cnt_2000]})
year_wise = year_wise.T #doing transpose of it
year_wise.rename(columns={0:'ppl_count'})
#to know how many people are on facebook based on their birth year
year_wise.plot.bar()
plt.title('Number of people on Facebook based on their the birth year')
plt.legend(loc='upper left')
plt.show()
#using gender dataframe to know how many people on facebook based on gender
gender.plot.barh(y='userid')
plt.xlabel('number of people')
plt.title('number of people on FB based on Gender')
plt.legend('no ppl',loc='upper left')
plt.show()
#dataframe is created by grouping data on age and finding mean
age_data = fb_data.groupby('age').mean()
age_data.head()
#to know on an average how long people have been using facebook
age_data.plot.line(y='tenure')
plt.ylabel('tenure')
plt.title('on average how long people on FB')
plt.show()
#based on age, on an average how many friends to the users have
age_data.plot.line(y='friend_count')
plt.ylabel('friend_count')
plt.title('on average number of friends based on age')
plt.show()
#separate dataframes are created based on age
data_below_18 = fb_data[fb_data['age']<=18]
data_bw_19_30 = fb_data[(fb_data['age']<=30)&(fb_data['age']>18)]
data_bw_31_60 = fb_data[(fb_data['age']<=60)&(fb_data['age']>30)]
data_bw_61_100 = fb_data[(fb_data['age']<=100)&(fb_data['age']>60)]
data_more_101 = fb_data[fb_data['age']>=101]
data_below_18.head()
data_bw_19_30.head()
data_bw_31_60.head()
data_bw_61_100.head()
data_more_101.head()
#to see how many have given more likes in age group 100 and above
sns.barplot(x='age',y='likes',data=data_more_101)
plt.title('max likes given by a age group more than 100')
plt.show()
#to know who have got max likes between age group 19 to 30 
sns.factorplot(x='age', y='likes_received', data=data_bw_19_30)
plt.title('max likes received by a age group between 19 and 30')
plt.show()
#to check how male and female users are on facebook in age group 31 to 60
sns.barplot(x='gender', y='age', data=data_bw_31_60)
plt.title('no of male and female between the age group 31 to 60')
plt.show()
sns.pairplot(data_below_18[['gender','likes_received','mobile_likes_received','www_likes_received']],
             hue='gender', diag_kind="hist")
# The histogram on the diagonal allows us to see the distribution of a single variable 
# while the scatter plots on the upper and lower triangles show the relationship (or lack thereof) between two variables
plt.show()
#relationship between age and friend request initiated, how many of have initiated more requests based on age
sns.lmplot(x='age', y='friendships_initiated', data=data_bw_61_100,fit_reg=False, aspect=2.5, x_jitter=.01)
plt.title('relation between age and friend request initiated')
plt.show()
sns.pairplot(fb_data[['gender','likes','mobile_likes','www_likes','likes_received','mobile_likes_received','www_likes_received']]
            , hue='gender', diag_kind="hist")
# The histogram on the diagonal allows us to see the distribution of a single variable 
# while the scatter plots on the upper and lower triangles show the relationship (or lack thereof) between two variables
plt.show()
#grouping data based on gender and finding sum of it
gender_on_likes = fb_data.groupby('gender').sum()
gender_on_likes
#to see which gender have received more likes
gender_on_likes.plot.bar(y='likes_received')
plt.ylabel('number of likes received')
plt.title('number of likes received on FB based on Gender')
plt.legend('no likes',loc='upper left')
plt.show()
#reading the csv file and trying to combine dob_year, dob_month, dob_day variables to get in yyyy-mm-dd format
fb_details = pd.read_csv('../input/pseudo_facebook.csv', parse_dates=[['dob_year','dob_month','dob_day']], 
                         index_col='dob_year_dob_month_dob_day')
#fb_details.rename(index={'dob_year_dob_month_dob_day':'dateOfBirth'},inplace=True)
fb_details.head()
#probability distribution of the variable age
age_mean = np.mean(fb_data['age'])
age_std = np.std(fb_data['age'])
pdf = stats.norm.pdf(fb_data['age'], age_mean, age_std)
plt.plot(fb_data['age'], pdf)
plt.hist(fb_data['age'], density=True)
plt.show()
#probability distribution of variable friend_count
frndcnt_mean = np.mean(fb_data['friend_count'])
frndcnt_std = np.std(fb_data['friend_count'])
pdf = stats.norm.pdf(fb_data['friend_count'], frndcnt_mean, frndcnt_std)
plt.plot(fb_data['friend_count'], pdf)
plt.hist(fb_data['friend_count'], density=True)
plt.show()
