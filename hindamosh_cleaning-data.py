# import the needed libraries

import pandas as pd 

import numpy as np

import re

import matplotlib.pyplot as plt

%matplotlib inline
#load data to pandas 

# application data

applcs = pd.read_csv("../input/Wuzzuf_Applications_Sample.csv")

# posts data

posts = pd.read_csv("../input/Wuzzuf_Job_Posts_Sample.csv")
# explor the top 5 rows

applcs.head()
posts.head()
# make a copy of our data

applcs_df = applcs

# The posts data

job_posts_df = posts
# fix the appication date

applcs_df.app_date = pd.to_datetime(applcs_df.app_date)
# test

applcs_df.dtypes
# set the index for app_date

applcs_df.index = applcs_df['app_date']

# remove the app_date column

applcs_df.drop(columns='app_date',axis= 1,inplace=True)

applcs_df["count"] = 1

# explore more info about the job-posts data

job_posts_df.info()
# drop the unneeded columns 

columns_drop = ['job_category2','job_category3','job_industry2','job_industry3']

job_posts_df.drop(columns=columns_drop, axis=1, inplace=True)

           



# rename the columns

job_posts_df.rename(columns={'job_category1': 'job_category', 'job_industry1': 'job_industry'}, inplace=True)

# convert post_date to datetime 

job_posts_df.post_date = pd.to_datetime(job_posts_df['post_date'])



# extract month name

job_posts_df['month'] = job_posts_df.post_date.apply(lambda x: x.month_name())



# extract  name of the day

job_posts_df['day'] = job_posts_df.post_date.apply(lambda x: x.day_name())
# check for data sample

job_posts_df.sample(5)
# check for null values

job_posts_df.isnull().sum()
# check for the null values in currency column

job_posts_df.loc[job_posts_df.currency.isnull()]
# fix the null values and converts it to Egyptian pound

l = job_posts_df.loc[job_posts_df.currency.isnull()].index

for i in l:

    job_posts_df.loc[i,'currency'] ='Egyptian Pound'
# check

job_posts_df.currency.isnull().sum()
# check for duplicates values

job_posts_df.duplicated().sum()
# get sample

job_posts_df.sample()
# check for the number of unique values in experience_years column

job_posts_df.experience_years.nunique()
# check for the experience years

job_posts_df.experience_years.unique()[::-10]
# fix the experience year column

l = job_posts_df.experience_years.values

range_exps =[]

for i in l:

    pattern =re.findall('\\b\\d+\\b', i)

    if len(pattern) == 0:

        #min_exp = min([int(s) for s in i if s.isdigit()])

        max_exp = max([int(s) for s in i if s.isdigit()])

    else:

        #min_exp = min([int(s) for s in pattern])

        max_exp = max([int(s) for s in pattern])

    # check for the + sign

    if '+' in i:

        max_exp += 3

    # add to our cleaned list

    if max_exp == 0 :

        range_exp = 'Fresh'

    elif 0 < max_exp <= 5:

        range_exp = 'below 5'

    elif 5 < max_exp <= 10:

        range_exp = '5-10'

    elif 10 < max_exp <= 15:

        range_exp = '10-15'

    elif 15 < max_exp <= 20:

        range_exp = '15-20'

    elif 20 < max_exp <= 25:

        range_exp = '20-25'

    else:

        range_exp = 'above 25'

    range_exps.append(range_exp)
# check 

len(range_exps) == job_posts_df.shape[0]
# so now we can add the cleaned range to our dataset

job_posts_df['experience_range'] = range_exps
# experience range new values

job_posts_df.experience_range.unique()
# check for some samples from the dataset

job_posts_df.sample(5,random_state=1)[['experience_years','experience_range']]
# now we can drop the uncleaned column of experience_years

job_posts_df.drop(columns= 'experience_years',inplace=True)
job_posts_df.sample(5, random_state=1)
from html.parser import HTMLParser



class MLStripper(HTMLParser):

    def __init__(self):

        self.reset()

        self.strict = False

        self.convert_charrefs= True

        self.fed = []

    def handle_data(self, d):

        self.fed.append(d)

    def get_data(self):

        return ''.join(self.fed).strip().replace('\r\n',' ').replace('\xa0','') # more cleaning for the extra \n\r or \xa0



def strip_tags(html):

    s = MLStripper()

    s.feed(html)

    return s.get_data()
# fix the job_description 

job_posts_df.job_description = job_posts_df.job_description.apply(lambda x: strip_tags(x) if isinstance(x,str) else x)
job_posts_df.job_description[0]
# fix the job_requirements type(x) == str

job_posts_df.job_requirements = job_posts_df.job_requirements.apply(lambda x: strip_tags(x) if isinstance(x,str) else x)
# check for our work

job_posts_df.loc[:,['job_description','job_requirements']].sample(5,random_state=1)
# check for the number of unique records

job_posts_df.city.nunique()
def strip_city_name(x):

    if x[-1] ==',':

        x = x[:-1]

    return x.lower().replace('- ',',').replace('&',',').replace('/',',').replace('.','').replace(', ',',').replace(' ,',',').replace(' and ',',').replace('el ','').replace('el-','').strip()
#job_posts_df.drop(columns='city_clean',inplace=True)
job_posts_df['city_clean'] = job_posts_df.city.apply(lambda x : strip_city_name(x))
# this part is hard coded as I don't know other way to do it

# collect the words relative for each group

# abroad  in relative to Egypt

abroad = ['SHARJAH','Riyadh', 'travel jeddah' ,'الرياض', 'ابها', 'Riyadh, Saudi Arabia' ,'Riyadh, KSA','Dubai','Middle East','Global']

# remote work either online or from home

remote = ['Remote (Work from Home)','Remote','Remotely from home',  'manalonline' ,'Online From Home', 'Work from home']

# unspecified job posts

unknwon =['To be definied','outside cairo']

# west of Egypt

west_region =['North Coast','North Cost','Marsa Matrouh','Alamein','North Cost']

# Upper Eegypt and Red Sea areas

upper_redsea_regions =['Hurghada','ASWAN','Marsa Alam','qina','El Fayoum','Bani Suief','El Menya', 

                       'red sea','Sohag','el minya','Aswan/ Qena','Minya - Assiut - Sohag',

                       'Assuit-Menia-Mansoura-El Bhira-Sohag','Upper Egypt & Red Sea','Upper Egypt / Red Sea',

                       'Upper Egypt','الغردقة','Minia','Qena','ain sokhna','Red Sea/ Sinai',

                       'Elwadi Elgded','Owainat East','east oweinat','East Owinat','East Owainat','Owinat East',

                       'Asyut','Ein sokhna','al fayoum','El Menia','luxor','al-minya','Ain El Sokhna','sharm elsheikh',

                       'EL Minia','al ain al sokhna', 'Minya','Wadi El Notron','new valley','Assuit',

                       'Assiut','fayoum']

# sinai area

sinai_region =['Sharm Elkheikh','sinai','South sinai','sharm el shaik','Ariesh','Arish']

# Cario districts

cairo_regions =['kairo','great cairo','maadi','maadi,cairo','Maddi','nasr city' ,'Doki','Mohand','El Mohndseen','Ain Shams',

                'Dokki - Mohandseen','Mohandeseen','Heliopolis','obour' ,'Obour city','nozha' ,"Ma'adi", 'Obour','zamalek',

                'Abou Rawash','Misr- El Gedida','Helwan','5th Settelment','maady','El-Obour','elobour','EL Obour city',

                'elharam','new nozha','Dokki' ,'Ain Shams/ Helwan' , 'El Obour' , 'Mohandessin','Mokattam','alabassia',

                'Mohandiseen','Badr','حلوان','zahraa el maadi','التجمع الأول', 'Haram','Obuor city', 'El-shrouk','مدينة بدر',

                'new egypt','El Obour Industrial City','مدينه نصر', 'newegypt','15th of May','El-Sherouk'] 
def city_cleaner(k):

    "take k city name as a str and edit it to be as one of the unified categries of Areas as above mentioned lists "

    if k[0] =='c' or k[0] == 'C' and 'Canal' not in k or 'Cairo' in k or 'القاهر' in k or 'Down Town' in k or k in cairo_regions:

        k = 'Cairo'

    elif 'Alex' in k or 'alex' in k or 'الاسكندرية' in k:

        k ='Alexandria'

    elif '6' in k or 'October' in k or 'أكتوبر' in k or 'sheikh zayed' in k or 'Shiekh' in k or 'Sheikh' in k:

        k = 'October'

    elif 'G' in k and 'iza' in k or 'g' in k and 'iza' in k or 'الجيز' in k or k == 'Giaz' or 'GIZA' in k or 'Jizah' in k or k == 'gize':

        k = 'Giza'

    elif '10' in k :

        k = '10th Ramadan'

    elif 'All ' in k or 'all ' in k or 'Any ' in k or k == 'مصر' or k == 'all' or k =='Egype' or k =='Egypt' or 'anywhere' in k or 'al city' in k or 'Any' in k:

        k = 'All country'

    elif 'Isma' in k or 'Isam' in k or 'Port' in k or 'port' in k or 'Sue' in k or 'الإسماعيلية' in k or 'siuz' in k or 'Canal' in k:

        k = 'Canal'

    elif k in sinai_region:

        k = 'Sinai'

    elif k in west_region:

        k = 'West'

    elif k in upper_redsea_regions:

        k = 'Upper Egypt & Red Sea'

    elif k in abroad:

        k = 'Abroad'

    elif k in remote:

        k = 'Remotely'

    elif k in unknwon:

        k = 'other'

    else:

        k = 'Delta'

    return k
# fix the city column

job_posts_df.city_clean = job_posts_df.city_clean.apply(lambda x: city_cleaner(x))
# compare our results

job_posts_df.loc[:,['city','city_clean']].sample(5,random_state =7)
# check for the work

job_posts_df.city_clean.value_counts()
job_posts_df.career_level.unique()
# fix the career_level

job_posts_df['career_level'] = job_posts_df['career_level'].apply(lambda x: x.split('(')[0])
job_posts_df.career_level.unique()
job_posts_df.columns
# save the data

#applcs_df.to_csv('clean_applications.csv',encoding='UTF-8',index=False)

#job_posts_df.to_csv('clean_job_posts.csv',encoding='UTF-8',index=False)