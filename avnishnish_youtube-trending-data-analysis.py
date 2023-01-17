import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import datetime as dt
from glob import glob
#Creating list of filenames
csv_files = glob('../input/*.csv')
json_files = glob('../input/*.json')

#Loading files into variables
df_list = list(map(lambda z: pd.read_csv(z,index_col='video_id'),
                                                             csv_files))
britain_js, germany_js, canada_js, france_js, usa_js = list(map(lambda a: json.load(open(a,'r')), 
                                                                            json_files))
df_list[4].head()
df_list[0].info()
def column_dropper(df):
    new_df = df.drop(columns=['description', 'tags', 'thumbnail_link'])
    return new_df

df_list2 = list(map(column_dropper, df_list)) 
df_list[0].head()
def category_dict_maker(js):
    items = js['items']
    item_id = []
    item_snippet_title = []
    for item in items:
        item_id.append(item['id']) 
        item_snippet_title.append(str(item['snippet']['title']))
    item_dict = dict(zip(item_id, item_snippet_title))
    return(item_dict)

brit_dict = category_dict_maker(britain_js)

def category_maker(value):
    for key in brit_dict:
        if str(value) == key:
            return (brit_dict[key])
        else:
            continue

def cat_applier(df):
    df['category'] = df.category_id.apply(func=category_maker)
    df.category = df.category.astype('category')
    return df.drop(columns=['category_id'])

df_list3 = list(map(cat_applier, df_list2))    
df_list3[0].head()

def string_convertor(string):
    yy=string[0:2]
    dd=string[3:5]
    mm=string[6:8]
    new_string = str("20"+yy+"-"+mm+"-"+dd)
    return new_string

def datetime_setter(df):
    df.trending_date = pd.to_datetime(df.trending_date.apply(string_convertor), errors='coerce')
    df.publish_time = pd.to_datetime(df.publish_time, errors='coerce')
    return df

df_list4 = list(map(datetime_setter, df_list3)) 
df_list4[0].head()
france, britain, canada, usa, germany = df_list4
britain['trending_delta'] = britain.trending_date - britain.publish_time
min_time = np.min(britain['trending_delta'])
max_time = np.max(britain['trending_delta'])


print("Fastest to trending:") 
print(britain[['title','trending_delta']].loc[britain['trending_delta']==min_time])
print("\nSlowest to trending:") ,
print(britain[['title','trending_delta']].loc[britain['trending_delta']==max_time],'\n')

print("Mean trending delta:", np.mean(britain['trending_delta']))
print("Median trending delta:", np.median(britain['trending_delta']))

sns.lmplot('views', 'likes', data=britain, hue='category', fit_reg=False);
plt.title('British Youtube Trending Section')
plt.xlabel('Views');
plt.ylabel('Likes');
plt.show()
sns.lmplot('views', 'likes', data=canada, hue='category', fit_reg=False);
plt.title('Canadian Youtube Trending Section')
plt.xlabel('Views');
plt.ylabel('Likes');
plt.show()
sns.countplot('category', data=britain)
plt.title('Category count plot for Britain')
plt.xlabel('Category')
plt.ylabel('Video Count')
plt.xticks(rotation=90)
plt.show()
sns.countplot('category', data=canada)
plt.title('Category count plot for Canada')
plt.xlabel('Category')
plt.ylabel('Video Count')
plt.xticks(rotation=90)
plt.show()