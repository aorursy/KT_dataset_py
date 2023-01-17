# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#!pip install --upgrade pip
#!pip install numpy pandas matplotlib seaborn wordcloud emoji --upgrade
import os          

os.getcwd()
os.chdir('/kaggle/')
        
os.getcwd()                     # Check the working directory again
os.listdir('/kaggle/input')
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import emoji
from collections import Counter
#!pip install seaborn
#conda install seaborn
#import seaborn as sns
def rawToDf(file, key):
    split_formats = {
        '12hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',
        '24hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        'custom' : ''
    }
    datetime_formats = {
        '12hr' : '%d/%m/%Y, %I:%M %p - ',
        '24hr' : '%d/%m/%Y, %H:%M - ',
        'custom': ''
    }
    
    with open(file, 'r') as raw_data:
        raw_string = ' '.join(raw_data.read().split('\n')) # converting the list split by newline char. as one whole string as there can be multi-line messages
        user_msg = re.split(split_formats[key], raw_string) [1:] # splits at all the date-time pattern, resulting in list of all the messages with user names
        date_time = re.findall(split_formats[key], raw_string) # finds all the date-time patterns
        
        df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg}) # exporting it to a df
        
    # converting date-time pattern which is of type String to type datetime,
    # format is to be specified for the whole string where the placeholders are extracted by the method 
    df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[key])
    
    # split user and msg 
    usernames = []
    msgs = []
    for i in df['user_msg']:
        a = re.split('([\w\W]+?):\s', i) # lazy pattern match to first {user_name}: pattern and spliting it aka each msg from a user
        if(a[1:]): # user typed messages
            usernames.append(a[1])
            msgs.append(a[2])
        else: # other notifications in the group(eg: someone was added, some left ...)
            usernames.append("grp_notif")
            msgs.append(a[0])

    # creating new columns         
    df['user'] = usernames
    df['msg'] = msgs

    # dropping the old user_msg col.
    df.drop('user_msg', axis=1, inplace=True)
    
    return df
df = rawToDf('input/whatsapp-chat/WhatsApp_Chat_with_ECE_2017.txt', '24hr')
df
df.shape # no. of msgs
me_df = df[df['user'] == 'Tapendra'] 
me_df
me = "Tapendra"
images = df[df['msg']=="<Media omitted> "] #no. of images, images are represented by <media omitted>
images.shape
df["user"].unique() #total user 
grp_notif = df[df['user']=="grp_notif"] #no. of grp notifications
grp_notif.shape
grp_notif
df.drop(images.index, inplace=True) #removing images
df.drop(grp_notif.index, inplace=True) #removing grp_notif
df.tail()
df.reset_index(inplace=True, drop=True)
df.shape
#Who is the most active member of the group
df.groupby("user")["msg"].count().sort_values(ascending=False)
# Count of all the emojis that Rasheed have used?

emoji_ctr = Counter()
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
for idx, row in df.iterrows():
    if row["user"] == 'Rasheed':
        emojis_found = r.findall(row["msg"])
        for emoji_found in emojis_found:
            emoji_ctr[emoji_found] += 1
for item in emoji_ctr.most_common(10):
    print(item[0] + " - " + str(item[1]))
# What can Jatin activity say about my sleep cycle?
df['hour'] = df['date_time'].apply(lambda x: x.hour)
df[df['user']=='Jatin'].groupby(['hour']).size().sort_index().plot(x="hour", kind='bar')
#What is the difference in Weekend vs Weekday usage pattern?
df['weekday'] = df['date_time'].apply(lambda x: x.day_name()) # can use day_name or weekday from datetime 
df['is_weekend'] = df.weekday.isin(['Sunday', 'Saturday'])
msgs_per_user = df['user'].value_counts(sort=True)
msgs_per_user
top5_users = msgs_per_user.index.tolist()[:5]
top5_users
df_top5 = df.copy()
df_top5 = df_top5[df_top5.user.isin(top5_users)]
df_top5.head()
plt.figure(figsize=(30,10))
sns.countplot(x="user", hue="weekday", data=df)
df_top5['is_weekend'] = df_top5.weekday.isin(['Sunday', 'Saturday'])
plt.figure(figsize=(20,10))
sns.countplot(x="user", hue="is_weekend", data=df_top5)
def word_count(val):
    return len(val.split())
df['no_of_words'] = df['msg'].apply(word_count)
df_top5['no_of_words'] = df_top5['msg'].apply(word_count)
total_words_weekday = df[df['is_weekend']==False]['no_of_words'].sum()
total_words_weekday
total_words_weekend = df[df['is_weekend']]['no_of_words'].sum()
total_words_weekend
total_words_weekday/5 # average words on a weekday
total_words_weekend/2 # average words on a weekend
df.groupby('user')['no_of_words'].sum().sort_values(ascending=False)
(df_top5.groupby('user')['no_of_words'].sum()/df_top5.groupby('user').size()).sort_values(ascending=False)
wordPerMsg_weekday_vs_weekend = (df_top5.groupby(['user', 'is_weekend'])['no_of_words'].sum()/df_top5.groupby(['user', 'is_weekend']).size())
wordPerMsg_weekday_vs_weekend
wordPerMsg_weekday_vs_weekend.plot(kind='barh')
#Most Usage - Time of Day
x = df.groupby(['hour', 'weekday'])['msg'].size().reset_index()
x2 = x.pivot("hour", 'weekday', 'msg')
x2.head()
days = ["Monday", 'Tuesday', "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.heatmap(x2[days].fillna(0), robust=True)
#In any group, do Yash Iiitnr have any inclination towards responding to someone?
my_msgs_index = np.array(df[df['user']== 'Yash Iiitnr'].index)
print(my_msgs_index, my_msgs_index.shape)
prev_msgs_index = my_msgs_index - 1
print(prev_msgs_index, prev_msgs_index.shape)
df_replies = df.iloc[prev_msgs_index].copy()
df_replies.shape
df_replies.groupby(["user"])["msg"].size().sort_values().plot(kind='barh')
#Which are the most common words?
comment_words = ' '
stopwords = STOPWORDS.update(['lo', 'ge', 'Lo', 'illa', 'yea', 'ella', 'en', 'na', 'En', 'yeah', 'alli', 'ide', 'okay', 'ok', 'will'])
  
for val in df.msg.values: 
    val = str(val) 
    tokens = val.split() 
        
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
wordcloud.to_image()
