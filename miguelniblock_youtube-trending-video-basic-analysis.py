import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))# we need to know the filename
# import the file using pandas read_csv function
usvids = pd.read_csv('../input/USvideos_modified.csv', index_col='video_id')
# let's see the columns
usvids.head(10)

# Let's see how many rows and columns we have,
# and how many of them have distinct values.

# Total number of rows and columns.
print(usvids.shape)
# Are there any duplicates?
print("video_id: "+ str(usvids.index.nunique()))
# Number of unique values for each column.
print(usvids.nunique())
#Looking for missing values and type of our data
usvids.info()
# Let's get a statistical summary of each numerical column.
usvids.describe()
plt.figure(figsize=(10,15))

plt.subplot(3,2,1)
plt.scatter(usvids.views, usvids.trend_day_count)
plt.xlabel('Views')
plt.ylabel('Trending Days Count')
plt.title('Views VS Trending Days')

plt.subplot(3,2,2)
plt.scatter(usvids.views, usvids['trend.publish.diff'])
plt.xlabel('Views')
plt.ylabel('Difference Between Publish & Trend Dates')
plt.title('Views VS Publish/Trend Difference')

plt.subplot(3,2,3)
plt.scatter(usvids.views, usvids['subscriber'])
plt.xlabel('Views')
plt.ylabel('Subscribers')
plt.title('Views VS Channel Subscribers')

plt.subplot(3,2,4)
plt.scatter(usvids.views, usvids['tags_count'])
plt.xlabel('Views')
plt.ylabel('Tags Count')
plt.title('Views VS Tags Count')

plt.subplot(3,2,5)
plt.scatter(usvids.views, usvids['tag_appeared_in_title_count'])
plt.xlabel('Views')
plt.ylabel('Tags in Title Count')
plt.title('Views VS Tags in Title')

plt.subplot(3,2,6)
plt.scatter(usvids.views, usvids['likes'])
plt.xlabel('Views')
plt.ylabel('Likes')
plt.title('Views VS Likes')

plt.tight_layout()
plt.show()
# We'll divide out dataset by wether the videos had their ratings disabled or not.
# We'll get a statistical summary of the relevant columns.
usvids.groupby('ratings_disabled').describe()[['views','tag_appeared_in_title_count','trend_day_count','trend.publish.diff','comment_count']]
# We'll divide out dataset by wether the videos had their comments disabled or not.
# We'll get a statistical summary of the relevant columns.
usvids.groupby('comments_disabled').describe()[['views','likes','dislikes','tag_appeared_in_title_count','trend_day_count','trend.publish.diff']]
plt.figure(figsize=(20,20))

plt.subplot(2,2,1)
usvids.groupby('ratings_disabled').agg(np.mean)['views'].plot(kind='bar',figsize=(10,10))
plt.title('Average Views- Ratings Disabled')

plt.subplot(2,2,2)
usvids.groupby('ratings_disabled').agg(np.mean)['comment_count'].plot(kind='bar',figsize=(10,10))
plt.title('Average Comments- Ratings Disabled')

plt.subplot(2,2,3)
usvids.groupby('comments_disabled').agg(np.mean)['likes'].plot(kind='bar',figsize=(10,10),color=['red','green'])
plt.title('Average Likes- Comments Disabled')

plt.subplot(2,2,4)
usvids.groupby('comments_disabled').agg(np.mean)['dislikes'].plot(kind='bar',figsize=(10,10),color=['red','green'])
plt.title('Average Dislikes- Comments Disabled')

plt.tight_layout()
plt.show()

# Perform T-Test and find P-Value
# Apply the natural logarithm to normalize the distributions
rat_dis = np.log(usvids[usvids.ratings_disabled == True].views)
rat_en = np.log(usvids[usvids.ratings_disabled == False].views)
from scipy.stats import ttest_ind
ttest_ind(rat_en, rat_dis, equal_var=False)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(usvids[usvids.ratings_disabled].views,range=(1,16000000),bins=30)
plt.axvline(np.mean(usvids[usvids.ratings_disabled].views),color='red')
plt.title('Distribution of Views in Videos with Ratings Disabled')
plt.xticks(rotation=90)

plt.subplot(1,2,2)
plt.hist(usvids[~usvids.ratings_disabled].views,range=(1,7000000),color='green',bins=30)
plt.title('Distribution of Views in Videos with Ratings Enabled')
plt.axvline(np.mean(usvids[~usvids.ratings_disabled].views),color='red')
plt.xticks(rotation=90)


plt.tight_layout()
plt.show()
usvids.publish_date = pd.to_datetime(usvids.publish_date)
# Let's plot these dates
pop_dates = usvids['publish_date'].value_counts().sort_index()
pop_dates = pop_dates[pop_dates.index > '2017-11-01']

plt.figure(figsize=(12,6))

plt.plot(pop_dates.index,pop_dates.values, color='red')
plt.xticks(rotation=90)
plt.title('Youtube\'s Trending Video Count by Date Published')
plt.axvline('2017-12-25',linestyle='dashed')
plt.axvline('2018-02-14',linestyle='dashed')
plt.text('2018-02-15',65,"Valentine's 2018",rotation=90)
plt.text('2017-12-26',65,"Christmas 2017",rotation=90)

plt.tight_layout()
plt.show()
# Let's see the most recurring publishing dates
print(usvids['publish_date'].value_counts().head(10))

#Let's visualize views by publish date of trending videos
dates_views = usvids.groupby('publish_date').agg(np.mean).sort_values('views',ascending=False).views.sort_index()
dates_views = dates_views[dates_views.index > '2017-11-01']
plt.figure(figsize=(12,6))

plt.plot(dates_views.index,dates_views.values, color='red')
plt.xticks(rotation=90)
plt.title('Youtube\'s Trending Video Views by Video Publish Date')
plt.axvline('2017-12-25',linestyle='dashed')
plt.axvline('2018-02-14',linestyle='dashed')
plt.text('2018-02-15',5000000,"Valentine's",rotation=90)
plt.text('2017-12-26',5000000,"Christmas",rotation=90)


plt.tight_layout()
plt.show()
# let's list the publishing dates with the highest number of views
dates_views.sort_values(ascending=False).head()
# separate each word in the tags column and add them onto a list of strings
# first split by '|' and send to a list.
tags = usvids.tags.str.split('|').tolist()
# then get rid of anything that isn't a list
tags = [x for x in tags if type(x) == list]

# that gave us a list of lists (of strings), so we must separate the items in each 
tags2 = []
tags3 = []
for item in tags:
    for string in item:
        # get rid of numbers and other types
        if type(string) == str:
            tags2.append(string)

def meaningless(x):
    words = ['to','the','a','of','and','on','in','for','is','&','with','you','video']
    return x in words

# now let's split these strings by the spaces between words
for multiple in tags2:
    singles = multiple.split()
    # then let's add these cleaned tags to the final list
    for tag in singles:
        # now let's make everything lowercase and get rid of spaces
        tag = tag.strip()
        tag = tag.lower()
        # now let's remove the meaningless tags   
        if not meaningless(tag):
            tags3.append(tag)

# let's bring that into a dataframe
tagsdf = pd.DataFrame(tags3,columns=['tags'])
# then count the values
tagcounts = tagsdf.tags.value_counts()

# now preparing a bar chart representing the top values
tagcountslice = tagcounts[:30].sort_values()
tagcountslice.plot(kind='barh',title='Most Popular Tags in Trending Videos',grid=True,fontsize=12,figsize=(11,8))
plt.xlabel('In How Many Videos the Tag Occurred')

plt.tight_layout()
plt.show()
# clean raw tags for each video and append them to a new list
# make another list with the views of each video
cleantagslist = []
tagsviews = []
count = 0
for rawtags in usvids.tags:
    try:
        cleantags = " ".join(" ".join(" ".join(" ".join(rawtags.split('|')).split()).split('(')).split(')')).strip().lower()
        cleantagslist.append(cleantags)
               
        count += 1
        tagsviews.append(usvids.views[count-1])
    except:
        ValueError
# let's show the cleaned tags for the first 5 videos
cleantagslist[:5]
# create a dataframe containing each video's cleaned tags and views
cleantagsdf = pd.DataFrame(columns=['tags','views'])
cleantagsdf['tags'] = cleantagslist
cleantagsdf['views'] = tagsviews
# now we have those cleaned tags in a dataframe along with their video views
cleantagsdf.head()
# make a list of unique tags. no repeated tags. no meaningless words
df = pd.DataFrame(" ".join(cleantagslist).split(),columns=['tags'])
uniquetagslist = df.tags.value_counts().keys()
uniquetagslist = [tag for tag in uniquetagslist if not meaningless(tag)]

# make a dataframe with each unique tag as the index and zeros on the 'views' column
# we will use this dataframe to count the views for each unique tag
tagsviewsdf = pd.DataFrame(index=uniquetagslist,columns=['views'])
tagsviewsdf = tagsviewsdf.views.fillna(0)
tagsviewsdf = pd.DataFrame(tagsviewsdf)

# show the dataframe where we'll count the views for each tag
tagsviewsdf.head()
tagsviewsdf.head()
# count the views for each unique tag and add them to above's dataframe
for unique in uniquetagslist:
    index = 0
    for tag in cleantagsdf.tags:
        index += 1
        if unique in tag:
            tagsviewsdf.views[index-1] += cleantagsdf.views[index-1]

# show the first tags along with their view count  
tagsviewsdf.head()
tagsviewsdf.views.sort_values(ascending=False)[:30]
# Now creating a bar chart of the top-viewed tags along with their view counts
tagsviewslice = tagsviewsdf.views.sort_values(ascending=False)[:30].sort_values()
tagsviewslice.plot(kind='barh',title='View Counts for Most-Viewed Tags in Trending Videos',grid=True,fontsize=12,figsize=(11,8))
plt.xlabel('Total Views By Videos Containing Each Tag')

plt.tight_layout()
plt.show()
