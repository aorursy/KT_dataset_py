# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualizations

import matplotlib.pyplot as plt # visualizations

from scipy import stats # stats.mode and stats.norm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# df is the dataframe object from pandas
df = pd.read_csv("../input/USvideos.csv")

# display the first 10 rows of df
df.head(10)
# just practicing here ...
date_pattern = r"[0-9]+\.[0-9]+\.[0-9]+"
date_pattern_2017 = r"17\.[0-9]+\.[0-9]+"
date_pattern_november = r"[0-9]+\.[0-9]+\.11"

df_with_no_tags = df[df.tags == "[none]"]
df_only_2017_trends = df[df.trending_date.str.contains(date_pattern_2017)]
df_only_november_trends = df[df.trending_date.str.contains(date_pattern_2017)]

new_filter = (df.trending_date.str.contains(date_pattern))
new_df = df[new_filter]

df.info()
print("-"*50)
df_with_no_tags.info()
print("-"*50)
df_only_2017_trends.info()
print("-"*50)
new_df.info()
print("-"*50)
# regex for each month
date_pattern_jan = r"[0-9]+\.[0-9]+\.01" 
date_pattern_feb = r"[0-9]+\.[0-9]+\.02" 
date_pattern_mar = r"[0-9]+\.[0-9]+\.03" 
date_pattern_apr = r"[0-9]+\.[0-9]+\.04" 
date_pattern_may = r"[0-9]+\.[0-9]+\.05" 
date_pattern_jun = r"[0-9]+\.[0-9]+\.06" 
date_pattern_jul = r"[0-9]+\.[0-9]+\.07" 
date_pattern_aug = r"[0-9]+\.[0-9]+\.08" 
date_pattern_sep = r"[0-9]+\.[0-9]+\.09" 
date_pattern_oct = r"[0-9]+\.[0-9]+\.10" 
date_pattern_nov = r"[0-9]+\.[0-9]+\.11" 
date_pattern_dec = r"[0-9]+\.[0-9]+\.12"
# filters for each month (returns true if contains pattern ... else false)
jan_filter = (df.trending_date.str.contains(date_pattern_jan))
feb_filter = (df.trending_date.str.contains(date_pattern_feb))
mar_filter = (df.trending_date.str.contains(date_pattern_mar))
apr_filter = (df.trending_date.str.contains(date_pattern_apr))
may_filter = (df.trending_date.str.contains(date_pattern_may))
jun_filter = (df.trending_date.str.contains(date_pattern_jun))
jul_filter = (df.trending_date.str.contains(date_pattern_jul))
aug_filter = (df.trending_date.str.contains(date_pattern_aug))
sep_filter = (df.trending_date.str.contains(date_pattern_sep))
oct_filter = (df.trending_date.str.contains(date_pattern_oct))
nov_filter = (df.trending_date.str.contains(date_pattern_nov))
dec_filter = (df.trending_date.str.contains(date_pattern_dec))
# make new dataframe, append 'month' column
jan_df = df[jan_filter]
jan_df['month'] = "jan"
feb_df = df[feb_filter]
feb_df['month'] = "feb"
mar_df = df[mar_filter]
mar_df['month'] = "mar"
apr_df = df[apr_filter]
apr_df['month'] = "apr"
may_df = df[may_filter]
may_df['month'] = "may"
jun_df = df[jun_filter]
jun_df['month'] = "jun"
jul_df = df[jul_filter]
jul_df['month'] = "jul"
aug_df = df[aug_filter]
aug_df['month'] = "aug"
sep_df = df[sep_filter]
sep_df['month'] = "sep"
oct_df = df[oct_filter]
oct_df['month'] = "oct"
nov_df = df[nov_filter]
nov_df['month'] = "nov"
dec_df = df[dec_filter]
dec_df['month'] = "dec"
# get info about each month
print("-"*50+"jan"+"-"*50)
jan_df.info()
print("-"*50+"feb"+"-"*50)
feb_df.info()
print("-"*50+"mar"+"-"*50)
mar_df.info()
print("-"*50+"apr"+"-"*50)
apr_df.info()
print("-"*50+"may"+"-"*50)
may_df.info()
print("-"*50+"jun"+"-"*50)
jun_df.info()
print("-"*50+"jul"+"-"*50)
jul_df.info()
print("-"*50+"aug"+"-"*50)
aug_df.info()
print("-"*50+"sep"+"-"*50)
sep_df.info()
print("-"*50+"oct"+"-"*50)
oct_df.info()
print("-"*50+"nov"+"-"*50)
nov_df.info()
print("-"*50+"dec"+"-"*50)
dec_df.info()
# look at the first 5 rows of jan_df
jan_df.head()
# recombine all the month data
month_array = [jan_df, feb_df, mar_df, apr_df, may_df, jun_df, jul_df, aug_df, sep_df, oct_df, nov_df, dec_df]
month_df = pd.concat(month_array)

sns.boxplot(x='month', y='views', data=month_df)
# get info about jul...oct
print("-"*50+"jul"+"-"*50)
jul_df.info()
print("-"*50+"aug"+"-"*50)
aug_df.info()
print("-"*50+"sep"+"-"*50)
sep_df.info()
print("-"*50+"oct"+"-"*50)
oct_df.info()

# look at the frist 5 rows of a df with just july for trending_date
df[(df.trending_date.str.contains(date_pattern_jul))].head()
# ignore regex warnings from pandas ... https://stackoverflow.com/q/39901550/5411712
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

weirdValuesRegex = r"[0-9]+\.[0-9]+\.([1][3-9]+|[2][0-9]|[3][0-9]|[4][0-9]|[5][0-9]|[6][0-9]|[7][0-9]|[8][0-9]|[9][0-9]|[^0][a-zA-Z]|[a-zA-Z][a-zA-Z])"
weirdValuesFilter = (df.trending_date.str.contains( weirdValuesRegex ))
df[weirdValuesFilter].head()
# filter for november in publish_time
new_nov_regex = r"^[0-9]+\-11+\-[0-9]+T"
new_nov_filter = (df.publish_time.str.contains( new_nov_regex ))
df[new_nov_filter].head()
# filter for july in publish_time
new_jul_regex = r"^[0-9]+\-07+\-[0-9]+T"
new_jul_filter = (df.publish_time.str.contains( new_jul_regex ))
df[new_jul_filter].head()
# regex for each publish_time month
date_pattern_jan = r"^[0-9]+\-01+\-[0-9]+T"
date_pattern_feb = r"^[0-9]+\-02+\-[0-9]+T"
date_pattern_mar = r"^[0-9]+\-03+\-[0-9]+T"
date_pattern_apr = r"^[0-9]+\-04+\-[0-9]+T"
date_pattern_may = r"^[0-9]+\-05+\-[0-9]+T"
date_pattern_jun = r"^[0-9]+\-06+\-[0-9]+T"
date_pattern_jul = r"^[0-9]+\-07+\-[0-9]+T"
date_pattern_aug = r"^[0-9]+\-08+\-[0-9]+T"
date_pattern_sep = r"^[0-9]+\-09+\-[0-9]+T"
date_pattern_oct = r"^[0-9]+\-10+\-[0-9]+T"
date_pattern_nov = r"^[0-9]+\-11+\-[0-9]+T"
date_pattern_dec = r"^[0-9]+\-12+\-[0-9]+T"

# filters for each publish_time month (returns true if contains pattern ... else false)
jan_filter = (df.publish_time.str.contains(date_pattern_jan))
feb_filter = (df.publish_time.str.contains(date_pattern_feb))
mar_filter = (df.publish_time.str.contains(date_pattern_mar))
apr_filter = (df.publish_time.str.contains(date_pattern_apr))
may_filter = (df.publish_time.str.contains(date_pattern_may))
jun_filter = (df.publish_time.str.contains(date_pattern_jun))
jul_filter = (df.publish_time.str.contains(date_pattern_jul))
aug_filter = (df.publish_time.str.contains(date_pattern_aug))
sep_filter = (df.publish_time.str.contains(date_pattern_sep))
oct_filter = (df.publish_time.str.contains(date_pattern_oct))
nov_filter = (df.publish_time.str.contains(date_pattern_nov))
dec_filter = (df.publish_time.str.contains(date_pattern_dec))

# make new dataframe (overwrite old variables), append 'month' column
jan_df = df[jan_filter]
jan_df['month'] = "jan"
feb_df = df[feb_filter]
feb_df['month'] = "feb"
mar_df = df[mar_filter]
mar_df['month'] = "mar"
apr_df = df[apr_filter]
apr_df['month'] = "apr"
may_df = df[may_filter]
may_df['month'] = "may"
jun_df = df[jun_filter]
jun_df['month'] = "jun"
jul_df = df[jul_filter]
jul_df['month'] = "jul"
aug_df = df[aug_filter]
aug_df['month'] = "aug"
sep_df = df[sep_filter]
sep_df['month'] = "sep"
oct_df = df[oct_filter]
oct_df['month'] = "oct"
nov_df = df[nov_filter]
nov_df['month'] = "nov"
dec_df = df[dec_filter]
dec_df['month'] = "dec"
# recombine all the month data
month_array = [jan_df, feb_df, mar_df, apr_df, may_df, jun_df, jul_df, aug_df, sep_df, oct_df, nov_df, dec_df]
month_df = pd.concat(month_array)

sns.boxplot(x='month', y='views', data=month_df)
# get info about jul...oct
print("-"*50+"jul"+"-"*50)
jul_df.info()
print("-"*50+"aug"+"-"*50)
aug_df.info()
print("-"*50+"sep"+"-"*50)
sep_df.info()
print("-"*50+"oct"+"-"*50)
oct_df.info()

jul_df.head(20) # there's only 10 videos in the month of july??? weird...
print(str(df.shape) + " \n\tis the output of `df.shape` \n\twhich is a " + str(type(df.shape)) + " \n\tthat tells us there are " + str(df.shape[0]) + " rows and " + str(df.shape[1]) + " columns")
print("\n"+"-"*70+"\n")
print(str(df.size) + " \n\tis the output of `df.size` \n\twhich is a " + str(type(df.size)) + "\n\tthat tells us there are " + str(df.size) + " total elements, \n\taka rows*cols = "+str(df.shape[0]) + "*" + str(df.shape[1]) + " = " +str(df.shape[0]*df.shape[1]))
print("\n"+"-"*70+"\n")
print(str(len(df)) + " \n\tis the output of `len(df)` \n\twhich is a " + str(type(len(df))) + "\n\tthat tells us there are " + str(df.size) + " rows")
print("\n"+"-"*70+"\n")
print(str(df.columns) + " \n\n\tis the output of `df.columns` \n\twhich is a " + str(type(df.columns)) + " \n\tthat shows us the column names in our dataframe... why is it not just an array? probs for extra fancy functions")
print("\n"+"-"*70+"\n")

print("here are the first 5 tags in df via the command `df['tags'][:5]`")
print(df['tags'][:5])

print("-"*70)
print("here are the first 5 views in df via the command `df['views'][:5]`")
print(df['views'][:5])

print("-"*70)
print("here are the second index for tags column in df via the command `df['tags'][:1]`\n")

print(df['tags'][1])
print("\n"+"="*70+"\n")

print("here are some terms")
print("df = dataframe ... it's basically a fancy Pandas object to represent a matrix. Why didn't they just call it a matrix? idk man.")
print("df['views'] gives us a single column. That's known as a 'series' ... just a 1D array (or 'list' in python)")
print("df['views'][1] gives us a single 'value' from the views array.")

print("\n"+"="*70+"\n")
print("here are the first 5 views and tags in df via the command `df[['views', 'tags']][:5]`")
print(df[['views', 'tags']][:5])

print("\n"+"="*70+"\n")
print("this is a good way to just get the 'features' you care about to optimize performance by having less data to move around.")
print("\n"+"="*70+"\n")
print("here is how to sort df by views and just see the top 5 videos with LEAST views")
print(df.sort_values(['views'])[['views', 'title']][:5])

print("\n"+"="*70+"\n")

print("here is how to sort df by views and just see the top 5 videos with MOST views")
print(df.sort_values(['views'], ascending=False)[['views', 'title']][:5])

viewsMean = np.mean(df['views'])
viewsMedian = np.median(df['views'])
viewsMode = stats.mode(df['views'])
viewsStd = df['views'].std() # standard deviation
viewsVar = df['views'].var() # variance
print("stats for `views`"
      + "\n\tmean \t=\t" + str(viewsMean)
      + "\n\tmedian \t=\t" + str(viewsMedian)
      + "\n\tmode \t=\t" + str(viewsMode) + "\n\t\t\t\t... meaning " + str(viewsMode[0]) + " shows up " + str(viewsMode[1]) + " times"
      + "\n\tstd \t=\t" + str(viewsStd)
      + "\n\tvar \t=\t" + str(viewsVar)
     )

# histogram for views with 50 bins
plt.hist(df['views'], 50)
plt.show()
# https://matplotlib.org/users/pyplot_tutorial.html
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20]) # [xmin, xmax, ymin, ymax]
plt.ylabel('y-axis')
plt.xlabel('x-axis')
plt.title('title')
plt.grid(True)
plt.text(1, 5, "some text $\mu=number,\ \sigma=number$")
plt.annotate('WOW! a point!', xy=(4, 16), xytext=(1, 17.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.show()
# xkcd style mode ;)
plt.xkcd() # https://matplotlib.org/xkcd/examples/showcase/xkcd.html
# https://matplotlib.org/users/pyplot_tutorial.html
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20]) # [xmin, xmax, ymin, ymax]
plt.ylabel('y-axis')
plt.xlabel('x-axis')
plt.title('title')
plt.grid(True)
plt.text(1, 5, "some text $\mu=number,\ \sigma=number$")
plt.annotate('WOW! a point!', xy=(4, 16), xytext=(1, 17.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.show()
# turn off xkcd mode
plt.rcdefaults()
plt.plot(df['views'], stats.norm.pdf(df['views']), "bo")
plt.plot(df['views'], stats.expon.pdf(df['views']), 'bo')
# shamelessly copying code...

plt.xkcd() # just for fun

df_yout = df

df_yout['likes_log'] = np.log(df_yout['likes'] + 1)
df_yout['views_log'] = np.log(df_yout['views'] + 1)
df_yout['dislikes_log'] = np.log(df_yout['dislikes'] + 1)
df_yout['comment_log'] = np.log(df_yout['comment_count'] + 1)

plt.figure(figsize = (12,6))

plt.subplot(221)
g1 = sns.distplot(df_yout['views_log'])
g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)

plt.subplot(222)
g4 = sns.distplot(df_yout['comment_log'])
g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)

plt.subplot(223)
g3 = sns.distplot(df_yout['dislikes_log'], color='r')
g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)

plt.subplot(224)
g2 = sns.distplot(df_yout['likes_log'],color='green')
g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)

plt.subplots_adjust(wspace=0.2, hspace=0.6)

plt.show()

plt.rcdefaults() # no more fun

# https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
        "The dog.",
        "The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print("vectorizer.vocabulary_: \n\t" + str(vectorizer.vocabulary_))
# TODO: figure out what IDF means
## IDF = Inverse Document Frequency: This downscales words that appear a lot across documents.
## https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html
print("vectorizer.idf_: \n\t" + str(vectorizer.idf_))
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print("vector.shape: \n\t" + str(vector.shape))
print("vector.toarray(): \n\t" + str(vector.toarray()))
# try the same stuff as above... but use the YouTube `title` data
# list of text documents
text = df['title']
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print("vectorizer.vocabulary_: \n\t" + str(vectorizer.vocabulary_))
print("vectorizer.idf_: \n\t" + str(vectorizer.idf_))
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print("vector.shape: \n\t" + str(vector.shape))
print("vector.toarray(): \n\t" + str(vector.toarray()))