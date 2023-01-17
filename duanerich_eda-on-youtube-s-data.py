#Generic imports

import numpy as np

import pandas as pd

import json



#Plotting

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

from pprint import pprint

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)



#Statistical imports

import statsmodels.api as sm

from sklearn.preprocessing import MultiLabelBinarizer
USvideos = pd.read_csv('../input/USvideos.csv',encoding='utf8',error_bad_lines = False)

GBvideos = pd.read_csv('../input/GBvideos.csv',encoding='utf8',error_bad_lines = False)

UScomments = pd.read_csv('../input/UScomments.csv',encoding='utf8',error_bad_lines = False) 

GBcomments = pd.read_csv('../input/GBcomments.csv',encoding='utf8',error_bad_lines = False) 



with open('../input/US_category_id.json') as file:    

    US_category_id = json.load(file)



with open('../input/GB_category_id.json') as file:    

    GB_category_id = json.load(file)
print('US video head:')

USvideos.sample(frac=0.01).head()
print('GB video head:')

GBvideos.sample(frac=0.01).head() #I'd like a random sample..
print(GBvideos['video_id'].nunique() == GBvideos.shape[0]) #True if video_id is unique

print(USvideos['video_id'].nunique() == USvideos.shape[0])
GBvideos['id_date'] = GBvideos['video_id'].astype(str) + GBvideos['date'].astype(str)

USvideos['id_date'] = USvideos['video_id'].astype(str) + USvideos['date'].astype(str)



print(GBvideos['id_date'].nunique() == GBvideos.shape[0])

print(USvideos['id_date'].nunique() == USvideos.shape[0])
dup_id_dates = USvideos['id_date'][USvideos['id_date'].duplicated()].values

logi = [e in dup_id_dates for e in USvideos['id_date'].values]

dups_df = USvideos[logi].sort_values(by='id_date')

dups_df
USvideos = USvideos[[not e for e in logi]] 

print(USvideos['id_date'].nunique() == USvideos.shape[0])
pprint(GB_category_id)
pprint(US_category_id)
def reform_js(json_obj):

    relevant = json_obj['items']

    cat_ids, assignable, cats = [], [], []

    for item in json_obj['items']:

        cat_ids.append(int(item['id']))

        cats.append(item['snippet']['title'])

        assignable.append(item['snippet']['assignable'])

    return pd.DataFrame({'category_id':cat_ids,'category':cats,'assignable':assignable})



US_category_iddf = reform_js(US_category_id)

GB_category_iddf = reform_js(GB_category_id)



print("US Category Mapper:")

print(US_category_iddf.head())

print("\nGB Category Mapper:")

print(GB_category_iddf.head())
USvideos = USvideos.merge(US_category_iddf, on = ('category_id')).drop('category_id',axis=1)

GBvideos = GBvideos.merge(GB_category_iddf, on = ('category_id')).drop('category_id',axis=1)
UScomments.head()
GBcomments.head()
def form_hist(given_list,top_n):

    """

    Returns a sorted histogram dataframe (with top_n rows) for a given list.

    """

    item_set = set(given_list)

    items = []

    counts = []

    for nm in item_set:

        items.append(nm)

        counts.append(given_list.count(nm))

    return pd.DataFrame({'count':counts,'items':items}).sort_values(by='count',ascending=False).head(top_n)



def create_hist(videos, num, title):

    """

    Plots our histogram

    """

    all_tags = videos['tags'].map(lambda k: k.lower().split('|')).values

    all_tags = [item for sublist in all_tags for item in sublist]



    counts = form_hist(all_tags,num)

    counts.columns = ['count','tags']

    plt.figure()

    sns.barplot(x = counts['tags'], y = counts['count'])

    plt.xticks(rotation=90)

    plt.ylabel('count')

    plt.title(title)



create_hist(USvideos,20,'US - Most Frequent Tags')

create_hist(GBvideos,20,'GB - Most Frequent Tags')
def tags_matter(videos, k):

    """

    This function accepts our videos dataframe (like USvideos or GBvideos) 

    and returns the linear regression coefficient statistics (along with labels)

    for the top k-most frequent tags.

    """



    #Determine the top k tags

    videos = videos.copy()

    all_tags = videos['tags'].map(lambda k: k.lower().split('|'))

    all_tags = [item for sublist in all_tags for item in sublist]

    counts = form_hist(all_tags,k)

    top_tags = counts['items'].values[:k]



    def filter_f(x):

        x = x.lower().split('|')

        return [e for e in x if e in top_tags]



    #Reduce tags to only the most frequent ones

    videos['tags'] = videos['tags'].map(filter_f)



    #Convert our data into the design matrix

    mlb = MultiLabelBinarizer()

    design = mlb.fit_transform(videos['tags'])

    design = sm.add_constant(design)

    

    #Fit linear regression

    ols = sm.OLS(videos['views'].values, design)

    fitting = ols.fit()

    labels = ['intercept'] + list(mlb.classes_)

    return fitting.summary(), labels



top_n_tags = 20

USsummary, uslabels = tags_matter(USvideos, top_n_tags)

print("US VIDEOS")

pprint(USsummary)
GBsummary, gblabels = tags_matter(GBvideos, top_n_tags)

print("GB VIDEOS")

pprint(GBsummary)
print(gblabels[14])
def form_curve(videos):

    dates = sorted(list(set(videos['date'])))

    max_diff = int(max(videos['date']))-int(min(videos['date']))

    curves = []

    weights = []

    

    #For each start date (all days except the last date), calculate the curve indicating indicating the probability a 

    #video will trend t days from now.

    for i in range(max_diff):

        dt = dates[i]

        curr_videos = set(videos['video_id'][videos['date']==dt])

        weights.append(len(curr_videos))

        curve = []

        for j in range(i + 1, max_diff + 1):

            run_videos = set(videos['video_id'][videos['date']==dates[j]])

            curve.append(np.mean([b in curr_videos for b in run_videos]))

        if curve:

            curves.append(curve)

        

    #Average all the curves together.

    final_curve = []

    for t in range(max_diff):

        denom = 0

        val = 0

        for i in range(len(curves)):

            if len(curves[i]) - 1 >= t:

                val += curves[i][t]*weights[i]

                denom += weights[i]

        final_curve.append(val/denom)

    

    return pd.Series(final_curve,index=list(range(1,len(final_curve)+1)))



plt.figure()

form_curve(USvideos).plot()

plt.xlabel('days')

plt.title('US - Probability trending t days from now, given video is trending now.')



plt.figure()

form_curve(GBvideos).plot()

plt.xlabel('days')

plt.title('GB - Probability trending t days from now, given video is trending now.')