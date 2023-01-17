import numpy as np # library for linear algebra

import pandas as pd # library for data processing, CSV file I/O

import matplotlib.pyplot as plt # library for data visualization

import seaborn as sns # library for data visualization

import warnings

warnings.filterwarnings('ignore')

# %matplotlib inline

# sns.set()
USVids = pd.read_csv("../input/USvideos.csv", error_bad_lines=False) # reading csv file containing YouTube US video data

USVids = USVids.drop_duplicates(subset='video_id', keep="last") # checking for possible duplicates, removing
USVids['enthusiasm'] = USVids['likes'] / (USVids['likes'] + USVids['dislikes']) # creating new column enthusiasm (measured per video as percentage of total ratings as likes)
CatIDAverageEnthusiasm = USVids.groupby(['category_id'])['enthusiasm'].mean() # grouping USVids by category_id, then obtaining average enthusiasm rating per category

CatIDAverageEnthusiasm
my_plot = CatIDAverageEnthusiasm.plot(kind='bar',title="Average Enthusiasm Ratings of Individual Video Categories") # plotting bar graph of average enthusiasm ratings for individual video categories
f = {'views': ['sum'], 'enthusiasm': ['mean']} # define dictionary

USVidsByChannel = USVids.groupby(['channel_title']).agg(f) # grouping USVids by channel_title and obtaining total views and average enthusiasm rating

USVidsByChannel
USVidsByChannel.plot(kind='scatter',title="Correlation Between Total Viewcount and Average Enthusiasm Rating for Individual Channels",x='views',y='enthusiasm'); # plotting scatter graph comparing individual channels' total views and average enthusiasm ratings
USVids.plot(kind='scatter',title="Correlation Between Total Viewcount and Average Enthusiasm Rating for Individual Videos",x='views', y='enthusiasm'); # plotting scatter graph comparing individual videos' total views and average enthusiasm ratings
SplitTitle = USVids.title.str.split('\s+',expand=True).stack() # creating new column SplitTitle by splitting column title in USVids; new rows created for each individual SplitTitle value (values for other attributes retained from original, pre-split row)

USVidsSplitTitle = USVids.join(pd.Series(index=SplitTitle.index.droplevel(1), data=SplitTitle.values, name='SplitTitle')) # creating USVidsSplitTitle by joining SplitTitle to USVids



USVidsSplitTitle['SplitTitle'] = USVidsSplitTitle['SplitTitle'].replace("\W","",regex=True) # filter out special characters from column SplitTitle

USVidsSplitTitle['SplitTitle'] = USVidsSplitTitle['SplitTitle'].str.lower() # converting all words to lowercase

USVidsSplitTitle = USVidsSplitTitle[USVidsSplitTitle['SplitTitle'] != ''] # removing rows with blank values

USVidsSplitTitle = USVidsSplitTitle.drop_duplicates() # removing repeated instances of the same word in a single video title

USVidsSplitTitle = USVidsSplitTitle[~USVidsSplitTitle['SplitTitle'].isin(["and","but","or",

                                                                          "the","a","an","is","was",

                                                                          "with","at","from","into","during","including","until","against","among","throughout",

                                                                          "despite","towards","upon","concerning","of","to","in","for","on","by",

                                                                          "about","like","through","over","before","between","after","since","without","under"])] # drop rows where SplitTitle value is a basic English article/linking verb/preposition



WordsPerTitle = USVidsSplitTitle.groupby(['video_id'])['video_id'].agg({'word_count':'count'}) # grouping USVidsSplitTitle by video_id and obtaining word count of video title

WordsPerTitle



USVidsSplitTitle = USVidsSplitTitle.join(WordsPerTitle, on=['video_id'], how='inner') # inner joining USVidsSplitTitle with WordsPerTitle

USVidsSplitTitle = USVidsSplitTitle.sort_values('video_id') # sorting by video_id
WordsByInstanceCount = USVidsSplitTitle.groupby('SplitTitle').agg({'SplitTitle':'count', 'views':'sum'}) # creating WordsByInstanceCount by grouping USVidsSplitTitle by SplitTitle and obtaining number of instances and sum of views for each SplitTitle/word

WordsByInstanceCount = WordsByInstanceCount.rename(columns={'SplitTitle':'word_frequency', 'views':'word_total_views'}) # renaming columns
WordsByInstanceCount['views_over_frequency'] = WordsByInstanceCount['word_total_views']/WordsByInstanceCount['word_frequency'] # creating new column views_over_frequency

WordsByInstanceCount = WordsByInstanceCount.sort_values('views_over_frequency', ascending=False) # sorting by views_over_frequency

WordsByInstanceCount
USVidsSplitTitleWithUniquenessScore = USVidsSplitTitle.join(WordsByInstanceCount, on=['SplitTitle'], how='inner') # inner joining USVidsSplitTitle with WordsByInstanceCount



USVidsFrequencySum = USVidsSplitTitleWithUniquenessScore.groupby(['video_id'])['word_frequency'].agg({'frequency_sum':'sum'}) # grouping by video_id, then obtaining frequency_sum by adding individual frequency scores for each word in a title



USVidsSplitTitleWithUniquenessScore = USVidsSplitTitleWithUniquenessScore.join(USVidsFrequencySum, on=['video_id'], how='inner') # inner joining USVidsSplitTitleWithUniquenessScore with USVidsFrequencySum

USVidsSplitTitleWithUniquenessScore = USVidsSplitTitleWithUniquenessScore.sort_values('video_id') # sorting by video_id

USVidsSplitTitleWithUniquenessScore['uniqueness'] = USVidsSplitTitleWithUniquenessScore['frequency_sum'] / USVidsSplitTitleWithUniquenessScore['word_count'] # adding new column uniqueness (measured as frequency_sum over total word_count per video)



USVidsSplitTitleWithUniquenessScoreGrouped = USVidsSplitTitleWithUniquenessScore.groupby(['video_id'], as_index=False).last() # creating USVidsSplitTitleWithUniquenessScoreGrouped by grouping USVidsSplitTitleWithUniquenessScore by video_id

USVidsSplitTitleWithUniquenessScoreGrouped = USVidsSplitTitleWithUniquenessScoreGrouped.filter(items=['video_id', 'title', 'views', 'word_count', 'frequency_sum', 'uniqueness']) # filtering to only include relevant columns

USVidsSplitTitleWithUniquenessScoreGrouped
USVidsSplitTitleWithUniquenessScoreGrouped.plot(kind='scatter',title="Correlation Between Uniqueness of Title Word Choice and Total Views per Video",x='uniqueness', y='views'); # plotting scatter graph comparing individual videos' total views and uniqueness rating
print("Average value for uniqueness:", USVidsSplitTitleWithUniquenessScoreGrouped['uniqueness'].mean()) # average value for uniqueness across all videos

sns.distplot(USVidsSplitTitleWithUniquenessScoreGrouped['uniqueness']); # plotting distplot for uniqueness