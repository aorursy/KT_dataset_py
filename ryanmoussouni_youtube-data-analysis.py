import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

data = pd.read_csv('../input/youtube-new/USvideos.csv')

print(data.columns)
## -- Checking for nan

print(data.isnull().sum(axis=0).to_string())
## -- Deleting entries with ratings disabled or video error

disabledIdx = np.union1d(np.where(data['ratings_disabled'])[0],np.where(data['video_error_or_removed'])[0])

data = data.drop(disabledIdx,axis=0)

data = data.drop(['ratings_disabled','video_error_or_removed'],axis = 1)

print('Proportion of lines deleted %f' %(len(disabledIdx)/data.shape[0]))
## -- Same video can appear in the dataset more than once

print(data.groupby(['video_id']).size().head(7).to_string())

print('\n On average, a video appears %i times in the dataset'%int(data.groupby(['video_id']).size().mean()))
## -- Engagement. Shows if you video has fostered a lot of reactions. 

alpha = 5 

data['engagementRate'] = (data['likes'] + data['dislikes'] + data['comment_count']*alpha) / ((2+alpha)*data['views']) #Comments value more than likes when it comes to commitment. 

print(data.columns)
## -- Quality. Shows how well was your video received

data['qualityScore'] = data['likes']/ (data['likes']+data['dislikes'])

print(data.columns)
## -- LogViews

data['logViews'] = np.log(data['views'])

print(data.columns)
## -- Views

data['logViews'].hist(bins = 50)

mean = data['views'].mean()

std = data['views'].std()

print(data['views'])

print('mean: %i' %int(mean))

print('std: %i' %int(std))

print('min: %i' %min(data['views']))
print('First Quantile: %i' %int(data['views'].quantile(0.1)))
 ## -- Engagement

data['engagementRate'].hist(bins = 50)

mean = data['engagementRate'].mean()

std = data['engagementRate'].std()

print(data['engagementRate'])

print('mean: %f' %mean)

print('std: %f' %std)
## -- Quality

data['qualityScore'].hist(bins = 50)

mean = data['qualityScore'].mean()

std = data['qualityScore'].std()

print(data['qualityScore'])

print('mean: %f' %mean)

print('std: %f' %std)
## This phenomenon is more visible on the box plot of the statistical series.

plt.show(data.boxplot('qualityScore',showfliers=False)) #outliers not shown
## -- correlation between likes views and dislikes

data[['views','likes','dislikes']].corr()
## -- Clickbait techniques

def capitalizedTitle(title: str)->bool:

    for word in title.split():

        if word.isupper():

            return True

    return False

data['capitalizedTitle'] = data['title'].apply(capitalizedTitle)

print('proportion of capitalized title : %i %% ' %int(data['capitalizedTitle'].sum()/data.shape[0] * 100))
capitalizedData = data[data['capitalizedTitle'] == True]

nonCapitalizedData = data[data['capitalizedTitle'] == False]

print('capitalizedData: \n \n',capitalizedData['title'].head(7).to_string())

print('\n \n nonCapitalizedData: \n \n',nonCapitalizedData['title'].head(7).to_string())

## -- Mean quality 

print('\n \n mean qualityScore: \n \n capitalizedData: %i %% \n nonCapitalizedData: %i %%'%(100*capitalizedData['qualityScore'].mean(),100*nonCapitalizedData['qualityScore'].mean()))



## -- Engagement Rate

print('\n \n mean engagementRate: \n \n capitalizedData: %f \n nonCapitalizedData: %f '%(1000*capitalizedData['engagementRate'].mean(),1000*nonCapitalizedData['engagementRate'].mean()))



## -- Views

print('\n \n mean views: \n \n capitalizedData: %i  \n nonCapitalizedData: %i '%(capitalizedData['views'].mean(),nonCapitalizedData['views'].mean()))
