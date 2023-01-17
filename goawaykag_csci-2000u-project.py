# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/all_kaggle_datasets.csv")
data.head()
#turn data['dateUpdated'] into datetime
data['dateUpdated']=pd.to_datetime(data["dateUpdated"])
#turn data['categories'] into df
# eval(data['categories'][77])['categories'][0]['description']
eval(data['categories'][4])
# data['categories'][0]['name']

#I give up
# Want numbers, select numbers
# then drop Ids
dataInNum=(data.select_dtypes(include=['float64','int','datetime'])
                .drop(['creatorUserId','currentDatasetVersionId','datasetId','ownerUserId'],1))
# Want title and overview as well, added
dataInNum['title']=data['title']
dataInNum['overview']=data['overview']
dataInNum.tail()
dataInNum.shape
ax=(data.set_index('dateUpdated')
                 .resample('1M')#per 1 Month
                 .size()        #row * columns, 
                 .plot(title='Datasets Per Month')
)
ax.set_ylabel("Datasets");
ax=(data.set_index('title').nlargest(8,'downloadCount')['downloadCount']
         .plot(kind='bar',title='Top 8 Dataset Downloads'))
ax.set_ylabel("Download Count");
pd.set_option('display.max_colwidth', -1)  
dataInNum.nlargest(8,'downloadCount')[['title','overview','viewCount']]
ax=(data.set_index('title').nlargest(8,'viewCount')['viewCount']
         .plot(kind='bar',title='Top 8 Dataset Views'))
ax.set_ylabel("View Counts");
dataInNum.nlargest(8,'viewCount')[['title','overview','viewCount']]
# dataInNum.nlargest(8,'viewCount')['categories']#[158]