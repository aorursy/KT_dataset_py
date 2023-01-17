import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv("../input/USvideos.csv")
data.head()
drop_list = ['video_id', 'trending_date', 'title', 'category_id', 'publish_time', 'tags', 'thumbnail_link', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed', 'description']
tempData = data.drop(drop_list, axis=1)
tempData.head()
f,ax = plt.subplots(figsize=(9, 7))
sns.heatmap(tempData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
tempData = tempData.sort_values(['likes']).reset_index(drop=True)
tempData.head()
