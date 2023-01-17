import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

us_videos = pd.read_csv("../input/USvideos.csv")
#print(us_videos.head(1));

def findTrumpVideos(x):
    #print(x.index);
    if 'trump' not in x.title.lower():
       # print(x.title)
        return x.title;
    
#print(us_videos[us_videos.title.str.contains('Trump')]);
us_videos_trump = us_videos.drop( us_videos[~us_videos.title.str.contains('Trump')].index);
us_videos_hilary = us_videos.drop( us_videos[~us_videos.title.str.contains('Hilary')].index);

print(us_videos_trump.tags.head(5));

tag = us_videos_trump.tags.str.split('|').tolist()
print(tag[10])

print(us_videos_trump.dislikes.sum());
print(us_videos_trump.likes.sum());
print(us_videos_trump.views.sum());