#importing everything that's needed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy
from scipy import stats
anime = pd.read_csv('../input/anime.csv')
rating = pd.read_csv('../input/rating.csv')
#Extracting top 20 anime 
animeid_top20 = anime.head(20)['anime_id']
print(animeid_top20)
#Extracting users who have watched top 20 anime 

UserRatingForTop20=rating.loc[rating['anime_id'].isin(animeid_top20)]
Userwithalltop20 = UserRatingForTop20['user_id']
anime.head()
Userwithalltop20
UserRatingForTop20.head()
x=Userwithalltop20.value_counts()
a=pd.DataFrame(x)
a.reset_index(level=0, inplace=True)
final_UserID=a[a['user_id'] > 15]['index']
#Getting the ratings for those users 
RatingsBytop20Watchers=rating.loc[rating['user_id'].isin(final_UserID)]

#Ignoring the -1s 
RatingsWithout = RatingsBytop20Watchers[RatingsBytop20Watchers['rating']>-1]
print(plt.hist(RatingsWithout['rating']))

#Evaluating Ratings by all users
ratingsforall=rating[rating['rating']>-1]
print(plt.hist(ratingsforall['rating']))

np.mean(ratingsforall['rating'])
np.mean(RatingsWithout['rating'])
stats.ttest_ind(ratingsforall['rating'], RatingsWithout['rating'],equal_var = False)
