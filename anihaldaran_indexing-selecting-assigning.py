import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.description
# Your code here
reviews.description[0]
# Your code here
reviews.loc[0]
#selcting first row of description column from reviews
reviews.loc[0,'description']
#another way of doing exercise 2
reviews.iloc[0]
# Your code here
reviews.loc[:9,'description']

#another way of executing task4
import pandas as pd
series=pd.Series([reviews.loc[0,'description'],reviews.loc[1,'description'],reviews.loc[2,'description'],reviews.loc[3,'description'],reviews.loc[4,'description'],reviews.loc[5,'description'],
                 reviews.loc[6,'description'],reviews.loc[7,'description'],reviews.loc[8,'description'],reviews.loc[9,'description']]
                 ,[0,1,2,3,4,5,6,7,8,9])
print(series)
# Your code here
pd.DataFrame({'country':[reviews.country[1],reviews.country[2],reviews.country[3],reviews.country[5],reviews.country[8]],
              'description':[reviews.description[1],reviews.description[2],reviews.description[3],reviews.description[5],reviews.description[8]],
              'designation':[reviews.designation[1],reviews.designation[2],reviews.designation[3],reviews.designation[5],reviews.designation[8]],
              'points':[reviews.points[1],reviews.points[2],reviews.points[3],reviews.points[5],reviews.points[8]],
              'price':[reviews.price[1],reviews.price[2],reviews.price[3],reviews.price[5],reviews.price[8]],
              'province':[reviews.province[1],reviews.province[2],reviews.province[3],reviews.province[5],reviews.province[8]],
              'region_1':[reviews.region_1[1],reviews.region_1[2],reviews.region_1[3],reviews.region_1[5],reviews.region_1[8]],
              'region_2':[reviews.region_2[1],reviews.region_2[2],reviews.region_2[3],reviews.region_2[5],reviews.region_2[8]],
              'taster_name':[reviews.taster_name[1],reviews.taster_name[2],reviews.taster_name[3],reviews.taster_name[5],reviews.taster_name[8]],
'taster_twitter_handle':[reviews.taster_twitter_handle[1],reviews.taster_twitter_handle[2],reviews.taster_twitter_handle[3],reviews.taster_twitter_handle[5],reviews.taster_twitter_handle[8]]},
            index=[1,2,3,5,8])              
# Your code here
pd.DataFrame({'country':[reviews.country[0],reviews.country[1],reviews.country[10],reviews.country[100]],
              'province':[reviews.province[0],reviews.province[1],reviews.province[10],reviews.province[100]],
              'region_1':[reviews.region_1[0],reviews.region_1[1],reviews.region_1[10],reviews.region_1[100]],
              'region_2':[reviews.region_2[0],reviews.region_2[1],reviews.region_2[10],reviews.region_2[100]]},
            index=[0,1,10,100])       
# Your code here
reviews.loc[0:99,['country','variety']]
# Your code here
reviews.loc[reviews.country=='Italy']
# Your code here
reviews.loc[reviews.region_2!='NaN']
# Your code here
reviews.points
#another way of doing exercise 10
reviews.loc[:,'points']
# Your code here
reviews.loc[:999,'points']
# Your code here
lenBegin=int(len(reviews)-1001)
lenEnd=int(len(reviews)-1)
reviews.loc[lenBegin:lenEnd,'points']
#another way of doing exercise 12
reviews.loc[(len(reviews)-1001):(len(reviews)-1),'points']
# Your code here
reviews.loc[reviews.country=='Italy',['points']]
# Your code here
reviews.loc[((reviews.country=='Italy') | (reviews.country=='France')) & (reviews.points>=90)] # '|' stands for OR and '&' stands for AND 