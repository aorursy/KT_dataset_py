#Load Basic Libraries
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
ramen = pd.read_csv("../input/ramen-ratings.csv")
ramen.head()
#How many entries brands are there?
ramen.groupby(['Brand']).count()['Review #']
print('Total Unique Brand Entries: ',pd.unique(ramen['Brand']).shape[0])
ramen.groupby(['Brand']).count()['Review #'].nlargest(10)
top_brand = list(ramen.groupby(['Brand']).count()['Review #'].nlargest(10).index)
df_agg = ramen.groupby(['Brand','Country']).count()
df_agg.iloc[df_agg.index.isin(top_brand,level = 'Brand')].sort_values(by='Review #',ascending=False).sort_index(level='Brand', sort_remaining=False)
ramen[pd.notna(ramen['Top Ten'])]
def get_year(val):
    if len(val)<4:
        return np.NaN #to handle '\n'
    return int(val[:4])

def get_rank(val):
    if len(val)<4:
        return np.NaN #to handle '\n'
    return int(val[6:])

#Remove NAN terms
df_rank = ramen[pd.notna(ramen['Top Ten'])].reset_index()
df_rank['Year of Rank'] = df_rank['Top Ten'].apply(lambda x: get_year(x))
df_rank['Rank'] = df_rank['Top Ten'].apply(lambda x: get_rank(x))
df_rank.dropna(inplace=True)
df_rank.sort_values(by=['Year of Rank','Rank'],ascending=[False,True])
ramen.loc[ramen['Stars'] == 'Unrated','Stars'] = 0 # To remove the unrated columns
ramen['Stars'] = ramen['Stars'].astype(float)
print("Reviews with 5 Star Ratings : ", ramen[ramen['Stars'] == 5].shape[0])
Brand_count = ramen[ramen['Stars'] == 5].groupby('Brand').count()['Stars'].nlargest(1).values[0]
Brand_name =  ramen[ramen['Stars'] == 5].groupby('Brand').count()['Stars'].nlargest(1).index.values[0]
print("Brand which has highest 5 star Rating: ", Brand_name," with Ratings: ", Brand_count)
nissin_data = ramen[ramen['Brand'] =='Nissin']
nissin_data['Stars'].describe()
nissin_data.boxplot(column=['Stars'], by='Country',fontsize=15,rot=90)