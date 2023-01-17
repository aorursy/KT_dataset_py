import os
print(os.listdir("../input"))
import glob
files = sorted( [i for i in glob.glob('../input/*videos.csv')] )
import pandas as pan
dflist = []
for i in files:
    temp = pan.read_csv(i)
    temp = temp.sort_values( 'views' , ascending=False ).drop_duplicates( 'video_id' , keep = 'first' )
    temp['country'] = i[9:11]
    dflist.append(temp)
main_df = pan.concat(dflist)
df_trended_in_countries = main_df[ main_df.duplicated ( 'video_id' , keep=False ) ] . sort_values( 'video_id' )[ [ 'video_id','title','category_id','country' ] ]
df_new = df_trended_in_countries.drop_duplicates( 'video_id' , keep = 'last' ) . drop( columns = ['country'] )
df_new['countries_trended_in'] = df_trended_in_countries.groupby( 'video_id' )[ 'country' ] . apply( lambda x: sorted([i for i in x]) ) . values
df_new['countries_trended_in'] = df_new[ 'countries_trended_in' ] . apply( lambda x: "[ %s ]" % ' - '.join(x) ) 
df_new.head(4)
import json
carid_dic_US = {}
with open( '../input/US_category_id.json' , 'r' ) as jfile:
    jdata = json.load(jfile)
    for i in jdata['items']:
        carid_dic_US[ i['id'] ] = i['snippet']['title']
df_new['video category']=df_new['category_id'] . apply( lambda x : carid_dic_US[str(x)] )
df_new = df_new . drop( columns = ['category_id'] )
df_new.head(4)
total_unique_vids = len( main_df.drop_duplicates( 'video_id' , keep = 'last' ) )
df_summary = pan.DataFrame(  { 'col1': [ 'Trended in more than one country' , 'Trended only in one country' ] , 'Number of videos' : [ len(df_new) ,   total_unique_vids - len(df_new) ] } )
df_summary.groupby('col1')['Number of videos'].sum().plot( kind='pie' , autopct='%.1f%%' , fontsize=18 , figsize=(10,10) ,  colormap = 'Set2' , explode = [0,.1] )
expo =  [ 0 for i in range( 3 , len( df_new.groupby('countries_trended_in')['countries_trended_in'] ) ) ] + [ .05 , .05 , .05 ]
df_new.groupby('countries_trended_in')['countries_trended_in'].size().sort_values().plot( kind='pie', autopct='%.1f%%' , fontsize=14 , figsize=(13,13) ,  colormap='brg' , explode=expo  )
df_new . groupby('countries_trended_in')['countries_trended_in'] . size() . apply(lambda x : x*100/len(df_new)) . sort_values(ascending=False)
sumCAUS = ( df_new[ df_new.countries_trended_in . isin( [ '[ CA - US ]'  ] ) ]
             . groupby('video category')['video category'] . size() . sort_values() . sum() )

print( 'Among the total of', total_unique_vids , 'trending videos during last five months,', sumCAUS ,'of them trended in mentioned countries:' , round(100*sumCAUS/total_unique_vids,1) , '%' )

( df_new[ df_new.countries_trended_in . isin( [ '[ CA - US ]'  ] ) ]
     . groupby('video category')['video category'] . size()
     . sort_values( ascending = True ) . apply( lambda x: 100*x/sumCAUS )
     . plot(kind = 'barh'  , figsize=(5,5) , fontsize=18 , colormap='summer' , title = 'Catgories of videos trended in both the US and Canada\n' ) ) . set_xlabel( "Probability [%]" , fontsize=18 )
sumGBUS = ( df_new[ df_new.countries_trended_in . isin( [ '[ GB - US ]'  ] ) ]
             . groupby('video category')['video category'] . size() . sort_values() . sum() )

print( 'Among the total of', total_unique_vids , 'trending videos during last five months,', sumGBUS ,'of them trended in mentioned countries:', round(100*sumGBUS/total_unique_vids,1) , '%' )

( df_new[ df_new.countries_trended_in . isin( [ '[ GB - US ]'  ] ) ]
     . groupby('video category')['video category'] . size()
     . sort_values( ascending = True ) . apply( lambda x: 100*x/sumGBUS )
     . plot(kind = 'barh'  , figsize=(5,5) , fontsize=18 , colormap='winter' , title = 'Catgories of videos trended in both the US and GB\n'  ) ) . set_xlabel("Probability [%]" , fontsize=18 )
sumCADEFR = ( df_new[ df_new.countries_trended_in . isin( [ '[ DE - FR ]' , '[ CA - DE - FR ]' , '[ CA - FR ]' , '[ CA - DE ]' ] ) ]
             . groupby('video category')['video category'] . size() . sort_values() . sum() )

print( 'Among the total of', total_unique_vids , 'trending videos during last five months,', sumCADEFR ,'of them trended in mentioned countries:', round(100*sumCADEFR/total_unique_vids,1) , '%' )

( df_new[ df_new.countries_trended_in . isin( [ '[ DE - FR ]' , '[ CA - DE - FR ]' , '[ CA - FR ]' , '[ CA - DE ]'  ] ) ]
     . groupby('video category')['video category'] . size()
     . sort_values( ascending = True ) . apply( lambda x: 100*x/sumCADEFR )
     . plot(kind = 'barh'  , figsize=(5,5) , colormap='cool' , title = 'Catgories of videos trended in Germany, France and Canada\n', fontsize=18 ) ) . set_xlabel("Probability [%]" , fontsize=18 )
import matplotlib.pyplot as plt
from math import pi

sumALL = ( df_new[ df_new.countries_trended_in . isin( [ '[ CA - DE - FR - GB - US ]' ] ) ]
             . groupby('video category')['video category'] . size() . sort_values() . sum() )

print( 'Among the total of', total_unique_vids , 'trending videos during last five months, only', sumCADEFR ,'of them could trend in all countries:', round(100*sumCADEFR/total_unique_vids,1) , '%' )

data_radar = ( df_new[ df_new.countries_trended_in . isin( [ '[ CA - DE - FR - GB - US ]' ] ) ]
     . groupby('video category')['video category'] . size()
     . sort_values( ascending = False ) . apply( lambda x: 100*x/sumALL ) )

categories = [i for i in data_radar.index.values]
values = [i for i in data_radar.values]
categories = categories[0:8]
values = values[0:8]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, polar=True)
plt.xticks(angles, categories, color='green' , fontsize=15)
plt.yticks([10,20,30], ['10%','20%','30%'], color="red", fontsize=18)
plt.ylim(0,50)
plt.title('Category of the videos which trended globally \n' , fontsize=16 , color = 'blue')
ax.plot(angles, values, linewidth=3, linestyle='solid')
ax.fill(angles, values, 'b', alpha=0.1)
ax.set_rlabel_position(0)
