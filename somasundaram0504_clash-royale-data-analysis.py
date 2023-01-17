# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Source data is in json format need to convert them to dataframe for further analysis

import json

from pandas.io.json import json_normalize
with open('../input/matches.txt') as file:

    CR = [ x.strip() for x in file.readlines()]
len(CR)
deserialize_cr =[json_normalize(eval(r1))for r1 in CR[0:10000]]

df_cr =pd.concat(deserialize_cr,ignore_index=True)

df_cr.columns = ['Left Clan','Left Deck','Left Player','Left Trophy','Right Clan','Right Deck','Right Player','Right Trophy','Result','Time','Type']

df_cr.head()
LD = [len(left_deck) for left_deck in df_cr['Left Deck']]

RD = [len(right_deck) for right_deck in df_cr['Right Deck']]

(set(LD),set(RD))
Left_Troops =  list(np.hstack([[x[0] for x in left_deck] for left_deck in df_cr['Left Deck']]))

Right_Troops = list(np.hstack([[x[0] for x in right_deck] for right_deck in df_cr['Right Deck']]))

distinct_troops = set(np.hstack([Left_Troops,Right_Troops]))

len(distinct_troops)
RightArmy_colNames = np.hstack([["Right Troop "+str(i+1) for i in range(8)],["Right Troop Count "+str(i+1) for i in range(8)]])

LeftArmy_colNames = np.hstack([["Left Troop "+str(i+1) for i in range(8)],["Left Troop Count "+str(i+1) for i in range(8)]])

RightArmy = pd.DataFrame(data=[np.hstack([[army[0] for army in x],[int(army[1]) for army in x]]) for x in df_cr['Right Deck']],columns = RightArmy_colNames)

LeftArmy = pd.DataFrame(data=[np.hstack([[army[0] for army in x],[int(army[1]) for army in x]]) for x in df_cr['Left Deck']],columns=LeftArmy_colNames)

finalCR_data = pd.concat([df_cr,LeftArmy,RightArmy],axis=1,join='inner')

finalCR_data.head(2)
finalCR_data['Left Crowns Won'] = [int(stars[0]) for stars in finalCR_data['Result']]

finalCR_data['Right Crowns Won'] = [int(stars[1]) for stars in finalCR_data['Result']]
finalCR_data.head()
finalCR_data['Battle Result'] = [ 'Left' if(left > right) else 'Right' if(left<right) else 'Tie' for left,right in zip(finalCR_data['Left Crowns Won'],finalCR_data['Right Crowns Won'])]
finalCR_data[['Result','Battle Result']].groupby('Battle Result').count().apply(lambda x:(x/x.sum())*100)
finalCR_data[(finalCR_data['Battle Result']=='Left')][['Left Troop 1','Left Troop 2','Left Troop 3','Left Troop 4','Left Troop 5','Left Troop 6','Left Troop 7','Left Troop 8','Result']].groupby(['Left Troop 1','Left Troop 2','Left Troop 3','Left Troop 4','Left Troop 5','Left Troop 6','Left Troop 7','Left Troop 8']).count().sort_values(by='Result',ascending=False).head(1)
finalCR_data[(finalCR_data['Battle Result']=='Right')][['Right Troop 1','Right Troop 2','Right Troop 3','Right Troop 4','Right Troop 5','Right Troop 6','Right Troop 7','Right Troop 8','Result']].groupby(['Right Troop 1','Right Troop 2','Right Troop 3','Right Troop 4','Right Troop 5','Right Troop 6','Right Troop 7','Right Troop 8']).count().sort_values(by='Result',ascending=False).head(1)
finalCR_data[(finalCR_data['Battle Result']=='Left') & (finalCR_data['Left Crowns Won']==3)][['Left Troop 1','Left Troop 2','Left Troop 3','Left Troop 4','Left Troop 5','Left Troop 6','Left Troop 7','Left Troop 8','Result']].groupby(['Left Troop 1','Left Troop 2','Left Troop 3','Left Troop 4','Left Troop 5','Left Troop 6','Left Troop 7','Left Troop 8']).count().sort_values(by='Result',ascending=False).head(1)
finalCR_data[(finalCR_data['Battle Result']=='Right') & (finalCR_data['Right Crowns Won']==3)][['Right Troop 1','Right Troop 2','Right Troop 3','Right Troop 4','Right Troop 5','Right Troop 6','Right Troop 7','Right Troop 8','Result']].groupby(['Right Troop 1','Right Troop 2','Right Troop 3','Right Troop 4','Right Troop 5','Right Troop 6','Right Troop 7','Right Troop 8']).count().sort_values(by='Result',ascending=False).head(1)
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
matches = finalCR_data[['Left Clan','Type']].groupby('Type',as_index=False).count()

sns.set_color_codes("muted")

sns.barplot(x="Type",y="Left Clan",data=matches,color="b")
finalCR_data[['Left Trophy','Right Trophy']] = finalCR_data[['Left Trophy','Right Trophy']].apply(pd.to_numeric)

finalCR_data['Left Category'] = pd.cut(finalCR_data['Left Trophy'],5)

finalCR_data['Right Category'] = pd.cut(finalCR_data['Right Trophy'],5)

finalCR_data[['Left Category']].drop_duplicates().sort_values(by=['Left Category'])
finalCR_data[['Right Category']].drop_duplicates().sort_values(by=['Right Category'])
player_categories = finalCR_data[['Left Category','Right Category','Result','Type']].groupby(['Right Category','Left Category','Type'],as_index=False).count().sort_values(by=['Left Category','Right Category'],ascending=True)

#player_categories

graph = sns.FacetGrid(player_categories,row='Left Category',col='Type',size=3.0,aspect =2.5,sharey=False)

graph.map(sns.barplot,'Right Category','Result',color='b',ci=None)
threecrowns = finalCR_data[(finalCR_data['Left Crowns Won']-finalCR_data['Right Crowns Won']==3) | (finalCR_data['Left Crowns Won']-finalCR_data['Right Crowns Won']==-3)][['Battle Result','Type','Result']].groupby(['Battle Result','Type'],as_index=False).count().sort_values(by='Type',ascending=True)

histgraph = sns.FacetGrid(threecrowns,col='Type',size=2.7,aspect=1.7,sharey=False)

histgraph.map(sns.barplot,'Battle Result','Result')
