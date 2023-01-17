import json



import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
with open('../input/data.json') as file:

    data = json.load(file)
decks = pd.DataFrame(data)

#decks['date'] = pd.to_datetime(decks['date'])

#card_col = ['card_{}'.format(str(i)) for i in range(30)]

#cards = pd.DataFrame([c for c in decks['cards']], columns=card_col)

#cards = cards.apply(np.sort, axis=1)

#decks = pd.concat([decks, cards], axis=1)

#decks = decks.drop('cards', axis=1)



decks.head()
with open('../input/refs.json', encoding="utf8") as file:

    data = json.load(file)
refs=pd.DataFrame(data)

refs.head()
del data
list_df=[]

for line in range(int(len(decks)/10)):

    deck=decks.iloc[line]["cards"]

    #print(deck,decks.iloc[line]["deck_class"])

    count_neutral=0

    for card in deck:

#         print(card,refs[refs["dbfId"]==card]["name"].values[0])

        if refs[refs["dbfId"]==card]["cardClass"].values[0]=="NEUTRAL":

            count_neutral +=1

    list_df.append([decks.iloc[line]["deck_class"],100*count_neutral/30,decks.iloc[line]["craft_cost"],decks.iloc[line]["rating"]])

    if line>0 and len(decks)%line==0 :

        print("step {}%".format(100*line/len(decks)))

    

    

df_partneutral=pd.DataFrame(list_df,columns=["Class","Part_neutral","Craft_cost","Rating"])

df_partneutral.head()
sns.pairplot(df_partneutral,hue='Class')
def fill_countcost(cost,count_cost):

    if cost>=9:

        count_cost[-1]=count_cost[-1]+1

    else:

        idx=cost/3

        count_cost[int(idx)]=count_cost[int(idx)]+1

        

    return count_cost
list_df=[]

for line in range(10000):

    deck=decks.iloc[line]["cards"]

    #print(deck,decks.iloc[line]["deck_class"])

    count_neutral=0

    count_cost=[0]*4

    condition_quit=False

    for card in deck:

        

        if refs[refs["dbfId"]==card]["cardClass"].values[0]=="NEUTRAL":

            count_neutral +=1

        

        

        

        

        if not np.isnan(refs[refs["dbfId"]==card]["cost"].values[0]):

            count_cost=fill_countcost(int(refs[refs["dbfId"]==card]["cost"].values[0]),count_cost)

        else:

            condition_quit=True

            break

    if not condition_quit:

        list_df.append([decks.iloc[line]["deck_class"]]+count_cost+[100*count_neutral/30,decks.iloc[line]["craft_cost"]])

    

    

    

        

        

card_col = ["cost_<3","cost_3-5","cost_6-8","cost_9+"]





df_cost=pd.DataFrame(list_df,columns=["Class"]+card_col+["Part_neutral","Craft_cost"])

df_cost.head()
sns.pairplot(df_cost,hue='Class')