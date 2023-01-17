import numpy as np

import pandas as pd

mtg = pd.read_json('../input/magic-the-gathering-cards/AllCards.json')
mtg.head()
mtg = pd.DataFrame.transpose(mtg)

mtg.head()
mtg.iloc[0]
features_to_consider = ["colorIdentity","convertedManaCost","manaCost","name","types","edhrecRank","power","toughness"]

df = mtg[features_to_consider]

df.head()
df = df.dropna(subset=['manaCost','convertedManaCost','colorIdentity'])
colorIdentity = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}

colorCount = {'Black': 0, 'Green': 0, 'Red': 0, 'Blue': 0, 'White': 0, 'Colorless':0, 'Multi':0}



for raw in df['colorIdentity']:

    if len(raw) == 0:

        colorCount['Colorless']+=1

    elif len(raw) > 1:

        colorCount['Multi']+=1

    else:

        for color in colorIdentity:

            if raw[0] == color:

                colorCount[colorIdentity[color]]+=1

colorCount
import matplotlib.pyplot as plt



labels = colorCount.keys()

sizes = colorCount.values()

colors = ['Black', 'green', 'red', 'blue','white','lightgray','gold']

explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1)



plt.title("Complete color pie",pad =30,fontsize=20)

plt.pie(sizes, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140,explode=explode,pctdistance=1.2)

plt.legend(labels=labels,frameon=False,bbox_to_anchor=(0.7, 0.3, 0.5, 0.5))

plt.axis('equal')

plt.show()
#add a "number of color" column:

df["colorNumber"] = mtg.colorIdentity.apply(len)

#create new columns, one for each color

for color in colorIdentity:

    df[color] = False

#set True/False in each column according to the matching colors

for color in colorIdentity:

    df[color] = mtg.colorIdentity.apply(lambda x: color in x)

    

df = df.drop(columns=['colorIdentity'])



df.head()
df['colorNumber'].plot.hist(bins=5)

plt.title("Number of color on each card",pad=10,fontsize=20)
for color in colorIdentity:

    df.loc[df[color]==True,"convertedManaCost"].plot.hist(color=colorIdentity[color],bins=range(0, 15))

    title = colorIdentity[color] + " (mean:" + str(round(df.loc[df[color]==True,"convertedManaCost"].mean(),2))+")"

    plt.title(title,pad=10,fontsize=20)

    plt.axes().set_facecolor("lightgrey")

    plt.show()
dfText = mtg[["text","type","name"]]

dfText["text"] = dfText["text"].apply(str)
from wordcloud import WordCloud



features_we_want = [dfText.name,dfText.type,dfText.text]

titles = ["Name of cards","Type of cards","Card text"]



for i,feature in enumerate(features_we_want):

    text = " ".join(txt for txt in feature)

    wordcloud = WordCloud(width=800, height=400,max_font_size=60, max_words=100, background_color="white").generate(text)



    plt.imshow(wordcloud, interpolation='bilinear')

    plt.title(titles[i],pad=10,fontsize=20)

    plt.axis("off")

    plt.show()
dfEdh = df.dropna(subset=['edhrecRank'])

dfEdh.index = dfEdh['edhrecRank']

dfEdh.head()
for color in colorIdentity:

    dfEdh.loc[dfEdh[color]==True,"convertedManaCost"].plot.hist(color=colorIdentity[color],bins=range(0, 15))

    title = colorIdentity[color] + " (mean:" + str(round(dfEdh.loc[dfEdh[color]==True,"convertedManaCost"].mean(),2))+")"

    plt.title(title,pad=10,fontsize=20)

    plt.axes().set_facecolor("lightgrey")

    plt.show()
colorIdentity = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}

colorCount = {'Black': 0, 'Green': 0, 'Red': 0, 'Blue': 0, 'White': 0}



#count colors

for color in colorIdentity:

    if color !='Colorless' and  color !='Multi':

        colorCount[colorIdentity[color]] = dfEdh[color].loc[(dfEdh[color]== True) & (dfEdh['colorNumber']==1)].count()   



colorCount
labels = colorCount.keys()

sizes = colorCount.values()

colors = ['Black', 'green', 'red', 'blue','white']

explode = (0.1,0.1,0.1,0.1,0.1)



plt.title("Edh color pie",pad =30,fontsize=20)

plt.pie(sizes, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140,explode=explode,pctdistance=1.2)

plt.legend(labels=labels,frameon=False,bbox_to_anchor=(0.7, 0.3, 0.5, 0.5))

plt.axis('equal')

plt.show()