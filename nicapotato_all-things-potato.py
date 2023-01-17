import pandas as pd
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
import nltk

# Visualization
from wordcloud import WordCloud, STOPWORDS
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv('../input/wikipedia-article-titles/titles.txt', sep='\n', dtype=str).rename(columns={"!":"text"})
df.text = df.text.str.replace(r"_"," ").str.lower()
df.text =df.text.str.replace(r"[^A-Za-z0-9\s]+","")
df = df.loc[df.text != "",:]
df.drop_duplicates(inplace=True)

potato = df.loc[df.text.str.contains(r"\bpotato|\bpotatoes|\bspud\b|\bspuds|\btater|/btattie|/bbyam",
                                     na=False)]
print("{}% of Wikipedia Articles are Potato Related".format(round((len(potato)/len(df))*100,3)))
print("Processing Ready")
stoptaters = {"potato","potatoe","potatoes","spud","taters","tater","tattie","yam","tuber"}
STOPWORDS.update(stoptaters)

# Function
def cloud(text, title):
    # Processing Text
    #stopwords = set(stopwords) # Redundant
    wordcloud = WordCloud(width=800, height=400,
                          #background_color='white',
                          stopwords=STOPWORDS,
                         ).generate(" ".join(text))
    
    # Output Visualization
    plt.figure(figsize=(18,5), facecolor='w')
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title(title, fontsize=25,color='k')
print("WordCloud Function Ready")
cloud(potato.text.values, "All Potato")
cloud(potato.loc[potato.text.str.contains(r"virus",na=False),"text"],"Virus")
cloud(potato.loc[potato.text.str.contains(r"sweet",na=False),"text"],"Sweet")
# Processing
pdf = pd.read_csv("../input/potato-pricing/Daily Retail Price of Potato.csv",
                  parse_dates=["Date"]).set_index("Date")
pdf.drop("Commodity_Name",axis=1, inplace=True)
pdf = pdf.pivot_table(values='Price',index="Date",columns="Centre_Name")
pdf = pdf.loc[pdf.index > "2000",:]
pdf.fillna(method='ffill',inplace=True)
f, ax = plt.subplots(figsize=[10,5])
pdf.rolling(window = 30).mean().plot.line(ax=ax)
ax.legend_.remove()
ax.set_title("Potato Prices in India")
ax.set_ylabel("Price")
plt.show()
edf = pd.read_csv("../input/sweet potato varietal identification/S2_File.csv")
edf.head()
