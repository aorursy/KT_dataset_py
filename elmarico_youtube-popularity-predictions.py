import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
%matplotlib inline
dfVideos = pd.read_csv("/kaggle/input/USA_videos.csv",error_bad_lines=False)
dfComments = pd.read_csv("/kaggle/input/USA_comments.csv",error_bad_lines=False)
print(dfVideos.head())
print(dfVideos.info())
print(dfVideos.describe())
# Dve liste gde cu skladistiti ID Kategorije i title Kategorije iz JSON fajla
categorieId = []
categorieTitle = []

# Citanje JSON Fajla
with open("/kaggle/input/USA_categories.json","r",encoding="utf-8") as fajl:
    categories = json.load(fajl)
    #print(categories)
    #print(len(categories["items"]))
    for i in categories["items"]:
        categorieId.append(i["id"]) # Dodavanje ID-a u listu za ID
        categorieTitle.append(i["snippet"]["title"]) # Dodavanje title u listu za Title


categorieId = list(map(int, categorieId)) # Konvertovanje ID tipa String u Int
categoriesDict = dict(zip(categorieId,categorieTitle)) # Spajanje lista u dict, key = id, value = title
categorieId
categoriesDict
print(sorted(dfVideos["category_id"].unique())) # Proveravanje svih id kategorija u dataframe-u radi provere
dfVideos["category_title"] = dfVideos["category_id"].map(categoriesDict) # Nova kolona u dataframe-u sa titlovima kategorija
dfVideos.head()

# ******* PROVERA NaN VREDNOSTI ********
print(dfVideos.isnull().sum())


# ******** CISCENJE TAGS I TITLE ********
dfVideos["title"] = dfVideos["title"].str.replace("[^a-zA-Z]"," ") # zamenjujemo sve special chars u space
dfVideos["title"] = dfVideos["title"].apply(lambda x: x.lower()) # konvertujem title u lower case
dfVideos["tags"] = dfVideos["tags"].str.replace("[^a-zA-Z]"," ") # special chars to space
dfVideos["tags"] = dfVideos["tags"].apply(lambda x: x.lower()) # tags to lower case



# ******** Dodavanje novih kolona i ciscenje ********
dfVideos["title_len"] = dfVideos["title"].apply(lambda x: len(x)) #Nova kolona-Duzina title-ova
dfVideos["tags_len"] = dfVideos["tags"].apply(lambda x: len(x)) # Nova kolona-Duzina tagova
dfVideos["tags_len"] = dfVideos["tags_len"].replace(6,0) # Zamenjivanje tags_len koje imaju vrednost 6, sa nulom. Jer length od 6 je zaporavo string [None]

init_notebook_mode(connected=True) 
categoryViews = dfVideos.groupby("category_title")["views"].sum().reset_index()
fig_1 = px.bar(categoryViews,x="category_title",y="views",title="Total views per category")
fig_1.show()
# Top 10 Najgledanijih kanala

mostViewedChannels = dfVideos.groupby(["channel_title","category_title"])["views"].sum().nlargest(10).reset_index()
vis_2 = px.bar(mostViewedChannels,x="channel_title",y="views",color="category_title",title="10 Most Watched Channels")
vis_2.show()
# Histogram za duzinu tagova
sns.set(style="darkgrid")

vis_3 = sns.distplot(dfVideos["tags_len"])
vis_3.set_title("Tags Length")
plt.show()
# Histogram za duzinu title
vis_4 = sns.distplot(dfVideos["title_len"])
vis_4.set_title("Title Length")
plt.show()
# Scatterplot za duzinu titlova i pregleda
plt.figure(figsize=(12,10))
vis_5 = sns.scatterplot(x="title_len",y="views",data=dfVideos,hue="category_title")
vis_5.set_title("Title Length affect on Views")
plt.show()
# Scatterplot za duzinu tagova i pregleda
plt.figure(figsize=(12,10))
vis_6 = sns.scatterplot(x="tags_len",y="views",data=dfVideos,hue="category_title")
vis_6.set_title("Tags Length affect on Views")
plt.show()
# Provera koliko tagova i koliki je title_len za 500 najpopularnijih klipova

tagsCheck = dfVideos.groupby(["tags_len","title_len","category_title"])["views"].nlargest(500).reset_index()
tagsCheck = tagsCheck.sort_values("views", ascending=False)
plt.figure(figsize=(15,10))
vis_7 = sns.scatterplot(x="tags_len",y="views",data=tagsCheck,sizes=(10,300),color="#4CB391",alpha=0.7)
vis_7.set_title("Title and Tags length affect on total Views")
vis_8 = sns.scatterplot(x="title_len",y="views",data=tagsCheck,ax=vis_7,alpha=0.5)
plt.show()

# Scatterplot povecanje pregleda u odnosu na lajkove i dislajkove
plt.figure(figsize=(15,10))
vis_9 = sns.scatterplot(x="views",y="likes",data=dfVideos,size="dislikes",sizes=(10,300),color="#4CB391",alpha=0.8)
vis_9.set_title("Likes/Dislikes affect on Views")
plt.show()
# Scatterplot povecanje pregleda u odnosu na broj komentara
plt.figure(figsize=(15,10))
vis_10 = sns.scatterplot(x="views",y="comment_total",data=dfVideos)
vis_10.set_title("Num. of Comments affect on Views")
plt.show()
# Dodeljujem ID Svakom kanalu
encoder = LabelEncoder()
dfVideos["channel_id"] = encoder.fit_transform(dfVideos["channel_title"])


vis_11 = sns.scatterplot(x="channel_id",y="views",data=dfVideos)
vis_11.set_title("Views based on Channel ID")
plt.show()
channel_id_sum = dfVideos.groupby(["channel_id"])["views"].sum().reset_index()
vis_12 = sns.scatterplot(x="channel_id",y="views",data=channel_id_sum)
vis_12.set_title("Total Views per channel ID")
plt.show()
# **** WORDCLOUD ****

titles = "".join(dfVideos["title"]) # pravim string titles u koji skladistim title iz dataframe-a
tags = "".join(dfVideos["tags"]) # isto za tags

def create_wordcloud(text, filename): # funkcija koja kreira wordcloud i cuva u folderu sliku
    stopwords = set(STOPWORDS)

    wc = WordCloud(background_color="white",
                   max_words=200,
                   stopwords=stopwords)

    wc.generate(text)
    wc.to_file(filename+".png")
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

create_wordcloud(titles,"titlesWordCloud") # wordlcoud za naslove video klipova
create_wordcloud(tags,"tagsWordCloud") # wordcloud za tagove video klipova
# TRAIN TEST SPLIT
X = dfVideos[["likes","dislikes","comment_total","category_id","tags_len","channel_id"]]
y = dfVideos["views"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
# LINEAR REGRESSION MODEL
lg = LinearRegression()
lg.fit(X_train,y_train) # FITTING DATA
predictions = lg.predict(X_test) # PREDICTIONS
print("Linear Regression")
print(lg.score(X_test,y_test)) # SCORE
# Random Forest Regressor

forest = RandomForestRegressor(n_estimators=40, random_state=42)
forest.fit(X_train,y_train)
forest_predictions = forest.predict(X_test)
print("Random Forest Regressor")
print(forest.score(X_test,y_test))
# Uticaj broja drveća na model

estimators = np.arange(10,200,10)
scores = []
model = RandomForestRegressor(n_jobs=-1)
for i in estimators:
    model.set_params(n_estimators = i)
    model.fit(X_train,y_train)
    scores.append(model.score(X_test,y_test))

plt.title("N_Estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
plt.show()