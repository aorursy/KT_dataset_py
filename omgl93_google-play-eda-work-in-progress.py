import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import warnings

import re

import PIL.Image as Image #Loading images for the wordcloud mask

from wordcloud import ImageColorGenerator #Also for the mask

import squarify

import missingno as msno # MissingNo 

warnings.filterwarnings('ignore')

%matplotlib inline
df_1 = pd.read_csv(r"../input/google-play-store-apps/googleplaystore.csv")

df_2 = pd.read_csv(r"../input/google-play-store-apps/googleplaystore_user_reviews.csv")
text = " ".join(app for app in df_1["App"])

text2 = re.sub('[^A-Za-z0-9]+', ' ', text)
char_mask = np.array(Image.open(("../input/imggoogle/Google-Play-Store-icon.jpg")))

image_colors = ImageColorGenerator(char_mask)
fig, ax = plt.subplots(figsize=(16,8))



wc = WordCloud(background_color="white", mask=char_mask).generate(text2)

plt.imshow(wc.recolor(color_func=image_colors), interpolation='bilinear')

plt.axis("off")
msno.matrix(df_1)
df_1.info()
msno.matrix(df_2)
df_2.info()
df_1.dropna(inplace=True)

df_1.drop_duplicates("App",inplace=True)
df_2.dropna(inplace=True)
#No unusual data

print(df_1["Type"].unique())
print(df_1["Size"].unique())
df_1["Size"] = df_1["Size"].apply(lambda x: x.replace("M","") if "M" in str(x) else x)

df_1["Size"] = df_1["Size"].apply(lambda x: x.replace(",","") if "," in str(x) else x)

df_1["Size"] = df_1["Size"].apply(lambda x: x.replace("Varies with device","0") if "Varies with device" in str(x) else x)

df_1["Size"] = df_1["Size"].apply(lambda x: float(str(x).replace("k",""))/1024 if "k" in str(x) else x)
print(df_1["Price"].unique())
df_1["Price"] = df_1["Price"].apply(lambda x: x.replace("$","") if "$" in str(x) else x)

df_1["Price"] = df_1["Price"].apply(lambda x: float(x))
print(df_1["Category"].unique())
print(df_1["Installs"].unique())
df_1["Installs"] = df_1["Installs"].apply(lambda x: x.replace("+", "") if "+" in str(x) else x)

df_1["Installs"] = df_1["Installs"].apply(lambda x: x.replace(",", "") if "," in str(x) else x)

df_1["Installs"] = df_1["Installs"].apply(lambda x: int(x))
df_1["Reviews"] = df_1["Reviews"].astype(int)
numerical_data = pd.DataFrame()

numerical_data["Reviews_Log"] = np.log(df_1["Reviews"])

numerical_data["Installs_Log"] = np.log(df_1["Installs"])

numerical_data["Rating"] = df_1["Rating"]

numerical_data["Price"] = df_1["Price"]

numerical_data["Type"] = df_1["Type"]

numerical_data["Size"] = df_1["Size"]
# Distribution of ratings over all the apps

sns.distplot(numerical_data["Rating"]).set_title("Distribution of ratings")
numerical_data_free = numerical_data[numerical_data["Type"] == "Free"]

numerical_data_paid = numerical_data[numerical_data["Type"] == "Paid"]

# Seperated per type

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

sns.distplot(numerical_data_free["Rating"],ax=ax1).set_title("Distribution of ratings - Free Apps")

sns.distplot(numerical_data_paid["Rating"],ax=ax2,color="red").set_title("Distribution of ratings - Paid Apps")

fig,ax1 = plt.subplots(figsize=(14,10))

p1= sns.catplot(x="Rating",y="Category",kind="box",data=df_1,ax=ax1)

plt.close(p1.fig)

plt.title("Distribution of ratings - Based on App Category")
best_cat = df_1.groupby("Category")["Rating"].mean().sort_values(ascending=False)
print(best_cat.iloc[:5])
print(best_cat.iloc[28:34])
cnt_rating = sns.catplot(x="Rating",y="Content Rating",data=df_1,kind="violin",hue="Type")

sns.color_palette("Spectral", 10)

cnt_rating.set_xticklabels(rotation=30, horizontalalignment='right')
content_mean = df_1.groupby("Content Rating")["Rating"].mean().sort_values(ascending=False)
print(content_mean)
most_rev = df_1.groupby("App")["Reviews"].sum().sort_values(ascending=False)[:10]
most_reviewed_apps = df_1[df_1["App"].isin(most_rev.index)]

most_reviewed_apps = most_reviewed_apps[["App","Reviews"]].sort_values(by="Reviews")
most_reviewed_apps
text_replace =  {"Security Master - Antivirus, VPN, AppLock, Booster" : "Security Master",

                "Clean Master- Space Cleaner & Antivirus": "Clean Master",

                "Messenger – Text and Video Chat for Free": "Messenger"}
most_reviewed_apps["App"] = most_reviewed_apps["App"].replace(text_replace)
most_reviewed_apps["Milions"] = round(most_reviewed_apps["Reviews"] / 1000000,2)

most_reviewed_apps["Milions"] = most_reviewed_apps["Milions"].apply(lambda x: str(x) + "M")

most_reviewed_apps["App"] = most_reviewed_apps["App"].map(str) + " " +  most_reviewed_apps["Milions"]

most_reviewed_apps["App"] = most_reviewed_apps["App"].replace(text_replace)
fig, ax = plt.subplots(figsize=(14,7))

squarify.plot(sizes=most_reviewed_apps["Reviews"], label=most_reviewed_apps["App"], alpha=.8,text_kwargs={'fontsize':10})

plt.axis('off')
num_corr = numerical_data.corr()

print(num_corr)
sns.heatmap(num_corr,annot=True,cmap = 'Reds')
sns.lmplot(x="Installs_Log", y="Reviews_Log", data=numerical_data,col="Type");
sns.jointplot(x="Price",y="Rating",data=numerical_data)
df_1["Price"].describe()
wow_price = df_1[df_1["Price"] > 200]
#Worst thing about it is that one has over 100k installs

print(wow_price["Installs"].sum())
print(wow_price["App"])

#Wow
wow_price["Money Spend"] = wow_price["Installs"] * wow_price["Price"]
print("The amount of money people spend on this :")

print(wow_price["Money Spend"].sum())
clean_price = df_1[df_1["Price"] < 200]

sns.jointplot(x="Price",y="Rating",data=clean_price)
ax,fig = plt.subplots(figsize=(15,7))

sns.countplot(y= df_1["Category"],order=df_1["Category"].value_counts().index)
df_split = pd.DataFrame(df_1)
list_of_comb = []

df_split["Combinations"] = df_split["Genres"].apply(lambda x: list_of_comb.append(x) if ";" in str(x) else x)
combination_data = pd.Series(list_of_comb)
sns.countplot(y = combination_data, order=combination_data.value_counts().iloc[:10].index)
df_split['A'], df_split['B'] = df_split['Genres'].str.split(';', 1).str
#Primary Genres

sns.countplot(y = df_split["A"], order=df_split["A"].value_counts().iloc[:15].index)
# Sec Genres

sns.countplot(y= df_split["B"], order=df_split["B"].value_counts().index)
df_games = pd.DataFrame(df_split[df_split["Category"] == "GAME"])
sns.countplot(y = df_games["A"], order = df_games["A"].value_counts().index)
mean_rating_games = df_games.groupby("A")["Rating"].mean().sort_values(ascending=False)

mean_rating_games
df_merge_col = pd.merge(df_1, df_2, on='App')
sentiment_df = df_merge_col[["App","Category","Sentiment","Translated_Review","Sentiment_Polarity","Sentiment_Subjectivity"]]
sentiment_df["Review_Len"] = sentiment_df["Translated_Review"].astype(str).apply(lambda x : len(x))
sentiment_df.head()
split = sentiment_df.groupby("Category")["Sentiment"].value_counts()
sent = sentiment_df.groupby("Category").count()
rating = split / sent["Sentiment"]
rating = pd.DataFrame(rating)

rating.rename(columns={"Sentiment" : "Percent"}, inplace=True)

rating.reset_index(level="Sentiment", inplace=True)
rating.head()
rating_positive = rating[rating["Sentiment"] == "Positive"]

rating_neutral = rating[rating["Sentiment"] == "Neutral"]

rating_negative = rating[rating["Sentiment"] == "Negative"]
fig,ax = plt.subplots(figsize=(15,7))

plt.bar(rating_positive.index,rating_positive["Percent"],color="forestgreen", label="Positive")

plt.bar(rating_neutral.index,rating_neutral["Percent"],bottom=rating_positive["Percent"],color="darkgrey",label="Neutral",)

plt.bar(rating_negative.index,rating_negative["Percent"],bottom=rating_positive["Percent"] + rating_neutral["Percent"],color="maroon",label="Negative")

plt.xticks(rotation='vertical')

plt.legend()
sort_positive_ratings = rating_positive.sort_values(by=["Percent"], ascending=False)
print(sort_positive_ratings.head())
print(sort_positive_ratings.tail())
positive_review = sentiment_df[sentiment_df["Sentiment"]=="Positive"]

neutral_review = sentiment_df[sentiment_df["Sentiment"]=="Neutral"]

negative_review = sentiment_df[sentiment_df["Sentiment"]=="Negative"]
text_positive = " ".join(review for review in positive_review["Translated_Review"].astype(str))

text_neutral = " ".join(review for review in neutral_review["Translated_Review"].astype(str))

text_negative = " ".join(review for review in negative_review["Translated_Review"].astype(str))
fig, ax = plt.subplots(figsize=(14,6))



wordcloud = WordCloud(background_color="white").generate(text_positive)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
fig, ax = plt.subplots(figsize=(14,6))



wordcloud = WordCloud(background_color="white").generate(text_neutral)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
fig, ax = plt.subplots(figsize=(14,6))



wordcloud = WordCloud(background_color="white").generate(text_negative)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
sentiment_df.dropna(inplace=True)
sns.distplot(sentiment_df["Sentiment_Polarity"]).set_title("Distribution of Sentiment Polarity")
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))

sns.distplot(positive_review["Sentiment_Polarity"],ax=ax1).set_title("Distribution of Sentiment Polarity - Positive Review")

sns.distplot(negative_review["Sentiment_Polarity"],color="r",ax=ax2).set_title("Distribution of Sentiment Polarity - Negative Review")
fig = plt.subplots(figsize=(12,6))

sns.boxplot(x = sentiment_df["Sentiment_Polarity"],y= sentiment_df["Category"]).set_title("Distribution of Sentiment Polarity per Category")
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))

sns.distplot(positive_review["Review_Len"],ax=ax1).set_title("Distribution of review lengths - Positive")

sns.distplot(neutral_review["Review_Len"],ax=ax2).set_title("Distribution of review lengths - Neutral")

sns.distplot(negative_review["Review_Len"],ax=ax3).set_title("Distribution of review lengths - Negative")