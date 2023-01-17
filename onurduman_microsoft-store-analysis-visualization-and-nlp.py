import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



import nltk

from nltk.corpus import stopwords

from textblob import Word

from textblob import TextBlob

msft = pd.read_csv("/kaggle/input/windows-store/msft.csv")

df = msft.copy()
df.head()
df.info()
df.describe().T
df[df.isna().any(axis=1)]
df.dropna(inplace = True)
rating_series = df["Rating"].value_counts()

labels = rating_series.index

sizes = rating_series.values

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]], subplot_titles=['Histogram', 'Pie'])



fig.add_trace(

    go.Histogram(x=df.Rating),

    row=1, col=1

)



fig.add_trace(

    go.Pie(values = sizes, labels = labels, textinfo='label+percent'),

    row=1, col=2

)





fig.update_layout(title_text='Rating ~ Frequency')

fig.show()
free_or_not = ["Free" if i == "Free" else "Paid" for i in df.Price]

df["FreeOrPaid"] = free_or_not

price_series = df.groupby("FreeOrPaid")["Name"].count()

fig = go.Figure(data=[go.Pie(values = price_series.values, labels = price_series.index, textinfo='label+percent')])

fig.show()
rating_by_general = df.groupby("Category").mean()

rating_by_general = rating_by_general.sort_values("Rating", ascending = False)



rating_by_free = df[df["Price"] == "Free"].groupby("Category").mean()

rating_by_free.columns = ["Rating_Free","No of people Rated_Free"]



rating_by_paid = df[df["Price"] != "Free"].groupby("Category").mean()

rating_by_paid.columns = ["Rating_Paid","No of people Rated_Paid"]



pd.concat([rating_by_general, rating_by_free, rating_by_paid], axis=1)

class_list = list()

for i in ["General", "Free", "Paid"]:

    for a in range(3):

        class_list.append(i)

        

rating_by_general2 = df.groupby("Category").mean().loc[["Developer Tools", "Books", "Business"]]

rating_by_free2 = df[df["Price"] == "Free"].groupby("Category").mean().loc[["Developer Tools", "Books", "Business"]]

rating_by_paid2 = df[df["Price"] != "Free"].groupby("Category").mean().loc[["Developer Tools", "Books", "Business"]]



df3 = pd.concat([rating_by_general2, rating_by_free2, rating_by_paid2], axis=0)

df3["Class"] = class_list



fig = px.bar(df3,x=df3.index.values, y="Rating", color="Class", barmode="group")

fig.update_layout()

fig.show()
df2 = df.groupby("Category")["Name"].count()





fig = px.bar(x=df2.index, y=df2.values)

fig.update_traces(marker_color='brown')

fig.show()
pd.pivot_table(df, index = "Category", columns = "Rating", values = "No of people Rated", aggfunc="mean")
df_new2 = pd.DataFrame(df[df["Rating"] <= 2.0].groupby(["Category","Rating"])["No of people Rated"].mean())

df_new2 = df_new2.reset_index(level=[0,1])



df_new3 = pd.DataFrame(df[df["Rating"] >= 4.0].groupby(["Category","Rating"])["No of people Rated"].mean())

df_new3 = df_new3.reset_index(level=[0,1])

fig = px.bar(df_new2, x="Category",y="No of people Rated", color="Rating")

fig.show()
fig = px.bar(df_new3, x="Category",y="No of people Rated", color="Rating")

fig.show()

df["Year"] = df["Date"].apply(lambda x: x.split("-")[2])

df["Month"] = df["Date"].apply(lambda x: x.split("-")[1])

df["Month"].replace(["01","02","03","04","05","06","07","08","09","10","11","12"],

                    ["January","February","March","April","May","June","July","August","September","October","November","December"],

                    inplace=True)

a = pd.DataFrame(df.groupby(["Year","Month","Category"])["Name"].count())

a = a.reset_index(level=[0,1,2])

fig = px.bar(a, x="Year",y="Name", color="Month")

fig.show()

fig = px.bar(a, y="Month",x="Name", color="Category")

fig.show()

b = df.groupby(["Year","Category"])["Rating","No of people Rated"].mean()

b = b.reset_index(level=[0,1])

fig = px.scatter(b, x="Year", y="Rating",color="Category",size="No of people Rated")

fig.show()
df_nlp = df.copy()

df_nlp.head()
df_nlp_series = df_nlp["Name"].apply(lambda x: " ".join(i.lower() for i in str(x).split()))
df_nlp_series = df_nlp_series.str.replace("[^\w\s]","") 

df_nlp_series = df_nlp_series.str.replace("\d","") # Numbers
stop_words = stopwords.words("english")

df_nlp_series = df_nlp_series.apply(lambda x: " ".join(i for i in x.split() if i not in stop_words)) # Stopwords
df_nlp_series = df_nlp_series.apply(lambda x: " ".join(Word(i).lemmatize() for i in x.split()))
df_nlp["Name"] = df_nlp_series

df_nlp.head()
a = " ".join(df_nlp[df_nlp["Category"] == "Music"]["Name"])

pd.Series(a.split()).value_counts()
word_cloud = WordCloud(max_font_size=50, background_color="white").generate(a)

plt.figure(figsize=(9,7))

plt.imshow(word_cloud, interpolation="bilinear")

plt.axis("off")

plt.show()