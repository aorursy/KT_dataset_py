import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn
data = pd.read_csv("../input/books.csv", error_bad_lines=False)



data.info()
data_to_plot = data.drop(["title", "authors", "isbn", "isbn13", "bookID", "language_code"], axis = 1)

data_to_plot.columns
data_to_plot.hist(figsize = (20, 20))

plt.show()
fig = plt.subplots(2,2, figsize = (20, 18))

for i in range (len(data_to_plot.columns)):

    plt.subplot(2, 2, i+1)

    sn.distplot(data_to_plot.iloc[:,i], color = "orange")
data_to_plot.corr()

plt.figure(figsize = (10, 10))

sn.heatmap(data_to_plot.corr(), linewidth = 0.5)
sn.jointplot(x = data["text_reviews_count"], y = data["ratings_count"], kind = "scatter")
sn.jointplot(x = data["average_rating"], y = data["# num_pages"], kind = "scatter")
plt.figure(figsize=(20,15))

books = data['title'].value_counts()[:10]

rating = data["average_rating"][:10]

sn.barplot(x = books, y = books.index)

plt.xlabel("# Occurances", fontsize = 20)

plt.ylabel("Books", fontsize = 20)

plt.xticks(fontsize = 17)

plt.yticks(fontsize = 17)



plt.show()
plt.figure(figsize = (12, 10))

language = data["language_code"].value_counts()

sn.barplot(x = language, y = language.index)

plt.yticks(fontsize = 15)

plt.xticks(fontsize = 12)

plt.ylabel("language code", fontsize = 15, color = "blue")

plt.xlabel("Count", fontsize = 15, color = "blue")
plt.figure(figsize = (12, 12))

plotting = data.sort_values(["ratings_count"], ascending = False)

sn.barplot(x = plotting["ratings_count"][:10], y = plotting["title"][:10])

plt.yticks(fontsize = 10)

plt.xticks(fontsize = 10)

plt.ylabel("title", fontsize = 15)

plt.xlabel("ratings_count", fontsize = 15)

fig = plt.figure(figsize = (15, 6))

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.xlabel("author", fontsize = 15, color = "green")

plt.ylabel("count", fontsize = 15, color = "green")

data["authors"].value_counts()[:10].plot.bar()
def graphing_rating(data):

    fig = plt.figure(figsize = (15, 6))

    plt.xticks(fontsize = 12)

    plt.yticks(fontsize = 12)

    plt.xlabel("author", fontsize = 15, color = "green")

    plt.ylabel("Book count", fontsize = 15, color = "green")

    data["authors"].value_counts()[:10].plot.bar()
condition1 = data[data["average_rating"] >= 4.3]

condition2 = data[data["average_rating"] >= 4.5]

condition3 = data[data["average_rating"] >= 4.7]

graphing_rating(condition1)
graphing_rating(condition2)
graphing_rating(condition3)