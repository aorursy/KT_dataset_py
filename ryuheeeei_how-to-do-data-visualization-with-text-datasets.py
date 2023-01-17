# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import some modules
import matplotlib.pyplot as plt
import seaborn as sns
import mglearn
from sklearn.feature_extraction.text import CountVectorizer
CommentsApril_df = pd.read_csv("../input/CommentsApril2018.csv")
ArticlesApril_df = pd.read_csv("../input/ArticlesApril2018.csv")
ArticlesApril_df.head()
CommentsApril_df.head()
# Visualization with new desk element
newDesk_group_dict = {}

# classify that the article has recommends
for i in range(len(ArticlesApril_df)):
    if ArticlesApril_df["newDesk"][i] in newDesk_group_dict.keys():
        newDesk_group_dict[ArticlesApril_df["newDesk"][i]] += 1
    else:
        newDesk_group_dict.setdefault(ArticlesApril_df["newDesk"][i], 1)
newDesk_group_dict
# I'll want to visualize how many articles have in each topic. 
newDesk_group_list = list(newDesk_group_dict.items())
newDesk_group_list.sort(key=lambda x: x[1], reverse=True)
x = []
y = []

for i in range(len(newDesk_group_list)):
    x.append(newDesk_group_list[i][0])
    y.append(newDesk_group_list[i][1])
plt.figure(figsize=(10, 10))
plt.title("The relationship between the category of articles and the number of articles")
plt.xlabel("Category")
plt.ylabel("Number of articles in each Topics")
plt.xticks(rotation=90)
sns.barplot(x, y)
plt.savefig("Category-articles.png")
# Visualization with newDesk element
comments_newDesk_group_dict = {}

# Count the number of comments by newDesk category. 
for i in range(len(CommentsApril_df)):
    if CommentsApril_df["newDesk"][i] in comments_newDesk_group_dict.keys():
        comments_newDesk_group_dict[CommentsApril_df["newDesk"][i]] += 1
    else:
        comments_newDesk_group_dict.setdefault(CommentsApril_df["newDesk"][i], 1)
# Let's take a look at comments_newDesk_group_dict
comments_newDesk_group_dict
# I'll want to visualize how many comments have in each topic. 
comments_newDesk_group_list = list(comments_newDesk_group_dict.items())
comments_newDesk_group_list.sort(key=lambda x: x[1], reverse=True)
comments_topic = []
n_comments = []

for i in range(len(comments_newDesk_group_list)):
    comments_topic.append(comments_newDesk_group_list[i][0])
    n_comments.append(comments_newDesk_group_list[i][1])
plt.figure(figsize=(10, 10))
plt.title("The relationship between the category of comments and the number of comments")
plt.xlabel("Category")
plt.ylabel("Number of comments in each Topics")
plt.xticks(rotation=90)
sns.barplot(comments_topic, n_comments)
plt.savefig("Category-comments.png")
df = pd.DataFrame.from_dict(comments_newDesk_group_dict, columns=["number of comments"] ,orient="index")
df["number of articles"] = newDesk_group_dict.values()
df.head()
w = 0.4
xtick = [i for i in range(len(df))]
xlabel = df.index

_, ax1 = plt.subplots(figsize=(20, 10))
ax1.bar(xtick, df["number of comments"], width=w/2, label='n of comments', align="center")
ax1.set_ylabel("n of comments")
plt.xticks(np.array(xtick) + w/2, xlabel, rotation=90)  
plt.legend(bbox_to_anchor=(1, 1))

ax2 = ax1.twinx()
ax2.bar(w /2 + np.array(xtick), df["number of articles"], width=w/2, color="orange", label='n of articles', align="center")
ax2.set_ylabel("n of articles")
plt.legend(bbox_to_anchor=(1, 0.95))

plt.title("number of comments and articles in each Topic")
plt.show()
plt.savefig("n_of_comments_articles.png")
# I want to do some stastitical analysis.
CommentsApril_df["recommendations"].describe()
plt.xlim(0, 200)
plt.title("The distribution of recommendations")
plt.xlabel("# of Recommendations")
plt.ylabel("# of Comments")
CommentsApril_df["recommendations"].hist(bins=max(CommentsApril_df["recommendations"]))
plt.grid(None)
plt.savefig("recommend-dist.png")
comment_list = list(CommentsApril_df["commentBody"])
comment_list[:5]
vect = CountVectorizer()
vect.fit(comment_list)
feature_names = vect.get_feature_names()
feature_names[:10]
feature_names[-10:]
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))
vect = CountVectorizer(max_features=1000, max_df=.15)
X = vect.fit_transform(comment_list)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(learning_method="batch", max_iter=10, random_state=0)
document_topics = lda.fit_transform(X)
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)
