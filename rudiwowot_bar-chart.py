import pandas as pd

df = pd.read_csv("/kaggle/input/hashtag-twitter/hashtag_PesanSambat.csv", index_col=0)
df.head()
df.columns
df.info()
# Grouping by username

df2 = df.groupby(by='username').sum()
df2
# sort value

most_likes = df2['likes'].sort_values(ascending=False)
most_likes
import matplotlib.pyplot as plt
plt.figure()
plt.bar(most_likes.index, most_likes)
plt.figure()
plt.bar(most_likes.index, most_likes)
plt.xticks(rotation=45, ha='right')
most_likes = most_likes.head(5)
plt.figure()
plt.bar(most_likes.index, most_likes)
plt.title("Most Likes Post")
plt.xlabel("username")
plt.ylabel("count")
plt.figure()
plt.bar(most_likes.index, most_likes, color='grey')
plt.title("Most Likes Post")
plt.xlabel("username")
plt.ylabel("count")