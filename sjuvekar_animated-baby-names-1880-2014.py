%matplotlib inline
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import matplotlib.animation as animation
names_df = pd.read_csv("../input/NationalNames.csv")
years = names_df.Year.unique()
print(years)
def get_wordcloud(year):
    year_df = names_df[names_df.Year == year]
    year_names = year_df.Name
    wordcloud = WordCloud(width=800, height=400, max_words=20).generate(" ".join(year_names))
    return wordcloud.to_image()
year = years[0]
fig = plt.figure(figsize=(16, 16))
im = plt.imshow(get_wordcloud(year), cmap=plt.get_cmap('jet'))
ax = plt.axis("off")

def updatefig(*args):
    global year
    year = 0 if year == len(years) - 1 else year+1
    im.set_data(get_wordcloud(year))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True, repeat=False)
plt.show()