import pandas as pd
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
df=pd.read_csv("../input/zuckerberg-testimony/mark.csv")

stopwords=set(STOPWORDS).union("going","want")
alice_mask = np.array(Image.open("../input/fbmask/fbmask.png"))
names = df["Text"]
#print(names)
wordcloud = WordCloud(max_words=150,stopwords=stopwords,max_font_size=70, width=800, height=300,mask=alice_mask,background_color ="white").generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud,interpolation="bilinear")
plt.title("Zuckerberg Testimony", fontsize=30)
plt.axis("off")
plt.savefig("mark",dpi=600)
plt.show()
