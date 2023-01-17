import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from IPython.display import Image
Image(filename='../input/yoda-pics-1/The-greatest-teacher-failure-is.-Master-Yoda-Star-Wars.png')
eiv=open('../input/star-wars-movie-scripts/SW_EpisodeIV.txt','r')
eiv=eiv.read()
print(eiv[:500])
eiv1=eiv.split("\n")
eiv1[:10]

print(len(eiv1)-1)
from PIL import Image
mask = np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/yoda.png"))
stop_words=set(STOPWORDS)
eiv_wc=WordCloud(width=800,height=500,mask=mask,random_state=21, max_font_size=110,stopwords=stop_words).generate(eiv)
fig=plt.figure(figsize=(16,8))
plt.imshow(eiv_wc)
