!pip install wikipedia
# Importing modules
import wikipedia
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import random
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS
# Function for grey colour of cloud
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

# Function that makes the cloud
def make_cloud(x, url):
    response = requests.get(url) # Requesting the url for image
    mask = np.array(Image.open(BytesIO(response.content))) # Converting image to numpy array to make mask
    cloud = WordCloud(background_color='black',
                      width=5000, height=5000, 
                      max_words=2000, max_font_size=200, 
                      min_font_size=1, mask=mask, stopwords=STOPWORDS)
    cloud.generate(x) # Generating WordCloud
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(cloud.recolor(color_func=grey_color_func, random_state=3), interpolation='bilinear') # Adding grey colour
    ax.set_axis_off()
    
    plt.show(cloud)
wikipedia.search('Michael Jordan')
df = wikipedia.page('Michael Jordan')
df_content = df.content
df.content
make_cloud(df_content, 'https://www.freepnglogos.com/uploads/logo-jordan-coloring-pages-3.jpg')
wikipedia.search('Avengers: Endgame')
df = wikipedia.page('Avengers: Endgame')
df_content = df.content
df.content
make_cloud(df_content, 'https://thumbs.dreamstime.com/b/logo-avengers-145259952.jpg')
make_cloud(df_content, 'https://3axis.co/user-images/yon2zgor.jpg')
make_cloud(df_content, 'https://i.pinimg.com/originals/89/2d/3f/892d3f42f51327ccd8954fe7dd3ba78b.jpg')