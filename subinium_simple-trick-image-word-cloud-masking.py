import numpy as np 

import requests

from PIL import Image

from io import BytesIO 

import matplotlib.pyplot as plt

from wordcloud import WordCloud
def read_img_from_url(url):

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))

    img_matrix = np.array(img)

    return img_matrix



def read_txt_from_url(url, *size):

    text = requests.get(url).text

    wc = WordCloud(background_color="white", max_words=100 , max_font_size=100, width=size[0], height=size[1], random_state=42)

    wc.generate(text)

    return wc.to_array()

    

img_url = "https://www.biography.com/.image/t_share/MTE4MDAzNDEwNzg5ODI4MTEw/barack-obama-12782369-1-402.jpg"

img_matrix = read_img_from_url(img_url)

txt_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

txt_matrix = read_txt_from_url(txt_url, *img_matrix.shape)



print(img_matrix.shape, txt_matrix.shape)
img_matrix[txt_matrix == 255] = 0

print(img_matrix.shape)
plt.figure(figsize=(10, 10), dpi=300)

plt.imshow(img_matrix)

plt.axis('off')

plt.show()