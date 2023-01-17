import urllib

import random



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

from wordcloud import WordCloud, ImageColorGenerator
df = pd.read_csv('../input/index.csv')



text = ' '.join(df['description'].values)

text[:300]
wordcloud = WordCloud(

    background_color="black",

    width=500,

    height=500,

    random_state=42,

)



wordcloud.generate(text)





plt.figure(figsize=(5, 5))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
def load_mask(mask_url):

    with urllib.request.urlopen(mask_url) as resp:

        mask = np.asarray(bytearray(resp.read()), dtype="uint8")

        mask = cv2.imdecode(mask, cv2.IMREAD_COLOR)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        

    return mask







mask = load_mask('https://i.imgur.com/nY3pUfe.png')



plt.imshow(mask)

plt.axis('off');
wordcloud = WordCloud(

    background_color="white", 

    mask=mask,

    max_words=500,

    random_state=42,

)



wordcloud.generate(text)





plt.figure(figsize=(8, 8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
wordcloud = WordCloud(

    background_color="white", 

    mask=mask,

    max_words=500,

    random_state=42,

    prefer_horizontal=0,

)



wordcloud.generate(text)





plt.figure(figsize=(8, 8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
colormaps = [

    'winter', 'spring', 'summer', 

    'autumn', 'gray', 'rainbow', 

    'Blues', 'jet', 'Oranges',

]



plt.figure(figsize=(16, 16))



for ind, colormap in enumerate(colormaps, 1):

    wordcloud = WordCloud(

        background_color="white", 

        mask=mask,

        max_words=200,

        colormap=colormap,

        random_state=42,

    )

    

    wordcloud.generate(text)



    plt.subplot(3, 3, ind)

    plt.imshow(wordcloud)

    plt.title(f'colormap: {colormap}', fontsize=16)

    plt.axis("off")
colored_mask = load_mask('https://i.imgur.com/HR1wh2N.png')



plt.imshow(colored_mask)

plt.axis('off');
wordcloud = WordCloud(

    background_color="white", 

    mask=colored_mask,

    max_words=500,

    random_state=42,

)

    

wordcloud.generate(text)



image_colors = ImageColorGenerator(colored_mask)



plt.figure(figsize=(8, 8))

plt.imshow(wordcloud.recolor(color_func=image_colors))

plt.axis("off")

plt.show()
wordcloud.to_file('example.png')
new_mask = load_mask('https://res.cloudinary.com/practicaldev/image/fetch/s--sKgAlMeY--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/i/9fzqwxykz0ha9emyrjin.jpg')



plt.imshow(new_mask)

plt.axis('off')

plt.show()



wordcloud = WordCloud( 

    mask=new_mask,

    random_state=42,

    max_font_size=50,

    colormap="Reds",

)

    

wordcloud.generate(text) 

wordcloud.to_file('love.png')



plt.figure(figsize=(8, 8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
new_mask = load_mask('https://www.dailyhostnews.com/wp-content/uploads/2018/07/Python-featured-1050x600.jpg')



plt.imshow(new_mask)

plt.axis('off')

plt.show()



wordcloud = WordCloud(

    background_color="white",

    mask=new_mask,

    random_state=42,

    max_font_size=50,

    max_words=1000,

)

    

wordcloud.generate(text) 



image_colors = ImageColorGenerator(new_mask)



plt.figure(figsize=(16, 8))

plt.imshow(wordcloud.recolor(color_func=image_colors))

plt.axis("off")

plt.show()
new_mask = load_mask("https://www.kaggle.com/static/images/site-logo.png")

new_mask = cv2.resize(new_mask, (new_mask.shape[1] * 5, new_mask.shape[0] * 5))



plt.imshow(new_mask)

plt.axis('off')

plt.show();



wordcloud = WordCloud(

    background_color="black",

    mask=new_mask,

    random_state=42,

    max_words=1000,

)

    

wordcloud.generate(text) 



image_colors = ImageColorGenerator(new_mask)



plt.figure(figsize=(16, 8))

plt.imshow(wordcloud.recolor(color_func=image_colors))

plt.axis("off")

plt.show()
new_mask = load_mask("https://luxuryasiainsider.files.wordpress.com/2018/01/google_logo.png?w=723")

# new_mask = cv2.resize(new_mask, (new_mask.shape[1] * 20, new_mask.shape[0] * 20))



plt.imshow(new_mask)

plt.axis('off')

plt.show();



wordcloud = WordCloud(

    background_color="white",

    mask=new_mask,

    random_state=42,

    max_words=1000,

)

    

wordcloud.generate(text) 



image_colors = ImageColorGenerator(new_mask)



plt.figure(figsize=(16, 8))

plt.imshow(wordcloud.recolor(color_func=image_colors))

plt.axis("off")

plt.show()