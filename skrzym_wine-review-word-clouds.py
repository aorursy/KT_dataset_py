# Typical imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re



# Not so typical

import matplotlib.image as image

import matplotlib.colors

from collections import defaultdict

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image

from IPython.display import Image as im

import squarify as sq

data = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col=0)

countries = data.country.value_counts()



# Limit top countries to those with more than 500 reviews

temp_dict = countries[countries>500].to_dict()

temp_dict['Other'] = countries[countries<501].sum()

less_countries = pd.Series(temp_dict)

less_countries.sort_values(ascending=False, inplace=True)



# Turn Series into DataFrame for display purposes

df = less_countries.to_frame()

df.columns=['Number of Reviews']

df.index.name = 'Country'

df
# New colors for tree map since base ones are bland

cmap = plt.cm.gist_rainbow_r

norm = matplotlib.colors.Normalize(vmin=0, vmax=15)

colors = [cmap(norm(value)) for value in range(15)]

np.random.shuffle(colors)



# Use squarify to plot the tree map with the custom colors

fig,ax = plt.subplots(1,1,figsize=(11,11))

sq.plot(sizes=less_countries.values, label=less_countries.index.values, alpha=0.5, ax=ax, color=colors)

plt.axis('off')

plt.title('Countries by Number of Wine Reviews')

plt.show()
descriptions = defaultdict(list)

data.apply(lambda x: descriptions[x.country].append(x.description), axis=1)

descriptions['Italy'][0:5]
unwanted_characters = re.compile('[^A-Za-z ]+')

for country in list(data.country.unique()):

    desc_string = ' '.join(descriptions[country])

    descriptions[country] = ' '.join([w.lower() for w in re.sub(unwanted_characters, ' ', desc_string).split() if len(w) > 3])
descriptions['Italy'][0:500]
wine_stopwords = ['drink','wine','wines','flavor','flavors','note','notes','palate','finish','hint','hints','show','shows']

for w in wine_stopwords:

    STOPWORDS.add(w)
def generate_country_wordcloud(words, mask_image, filename=None, colormap='jet'):

    mask = np.array(Image.open(mask_image))

    wc = WordCloud(background_color="white", max_words=3000, mask=mask, stopwords=STOPWORDS, colormap=colormap)

    wc.generate(words)

    if filename:

        wc.to_file(filename)

    return wc
masks = dict()

masks['Argentina'] = '../input/outline-images-of-countries/argentina_bw_map.jpg'

masks['Australia'] = '../input/outline-images-of-countries/australia_bw_map.jpg'

masks['Austria'] = '../input/outline-images-of-countries/austria_bw_map.jpg'

masks['Chile'] = '../input/outline-images-of-countries/chile_bw_map.jpg'

masks['France'] = '../input/outline-images-of-countries/france_bw_map.jpg'

masks['Italy'] = '../input/outline-images-of-countries/italy_bw_map.jpg'

masks['Portugal'] = '../input/outline-images-of-countries/portugal_bw_map.jpg'

masks['Spain'] = '../input/outline-images-of-countries/spain_bw_map.jpg'

masks['US'] = '../input/outline-images-of-countries/usa_bw_map.jpg'

masks['Germany'] = '../input/outline-images-of-countries/germany_bw_map.jpg'

masks['Israel'] = '../input/outline-images-of-countries/israel_bw_map.jpg'

masks['New Zealand'] = '../input/outline-images-of-countries/newzealand_bw_map.jpg'

masks['South Africa'] = '../input/outline-images-of-countries/southafrica_bw_map.jpg'

us_wc = generate_country_wordcloud(descriptions['US'], masks['US'], 'US.jpg')

us_wc.to_image()
france_wc = generate_country_wordcloud(descriptions['France'], masks['France'], 'France.jpg')

france_wc.to_image()
italy_wc = generate_country_wordcloud(descriptions['Italy'], masks['Italy'], 'Italy.jpg')

italy_wc.to_image()
spain_wc = generate_country_wordcloud(descriptions['Spain'], masks['Spain'], 'Spain.jpg')

spain_wc.to_image()
portugal_wc = generate_country_wordcloud(descriptions['Portugal'], masks['Portugal'], 'Portugal.jpg')

portugal_wc.to_image()
chile_wc = generate_country_wordcloud(descriptions['Chile'], masks['Chile'], 'Chile.jpg')

chile_wc.to_image()
argentina_wc = generate_country_wordcloud(descriptions['Argentina'], masks['Argentina'], 'Argentina.jpg')

argentina_wc.to_image()
austria_wc = generate_country_wordcloud(descriptions['Austria'], masks['Austria'], 'Austria.jpg')

austria_wc.to_image()
australia_wc = generate_country_wordcloud(descriptions['Australia'], masks['Australia'], 'Australia.jpg')

australia_wc.to_image()
germany_wc = generate_country_wordcloud(descriptions['Germany'], masks['Germany'], 'Germany.jpg')

germany_wc.to_image()
nz_wc = generate_country_wordcloud(descriptions['New Zealand'], masks['New Zealand'], 'NewZealand.jpg')

nz_wc.to_image()
south_africa_wc = generate_country_wordcloud(descriptions['South Africa'], masks['South Africa'], 'SouthAfrica.jpg')

south_africa_wc.to_image()
israel_wc = generate_country_wordcloud(descriptions['Israel'], masks['Israel'], 'Israel.jpg')

israel_wc.to_image()
flags = dict()

flags['Argentina'] = '../input/world-flags/Argentina.png'

flags['Australia'] = '../input/world-flags/Australia.png'

flags['Austria'] = '../input/world-flags/Austria.png'

flags['Chile'] = '../input/world-flags/Chile.png'

flags['France'] = '../input/world-flags/France.png'

flags['Italy'] = '../input/world-flags/Italy.png'

flags['Portugal'] = '../input/world-flags/Portugal.png'

flags['Spain'] = '../input/world-flags/Spain.png'

flags['US'] = '../input/world-flags/United_States.png'

flags['Germany'] = '../input/world-flags/Germany.png'

flags['Israel'] = '../input/world-flags/Israel.png'

flags['New Zealand'] = '../input/world-flags/New_Zealand.png'

flags['South Africa'] = '../input/world-flags/South_Africa.png'

# Developed to see what the flag masks would look like without blocking out the white areas of the flags.

# Turns out its not as nice IMO but I will leave it here for others to mess with.

def replace_color(img_data, old_color, new_color):

    

    r1, g1, b1 = old_color # Original value

    r2, g2, b2 = new_color # Value that we want to replace it with



    red, green, blue = img_data[:,:,0], img_data[:,:,1], img_data[:,:,2]

    mask = (red == r1) & (green == g1) & (blue == b1)

    img_data[:,:,:3][mask] = [r2, g2, b2]

    

    return img_data
def generate_flag_wordcloud(words, flag_image, filename=None):

    mask = np.array(Image.open(flag_image))

    #mask = replace_color(mask, (255, 255, 255), (200, 200, 200))

    wc = WordCloud(background_color="white", max_words=3000, mask=mask, stopwords=STOPWORDS, 

                   max_font_size=50, random_state=42)

    wc.generate(words)

    image_colors = ImageColorGenerator(mask)

    wc.recolor(color_func=image_colors)

    if filename:

        wc.to_file(filename)

    return wc
us_wc_flag = generate_flag_wordcloud(descriptions['US'], flags['US'], 'US_flag.jpg')

us_wc_flag.to_image()
france_wc_flag = generate_flag_wordcloud(descriptions['France'], flags['France'], 'France_flag.jpg')

france_wc_flag.to_image()
italy_wc_flag = generate_flag_wordcloud(descriptions['Italy'], flags['Italy'], 'Italy_flag.jpg')

italy_wc_flag.to_image()
spain_wc_flag = generate_flag_wordcloud(descriptions['Spain'], flags['Spain'], 'Spain_flag.jpg')

spain_wc_flag.to_image()
portugal_wc_flag = generate_flag_wordcloud(descriptions['Portugal'], flags['Portugal'], 'Portugal_flag.jpg')

portugal_wc_flag.to_image()
chile_wc_flag = generate_flag_wordcloud(descriptions['Chile'], flags['Chile'], 'Chile_flag.jpg')

chile_wc_flag.to_image()
argentina_wc_flag = generate_flag_wordcloud(descriptions['Argentina'], flags['Argentina'], 'Argentina_flag.jpg')

argentina_wc_flag.to_image()
austria_wc_flag = generate_flag_wordcloud(descriptions['Austria'], flags['Austria'], 'Austria_flag.jpg')

austria_wc_flag.to_image()
australia_wc_flag = generate_flag_wordcloud(descriptions['Australia'], flags['Australia'], 'Australia_flag.jpg')

australia_wc_flag.to_image()
germany_wc_flag = generate_flag_wordcloud(descriptions['Germany'], flags['Germany'], 'Germany_flag.jpg')

germany_wc_flag.to_image()
nz_wc_flag = generate_flag_wordcloud(descriptions['New Zealand'], flags['New Zealand'], 'NewZealand_flag.jpg')

nz_wc_flag.to_image()
south_africa_wc_flag = generate_flag_wordcloud(descriptions['South Africa'], flags['South Africa'], 'SouthAfrica_flag.jpg')

south_africa_wc_flag.to_image()
israel_wc_flag = generate_flag_wordcloud(descriptions['Israel'], flags['Israel'], 'Israel_flag.jpg')

israel_wc_flag.to_image()