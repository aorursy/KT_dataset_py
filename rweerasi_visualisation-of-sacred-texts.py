import numpy as np

from PIL import Image

from os import path

import matplotlib.pyplot as plt

import pylab

import random

from wordcloud import WordCloud, STOPWORDS



def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)



%pylab inline

pylab.rcParams['figure.figsize'] = (18,10)



mask = np.array(Image.open("../input/buddha.png"))



text = open("../input/buddha.txt").read()[28:]



wc = WordCloud(max_words=2000, mask=mask, stopwords=STOPWORDS, margin=5,

               random_state=1).generate(text)



default_colors = wc.to_array()

plt.title("The Gospel of Buddha as Buddha")

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))

plt.axis("off")

plt.show()
import numpy as np

from PIL import Image

from os import path

import matplotlib.pyplot as plt

import pylab

import random

from wordcloud import WordCloud, STOPWORDS



def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)



%pylab inline

pylab.rcParams['figure.figsize'] = (15,7)



mask = np.array(Image.open("../input/JC.jpg"))



text = open("../input/kingjamesbible.txt").read()[33:]



wc = WordCloud(max_words=2000, mask=mask, stopwords=STOPWORDS, margin=5,

               random_state=1).generate(text)



default_colors = wc.to_array()

plt.title("The Bible as Jesus")

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))

plt.axis("off")

plt.show()
import numpy as np

from PIL import Image

from os import path

import matplotlib.pyplot as plt

import pylab

import random

from wordcloud import WordCloud, STOPWORDS



def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)



%pylab inline

pylab.rcParams['figure.figsize'] = (15,7)



mask = np.array(Image.open("../input/LDS.jpg"))



text = open("../input/mormon.txt").read()[35:]



wc = WordCloud(max_words=2000, mask=mask, stopwords=STOPWORDS, margin=5,

               random_state=1).generate(text)



default_colors = wc.to_array()

plt.title("The Book of Mormon as Latter Day Saints Church")

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))

plt.axis("off")

plt.show()
import numpy as np

from PIL import Image

from os import path

import matplotlib.pyplot as plt

import pylab

import random

from wordcloud import WordCloud, STOPWORDS



def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)



%pylab inline

pylab.rcParams['figure.figsize'] = (15,7)



mask = np.array(Image.open("../input/mosque.png"))



text = open("../input/quran.txt").read()[36:]



wc = WordCloud(max_words=2000, mask=mask, stopwords=STOPWORDS, margin=5,

               random_state=1).generate(text)



default_colors = wc.to_array()

plt.title("The Quran as Islamic Mosque")

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))

plt.axis("off")

plt.show()