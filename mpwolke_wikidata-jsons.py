#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS_pKmxJrjljhJOnCorhX34L8VtLdjkQJTprD9825MyRBzKkjcW',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS5kJ7nNH9pmogBtrtEtKz6XfhjaU5dM71Rq0PphiAzrEC5UCHP',width=400,height=400)
df = pd.read_json('../input/wikidata-jsons/wikidata_proc_json 2/wikidata_rev_type_dict.json', encoding='ISO-8859-2')
df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F85988%2Ffe73a6cc-6e22-cc18-10e9-de72f11236ff.png?ixlib=rb-1.2.2&auto=format&gif-q=60&q=75&s=c3110864454256b1953227de4ef3c5f4',width=400,height=400)
df1 = pd.read_json('../input/wikidata-jsons/wikidata_proc_json 2/wikidata_type_dict.json', encoding='ISO-8859-2')
df1.head() 
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT30QCqhg1SXJM7_CYYLyu5VqLmK5_olhNx8eXt1no-cmvDaCJZ',width=400,height=400)
x = ['Q5354802', 'Q15726688']

y = ['P991', 'P17']

plt.barh(x,y)

plt.title("Q5354802")

plt.xlabel("P991")

plt.ylabel('Q5354802')

plt.style.use('seaborn-dark-palette')
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df1.Q676050)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQr1o9RH1GtZatMSNynlX52TDojxNhA3Dpn372fbJADg9nr_Pkl',width=400,height=400)
#Polar Axis from Saurav Anand @saurav9786

r = np.arange(0, 2, 0.01)

theta = 2 * np.pi * r



ax = plt.subplot(111, projection='polar')

ax.plot(theta, r)

ax.set_rmax(2)

ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks

ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line

ax.grid(True)



ax.set_title("A line plot on a polar axis", va='bottom')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR9sJsge4D3GI4RMjjatPjQNaQuhUBI1BY4aRO4j7rTBxQQq4QK',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTAfBuclafvCmYhZe7sfdLJCc-n0hFYRmYTPlAIN-vm3fyxYHaG',width=400,height=400)