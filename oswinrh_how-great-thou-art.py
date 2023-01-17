import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd 

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/t_asv.csv')

df.head(10)
def lower_column_t(data):

    values = data['t']

    values = values.lower()

    data['t'] = values

    return data
df = df.apply(lower_column_t, axis=1)

df.head(10)
mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)

mpl.rcParams['font.size']=12                #10 

mpl.rcParams['savefig.dpi']=100             #72 

mpl.rcParams['figure.subplot.bottom']=.1 





stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=80, 

                          random_state=42

                         ).generate(str(df['t']))



fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()