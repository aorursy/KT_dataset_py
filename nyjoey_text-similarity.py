# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from gensim.models import Word2Vec

from gensim.similarities import WmdSimilarity

from nltk import word_tokenize

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%matplotlib inline
train = pd.read_csv('/kaggle/input/text-similarity/train.csv')

test = pd.read_csv('/kaggle/input/text-similarity/test.csv')
# Remove stopwords.

stop_words = stopwords.words('english')

x_list = []

y_list = []

x_list = [w.lower() for w in train['description_x'] if w not in stop_words]

y_list = [w.lower() for w in train['description_y'] if w not in stop_words]

test_x_list = []

test_y_list = []

test_x_list = [w.lower() for w in train['description_x'] if w not in stop_words]

test_y_list = [w.lower() for w in train['description_y'] if w not in stop_words]
train.head()

test.head()

x_list

y_list

z_list = x_list + y_list

z_list
model = Word2Vec(z_list, min_count=1,size= 200,workers=3, window =3, sg = 1)
model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.

distance = model.wmdistance(test_x_list[0], test_y_list[0])

print('distance = %.4f' % distance)