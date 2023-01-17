# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# To store data
import pandas as pd

# To do linear algebra
import numpy as np

# To plot graphs
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

# To create nicer graphs
import seaborn as sns

# To create interactive graphs
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# To vectorize texts
from sklearn.feature_extraction.text import CountVectorizer
# To decompose texts
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
# To visualize high dimensional dataset
from sklearn.manifold import TSNE

# To tag words
from textblob import TextBlob

# To use new datatypes
from collections import Counter

df = pd.read_json('../input/Sarcasm_Headlines_Dataset.json', lines=True).rename(columns={'headline':'text'})
print('DataFrame Shape: {}'.format(df.shape))
df.head()
