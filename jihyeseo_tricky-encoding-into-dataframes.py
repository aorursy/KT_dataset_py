# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
filenames = filenames.split('\n')
filenames
def probeEncoding(filename, maxSize):
    for size in range(1,maxSize):
        length = 10 ** size
        with open("../input/" + filename, 'rb') as rawdata:
            result = chardet.detect(rawdata.read(length))

        # check what the character encoding might be
        print(size, length, result)
for f in filenames:
    probeEncoding(f,6)
Nutritions_US = pd.read_csv('../input/Nutritions_US.csv',encoding = 'ISO-8859-1')
Nutritions_US.sample(15)
Nutritions_US.dtypes
drugs_product = pd.read_csv('../input/Drugs_product.csv',encoding = 'ISO-8859-1')
drugs_product.head()
drugs_product.dtypes
drugs_product.ROUTENAME.value_counts().plot.bar()
