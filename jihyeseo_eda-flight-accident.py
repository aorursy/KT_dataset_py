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
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
for size in range(1,9):
    length = 10 ** size
    with open("../input/AviationData.csv", 'rb') as rawdata:
        result = chardet.detect(rawdata.read(length))

    # check what the character encoding might be
    print(size, length, result)
df = pd.read_csv('../input/AviationData.csv', encoding = 'latin1')
df.dtypes
df['Event.Date'] = pd.to_datetime(df['Event.Date'], infer_datetime_format = True)
df = df.set_index('Event.Date')
df['Total.Serious.Injuries'].plot.line()
df.shape
df['Total.Uninjured'].plot.line()
df['Total.Fatal.Injuries'].plot.line()
df['Total.Fatal.Injuries'].hist(bins = 50)
df.Make.unique()
df['Engine.Type'].unique()
df['Purpose.of.Flight'].unique()
df['Weather.Condition'].unique()
