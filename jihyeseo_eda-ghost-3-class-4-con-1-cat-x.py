#https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo/data
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

df = pd.read_csv('../input/train.csv')
dg = pd.read_csv('../input/test.csv')
dh = pd.read_csv('../input/sample_submission.csv')
df.describe(include = 'O').transpose()
df.describe(exclude = 'O').transpose()
dg.sample(4)
sns.lmplot(y='hair_length', x='bone_length', hue='type', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':1})
df.columns
sns.pairplot(df[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul',
       'color', 'type']], hue="type")
sns.countplot(x="color", hue="type", data=df)
