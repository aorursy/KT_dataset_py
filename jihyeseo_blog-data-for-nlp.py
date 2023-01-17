import math
import seaborn as snsº
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
df = pd.read_csv('../input/blogtext.csv')
df.head()
df.date = pd.to_datetime(df.date , infer_datetime_format = True, errors = 'coerce')
df.dtypes
df.describe(include = 'O').transpose()
df.describe(exclude = 'O').transpose()
#https://stackoverflow.com/questions/27472548/pandas-scatter-plotting-datetime
df.plot(x = 'date', y ='age', style = '.' )    
#https://stackoverflow.com/questions/27472548/pandas-scatter-plotting-datetime
df.plot(x = 'date', y ='age', style = '.')#, color = 'gender' )    
df.shape
df.columns
dg = df[['gender','age','date']]

g = sns.PairGrid(dg, hue="gender")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()
