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
df= pd.read_csv('../input/database.csv')

df.head()
df.describe(include = 'O').transpose()
df['Aircraft Type'].value_counts().plot.bar()
# highly unbalanced dataset
df['Warning Issued'].value_counts().plot.bar()
# needs to clean n, y into N, Y
df.groupby('Incident Year')['Record ID'].count().plot.line()
df.groupby('Incident Month')['Record ID'].count().plot.line()