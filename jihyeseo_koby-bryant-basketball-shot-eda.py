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
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
import chardet
np.random.seed(0)
df = pd.read_csv('../input/data.csv') 
dh = pd.read_csv('../input/sample_submission.csv')
df.isnull().sum()
df.sample(10)
df.game_date.head()

df.game_date = pd.to_datetime(df.game_date, format = '%Y-%m-%d')
df.describe(include = 'O').transpose()
df.hist()
