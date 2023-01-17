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
filenames = filenames.split('\n')

dfs = dict()
for f in  filenames:
    dfs[f[:-4]] = pd.read_csv('../input/'+ f)
