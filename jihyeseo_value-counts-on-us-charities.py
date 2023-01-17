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

df1 = pd.read_csv("../input/eo1.csv")
df2 = pd.read_csv("../input/eo2.csv")
df3 = pd.read_csv("../input/eo3.csv")
df4 = pd.read_csv("../input/eo4.csv")
df_pr = pd.read_csv("../input/eo_pr.csv")
df_xx = pd.read_csv("../input/eo_xx.csv")
df1.head()
df2.head()
df3.head()
df4.head()
df_pr.head()
df_xx.head()
df = pd.concat([df1,df2,df3,df4,df_pr, df_xx])
df.head()
len(df)
len(df1)
df.STATE.value_counts().head(10)
df.columns
df.T.apply(lambda x: x.nunique(), axis=1).sort_values()
df.STATUS.value_counts().head(10)
df.DEDUCTIBILITY.value_counts().head(10)
df.PF_FILING_REQ_CD.value_counts().head(10)
df.ORGANIZATION.value_counts().head(10)
df.AFFILIATION.value_counts().head(10)
df.FOUNDATION.value_counts().head(10)
