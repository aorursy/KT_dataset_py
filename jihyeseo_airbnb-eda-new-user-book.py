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
df = pd.read_csv('../input/train_users_2.csv')
dg = pd.read_csv('../input/test_users.csv')
dh = pd.read_csv('../input/sample_submission.csv')
di = pd.read_csv('../input/sessions.csv')
dj = pd.read_csv('../input/countries.csv')
dk = pd.read_csv('../input/age_gender_bkts.csv')

df.head()
dk.head()
dj.head()
di.head()
df.columns
dg.columns
# predict country_destination
df.country_destination.value_counts().plot.bar()
