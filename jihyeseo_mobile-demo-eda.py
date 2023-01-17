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
dg = pd.read_csv('../input/gender_age_test.csv')
df = pd.read_csv('../input/gender_age_train.csv')

di = pd.read_csv('../input/app_events.csv')
dj = pd.read_csv('../input/app_labels.csv')
dh = pd.read_csv('../input/events.csv')

dk = pd.read_csv('../input/sample_submission.csv')
dl = pd.read_csv('../input/label_categories.csv')
dm = pd.read_csv('../input/phone_brand_device_model.csv') 
df.head()
df.group.value_counts().sort_index().plot.bar()
dg.head()
dh.head()
dh.timestamp = pd.to_datetime(dh.timestamp, format = '%Y-%m-%d %H:%M:%S')
di.head()
dj.head()
dj.label_id.value_counts().plot.bar()
dk.head()
dl.head()
dl.category.value_counts().plot.bar()
dm.head()
dm.phone_brand.value_counts().plot.bar()
