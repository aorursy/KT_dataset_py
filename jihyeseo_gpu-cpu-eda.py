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
df = pd.read_csv('../input/Intel_CPUs.csv')
dg = pd.read_csv('../input/All_GPUs.csv')
df.dtypes
dg.dtypes
#dg.Architecture.value_counts().plot.bar()
dg.Release_Date.sample(15)
dg.Release_Price.sample(15)
dg.Release_Date = dg.Release_Date.str.strip()
dg.Release_Date = pd.to_datetime(dg.Release_Date, infer_datetime_format=True, errors ='coerce')
# https://stackoverflow.com/questions/32332984/python-pandas-parse-datetime-string-with-months-names 
sorted(dg.Release_Date.unique())
# https://stackoverflow.com/questions/34505579/pandas-to-datetime-valueerror-unknown-string-format
dg.head()
dg.dtypes
dg = dg.set_index('Release_Date')
dg.DisplayPort_Connection.plot()
dg.HDMI_Connection.plot()
dg.Open_GL.plot()
dg.DVI_Connection.plot()
dg.Manufacturer.value_counts().plot.bar()
