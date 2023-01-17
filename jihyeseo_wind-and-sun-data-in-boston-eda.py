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
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
df = pd.read_csv('../input/tmy3.csv')
df.head()
df.shape
dg = pd.read_csv('../input/TMY3_StationsMeta.csv')
dg.head()
sns.lmplot(y='Latitude', x='Longitude', hue='Pool', 
           data=dg, 
           fit_reg=False, scatter_kws={'alpha':0.1})
dg.dtypes
dg.shape
dg.Pool.value_counts().plot.bar()
dg.State.value_counts().plot.bar()
dg[dg.State == 'MA']
# Let me choose MARTHAS VINEYARD and PROVINCETOWN (AWOS), BOSTON LOGAN INT'L ARPT
db = df[df['station_number'].isin([725066, 725073, 725090])]
db.columns
db.groupby('station_number')['Pressure (mbar)'].mean()
db.groupby('station_number')['Wspd (m/s)'].mean()
db['date_parsed'] = pd.to_datetime(db['Date (MM/DD/YYYY)'], infer_datetime_format = True)

dgroup = db.groupby(['date_parsed' , 'station_number'])['DNI (W/m^2)'].mean()
dgroup.plot()
dgroup = dgroup.reset_index().head()
dgroup
dgroup = dgroup.set_index('date_parsed')
dgroup['DNI (W/m^2)'].plot.line()
