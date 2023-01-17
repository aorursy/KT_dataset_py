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
df = pd.read_csv('../input/boe_balance_sheet.csv', skiprows = [272] + list(range(314,327)), thousands = ',')
df.rename(columns = {'Unnamed: 0': 'Year'}, inplace=  True)
df = df.set_index('Year')
df.head()
df.dtypes
df['Check Assets=Liabilities'].unique()

df['Total Assets, £'].plot.line()
df['Government debt '].plot.line()
df['Other Government securities '].plot.line()
df['Other securities '].plot.line()
df['Coin and bullion'].plot.line()
df['Notes in the Bank '].plot.line()
df['Capital '].plot.line()





df.columns
