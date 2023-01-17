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
filenames = filenames.split('\n')
print(filenames)
df = pd.read_csv('../input/items.txt', sep =';', header = None)
df = df.rename(columns = {0:'itemId', 1:'itemDesc', 2:'item1',3:'item2',4:'item3',
                     5:'item4',6:'item5',7:'item6',8:'item7',9:'item8'})
df.head()
dg = pd.read_csv('../input/ratings.txt', sep =';', header = None)
dg = dg.rename(columns = {0:'contextId', 1:'itemId', 2:'rating',3:'userId'})
dg.head()
dh = pd.read_csv('../input/usersDescription.txt', sep =';', header = None)
dh = dh.rename(columns = {0:'contextId', 1:'age', 2:'man', 3:'woman', 50:'userId'})
for i in range(3,16):
    dh = dh.rename(columns = {i+1:'SPC'+str(i-2)})
for i in range(16,29):
    dh = dh.rename(columns = {i+1:'userSpecialty'+str(i-15)})
for i in range(29,39):
    dh = dh.rename(columns = {i+1:'userPreference'+str(i-28)})
for i in range(39, 47):
    dh = dh.rename(columns = {i+1:'userHighDegree'+str(i-38)})
for i in range(47,49):
    dh = dh.rename(columns = {i+1:'weatherSeason'+str(i-46)})
            
    
dh.head()
sorted(list(dh.columns))
df.head()
dg.head()
dh.head()
dfg = df.join(dg, on ='itemId', lsuffix='_item', rsuffix='_rating')
dfg.head()
