# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
filenames = check_output(["ls", "../input"]).decode("utf8").split('\n')

# Any results you write to the current directory are saved as output.
filenames
dfs = []
for f in filenames[:-1]:
    df = pd.read_csv('../input/' + f)
    dfs.append(df)
dfs[0].head()
dfs[0].dtypes
dfs[0].shape
dfs[1].shape
dfs[1].head()
dfs[1].party.value_counts().plot.bar()
dfs[2].head()
dfs[2].author_state.value_counts().plot.bar()
dfs[3].head()
dfs[3]['ynRatio'] = dfs[3].yes_votes / dfs[3].no_votes
dfs[3].ynRatio.plot.line()
