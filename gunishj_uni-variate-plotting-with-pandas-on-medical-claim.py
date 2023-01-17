# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
claim = pd.read_csv('../input/assembled-workers-compensation-claims-beginning-2000.csv')
claim.head()
claim['Current Claim Status'].value_counts().plot.bar()
claim['District Name'].value_counts().plot.bar()
(claim['District Name'].value_counts()/ len(claim)).plot.bar()
claim[(claim['Age at Injury'] >= 20) & (claim['Age at Injury'] <= 70)]['Age at Injury'].value_counts().sort_index().plot.bar()
claim['Age at Injury'].value_counts().sort_index().plot.line()
claim['Age at Injury'].value_counts().sort_index().plot.area()
claim.head()
claim[claim['Birth Year'] > 1750]['Birth Year'].plot.hist()
claim['Birth Year'].plot.hist()