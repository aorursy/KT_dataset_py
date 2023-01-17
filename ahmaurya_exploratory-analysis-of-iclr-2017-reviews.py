# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iclr2017_papers_df = pd.read_csv('../input/iclr2017_papers.csv')

iclr2017_papers_df.sample(10)
iclr2017_conversations_df = pd.read_csv('../input/iclr2017_conversations.csv')

iclr2017_conversations_df.sample(10)
from functools import reduce

keywords = iclr2017_papers_df['keywords'].astype(str).apply(lambda x: [s.strip() for s in x.split(',') if s != 'nan'])

keywords = reduce(lambda x, y: x+y, keywords)

keywords = pd.Series(keywords).value_counts()

keywords
iclr2017_conversations_df.rating.value_counts()
iclr2017_conversations_df.decision.value_counts()