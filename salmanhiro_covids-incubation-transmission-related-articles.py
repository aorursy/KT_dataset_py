# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

df.shape
df.head()
df.tail()
title = df.copy()
title = title.dropna(subset=['title'])
title.head()
title['title'] = title['title'].str.replace('[^a-zA-Z]', ' ', regex=True)

title['title'] = title['title'].str.lower()
title.tail()
title['keyword_transmission'] = title['title'].str.find('transmission') 
title.head()
included_transmission = title.loc[title['keyword_transmission'] != -1]

included_transmission
title['keyword_transmit'] = title['title'].str.find('transmit')

included_transmit = title.loc[title['keyword_transmit'] != -1]

included_transmit
title['keyword_incubation'] = title['title'].str.find('incubation')

included_incubation = title.loc[title['keyword_incubation'] != -1]

included_incubation