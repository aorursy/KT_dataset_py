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
gps_data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
gps_data['Reviews'] = pd.to_numeric(gps_data['Reviews'],errors = 'coerce')
top_reviewed_apps = gps_data.groupby('App',as_index=False)['Reviews'].max().sort_values('Reviews',ascending=False).head(10)
print(top_reviewed_apps.plot(kind='bar',x='App',y='Reviews'))