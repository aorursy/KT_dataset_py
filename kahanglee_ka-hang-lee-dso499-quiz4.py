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
consumption = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')

consumption.head(5)
consumption.columns
consumption.shape
consumption.shape[0]
consumption[consumption.age < 22].shape[0]
consumption.iloc[:5]
consumption.Dalc.mean()
consumption.Walc.mean()
consumption.absences.max()
consumption.absences.sort_values(ascending=False).iloc[0]
index = consumption.absences.idxmax()

consumption.loc[index].Dalc
consumption.groupby('Dalc').health.mean()
consumption.groupby('goout').Walc.mean().plot()
consumption.groupby('age').studytime.mean().plot.bar()
consumption.groupby('sex').Walc.mean().plot.pie()