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

df = pd.read_csv('../input/train.csv')


#print mean, stddev, range of data
for col in df[df>0]:
    print(df[col].describe()[['mean','std','min','max']])
    print("")

#print top 10 most frequently checked in places
print ("top 10 most frequently checked in places")
df.groupby('place_id')['row_id'].nunique().sort_values(ascending=False).head(10)
#print bottom 10 least frequently checked in places
print ("bottom 10 most frequently checked in places")
df.groupby('place_id')['row_id'].nunique().sort_values(ascending=False).tail(10)

df.loc[df['place_id'] == 8772469670]['time'].hist()
