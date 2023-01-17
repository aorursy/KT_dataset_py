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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
def dealing(df):

    df['hour'] = df.datetime.dt.hour + 24*(df.datetime.dt.day - 1)

    df['sms'] = df['smsin'] + df['smsout']

    df['call'] = df['callin'] + df['callout']

    return df[['hour', 'CellID', 'sms', 'call', 'internet']]
fileList = ['sms-call-internet-mi-2013-11-0{}.csv'.format(x+1) for x in range(7)]

fileList
totalSelectedGroup = pd.DataFrame({})

for file in fileList:

    chunks_df = pd.read_csv('../input/'+file, parse_dates=['datetime'], chunksize=1000)

    chunks_df_piece = [dealing(chunk) for chunk in chunks_df]

    chunks_groupby = [piece.groupby(['hour', 'CellID'], as_index = False).sum() for piece in chunks_df_piece]

    total_groupby = pd.concat(chunks_groupby).groupby(['hour', 'CellID'], as_index = False).sum()

    totalSelectedGroup = totalSelectedGroup.append(total_groupby)

totalSelectedGroup.sort_values(by=['hour', 'CellID'], inplace=True)

totalSelectedGroup.head()