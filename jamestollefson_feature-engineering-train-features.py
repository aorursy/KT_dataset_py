# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.options.mode.chained_assignment = None



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_json('../input/train.json')

train = train.iloc[:10, :]



def newfeat(name, df, series):

    """Create a Series for my feature building loop to fill"""

    feature = pd.Series(0, df.index, name=name)

    """Now populate the new Series with numeric values"""

    for row, word in enumerate(series):

        if name in word:

            feature.iloc[row] = 1

    df[name] = feature

    return(df)

   

train = newfeat('Elevator', train, train.features)

train = newfeat('Dogs Allowed', train, train.features)

train = newfeat('Cats Allowed', train, train.features)



print(train)
train['pet_friendly'] = train['Cats Allowed'] + train['Dogs Allowed']

print(train['pet_friendly'])