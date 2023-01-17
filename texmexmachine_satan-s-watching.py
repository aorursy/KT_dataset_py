# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



df= pd.read_csv('../input/gifts.csv')



#df

df.GiftId.unique()

rawGifts = df.GiftId.apply(lambda x : x.split("_")[0])

print (type(rawGifts))

#print (rawGifts.column.values)

rawGifts.value_counts()