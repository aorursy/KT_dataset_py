# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from ggplot import *

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")



a = len(df.id.unique())

b = len(df.timestamp.unique())



# Any results you write to the current directory are saved as output.
import kagglegym



# Create environment

env = kagglegym.make()

# Get first observation

observation = env.reset()

# Look at first few rows of the train dataframe

traindf = observation.train

traindf.head()
#Let's see how many different ids we have

traindf.id.unique().shape



#Now let's grab 1 instrument (id=10)

id10 = traindf[traindf.id == 10]

id10.shape

id10.head()
#We can see a few columns with no data at all

nans_per_column = id10.isnull().sum(axis=0)

empty_columns = nans_per_column[nans_per_column == 116].index

empty_columns
from ggplot import *

import pandas as pd

#Let's plot the derived values

id10derived = pd.melt(id10, id_vars=['id','timestamp'], value_vars=['derived_0','derived_1','derived_2','derived_3','derived_4'])

print( ggplot(id10derived, aes('timestamp','value',fill='variable')) + geom_line() )
import tensorflow as tf