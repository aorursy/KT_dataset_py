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
import numpy as np

import pandas as pd

import sklearn as skl

import matplotlib as plt



dt = pd.read_json('../input/train.json')

dt.head(5)

import numpy as np

import pandas as pd

import sklearn as skl

import matplotlib as plt



dt = pd.read_json('../input/train.json')

dt.head(5)



import numpy as np

import pandas as pd

import sklearn as skl

import matplotlib as plt



dt = pd.read_json('../input/train.json')

#dt.head(5)

#print (dt['bathrooms'],dt['bedrooms'])



dt.drop(['building_id', 'manager_id', 'created', 'latitude', 'listing_id', 'longitude' ], axis = 1, level = None, inplace = True, errors = 'raise')

#dt.head(5)



#temp3 = pd.crosstab(dt['bedrooms'], dt['price'])

#temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

dt.apply(lambda x: sum(x.isnull()),axis=0) 
