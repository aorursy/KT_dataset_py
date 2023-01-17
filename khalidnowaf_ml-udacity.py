# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


store = pd.read_csv('../input/googleplaystore.csv')
review = pd.read_csv('../input/googleplaystore_user_reviews.csv')


Y = store['Rating']
data = store.drop('Rating', axis=1)

## simple cleaing 
data['Installs'] = data['Installs'].apply(lambda i: i.replace(',', '').replace('+',''))


data['Size'] = data['Size'].apply(lambda i: -1 if i == 'Varies with device' else i)
                                   
# data['Size'] = data['Size'].apply(lambda i: if 'M' in i:  float(i[:-1]) * 1024
#                                             elif 'k' in i:  float(i[:-1])
#                                             else : -1
#                                  )


data.head()

print(data['Size'])


review.head(n=10)

# How Many NaNs ?? 
# TODO: Clean Data
## Clean 
## how many row 
## TODO: 
    ## parse rating 
    ## Parse Install
    ## Parse Size
    ## how many Category --> Genres
    


    

