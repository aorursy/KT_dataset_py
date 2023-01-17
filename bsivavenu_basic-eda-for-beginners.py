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
data = pd.read_csv('../input/movies_metadata.csv')

# cast_crew = pd.read_csv('../input/credits.csv')

# links = pd.read_csv('../input/links.csv')

data.head(2)
data['adult'] = data['adult'].map({'False':0,'True':1})
data[data['adult']==1].head()
data.isnull().sum().sort_values(ascending = False)
data.select_dtypes(include =['number'] ).columns
data.select_dtypes(include =['object'] ).columns
data.genres