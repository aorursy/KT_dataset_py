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
data = pd.read_table('../input/upc_corpus.csv', delimiter=',' , dtype={"ean": np.str , "name": np.str})

data.head()

# Dim => (1048575, 2)

print(data.shape)

# Column => Index(['ean', 'name'], dtype='object')

print(data.columns)

# ean :  

print(data.describe(include=['O']))

data = data[data.ean.str.len()==12]

#(415308, 2)

print(data.shape)

#                 ean    name

#count         415308  415284

#unique        414250  395056

#top     799439600164   shoes

#freq              14     312

print(data.describe(include=['O']))
# If no name : No interest

data = data[data.name.notnull()]

#(415284,2)

print(data.shape)

#                 ean    name

#count         415284    415284

#unique        414227    395056

#top     799439600164   shoes

#freq              14     312

print(data.describe(include=['O']))
dataTest = data[data.ean=='799439600164']

print(dataTest)
# See record with name='test'

# look for ignorecase

dataTest = data[data.name.str.count('[T|t][E|e][S|s][T|t]')>1]

print(dataTest.describe())

print(dataTest)

#

data = data[data.name.str.count('[T|t][E|e][S|s][T|t]')<2]
print(data)

print(data.describe(include=['O']))