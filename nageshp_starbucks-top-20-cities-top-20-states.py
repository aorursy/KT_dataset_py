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
fileName = '../input/directory.csv'

starbucksDF = pd.read_csv(fileName);
starbucksStoreByState = starbucksDF.groupby(['Country', 'State/Province']).count().reset_index().iloc[:,range(3)]

starbucksStoreByState["Country-State"] = starbucksStoreByState["Country"]+ "-" + starbucksStoreByState["State/Province"]

starbucksStoreByState.rename(columns={'Brand': 'StoreCount'}, inplace=True)

starbucksStoreByState.columns
starbucksStoreByStateOrderedTop20 = starbucksStoreByState.sort_values(['StoreCount'], ascending=[0]).head(20)
import matplotlib.pyplot as plt

import numpy as np



# The X axis can just be numbered 0,1,2,3...

x = np.arange(len(starbucksStoreByStateOrderedTop20["Country-State"]))



plt.bar(x, starbucksStoreByStateOrderedTop20["StoreCount"])

plt.xticks(x, starbucksStoreByStateOrderedTop20["Country-State"], rotation=90)



plt.figure(figsize=(2000,1000))
# California state has highest number

# England on #3
starbucksStoreByCityOrderedTop20 = starbucksDF.groupby(['Country', 'State/Province', 'City']).count().reset_index().sort_values(['Brand'], ascending=[0]).iloc[:,range(4)].head(20)

starbucksStoreByCityOrderedTop20["Country-State-City"] = starbucksStoreByCityOrderedTop20["Country"]+ "-" + starbucksStoreByCityOrderedTop20["State/Province"]+ "-" + starbucksStoreByCityOrderedTop20["City"]

starbucksStoreByCityOrderedTop20.rename(columns={'Brand': 'StoreCount'}, inplace=True)
import matplotlib.pyplot as plt

import numpy as np



# The X axis can just be numbered 0,1,2,3...

x = np.arange(len(starbucksStoreByCityOrderedTop20["Country-State-City"]))



plt.bar(x, starbucksStoreByCityOrderedTop20["StoreCount"])

plt.xticks(x, starbucksStoreByCityOrderedTop20["Country-State-City"], rotation=90)



plt.figure(figsize=(2,1))
# Interesting - Seoul city has more starbucks stores than New York city