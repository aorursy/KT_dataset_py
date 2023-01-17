print## This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load in the data and show the number of rows and columns:

dataset = pd.read_csv("../input/train.csv")

print(dataset.shape)
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
# 1 - View the dataframe headers:

print("1:")

print(dataset[0:1])



# 2 - Subset the dataframe to extract only the tree cover:

type = dataset['Cover_Type']



# 3 - Check that you have succeeded by printing the header of this:

print("2:")

print(dataset.Cover_Type[0:4])

print(type[0:4])
#type[1].value_counts()



#df = pd.Dataframe({'a':list('abssbab')})

#df.groupby('a').count()



x = dataset.Cover_Type.apply(pd.value_counts)



print(x[0:4])
# print(x)

# print(x.dtypes)

print(x.sum(axis=0))
dataset = pd.read_csv("../input/test.csv")
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