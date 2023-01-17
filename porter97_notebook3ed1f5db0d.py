# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df=pd.read_csv("../input/xAPI-Edu-Data.csv")



#lets start out slow by looking at the variables we have

df.head()









# Any results you write to the current directory are saved as output.
## now we can look a little bit more at the distributions in histogram format to check out for

## interesting relationships

df.hist()

##most of these are fairly straightforward, only a few people actively participate in discussion, 

##same with viewed announcements, but the bimodal distribution of raised hands is interesting, so 

##lets explore it further

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

df=pd.read_csv("../input/xAPI-Edu-Data.csv")

ax = sns.boxplot(x = df.PlaceofBirth, y = df.raisedhands, data = df)

plt.show









import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

df=pd.read_csv("../input/xAPI-Edu-Data.csv")

ax = sns.barplot(x = df.PlaceofBirth, y = df.raisedhands, data = df)

plt.show

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

df=pd.read_csv("../input/xAPI-Edu-Data.csv")

df.groupby('StageID').hist()