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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("ggplot")

mine=pd.read_csv("../input/FullData.csv")

type(mine['Name'])

mine
#mine.plot(x="Age",y="Long_Shots",style="o")

sns.jointplot(mine["Long_Shots"],mine['Age'],kind='kde',alpha=0.7)
df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])

df.plot(width=0.25, align='center')

ax=df.plot.scatter(x='a',y='c',color="Red",label='Malaria',width=0.25, align='center')

df.plot.scatter(x='b',y='d',color="DarkGreen",label="HIV/AIDS",ax=ax)


