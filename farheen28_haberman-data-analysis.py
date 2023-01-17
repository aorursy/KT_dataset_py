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
#Given the dataset, we first the data and try to interpret it, and try to understand all the information we can.

haberman=pd.read_csv("../input/haberman.csv")

haberman.head()

#number of data-points and features.

print(haberman.shape)
#features in our dataset.

print(haberman.columns)
haberman.head(5)
import matplotlib.pyplot as plt

haberman.plot()

plt.show()
#histograms

haberman.hist()

plt.show()
haberman['status'].value_counts()
haberman.plot(kind='scatter', x='age', y='axil_nodes');

plt.show();
import seaborn as sns

#pairwise relationship between the different features of the dataset, age, year, axiliary, survived.

plt.close();

sns.pairplot(haberman, hue = 'status')

plt.show()

#3D scattered plot.

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



fig=plt.figure()

ax=fig.add_subplot(111, projection='3d')



x=haberman["age"]

y=haberman["operation_year"]

z=haberman["axil_nodes"]



ax.scatter(x,y,z,marker='o', c='r');



ax.set_xlabel('age')

ax.set_ylabel('operation_year')

ax.set_zlabel('')



plt.show()
import numpy as np

plt.plot(haberman['axil_nodes'], np.zeros_like(haberman['axil_nodes']), 'o')

plt.show()