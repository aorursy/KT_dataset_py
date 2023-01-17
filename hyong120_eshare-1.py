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
train = pd.read_csv('../input/train_1.csv')
train.describe()
train.head()
values = train.values

values[0:10]
value = values[:,1:]

value[0:10]
columns = train.columns

column = columns[1:]

column[:10]
index = values[:,0:1]

index[:10]
train_new = pd.DataFrame(value,index=index,columns=column)
train_new.head()
np.save('train',train_new)
import matplotlib.pyplot as plt

plt.style.use('ggplot')
first = train_new[0:1]

first
first_tem = first.T

first_tem.plot()

plt.show()
mean = describe.mean()
plt.figure(figsize=(80,60))

mean.plot()

plt.show()
train_new.fillna(value=mean)
mean1 = train_new.mean(1)
mean1.plot()

plt.show()