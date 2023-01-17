# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris_data = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

iris_data.head()
iris_data['species'].value_counts()
label_encoding = {'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}
iris_data['species'] = iris_data['species'].map(label_encoding)
iris_data.head()
groups = iris_data.groupby('species')
group_0 = groups.get_group(0)

group_1 = groups.get_group(1)

group_2 = groups.get_group(2)
def get_probability(x,df):

    cols=df.columns

    for i in range(len(x)):

        count=0.0

        for j in df[cols[i]]:

            if j==x[i]:

                count+=1

        prob=count/len(df['sepal_length'])

        x[i]=prob

    z=1.0

    for i in x:

        z=z*i

    return z
new_example = [5.0, 3.1, 1.4, 0.5]



y_1=get_probability(new_example,group_0)

y_2=get_probability(new_example,group_1)

y_3=get_probability(new_example,group_2)



print(y_1,y_2,y_3)