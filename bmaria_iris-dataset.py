# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



dataset = pd.read_csv('../input/Iris.csv',index_col='Id')



X=pd.DataFrame(dataset, columns =["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])

dataset['Species']=dataset['Species'].astype('category')

y=pd.Categorical(dataset['Species']).codes



X=X.as_matrix()
for t,marker,c in zip(range(3),"<ox","rgb"):

    # We plot each class on its own to get different colored markers

    plt.scatter(X[y == t,0], X[y == t,1], marker=marker, c=c)