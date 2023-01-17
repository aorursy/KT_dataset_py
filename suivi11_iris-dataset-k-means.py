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
dat = pd.read_csv('../input/Iris.csv')
X = dat.drop('Species', axis=1)

y = dat['Species']



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=3).fit(X)
def category_to_label(category):

    if category == 2:

        return 'Iris-setosa'

    elif category == 1:

        return 'Iris-versicolor'

    elif category == 0:

        return 'Iris-virginica'

    else:

        raise ValueError('Category out of range')

        

y_pred = pd.Series([category_to_label(category) for category in kmeans.labels_])
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y, y_pred))

print(classification_report(y, y_pred))
