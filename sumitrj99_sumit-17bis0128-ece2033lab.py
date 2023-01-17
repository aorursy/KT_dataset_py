import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets

import os
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
df['target'] = iris.target
df.head()
df.describe()[3:]
df.describe()[1:3]
df0 = df[df['target']==0]
df1 = df[df['target']==1]
df2 = df[df['target']==2]
sns.set_style("whitegrid");
sns.pairplot(df,hue="target",height=3);
print('The diagonal plot which showcases the histogram. The histogram allows us to see the PDF/Probability distribution of a single variable')
plt.show()
plt.plot(df['petal_len'])
def findflower(petal_length):
    f = 0
    if (petal_length < 1.9):
        f =  'Setosa'
    elif (petal_length > 3.2 and petal_length < 5):
        f =  'Versicolor'
    elif (petal_length > 5):
        f =  'Virginica'
    return f