#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

# births.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/python-data-science-handbook/notebooks/data/births.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'births.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
dfcorr=df.corr()

dfcorr
sns.heatmap(dfcorr,annot=True,cmap='winter_r')

plt.show()
fig=sns.lmplot(x="births", y="year",data=df)
df.plot.area(y=['year','births'],alpha=0.4,figsize=(12, 6));
sns.factorplot('births','month',hue='year',data=df)

plt.show()
pd.crosstab([df.year],df.births).style.background_gradient(cmap='summer_r')
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.01-classification-3.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.08-decision-tree-levels.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.01-clustering-2.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.03-bias-variance.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.01-dimesionality-2.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.09-PCA-rotation.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.08-decision-tree-overfitting.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.05-gaussian-NB.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/Data_Science_VD.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/03.08-split-apply-combine.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.02-samples-features.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/PDSH-cover.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.09-digits-pixel-components.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/array_vs_list.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.01-regression-1.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.12-covariance-type.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.11-expectation-maximization.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.06-gaussian-basis.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.03-5-fold-CV.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/02.05-broadcasting.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/cint_vs_pyint.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.09-digits-pca-components.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures//05.03-learning-curve.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.10-LLE-vs-MDS.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/python-data-science-handbook/notebooks/figures/05.03-validation-curve.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)