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
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
y = train.label

y.shape
X = np.array(train)

X = X[:,1:]

X.shape
import matplotlib.pyplot as plt

def plot_image(image):

    image = image.reshape(28,28)

    plt.imshow(image,cmap='gray')
plot_image(X[5])
from sklearn.linear_model import LogisticRegression as logic
mod