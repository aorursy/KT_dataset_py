# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")

def getLabelAndImageFromRow(row):
    return row[0], np.reshape(row[1:], (28, 28))

def drawData(row):
    label, image = getLabelAndImageFromRow(row)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
    
def drawDataList(dataList):
    ITEMS_EACH_ROW = 4
    size = len(dataList)
    num_of_row = np.ceil(float(size)/ITEMS_EACH_ROW)
    
    for index, row in enumerate(dataList):
        plt.subplot(num_of_row, ITEMS_EACH_ROW, index + 1)
        drawData(row)
        
drawDataList(trainData.values[:8])
from sklearn.svm import SVC
clf = SVC(gamma=0.001)
clf.fit(trainData.values[:100, 1:], trainData.values[:100, 0])
