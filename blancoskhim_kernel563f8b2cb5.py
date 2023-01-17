# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
cancerDataSet=pd.read_csv('../input/wisconsin_breast_cancer.csv')

cancerDataSet.fillna(0,inplace=True) #removes non numbers

print(cancerDataSet['nuclei'])



cancerDataSet.head(10)
from sklearn.model_selection import train_test_split

x=cancerDataSet.iloc[:, 1:10]

y=cancerDataSet.iloc[:,10:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y,random_state=1296429)

print(x_train.shape,y_train.shape)
from sklearn.linear_model import SGDClassifier

my_classifier=SGDClassifier(random_state=1296429)



from sklearn.model_selection import cross_val_score

def findcolumns(alt, x_train):

    cols=list()

    

    if(alt&1==1):

      cols.append(x_train.columns[0])

    if(alt&2==2):

      cols.append(x_train.columns[1])

    if(alt&4==4):

      cols.append(x_train.columns[2])

    if(alt&8==8):

      cols.append(x_train.columns[3])

    if(alt&16==16):

      cols.append(x_train.columns[4])

    if(alt&32==32):

      cols.append(x_train.columns[5])

    if(alt&64==64):

      cols.append(x_train.columns[6])

    if(alt&128==128):

      cols.append(x_train.columns[7])

    if(alt&256==256):

      cols.append(x_train.columns[8])

    return cols

   

subsets=list()

for alt in range(1,512):

    subset = findcolumns(alt, x_train)

    print(alt, subset)

    subsets.append(subset)

  

print(len(subsets))
scores=list()  

maxscore=0

best_subset=0

for subset in subsets:

    cvscore=cross_val_score(my_classifier,x_train[subset],y_train,cv=10,scoring="accuracy")

    scores.append(np.mean(cvscore))

    if scores[-1]>maxscore:

        max_score=scores[-1]

        best_subset=subset

    print(subset, scores[-1])

    
print(scores)

print(max(scores))

print(maxscore)

print(bestsubset)

# the "thickness,shape,adhesion,nuclei,nucleoli,mitosis" subset looks the best given the cv scores.

# the accuracy of this subset is 0.9768762816131238
import matplotlib.pyplot as plt

colors=(0,0,0)

area=np.pi*3

plt.scatter(scores, y_test, s=area, c=colors, alpha=0.5)

plt.title('scatter-plot')

plt.xlabel('cross validation scores')

plt.ylabel('test results')

plt.show()