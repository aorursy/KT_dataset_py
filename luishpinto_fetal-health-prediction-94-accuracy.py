# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt



import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'

matplotlib.rcParams['font.family'] = 'sans-serif'

matplotlib.rcParams['font.size'] = 12
# preprocessing libraries

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



# model selection libraries

from sklearn.model_selection import train_test_split



# machine learning libraries

from sklearn.ensemble import RandomForestClassifier



# postprocessing and checking-results libraries

from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
def plotConfusionMatrix(dtrue,dpred,classes,title = 'Confusion Matrix',\

                        width = 0.75,cmap = plt.cm.Blues):

  

    cm = confusion_matrix(dtrue,dpred)

    cm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]



    fig,ax = plt.subplots(figsize = (np.shape(classes)[0] * width,\

                                       np.shape(classes)[0] * width))

    im = ax.imshow(cm,interpolation = 'nearest',cmap = cmap)



    ax.set(xticks = np.arange(cm.shape[1]),

           yticks = np.arange(cm.shape[0]),

           xticklabels = classes,

           yticklabels = classes,

           title = title,

           aspect = 'equal')

    

    ax.set_ylabel('True',labelpad = 20)

    ax.set_xlabel('Predicted',labelpad = 20)



    plt.setp(ax.get_xticklabels(),rotation = 90,ha = 'right',

             va = 'center',rotation_mode = 'anchor')



    fmt = '.2f'



    thresh = cm.max() / 2.0



    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j,i,format(cm[i,j],fmt),ha = 'center',va = 'center',

                    color = 'white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()

    plt.show()
df = pd.read_csv('../input/fetal-health-classification/fetal_health.csv')

df.drop_duplicates(inplace = True)



y = LabelEncoder().fit_transform(df['fetal_health'])

X = df.drop(columns = ['fetal_health'],axis = 1)
count = np.zeros(3)

for i in range(3):

    count[i] = np.where(y == i)[0].size

    

plt.subplots(figsize = (6.0,6.0))

plt.bar(np.arange(3),count,color = 'orange',edgecolor = 'black')

plt.xticks(np.arange(3),('N','S','P'))

plt.xlabel('Fetal State')

plt.ylabel('Number of Instances')

plt.show()
scaler = StandardScaler().fit(X)

Xnorm = scaler.transform(X)
Xtrain,Xtest,ytrain,ytest = train_test_split(Xnorm,y,test_size = 0.30,stratify = y,shuffle = True,random_state = 21)
Xtrain,ytrain = RandomOverSampler(random_state = 21).fit_resample(Xtrain,ytrain)
clf = RandomForestClassifier(random_state = 21).fit(Xtrain,ytrain)
ypred = clf.predict(Xtest)
print(classification_report(ytest,ypred))
plotConfusionMatrix(ytest,ypred,classes = np.array(['N','S','P']),width = 1.5,cmap = plt.cm.binary)