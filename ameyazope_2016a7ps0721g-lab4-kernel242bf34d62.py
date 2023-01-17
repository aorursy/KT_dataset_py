import numpy as np

import pandas as pd



np.random.seed(12)



from PIL import Image

from numpy import asarray

import matplotlib.pyplot as plt

from matplotlib import image

%matplotlib inline



from skimage import io



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



import warnings

warnings.filterwarnings('ignore')
trainNpy = np.load('/kaggle/input/eval-lab-4-f464-v2/train.npy',allow_pickle=True)

testNpy = np.load('/kaggle/input/eval-lab-4-f464-v2/test.npy',allow_pickle=True)
from skimage import color

from tqdm import tqdm_notebook as tqdm

from skimage.color import separate_stains, hdx_from_rgb



trainX = list(list())

trainY = list()



for i in tqdm(range(len(trainNpy))):

    trainX.append(color.separate_stains(trainNpy[i][1],hdx_from_rgb).flatten())

    trainY.append(trainNpy[i][0])

    

    

testX = list(list())

for i in tqdm(range(len(testNpy))):

    testX.append(color.separate_stains(testNpy[i][1],hdx_from_rgb).flatten())
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(trainX,trainY,test_size=0.3,stratify=trainY,random_state=36)#random_state=576 gives 0.526 on public leaderboard

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0,C=2,n_jobs=3)

clf.fit(np.array(trainX),trainY)

y_pred = clf.predict(x_test)

print(f1_score(y_test,y_pred,average='micro'))
y_ans = clf.predict(testX)

idw = np.load('test.npy',allow_pickle=True)

idq = np.array([],dtype=int)



for i in range(len(idw)):

    idq = np.append(idq,int(idw[i][0]))



data_fin = pd.DataFrame(idq,columns=['ImageId'])

data_fin['Celebrity'] = y_ans

data_fin.to_csv('submission2 with seperate stains and full dataset as train.csv',index=False)
from sklearn.metrics import confusion_matrix

from sklearn import metrics



print(metrics.classification_report(y_test,y_pred))
import numpy as np

import pandas as pd



np.random.seed(12)



from PIL import Image

from numpy import asarray

import matplotlib.pyplot as plt

from matplotlib import image

%matplotlib inline



from skimage import io



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



import warnings

warnings.filterwarnings('ignore')
trainNpy = np.load('/kaggle/input/eval-lab-4-f464-v2/train.npy',allow_pickle=True)

testNpy = np.load('/kaggle/input/eval-lab-4-f464-v2/test.npy',allow_pickle=True)
trainNpy = np.load('train.npy',allow_pickle=True)

testNpy = np.load('test.npy',allow_pickle=True)



from skimage import color

from tqdm import tqdm_notebook as tqdm

from skimage.color import separate_stains, hdx_from_rgb



trainX = list(list())

trainY = list()



for i in tqdm(range(len(trainNpy))):

    trainX.append(color.separate_stains(trainNpy[i][1],hdx_from_rgb).flatten())

    trainY.append(trainNpy[i][0])

    

    

testX = list(list())

for i in tqdm(range(len(testNpy))):

    testX.append(color.separate_stains(testNpy[i][1],hdx_from_rgb).flatten())

    

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(trainX,trainY,test_size=0.3,stratify=trainY,random_state=36)#random_state=576 gives 0.526 on public leaderboard





from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0,C=2)

clf.fit(np.array(x_train),y_train)

y_predSS = clf.predict(x_test)



y_ansSS = clf.predict(testX)

trainNpy = np.load('train.npy',allow_pickle=True)

testNpy = np.load('test.npy',allow_pickle=True)



from skimage import color

from tqdm import tqdm_notebook as tqdm

from skimage.color import separate_stains, hdx_from_rgb



trainX = list(list())

trainY = list()



for i in tqdm(range(len(trainNpy))):

    trainX.append(color.separate_stains(trainNpy[i][1],hdx_from_rgb).flatten())

    trainY.append(trainNpy[i][0])

    

    

testX = list(list())

for i in tqdm(range(len(testNpy))):

    testX.append(color.separate_stains(testNpy[i][1],hdx_from_rgb).flatten())

    

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(trainX,trainY,test_size=0.3,stratify=trainY,random_state=36)#random_state=576 gives 0.526 on public leaderboard



from sklearn.ensemble import ExtraTreesClassifier



clf = ExtraTreesClassifier(random_state=1477,n_estimators=3000)

clf.fit(np.array(x_train),y_train)

y_predET = clf.predict(x_test)



y_ansET = clf.predict(testX)

trainNpy = np.load('train.npy',allow_pickle=True)

testNpy = np.load('test.npy',allow_pickle=True)



from skimage import color

from tqdm import tqdm_notebook as tqdm



trainX = list(list())

trainY = list()



for i in tqdm(range(len(trainNpy))):

    trainX.append(color.rgb2ydbdr(trainNpy[i][1]).flatten())

    trainY.append(trainNpy[i][0])

    

    

testX = list(list())

for i in tqdm(range(len(testNpy))):

    testX.append(color.rgb2ydbdr(testNpy[i][1]).flatten())

    

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(trainX,trainY,test_size=0.3,stratify=trainY,random_state=36)#random_state=576 gives 0.526 on public leaderboard



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0)

clf.fit(np.array(x_train),y_train)



y_predYDBDR = clf.predict(x_test)



y_ansYDBDR = clf.predict(testX)

y_ans = [0]*len(y_ansSS)

for i in range(len(y_ansSS)):

    y_ans[i] = y_ansSS[i]

    if y_ansSS[i]=='a_r__rahman' or y_ansSS[i]=='ajay_devgn' or y_ansSS[i]=='akshay_kumar' or y_ansSS[i]=='fawad_khan' or y_ansSS[i]=='irrfan_khan' or y_ansSS[i]=='kapil_sharma' or        y_ansSS[i]=='mahendra_singh_dhoni' or y_ansSS[i]=='naseeruddin_shah' or y_ansSS[i]=='preity_zinta' or y_ansSS[i]=='sonam_kapoor' or y_ansSS[i]=='vidya_balan' :

        y_ans[i] = y_ansSS[i]

    elif y_ansET[i]=='virat_kohli':

        y_ans[i] = y_ansET[i]

    elif y_ansYDBDR[i]=='aamir_khan' or y_ansYDBDR[i]=='kangana_ranaut' or y_ansYDBDR[i]=='kriti_sanon' or y_ansYDBDR[i]=='ranbir_kapoor' or y_ansYDBDR[i]=='saif_ali_khan' or y_ansYDBDR[i]=='shahid_kapoor' or y_ansYDBDR[i]=='shraddha_kapoor':

        y_ans[i] = y_ansYDBDR[i]

        
idw = np.load('test.npy',allow_pickle=True)

idq = np.array([],dtype=int)



for i in range(len(idw)):

    idq = np.append(idq,int(idw[i][0]))



data_fin = pd.DataFrame(idq,columns=['ImageId'])

data_fin['Celebrity'] = y_ans

data_fin.to_csv('submission2 with Stacking.csv',index=False)
from sklearn.metrics import confusion_matrix

from sklearn import metrics



print(metrics.classification_report(y_test,y_pred))