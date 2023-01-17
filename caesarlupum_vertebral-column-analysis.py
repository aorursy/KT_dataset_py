import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix , classification_report

from sklearn.cluster import KMeans

import sklearn.metrics as cma

from sklearn.model_selection import KFold

from sklearn import model_selection

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
df2 = pd.read_csv('../input/vertebralcolumndataset/column_3C.csv')

df2.describe()

df2.info()

df2['class'] = df2['class'].map({'Normal': 0, 'Hernia': 1, 'Spondylolisthesis': 2})
sns.pairplot(df2, hue="class", size=3, diag_kind="kde")
def euclidian(p1, p2): 

    dist = 0

    for i in range(len(p1)):

        dist = dist + np.square(p1[i]-p2[i])

    dist = np.sqrt(dist)

    return dist;



def manhattan(p1, p2): 

    dist = 0

    for i in range(len(p1)):

        dist = dist + abs(p1[i]-p2[i])

    return dist;

def dNN_2(X_train,y_train, X_test,dist='euclidian',q=2):

    pred = []

    if isinstance(X_test, np.ndarray):

        X_test=pd.DataFrame(X_test)

    if isinstance(X_train, np.ndarray):

        X_train=pd.DataFrame(X_train)

    vetMean = df2.reset_index().groupby( [ "class"],as_index=False ).agg({'pelvic_incidence': [np.mean],

                            'pelvic_tilt': [np.mean],

                            'lumbar_lordosis_angle': [np.mean],

                            'sacral_slope': [np.mean],

                            'pelvic_radius': [np.mean],

                            'degree_spondylolisthesis': [np.mean]

                          }, as_index=False )

    

    for i in range(len(X_test)):    

        # Calculando as distâncias para nosso test-point

        novadist = np.zeros(len(y_train))

        novadistc0 = np.zeros(len(y_train))

        novadistc1 = np.zeros(len(y_train))

        novadistc2 = np.zeros(len(y_train))

        

        if dist=='euclidian':

            for l in range(len(y_train)):

                for j1 in range(len(vetMean)):

                    novadistc0[l] = euclidian(vetMean.iloc[0,1:], X_test.iloc[i,:])

                    novadistc1[l] = euclidian(vetMean.iloc[1,1:], X_test.iloc[i,:])

                    novadistc2[l] = euclidian(vetMean.iloc[2,1:], X_test.iloc[i,:])

                    

                    novadist[l] = np.minimum(novadistc2[l]  , np.minimum( novadistc0[l],novadistc1[l]  )  )



            if novadistc0[l] <= novadistc1[l] and  novadistc0[l] <= novadistc2[l]:

                pred.append(0)

            elif novadistc1[l] <= novadistc0[l] and  novadistc1[l] <= novadistc2[l]:

                pred.append(1)

            else:

                novadistc2[l] <= novadistc0[l] and  novadistc2[l] <= novadistc1[l]

                pred.append(2)

        novadist = np.array([novadist, y_train])

    return pred
def kNN(X_train,y_train, X_test, k, dist='euclidian',q=2):

    pred = []

    # Testando o tipo de dado recebido

    if isinstance(X_test, np.ndarray):

        X_test=pd.DataFrame(X_test)

    if isinstance(X_train, np.ndarray):

        X_train=pd.DataFrame(X_train)

        

    for i in range(len(X_test)):    

        # Calculando as distancias para os pontos de teste

        novadist = np.zeros(len(y_train))



        if dist=='euclidian':

            for j in range(len(y_train)):

                novadist[j] = euclidian(X_train.iloc[j,:], X_test.iloc[i,:])

    

        novadist = np.array([novadist, y_train])



        ## Encontrando os k vizinhos mais próximos

        idx = np.argsort(novadist[0,:])



        # Sorteio de todos novadist

        novadist = novadist[:,idx]



        # Contando os labels vizinhos e pegando o label com max count

        # Define um dict para os counts

        c = {'0':0,'1':0,'2':0 }

        # Update counts no dict 

        for j in range(k):

            c[str(int(novadist[1,j]))] = c[str(int(novadist[1,j]))] + 1



        key_max = max(c.keys(), key=(lambda k: c[k]))

        pred.append(int(key_max))

    return pred
all_X = df2[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis']]

all_y = df2['class']





df2=df2[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis', 'class']]

train_data,test_data = train_test_split(df2,train_size = 0.8,random_state=2)

X_train = train_data[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis']]

y_train = train_data['class']

X_test = test_data[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis']]

y_test = test_data['class']



def transform2(i):

    if i == 0:

        return 'Normal'

    if i == 1:

        return 'Hernia'

    if i == 2:

        return 'Spondylolisthesis'
x = all_X.values

y = all_y.values



scores = []

cv = KFold(n_splits=5, random_state=42, shuffle=False)

for train_index, test_index in cv.split(x):

    X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

    best_svr =  dNN_2(X_train,y_train, X_test)

    cm = confusion_matrix(y_test, best_svr)

    scores.append(cm)        



print('Confusion Matrix ',scores)

print('\n')

a = cm.shape

corrPred = 0

falsePred = 0



for row in range(a[0]):

    for c in range(a[1]):

        if row == c:

            corrPred +=cm[row,c]

        else:

            falsePred += cm[row,c]

print('True pred: ', corrPred)

print('False pred', falsePred)  
x = all_X.values

y = all_y.values



scores = []

cv = KFold(n_splits=5, random_state=42, shuffle=False)

for train_index, test_index in cv.split(x):



    X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

    best_svr =  kNN(X_train,y_train, X_test, 5, dist='euclidian',q=5)

    cm = confusion_matrix(y_test, best_svr)

    scores.append(cm)        

print('Confusion Matrix ',scores)

print('\n')

a = cm.shape

corrPred = 0

falsePred = 0



for row in range(a[0]):

    for c in range(a[1]):

        if row == c:

            corrPred +=cm[row,c]

        else:

            falsePred += cm[row,c]

print('True pred: ', corrPred)

print('False pred', falsePred)  