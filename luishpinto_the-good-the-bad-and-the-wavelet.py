import numpy as np
import matplotlib.pyplot as plt



import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'

matplotlib.rcParams['font.family'] = 'sans-serif'

matplotlib.rcParams['font.size'] = 10
from scipy import signal
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.models import *

from keras.layers import *

from keras.utils import to_categorical
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
fscan = 200.0



rpm = 1650.0

f = rpm / 60.0

amp = 1.0



bpfo = 0.4 * 12 * rpm



print('Bearing RMP: {:.1f} rpm'.format(rpm))

print('Bearing natural frequencies and harmonics: {:.1f} Hz'.format(f))

print('Bearing acceleration amplitude: {:.1f} mm/s²'.format(amp))

print('Ball pass frequency outer: {:.1f} Hz'.format(bpfo))
t = np.linspace(0.0,0.1,int(fscan) + 1)
np.random.seed()



A = np.array([],dtype = float)

y = np.array([],dtype = int)



ampo = np.array([0.0,0.5,1.5,3.0])

sample = np.array([100,100,200,50])

rul = np.array([0,1,2,3])

rultxt = np.array(['long-life','medium-life','short-life','failure'])



n = sample.sum()



np.random.seed()



for i,j,k in zip(sample,ampo,rul):

  for u in range(i):

    ph = np.pi * (4.0 * np.random.rand(2) - 2.0)

    A = np.append(A,np.array([amp * np.sin(2.0 * np.pi * f * t + ph[0]) + \

                              j * np.sin(2.0 * np.pi * bpfo * t + ph[1])]) + \

                              0.5 * np.random.randn(t.shape[0]))

    y = np.append(y,k)



A = A.reshape(n,-1)
scaler = StandardScaler().fit(A)

Anorm = scaler.transform(A)
plt.subplots(figsize = (5.0,5.0))

plt.plot(t,Anorm[-1],'k-',lw = 0.75)

plt.xlabel('Time [s]')

plt.ylabel('Acceleration [mm/s²]')

plt.show()
N = 32



width = np.arange(1,N + 1)



cwt = np.array([],dtype = float)



for i in range(n):

  cwt = np.append(cwt,signal.cwt(Anorm[i],signal.ricker,width))



cwt = cwt.reshape(n,N,-1)
X = resize(cwt,(n,64,64))
plt.subplots(1,sample.shape[0],figsize = (3.0 * sample.shape[0],3.0))

j = 1

for i in (sample.cumsum() - int(0.5 * sample[0])):

  plt.subplot(1,sample.shape[0],j)

  plt.imshow(X[i],vmax = 3.0,vmin = -3.0,cmap = 'binary')

  plt.xticks([])

  plt.yticks([])

  plt.title('{} spectrum'.format(rultxt[j - 1]))

  j += 1



plt.tight_layout()

plt.show()
label = to_categorical(y)
Ztrain,Ztest,ytrain,ytest = train_test_split(X.reshape(n,-1),label,test_size = 0.25)

Ztrain = Ztrain.reshape(-1,64,64,1)

Ztest = Ztest.reshape(-1,64,64,1)
model = Sequential()



model.add(Conv2D(128,(8,8),activation = 'relu',input_shape = (64,64,1)))

model.add(MaxPooling2D((4,4)))

model.add(Dropout(0.50))

model.add(Conv2D(64,(8,8),activation = 'relu'))

model.add(MaxPooling2D((4,4)))

model.add(Dropout(0.50))

model.add(Flatten())

model.add(Dense(1024,activation = 'relu'))

model.add(Dropout(0.50))

model.add(Dense(4,activation = 'softmax'))
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
print(model.summary())
hist = model.fit(Ztrain,ytrain,validation_data = (Ztest,ytest),epochs = 100,verbose = 0)
plt.subplots(1,2,figsize = (9.0,4.5),sharex = True)

plt.subplot(1,2,1)

plt.plot(hist.epoch,hist.history['accuracy'],\

         color = 'black',lw = 2.50)

plt.title('1D-CNN Accuracy')

plt.xlabel('Epoch')

plt.subplot(1,2,2)

plt.plot(hist.epoch,hist.history['loss'],\

         color = 'black',lw = 2.50)

plt.title('1D-CNN Loss')

plt.xlabel('Epoch')

# plt.tight_layout()

plt.show()
print(model.evaluate(Ztest,ytest,verbose = 2))
print(classification_report(np.argmax(ytest,axis = -1),\

                            np.argmax(model.predict(Ztest),axis = -1)))
plotConfusionMatrix(np.argmax(ytest,axis = -1),np.argmax(model.predict(Ztest),axis = -1),\

                    ['long','medium','short','urgent'],

                    title = 'Confusion Matrix ( RECALL )',width = 1.5,cmap = plt.cm.Blues)

plt.show()