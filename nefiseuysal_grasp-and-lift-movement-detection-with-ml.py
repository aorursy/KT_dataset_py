import zipfile
with zipfile.ZipFile("../input/grasp-and-lift-eeg-detection/test.zip","r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/grasp-and-lift-eeg-detection/train.zip","r") as z:
    z.extractall(".")
import pandas as pd
import matplotlib.pyplot as plt
train_set_labels = pd.read_csv("train/subj1_series1_events.csv")
train_set_signals = pd.read_csv("train/subj1_series1_data.csv")
train_set_signals.head()
axis = plt.gca()
downSampleToShow = 500
train_set_signals[::downSampleToShow].plot(x="id", y="Fp1", ax=axis)
train_set_signals[::downSampleToShow].plot(x="id", y="PO10", ax=axis, figsize=(15,5))
train_set_labels[::downSampleToShow].plot(figsize=(15,5))
plt.show()
eeg_channels = train_set_signals.columns.drop('id')
labels = train_set_labels.columns.drop('id')
train_set_complete = pd.concat([train_set_signals,train_set_labels], axis=1)
train_set_complete.insert(0, "order", range(0, len(train_set_complete)))
train_set_complete.head()
def highlight(indices,ax,color):
    i=0
    while i<len(indices):
        ax.axvspan(indices[i]-0.5, indices[i]+0.5, facecolor=color, edgecolor='none', alpha=.35)
        i+=1
secondsToShow = 8
channelsToShow = 3
labelsToShow = 6

sample_set = train_set_complete[train_set_complete["order"] < secondsToShow*500].drop("id", axis=1).set_index("order") #sample rate is 500hz 
colors=["red","purple","black","green", "yellow", "blue"]
axes = sample_set.plot(y=eeg_channels[:channelsToShow],subplots=True, figsize=(15,10))
for i in range(0, len(labels)):
    print(labels[i], "=", colors[i])
    
for axis in axes:    
    colorindex = 0
    for label in labels[:labelsToShow]:
        highlight(sample_set[sample_set[label]==1].index, axis, colors[colorindex])        
        colorindex = colorindex + 1
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

sub1_events_file = 'train/subj1_series1_events.csv'
sub1_data_file = 'train/subj1_series1_data.csv'

sub1_events = pd.read_csv(sub1_events_file)
sub1_data = pd.read_csv(sub1_data_file)

sub1 = pd.concat([sub1_events, sub1_data], axis = 1)
sub1["time"] = range(0, len(sub1))

sample_sub1 = sub1[sub1["time"] < 5000]

event = "HandStart"
event1 = "FirstDigitTouch"
EventColors = ["lightgrey", "green","blue"]

plot_columns = ["O1", "O2", "C3", "C4"]

fig, axes = plt.subplots(nrows=len(plot_columns), ncols=1)
fig.suptitle(event)
for (i, y) in enumerate(plot_columns):
    # Plot all the columns
    sample_sub1.plot(kind="scatter", x="time", y=y, edgecolors='none', ax=axes[i], figsize=(10,8), c=sample_sub1[event].apply(EventColors.__getitem__))
   

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from glob import glob
import scipy
from scipy.signal import butter, lfilter, convolve, boxcar
from scipy.signal import freqz
from scipy.fftpack import fft, ifft
import os

from sklearn.preprocessing import StandardScaler



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
def data_preprocess_train(X):
    X_prep=scaler.fit_transform(X)
    #do here your preprocessing
    return X_prep
def data_preprocess_test(X):
    X_prep=scaler.transform(X)
    #do here your preprocessing
    return X_prep
subjects = range(1,4)
from glob import glob
import pandas as pd
ids_tot = []
pred_tot = []
X_train_butter = []
from sklearn.model_selection import train_test_split
import numpy as  np

###loop on subjects and 8 series for train data + 2 series for test data
y_raw= []
raw = []
y_rawt= []
rawt = []
for subject in subjects:
    
    ################ READ DATA ################################################
    fnames =  sorted(glob('train/subj%d_series*_data.csv' % (subject)))[:6]


#    fnames =  glob('../input/train/subj1_series1_events.csv')
#    fnames =  glob('../input/train/subj1_series1_data.csv')
    for fname in fnames:
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)

    for fname in fnames:
      with open(fname) as myfile:
        head = [next(myfile) for x in range(10)]
        
        ################ READ TEST DATA ################################################
    tnames =  sorted(glob('train/subj%d_series*_data.csv' % (subject)))


#    fnames =  glob('../input/train/subj1_series1_events.csv')
#    fnames =  glob('../input/train/subj1_series1_data.csv')
    for fname in tnames:
      datat,labelst=prepare_data_train(fname)
      rawt.append(datat)
      y_rawt.append(labelst)

    for fname in tnames:
      with open(fname) as myfile:
        head = [next(myfile) for x in range(10)]
      
        
X = pd.concat(raw)
y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
X_train =np.asarray(X.astype(float))
ytrain = np.asarray(y.astype(float))



from sklearn.preprocessing import StandardScaler,Normalizer
scaler= StandardScaler()
def data_preprocess_train(X):
    X_prep=scaler.fit_transform(X)
    #do here your preprocessing
    return X_prep
fs = 500.0
lowcut = 7.0
highcut = 30.0

x_train_butter = []
for i in range(0,32):
    x_train_butter.append( butter_bandpass_filter(X_train[:,i], lowcut, highcut, fs, order=6))
x_train_butter=np.array(x_train_butter).T
xtrain=data_preprocess_train(x_train_butter)
splitrate=xtrain.shape[0]//5
xval=xtrain[:splitrate]
yval=ytrain[:splitrate]
xtrain=xtrain[splitrate:]
ytrain=ytrain[splitrate:]
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(X_train[:10000,1], label='signal', color="b", alpha=0.5,)
ax.plot(x_train_butter[:10000,1], label='reconstructed signal',color="k")
ax.legend(loc='upper left')
ax.set_title('Denoising with Butterworth')
plt.show()

from sklearn.model_selection import train_test_split
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from glob import glob
import scipy
from scipy.signal import butter, lfilter, convolve, boxcar
from scipy.signal import freqz
from scipy.fftpack import fft, ifft
import os

from sklearn.preprocessing import StandardScaler

def wavelet_denoising(x, wavelet='db2', level=3):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)
#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data

def data_preprocess_train(X):
    X_prep=scaler.fit_transform(X)
    #do here your preprocessing
    return X_prep
def data_preprocess_test(X):
    X_prep=scaler.transform(X)
    #do here your preprocessing
    return X_prep
subjects = range(1,6)
from glob import glob
import pandas as pd
ids_tot = []
pred_tot = []
X_train_butter = []
from sklearn.model_selection import train_test_split
import numpy as  np

###loop on subjects and 8 series for train data + 2 series for test data
y_raw= []
raw = []
y_rawt= []
rawt = []
for subject in subjects:
    
    ################ READ DATA ################################################
    fnames =  sorted(glob('train/subj%d_series*_data.csv' % (subject)))#[:6]


#    fnames =  glob('../input/train/subj1_series1_events.csv')
#    fnames =  glob('../input/train/subj1_series1_data.csv')
    for fname in fnames:
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)

    for fname in fnames:
      with open(fname) as myfile:
        head = [next(myfile) for x in range(10)]
        

      
        
X = pd.concat(raw)
y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
X_train =np.asarray(X.astype(float))
y_train = np.asarray(y.astype(float))



from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
scaler= StandardScaler()
def data_preprocess_train(X):
    X_prep=scaler.fit_transform(X)
    #do here your preprocessing
    return X_prep

x_train_butter=wavelet_denoising(X_train)
x_train=data_preprocess_train(x_train_butter)
splitrate=-x_train.shape[0]//5*2
xval=x_train[splitrate:splitrate//2]
yval=y_train[splitrate:splitrate//2]
xtest=x_train[splitrate//2:]
ytest=y_train[splitrate//2:]
xtrain=x_train[:splitrate]
ytrain=y_train[:splitrate]

"""
#for  ml algoritm
x_train_butter=wavelet_denoising(X_train)
x_train=data_preprocess_train(x_train_butter)
splitrate=x_train.shape[0]//5
xval=x_train[:splitrate]
yval=y_train[:splitrate]
xtrain=x_train[splitrate:]
ytrain=y_train[splitrate:]"""
import pywt
import pandas as pd
import numpy as np
def wavelet_denoising(x, wavelet='db2', level=3):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

signal=pd.read_csv('train/subj1_series1_data.csv')
signal = signal.drop("id", axis=1)
filtered = wavelet_denoising(signal, wavelet='db2', level=3)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(signal.iloc[:10000,1], label='signal', color="b", alpha=0.5,)
ax.plot(filtered[:10000,1], label='reconstructed signal',color="k")
ax.legend(loc='upper left')
ax.set_title('Denoising with DWT')
plt.show()
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score
lr = SGDClassifier()
#clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf =OneVsRestClassifier(lr)
import time
start=time.time()
clf.fit(xtrain,ytrain)

print('training time taken: ',round(time.time()-start,0),'seconds')

from sklearn.metrics import roc_curve, auc
y_score = clf.decision_function(xval)
#y_score = clf.predict(xval)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(yval[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(6):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import label_ranking_average_precision_score
predictions = clf.predict(xval)
# accuracy
print("roc_auc:",sum(roc_auc.values())/6)
print("Accuracy = ",accuracy_score(yval,predictions))
print("Hamming Loss = ",hamming_loss(yval,predictions))
print("label_ranking_average_precision_score",label_ranking_average_precision_score(yval,predictions))
from sklearn.metrics import multilabel_confusion_matrix
cm=multilabel_confusion_matrix(yval, predictions)
print(cm)
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm[0])
plt.show()
from skmultilearn.problem_transform import ClassifierChain
import time
from sklearn.naive_bayes import GaussianNB

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
cls = ClassifierChain(GaussianNB())

# train
start=time.time()
cls.fit(xtrain,ytrain)

print('training time taken: ',round(time.time()-start,0),'seconds')
from sklearn.metrics import roc_curve, auc
y_score = cls.predict(xval)
ycls=y_score.toarray()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(yval[:, i], ycls[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class

for i in range(6):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import label_ranking_average_precision_score

predictionc = cls.predict(xval)
print("roc_auc:",sum(roc_auc.values())/6)
print("Accuracy = ",accuracy_score(yval,predictionc))
print("Hamming Loss = ",hamming_loss(yval,predictionc))
print("label_ranking_average_precision_score",label_ranking_average_precision_score(yval,predictionc.toarray()))
from sklearn.metrics import multilabel_confusion_matrix
cm=multilabel_confusion_matrix(yval, predictionc)
print(cm)
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm[1])
plt.show()
import time
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
start=time.time()
classifier.fit(xtrain,ytrain)

print('training time taken: ',round(time.time()-start,0),'seconds')


from sklearn.metrics import roc_curve, auc
y_score = classifier.predict(xval)
ycls=y_score.toarray()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(yval[:, i], ycls[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(6):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
# predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import label_ranking_average_precision_score

prediction = classifier.predict(xval)

print("roc_auc:",sum(roc_auc.values())/6)
print("Accuracy = ",accuracy_score(yval,prediction))
print("Hamming Loss = ",hamming_loss(yval,prediction))
print("label_ranking_average_precision_score",label_ranking_average_precision_score(yval,prediction.toarray()))


from sklearn.metrics import multilabel_confusion_matrix
cm=multilabel_confusion_matrix(yval, prediction)
print(cm)
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm[1])
plt.show()