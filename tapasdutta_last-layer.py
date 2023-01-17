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

from sklearn.metrics import confusion_matrix



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ans_1=np.load('/kaggle/input/answers_last_fold1.npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_fold1.npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold1.npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_best_fold1.npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold2.npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_fold2.npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold2.npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_best_fold2.npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold3 (1).npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_fold3 (1).npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold3 (1).npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_best_fold3 (1).npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold4.npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_fold4.npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold4.npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_best_fold4.npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold5.npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_fold5.npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
ans_1=np.load('/kaggle/input/answers_last_fold5.npy',allow_pickle=True).item()

fold_1_pre=np.load('/kaggle/input/predictions_last_best_fold5.npy',allow_pickle=True).item()

ans_1=list(ans_1.values())[0]

fold_1_pre=list(fold_1_pre.values())[0]

print(confusion_matrix(ans_1,fold_1_pre))
fnl_time1=list(np.load('/kaggle/input/times_last_fold1.npy',allow_pickle=True).item().values())[0]

fnl_time2=list(np.load('/kaggle/input/times_last_fold2.npy',allow_pickle=True).item().values())[0]

fnl_time3=list(np.load('/kaggle/input/times_last_fold3 (1).npy',allow_pickle=True).item().values())[0]

fnl_time4=list(np.load('/kaggle/input/times_last_fold4.npy',allow_pickle=True).item().values())[0]

fnl_time5=list(np.load('/kaggle/input/times_last_fold5.npy',allow_pickle=True).item().values())[0]

np.mean([fnl_time1,fnl_time2,fnl_time3,fnl_time4,fnl_time5])
fnl_acc3=list(np.load('/kaggle/input/final_accuracy_last_fold3 (1).npy',allow_pickle=True).item().values())[0]

fnl_acc5=list(np.load('/kaggle/input/final_accuracy_last_fold5.npy',allow_pickle=True).item().values())[0]

fnl_acc4=list(np.load('/kaggle/input/final_accuracy_last_fold4.npy',allow_pickle=True).item().values())[0]

fnl_acc2=list(np.load('/kaggle/input/final_accuracy_last_fold2.npy',allow_pickle=True).item().values())[0]

fnl_acc1=list(np.load('/kaggle/input/final_accuracy_last_fold1.npy',allow_pickle=True).item().values())[0]

np.mean([fnl_acc1,fnl_acc2,fnl_acc3,fnl_acc4,fnl_acc5])
fnl_acc3=list(np.load('/kaggle/input/best_accuracy_last_fold3 (1).npy',allow_pickle=True).item().values())[0]

fnl_acc5=list(np.load('/kaggle/input/best_accuracy_last_fold5.npy',allow_pickle=True).item().values())[0]

fnl_acc4=list(np.load('/kaggle/input/best_accuracy_last_fold4.npy',allow_pickle=True).item().values())[0]

fnl_acc2=list(np.load('/kaggle/input/best_accuracy_last_fold2.npy',allow_pickle=True).item().values())[0]

fnl_acc1=list(np.load('/kaggle/input/best_accuracy_last_fold1.npy',allow_pickle=True).item().values())[0]

np.mean([fnl_acc1,fnl_acc2,fnl_acc3,fnl_acc4,fnl_acc5])
history={}

history['fold1']=np.load('/kaggle/input/history_last_fold1.npy',allow_pickle=True).item()

history['fold2']=np.load('/kaggle/input/history_last_fold2.npy',allow_pickle=True).item()

history['fold3']=np.load('/kaggle/input/history_last_fold3 (1).npy',allow_pickle=True).item()

history['fold4']=np.load('/kaggle/input/history_last_fold4.npy',allow_pickle=True).item()

history['fold5']=np.load('/kaggle/input/history_last_fold5.npy',allow_pickle=True).item()

from matplotlib import pyplot as plt

for i in range(5):

    fold='fold'+str(i+1)

    fold1='fold_'+str(i+1)

    plt.plot(history[fold][fold1]['accuracy'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Training all layers')

    plt.show()
from matplotlib import pyplot as plt

for i in range(5):

    fold='fold'+str(i+1)

    fold1='fold_'+str(i+1)

    plt.plot(history[fold][fold1]['loss'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('Training all layers')

    plt.show()
from matplotlib import pyplot as plt

for i in range(5):

    fold='fold'+str(i+1)

    fold1='fold_'+str(i+1)

    plt.plot(history[fold][fold1]['val_loss'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('val_loss')

    plt.title('Training all layers')

    plt.show()
from matplotlib import pyplot as plt

for i in range(5):

    fold='fold'+str(i+1)

    fold1='fold_'+str(i+1)

    plt.plot(history[fold][fold1]['val_accuracy'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('val_accuracy')

    plt.title('Training all layers')

    plt.show()