# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from pandas.tools.plotting import scatter_matrix

from pandas.plotting import autocorrelation_plot



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from mpl_toolkits.mplot3d import axes3d, Axes3D

import seaborn as sns



from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.svm import SVC

from sklearn.neighbors import NearestCentroid

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import metrics



from itertools import product

from sklearn import svm



import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
mmscaler = MinMaxScaler()

standard_scaler = StandardScaler()

robust_scaler = RobustScaler()



musicdata = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_train.csv')



#colab

#musicdata = pd.read_csv('/content/input2/data_train.csv')





plot_label =['tempo', 'beats', 'chroma_stft', 'rmse',

        'spectral_centroid', 'spectral_bandwidth', 'rolloff',

        'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',

        'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',

        'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',

        'mfcc20']



#더 간편한 조원 군의 코드를 참고했습니다.

data = musicdata.loc[:,:].values[:,1:-1]

label = musicdata.values[:,-1]



scale_data = scale(data)

mmscaler_data = mmscaler.fit_transform(data)

standard_data = standard_scaler.fit_transform(data)

robust_data = robust_scaler.fit_transform(data)



scale_df = pd.DataFrame(scale_data,columns=plot_label)

mm_df = pd.DataFrame(mmscaler_data,columns=plot_label)

stand_df = pd.DataFrame(standard_data,columns=plot_label)

robust_df = pd.DataFrame(robust_data,columns=plot_label)



fig, ax0 = plt.subplots(ncols=1,figsize=(10,10))

ax0.set_title("scale")

sns.kdeplot(scale_df['tempo'],ax=ax0)

sns.kdeplot(scale_df['beats'],ax=ax0)

sns.kdeplot(scale_df['chroma_stft'],ax=ax0)

sns.kdeplot(scale_df['rmse'],ax=ax0)

sns.kdeplot(scale_df['spectral_centroid'],ax=ax0)

sns.kdeplot(scale_df['spectral_bandwidth'],ax=ax0)

sns.kdeplot(scale_df['mfcc1'],ax=ax0)

sns.kdeplot(scale_df['mfcc2'],ax=ax0)

sns.kdeplot(scale_df['mfcc3'],ax=ax0)

sns.kdeplot(scale_df['mfcc4'],ax=ax0)

sns.kdeplot(scale_df['mfcc5'],ax=ax0)

sns.kdeplot(scale_df['mfcc6'],ax=ax0)

sns.kdeplot(scale_df['mfcc7'],ax=ax0)

sns.kdeplot(scale_df['mfcc8'],ax=ax0)

sns.kdeplot(scale_df['mfcc9'],ax=ax0)

sns.kdeplot(scale_df['mfcc10'],ax=ax0)

sns.kdeplot(scale_df['mfcc11'],ax=ax0)

sns.kdeplot(scale_df['mfcc12'],ax=ax0)

sns.kdeplot(scale_df['mfcc13'],ax=ax0)

sns.kdeplot(scale_df['mfcc14'],ax=ax0)

sns.kdeplot(scale_df['mfcc15'],ax=ax0)

sns.kdeplot(scale_df['mfcc16'],ax=ax0)

sns.kdeplot(scale_df['mfcc17'],ax=ax0)

sns.kdeplot(scale_df['mfcc18'],ax=ax0)

sns.kdeplot(scale_df['mfcc19'],ax=ax0)

sns.kdeplot(scale_df['mfcc20'],ax=ax0)

plt.show()



fig, ax1 = plt.subplots(ncols=1,figsize=(10,10))

ax1.set_title("MinMaxScaler")

sns.kdeplot(mm_df['tempo'],ax=ax1)

sns.kdeplot(mm_df['beats'],ax=ax1)

sns.kdeplot(mm_df['chroma_stft'],ax=ax1)

sns.kdeplot(mm_df['rmse'],ax=ax1)

sns.kdeplot(mm_df['spectral_centroid'],ax=ax1)

sns.kdeplot(mm_df['spectral_bandwidth'],ax=ax1)

sns.kdeplot(mm_df['mfcc1'],ax=ax1)

sns.kdeplot(mm_df['mfcc2'],ax=ax1)

sns.kdeplot(mm_df['mfcc3'],ax=ax1)

sns.kdeplot(mm_df['mfcc4'],ax=ax1)

sns.kdeplot(mm_df['mfcc5'],ax=ax1)

sns.kdeplot(mm_df['mfcc6'],ax=ax1)

sns.kdeplot(mm_df['mfcc7'],ax=ax1)

sns.kdeplot(mm_df['mfcc8'],ax=ax1)

sns.kdeplot(mm_df['mfcc9'],ax=ax1)

sns.kdeplot(mm_df['mfcc10'],ax=ax1)

sns.kdeplot(mm_df['mfcc11'],ax=ax1)

sns.kdeplot(mm_df['mfcc12'],ax=ax1)

sns.kdeplot(mm_df['mfcc13'],ax=ax1)

sns.kdeplot(mm_df['mfcc14'],ax=ax1)

sns.kdeplot(mm_df['mfcc15'],ax=ax1)

sns.kdeplot(mm_df['mfcc16'],ax=ax1)

sns.kdeplot(mm_df['mfcc17'],ax=ax1)

sns.kdeplot(mm_df['mfcc18'],ax=ax1)

sns.kdeplot(mm_df['mfcc19'],ax=ax1)

sns.kdeplot(mm_df['mfcc20'],ax=ax1)

plt.show()



fig, ax2 = plt.subplots(ncols=1,figsize=(10,10))

ax2.set_title("standard_scaler")

sns.kdeplot(stand_df['tempo'],ax=ax2)

sns.kdeplot(stand_df['beats'],ax=ax2)

sns.kdeplot(stand_df['chroma_stft'],ax=ax2)

sns.kdeplot(stand_df['rmse'],ax=ax2)

sns.kdeplot(stand_df['spectral_centroid'],ax=ax2)

sns.kdeplot(stand_df['spectral_bandwidth'],ax=ax2)

sns.kdeplot(stand_df['mfcc1'],ax=ax2)

sns.kdeplot(stand_df['mfcc2'],ax=ax2)

sns.kdeplot(stand_df['mfcc3'],ax=ax2)

sns.kdeplot(stand_df['mfcc4'],ax=ax2)

sns.kdeplot(stand_df['mfcc5'],ax=ax2)

sns.kdeplot(stand_df['mfcc6'],ax=ax2)

sns.kdeplot(stand_df['mfcc7'],ax=ax2)

sns.kdeplot(stand_df['mfcc8'],ax=ax2)

sns.kdeplot(stand_df['mfcc9'],ax=ax2)

sns.kdeplot(stand_df['mfcc10'],ax=ax2)

sns.kdeplot(stand_df['mfcc11'],ax=ax2)

sns.kdeplot(stand_df['mfcc12'],ax=ax2)

sns.kdeplot(stand_df['mfcc13'],ax=ax2)

sns.kdeplot(stand_df['mfcc14'],ax=ax2)

sns.kdeplot(stand_df['mfcc15'],ax=ax2)

sns.kdeplot(stand_df['mfcc16'],ax=ax2)

sns.kdeplot(stand_df['mfcc17'],ax=ax2)

sns.kdeplot(stand_df['mfcc18'],ax=ax2)

sns.kdeplot(stand_df['mfcc19'],ax=ax2)

sns.kdeplot(stand_df['mfcc20'],ax=ax2)

plt.show()



fig, ax3 = plt.subplots(ncols=1,figsize=(10,10))

ax3.set_title("robust_scaler")

sns.kdeplot(robust_df['tempo'],ax=ax3)

sns.kdeplot(robust_df['beats'],ax=ax3)

sns.kdeplot(robust_df['chroma_stft'],ax=ax3)

sns.kdeplot(robust_df['rmse'],ax=ax3)

sns.kdeplot(robust_df['spectral_centroid'],ax=ax3)

sns.kdeplot(robust_df['spectral_bandwidth'],ax=ax3)

sns.kdeplot(robust_df['mfcc1'],ax=ax3)

sns.kdeplot(robust_df['mfcc2'],ax=ax3)

sns.kdeplot(robust_df['mfcc3'],ax=ax3)

sns.kdeplot(robust_df['mfcc4'],ax=ax3)

sns.kdeplot(robust_df['mfcc5'],ax=ax3)

sns.kdeplot(robust_df['mfcc6'],ax=ax3)

sns.kdeplot(robust_df['mfcc7'],ax=ax3)

sns.kdeplot(robust_df['mfcc8'],ax=ax3)

sns.kdeplot(robust_df['mfcc9'],ax=ax3)

sns.kdeplot(robust_df['mfcc10'],ax=ax3)

sns.kdeplot(robust_df['mfcc11'],ax=ax3)

sns.kdeplot(robust_df['mfcc12'],ax=ax3)

sns.kdeplot(robust_df['mfcc13'],ax=ax3)

sns.kdeplot(robust_df['mfcc14'],ax=ax3)

sns.kdeplot(robust_df['mfcc15'],ax=ax3)

sns.kdeplot(robust_df['mfcc16'],ax=ax3)

sns.kdeplot(robust_df['mfcc17'],ax=ax3)

sns.kdeplot(robust_df['mfcc18'],ax=ax3)

sns.kdeplot(robust_df['mfcc19'],ax=ax3)

sns.kdeplot(robust_df['mfcc20'],ax=ax3)

plt.show()
sX_train, sX_test, sy_train, sy_test = train_test_split(scale_data,label,test_size=0.25, random_state=42)

mX_train, mX_test, my_train, my_test = train_test_split(mmscaler_data,label,test_size=0.25, random_state=42)

stX_train, stX_test, sty_train, sty_test = train_test_split(standard_data,label,test_size=0.25, random_state=42)

roX_train, roX_test, roy_train, roy_test = train_test_split(robust_data,label,test_size=0.25, random_state=42)
parameters = {'kernel':['rbf'], 'C': [10], 'gamma': [ 0.05] ,'class_weight' : ['balanced']}

svc = svm.SVC(gamma="scale")

clf = GridSearchCV(svc, parameters, cv=5)



s=clf.fit(sX_train,sy_train)

m=clf.fit(mX_train,my_train)

st=clf.fit(stX_train,sty_train)

ro=clf.fit(roX_train,roy_train)



sy_predict=s.predict(sX_test)

my_predict=m.predict(mX_test)

sty_predict=st.predict(stX_test)

roy_predict=ro.predict(roX_test)
mat1 = confusion_matrix(sy_test,sy_predict)

sns.heatmap(mat1.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');



print(classification_report(sy_test, sy_predict))
mat2 = confusion_matrix(my_test,my_predict)

sns.heatmap(mat2.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');



print(classification_report(my_test, my_predict))
mat3 = confusion_matrix(sty_test,sty_predict)

sns.heatmap(mat3.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');



print(classification_report(sty_test, sty_predict))
mat4 = confusion_matrix(roy_test,roy_predict)

sns.heatmap(mat4.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');



print(classification_report(roy_test, roy_predict))
parameters = {'kernel':['rbf'], 'C': [1,5,10,50,100,50,1000], 'gamma': [ 0.1,0.05,0.01,0.005,0.001,0.0005,0.0001] ,'class_weight' : ['balanced']}

svc = svm.SVC(gamma="scale")

clf = GridSearchCV(svc, parameters, cv=5)



s=clf.fit(sX_train,sy_train)

m=clf.fit(mX_train,my_train)

st=clf.fit(stX_train,sty_train)

ro=clf.fit(roX_train,roy_train)



sy_predict=s.predict(sX_test)

my_predict=m.predict(mX_test)

sty_predict=st.predict(stX_test)

roy_predict=ro.predict(roX_test)
print(s.best_params_)

print(classification_report(sy_test, sy_predict))



print(m.best_params_)

print(classification_report(my_test, my_predict))



print(st.best_params_)

print(classification_report(sty_test, sty_predict))



print(ro.best_params_)

print(classification_report(roy_test, roy_predict))