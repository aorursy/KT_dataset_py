# Feature slection using imporved Fscore 
# about the data_set 

  #microarray data

    #http://csse.szu.edu.cn/staff/zhuzx/Datasets.html

  #"Zexuan Zhu, Y. S. Ong and M. Dash, “Markov Blanket-Embedded Genetic Algorithm for Gene Selection”, Pattern Recognition, Vol. 49, No. 11, 3236-3248, 2007."
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from scipy.io import arff

import pandas as pd

data =pd.read_csv('/kaggle/input/mll-data-csv/csv_result-MLL.csv')

data.head()
data.shape # there is 12583 feature present in this data set
data=data.dropna()

data=data.drop(columns=['id'])
data.head()
data.groupby("class").size()
Class={

    'ALL':    24,

'AML' :   28,

'MLL' :   20

}
from IPython.display import Image

Image(filename='/kaggle/input/f-score/f-score.png') # fscore 
from IPython.display import Image

Image(filename='/kaggle/input/improved-fscore/imporved  F score.png') # imporved f score
Activity_name=list(Class.keys())
Activity_name[0]

def D(col):

    df=data[[col,'class']]

    d_value=0.0

    total_len=df.shape[0]

    for ix in range(0,len(Activity_name)):

        for ixx in range(ix+1,len(Activity_name)):



            class_A=df[df['class']==Activity_name[ix]]

            class_B=df[df['class']==Activity_name[ixx]]

            #print("add")

            #print(((class_A.shape[0]+class_B.shape[0])*np.square(np.mean(class_A.iloc[:,0].values)-np.mean(class_B.iloc[:,0].values)))/float(total_len))

            #print(d_value)

            #print("value")

            d_value=d_value+((class_A.shape[0]+class_B.shape[0])*(np.square(np.mean(class_A.iloc[:,0].values)-np.mean(class_B.iloc[:,0].values))))/float(total_len)

            #print(d_value)

    return d_value
def S(col):

    df=data[[col,'class']]

    sx=0

    for ix in range(0,len(Activity_name)):



        in_class=df[df['class']==Activity_name[ix]]



        means=(np.mean(in_class.iloc[:,0].values))



        s=0

        mins=np.square(in_class.iloc[0,0]-means) 

        maxs=np.square(in_class.iloc[0,0]-means) 

        #print(mins,maxs)



        for ix in in_class.iloc[1:,0]:

            s=s+np.square(ix-means)

            m=np.square(ix-means)

            #print('value of m',m)

            if mins>m:

                mins=m

            if maxs<m:

                maxs=m

        s=(1/float(in_class.shape[0]))*(s-mins)/float(maxs-mins)

        sx=sx+s

    return sx

def Fdf(col):

    Dx=D(col)

    Sx=S(col)

    return Dx/Sx
columns=data.columns[:-1]
len(columns)
feature={}

for ix in columns:

    #print(ix)

    feature[ix]=Fdf(ix)
ns_df = pd.DataFrame(columns=['features','F_Scores'])

ns_df['features']=feature.keys()

ns_df['F_Scores']=feature.values()

ns_df_sorted = ns_df.sort_values(['F_Scores','features'], ascending =[False, True])
import matplotlib.pyplot as plt

plt.plot(list(ns_df_sorted.F_Scores[:200]),label ='F-score')

#plt.axhline(y=, color='r', linestyle='-',label ='thershold')

# plt.axvline(x=350, color='g', linestyle='-',label ='index no of features')

# plt.legend()

#plt.show()
from sklearn import svm

from sklearn.model_selection import train_test_split
def no_feature(x):

    fe=ns_df_sorted.iloc[:x,0]

    X=data[fe]

    y=data['class']

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.4, random_state=42)

    svc=svm.LinearSVC()

    svc.fit(X_train,y_train.astype('category').cat.codes)

    acc=svc.score(X_test,y_test.astype('category').cat.codes)

    return acc
features=[10,20,40,50,60,70,80,100,120,180,200,250,300,400,500,600]

accuracy=[]

for ix in features:

    accuracy.append(no_feature(ix))
import matplotlib.pyplot as plt

plt.plot(features,accuracy,label ='accuracy')

plt.ylabel('accuracy')

plt.xlabel('no of features')

plt.legend()

plt.show()
# we select the 50 features with 100% acc
x=50
fe=ns_df_sorted.iloc[:x,0]

X=data[fe]

y=data['class']

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.4, random_state=42)

svc=svm.LinearSVC()

svc.fit(X_train,y_train.astype('category').cat.codes)

acc=svc.score(X_test,y_test.astype('category').cat.codes)

print(acc)
y_pred=svc.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_pred,y_test.astype('category').cat.codes))