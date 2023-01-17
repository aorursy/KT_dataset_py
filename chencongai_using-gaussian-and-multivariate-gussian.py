

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



#firstly we will play with the data to understand better the data dimensions

"""

DataFrame Structure : 

    data = {'name':["张三","李四","王五","马六"],

       "student_id":[1,2,3,4],

       "scores":[30,42,110,150]}

"""



data = pd.read_csv("../input/creditcard.csv")

print(data)

m = data.shape[0]

n = data.shape[1]

print("data shape:",data.shape)

print(data['Time'][1])



#check if NaN exist in our datasets

print(np.any(data.isnull()) == True)



#we don't have any Nan, so continue



data_class = pd.value_counts(data['Class'])

print("data_class:",data_class)

data_class_len = len(data_class)

print("data_class_len:",data_class_len)

print("data_class shape:",data_class.shape)

#there are 284315 genuine data

print("data_class[0]:",data_class[0])

#there are 492 fraud data

print("data_class[1]",data_class[1])



#we can see from the data visualization , the data is clearly skewed

data_class.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")



# Any results you write to the current directory are saved as output.
print(type(data))

#transform dataframe to numpy matrix

data_matrix = data.values



'''

Next we need to plot every feature to check their distribution for genuine and fraud class.

because we will use gaussian distribution to detect the anormaly , so it's better to make sure every features has gaussian distribution,

otherwise, we need to transform the data to gaussian distribution using log or some other method.

'''

v_features = data.columns

plt.figure(figsize=(12,31*4))

gs = gridspec.GridSpec(31,1)



for i, col in enumerate(v_features):

    ax = plt.subplot(gs[i])

    sns.distplot(data[col][data['Class']==0],color='g',label='Genuine Class')

    sns.distplot(data[col][data['Class']==1],color='r',label='Fraud Class')

    ax.legend()

plt.show()
'''

we can see from the features' data visualization , these features :'V1', 'V5', 'V6','V7','V8','V13','V15','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'

will not help us to find the anormaly, why ? 

   1. The genuine class has the same distribution compared to the fraud class

   2. For example : These features 'V1','V2','V5','V6','V7','V21','Amount', they have a large part of the intersection for genuine and fraud class

'''

data = data.drop(['Time','V1', 'V5', 'V6','V7','V8','V13','V15','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'],axis=1)

print(data.columns)
'''

data_class_0 = data[data.Class==0]

data_class_1 = data[data.Class==1]



data_class_0_gaussian_v2 = np.power(data_class_0.V2,-3)

data_class_1_gaussian_v2 = np.power(data_class_1.V2,-3)

data_class_0_gaussian_v3 = np.power(data_class_0.V3,1)

data_class_0_gaussian_v4 = np.power(data_class_0.V19,1)

#data_class_0_log = data_class_0_log.dropna(

#    axis=0,     # 0: row, 1: column

#    how='any'   # 'any': dorp if any Nan exist; 'all': drop if all row are Nan

#    )

#sns.set_style('darkgrid')

#sns.distplot(data_class_0_gaussian_v2,color='r')

#sns.distplot(data_class_0_gaussian_v3,color='r')

#sns.distplot(data_class_0_gaussian_v4,color='r')

'''

#we can see our features has gaussian distribution 

v_features = data.columns

fig, axList = plt.subplots(13,2,figsize=(12,36))

for i, col in enumerate(v_features):

  data.query("Class==1").hist(column=col,bins=np.linspace(-10,10,20),ax=axList[i][0],label='Fraud')

  #ax1.legend()

  data.query("Class==0").hist(column=col,bins=np.linspace(-10,10,20),ax=axList[i][1],label='Genuine')

  #plt.legend()

plt.show()



'''

Next we will create our gaussian distribution with the dataset:

Important point:

1. Because the data is skewed, classification accuracy would not be a good evaluation metrics, so what's the good evaluation metric to use,

possible evaluation metrics : 

      - True positive, false positive , false negative, true negative

      - Precision/Recall

      - F1-score

      

2. we need to use the cross validation dataset to choose our parameter epsilon for gaussian distribution



if p(x) < eplion: y = 1 , Anomaly 



if p(x) > eplion: y = 0 , Normal 



From above , we can see that there are 284315 genuine data and 492 fraud data

  Traning set :276315 genuine

  CV : 4000 genuine, 242 fraud

  Test set :4000 genuine, 250 fraud

'''

#calculate mu and sigma using Training set



#(284807, 13)

#print(data.shape)

#m = data.shape[0]

#number of features

n = data.shape[1]

data_class_normal = data[data.Class==0]

data_class_anomaly = data[data.Class==1]

#delete Class which will not be used for calculate the p(x)

#data_class_normal = data_class_normal.drop(['Class'],axis=1)

#data_class_anomaly = data_class_anomaly.drop(['Class'],axis=1)

m_train = 276315

m_cv = 4000

m_cv_anomaly = 242

m_test = 4000

m_test_anomaly = 250

data_train = data_class_normal[0:m_train][:]



data_cv = data_class_normal[m_train:m_train+m_cv][:]

data_cv_anomaly = data_class_anomaly[0:m_cv_anomaly][:]

data_cv_combined = np.vstack((data_cv,data_cv_anomaly))

#get the last column 'Class'

data_cv_combined_y = data_cv_combined[:, n-1:]

#get the first 12 rows for calculating the normal distribution p(x)

data_cv_combined = np.delete(data_cv_combined, -1, axis=1)



data_test = data_class_normal[m_train+m_cv:m_train+m_cv+m_test][:]

data_test_anomaly = data_class_anomaly[m_cv_anomaly:m_cv_anomaly+m_test_anomaly][:]

data_test_combined = np.vstack((data_test,data_test_anomaly))

#get the last column 'Class'

data_test_combined_y = data_test_combined[:, n-1:]

#get the first 12 rows for calculating the normal distribution p(x)

data_test_combined = np.delete(data_test_combined, -1, axis=1)





#(276315, 13)

print('data_train',data_train.shape)

#(4000, 13)

print('data_cv',data_cv.shape)

#(242, 13)

print('data_cv_anomaly',data_cv_anomaly.shape)

#(4242, 13)

print('data_cv_combined',data_cv_combined.shape)

#(4000, 13)

print('data_test',data_test.shape)

#(250, 13)

print('data_test_anomaly',data_test_anomaly.shape)

#(4250, 13)

print('data_test_combined',data_test_combined.shape)



#caculate mu and sigma using data_train

mu = np.mean(data_train,axis=0)

#sigma = np.sqrt(np.sum(((data_train - mu) ** 2),axis=0) / data_train.shape[0])

sigma = np.std(data_train,ddof=0,axis=0)

print('mu shape : ',mu.shape)

print('sigma shape:',sigma.shape)

mu=mu.values.reshape(1,n)

mu = np.delete(mu, -1, axis=1)

sigma = sigma.values.reshape(1,n)

sigma = np.delete(sigma, -1, axis=1)

print('mu shape : ',mu.shape)

print('sigma shape:',sigma.shape)



def selectThreshold(yval,pval):

  bestEpsilon = 0.

  bestF1 = 0.

  F1 = 0.

  step = (np.max(pval)-np.min(pval))/1000

  #print('step:',step)

  #print('minpval : ',np.min(pval))

  #print('maxpval : ',np.max(pval))

  for epsilon in np.arange(np.min(pval),np.max(pval),step):

    cvPrecision = (pval < epsilon)

    # sum return int, so need to transform to float

    tp = np.sum((cvPrecision == 1) & (yval == 1)).astype(float)

    fp = np.sum((cvPrecision == 1) & (yval == 0)).astype(float)

    fn = np.sum((cvPrecision == 1) & (yval == 0)).astype(float)



    precision = tp/(tp+fp)

    recision = tp/(tp+fn)

    #calculate F1

    F1 = (2*precision*recision)/(precision+recision)

    #print('precision,recision,F1',precision,',',recision,',',F1)

    if F1 > bestF1:

      bestF1 = F1

      bestEpsilon = epsilon



  return bestEpsilon,bestF1

#print(pval)



#calculate using CV datasets

pval = np.exp(-((data_cv_combined-mu)**2)/2*(sigma**2))/(np.sqrt(2*math.pi)*sigma)



print(pval.shape)

bestEpsilon,bestF1 = selectThreshold(data_cv_combined_y,pval)

print('bestEpsilon: ',bestEpsilon)

print('bestF1:',bestF1)