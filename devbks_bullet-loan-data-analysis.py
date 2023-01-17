# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

import os



%matplotlib inline
loandf=pd.read_csv('../input/Loan payments data.csv')
loandf.head()
loandf.info()
loandf['loan_status'].unique()
sns.set_style('darkgrid')

sns.countplot(loandf['loan_status'], palette='Spectral')
loandf[['loan_status','Principal','Loan_ID']].groupby(['loan_status','Principal']).agg(['count'])
fig=plt.figure(figsize=(12,6))

sns.distplot(loandf['Principal'], bins=40)
ax= sns.countplot(loandf['terms'], palette='Spectral')

ax.set_title('Term Counts')
fig, ax=plt.subplots(figsize=(12,4))

sns.countplot(x='terms', hue='loan_status', data=loandf, palette='Spectral')

ax.set_title('Term counts by Loan Status')

ax.legend(loc='upper left')

loandf['Days to pay']= (pd.DatetimeIndex(loandf['paid_off_time']).normalize()

                        -pd.DatetimeIndex(loandf['effective_date']).normalize())/np.timedelta64(1,'D')
loandf['paid_off_date'] = pd.DatetimeIndex(loandf['paid_off_time']).normalize()
fig, ax=plt.subplots(figsize=(15,6))

ax=sns.countplot(x='Days to pay',hue='terms',data=loandf)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
fig, ax=plt.subplots(figsize=(15,6))

ax=sns.countplot(x='Days to pay', hue='terms', data=loandf[loandf['loan_status']== 'PAIDOFF'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
tmp = loandf.loc[(loandf['Days to pay'] > 30) & (loandf['loan_status'] == 'PAIDOFF')]

print("{}: Incorrect status: {} observations")

print(tmp[['loan_status', 'terms', 'effective_date', 'due_date', 'paid_off_time']])
## exploring demographic

fig, axs=plt.subplots(3,2, figsize=(20,15))



sns.distplot(loandf['age'], ax=axs[0][0])

axs[0][0].set_title("Total age distribution across dataset")



sns.boxplot(x='loan_status', y='age', data=loandf, ax=axs[0][1])

axs[0][1].set_title("Age distribution by loan status")



sns.countplot(x='education', data=loandf, ax=axs[1][0])

axs[1][0].set_title("Education count")





sns.countplot(x='education', data=loandf, hue='loan_status', ax=axs[1][1])

axs[1][1].set_title("Education by loan status")

axs[1][1].legend(loc='upper right')





sns.countplot(x='Gender', data=loandf, ax=axs[2][0])

axs[2][0].set_title(" Gender")



sns.countplot(x='Gender', data=loandf, hue='education', ax=axs[2][1])

axs[2][1].set_title("Education of the gender")
## exploring gender +education 

pd.crosstab(loandf['loan_status'], loandf['Gender'] + "_" + loandf['education'], margins=True)
pd.crosstab(loandf['loan_status'],loandf['Gender']+"_"+loandf['education'],margins=True,normalize='all')
pd.crosstab(loandf['loan_status'],loandf['Gender']+"_"+loandf['education'],margins=True,normalize='index')
pd.crosstab(loandf['loan_status'],loandf['Gender']+"_"+loandf['education'],margins=True,normalize='columns')
loandf.loc[(loandf['loan_status'] =='PAIDOFF' ) &(loandf['Days to pay']>30),'loan_status']='COLLECTION_PAIDOFF'
smap= {"PAIDOFF": 1, "COLLECTION": 2, "COLLECTION_PAIDOFF": 2 }

loandf['loan_status_trgt'] = loandf['loan_status'].map(smap)



fig, axs=plt.subplots(1,2,figsize=(12,5))



sns.countplot(x='loan_status',data=loandf,ax=axs[0])

axs[0].set_title('Count with original targets')



sns.countplot(x='loan_status_trgt', data=loandf, ax=axs[1])

axs[1].set_title('Count with new targets')
dummies=pd.get_dummies(loandf['education']).rename(columns=lambda x:'is_' +str(x))

loandf=pd.concat([loandf,dummies],axis=1)

loandf.drop(['education'],axis=1,inplace=True)

dummies=pd.get_dummies(loandf['Gender']).rename(columns=lambda x:'is_' +str(x))

loandf=pd.concat([loandf,dummies],axis=1)

loandf.drop(['Gender'],axis=1,inplace=True)
loandf.drop(['Loan_ID', 'loan_status', 'effective_date', 'due_date',

             'paid_off_time', 'past_due_days', 'paid_off_date', 'Days to pay'], axis=1,inplace=True)
dummyvar=['is_female','is_Master or Above']

loandf.drop(dummyvar,axis=1, inplace=True)
#create our inputs and target variable

X=loandf.drop('loan_status_trgt',axis=1)

y=loandf['loan_status_trgt']
#import ML libraries



from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
## funciton to evaluate our models



def eval_model(model, data, target, splitratio):

    trainX, testX, trainY, testY = train_test_split(data, target, train_size=splitratio, random_state=0)

    model.fit(trainX,trainY)

    return model.score(testX,testY)
import warnings

warnings.filterwarnings("ignore")



num_estimator=np.array([1,5,10,50,100,250,500])

num_sample=5

num_grid=len(num_estimator)

score_mean=np.zeros(num_grid)

score_sigma=np.zeros(num_grid)

j=0







print("RandomForestClassification Starting")

for x in num_estimator:

    score_array = np.zeros(num_sample) # Initialize

    for i in range(0,num_sample):

        rf_class = RandomForestClassifier(n_estimators=x, n_jobs=1, criterion="gini")

        score_array[i] = eval_model(rf_class, X, y, 0.8)

        print("Try {} with n_estimators = {} and score = {}".format( i, x, score_array[i]))

    score_mean[j], score_sigma[j] = np.mean(score_array), np.std(score_array)

    j=j+1



print("RandomForestClassification Done!")
fig = plt.figure(figsize=(12,6))

plt.errorbar(num_estimator, score_mean, yerr=score_sigma, fmt='k.-')

plt.xscale("log")

plt.xlabel("number of estimators",size = 16)

plt.ylabel("accuracy",size = 16)

plt.xlim(0.9,600)

plt.ylim(0.3,0.8)

plt.title("Random Forest Classifier", size = 18)

plt.grid(which="both")
#SVM linear



C_ar = np.array([0.5, 0.1, 1, 5, 10])

score_ar = np.zeros(len(C_ar))

i=0

for C_val in C_ar:

    svc_class = svm.SVC(kernel='linear', random_state=1, C = C_val)

    score_array[i] = eval_model(svc_class, X, y, 0.8)

    i=i+1



score_mu, score_sigma = np.mean(score_ar), np.std(score_ar)



fig = plt.figure(figsize=(12,6))

plt.errorbar(C_ar, score_ar, yerr=score_sigma, fmt='k.-')

plt.xlabel("C assignment",size = 16)

plt.ylabel("accuracy",size = 16)

plt.title("SVM Classifier (Linear)", size = 18)

plt.grid(which="both")
#adjusting our gamma

gamma_ar = np.array([0.001, 0.01, 0.1, 1, 10])

score_ar = np.zeros(len(gamma_ar))

score_mean = np.zeros(len(gamma_ar))

score_sigma = np.zeros(len(gamma_ar))

i=0

for l in gamma_ar:

    svc_class = svm.SVC(kernel='rbf', random_state=1, gamma = l)

    score_array[i] = eval_model(svc_class, X, y, 0.8)

    score_mean[i], score_sigma[i] = np.mean(score_ar[i]), np.std(score_ar[i])

    i=i+1





fig = plt.figure(figsize=(12,6))

plt.errorbar(gamma_ar, score_mean, yerr=score_sigma, fmt='k.-')

plt.xscale('log')

plt.xlabel("Gamma",size = 16)

plt.ylabel("accuracy",size = 16)

plt.title("SVM Classifier (RBF)", size = 18)

plt.grid(which="both")
#keras



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD



# Change to np.array type

new_x = np.array(X)

new_y = np.array(y)



# fix random seed for reproducibility

np.random.seed(1234)



model = Sequential()

model.add(Dense(64, input_dim=7, init='uniform', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])




model.fit(new_x, new_y, epochs=150, batch_size=20)

scores = model.evaluate(new_x, new_y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


