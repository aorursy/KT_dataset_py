%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import os

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import KFold

from sklearn.metrics import log_loss
gatrain = pd.read_csv('../input/gender_age_train.csv')

gatest = pd.read_csv('../input/gender_age_test.csv')

gatrain.head(3)
letarget = LabelEncoder().fit(gatrain.group.values)

y = letarget.transform(gatrain.group.values)

n_classes = len(letarget.classes_)
phone = pd.read_csv('../input/phone_brand_device_model.csv',encoding='utf-8')

phone.head(3)
phone = phone.drop_duplicates('device_id', keep='first')
lebrand = LabelEncoder().fit(phone.phone_brand)

phone['brand'] = lebrand.transform(phone.phone_brand)

m = phone.phone_brand.str.cat(phone.device_model)

lemodel = LabelEncoder().fit(m)

phone['model'] = lemodel.transform(m)
train = gatrain.merge(phone[['device_id','brand','model']], how='left',on='device_id')
class GenderAgeGroupProb(object):

    def __init__(self):

        pass

    

    def fit(self, df, by, n_smoothing, weights):

        self.by = by

        self.n_smoothing = n_smoothing

        self.weights = np.divide(weights,sum(weights))

        self.classes_ = sorted(df['group'].unique())

        self.n_classes_ = len(self.classes_)

        

        self.group_freq = df['group'].value_counts().sort_index()/df.shape[0]

        

        self.prob_by = []

        for i,b in enumerate(self.by):

            c = df.groupby([b,'group']).size().unstack().fillna(0)

            total = c.sum(axis=1)

            prob = (c.add(self.n_smoothing[i]*self.group_freq)).div(total+self.n_smoothing[i], axis=0)

            self.prob_by.append(prob)

        return self

    

    def predict_proba(self, df):

        pred = pd.DataFrame(np.zeros((len(df.index),self.n_classes_)),columns=self.classes_,index=df.index)

        pred_by = []

        for i,b in enumerate(self.by):

            pred_by.append(df[[b]].merge(self.prob_by[i], how='left',

                                      left_on=b, right_index=True).fillna(self.group_freq)[self.classes_])

            pred = pred.radd(pred_by[i].values*self.weights[i])

        

        pred.loc[pred.iloc[:,0].isnull(),:] = self.group_freq

        return pred[self.classes_].values

    

def score(ptrain, by, n_smoothing, weights=[0.5,0.5]):

    kf = KFold(ptrain.shape[0], n_folds=10, shuffle=True, random_state=0)

    pred = np.zeros((ptrain.shape[0],n_classes))

    for itrain, itest in kf:

        train = ptrain.iloc[itrain,:]

        test = ptrain.iloc[itest,:]

        ytrain, ytest = y[itrain], y[itest]

        clf = GenderAgeGroupProb().fit(train,by,n_smoothing,weights)

        pred[itest,:] = clf.predict_proba(test)

    return log_loss(y, pred)
n_smoothing = [1,5,10,15,20,50,100]

res = [score(train,['brand','model'],[s,s],[.5,.5]) for s in n_smoothing]

plt.plot(n_smoothing, res)

plt.title('Best score {:.5f} at n_smoothing = {}'.format(np.min(res),n_smoothing[np.argmin(res)]))

plt.xlabel('n_smoothing')
brand_weight = [0,0.2,0.4,0.6,0.8,1.0]

res = [score(train,['brand','model'],[15,15],[b,1-b]) for b in brand_weight]

plt.plot(brand_weight, res)

plt.title('Best score {:.5f} at brand_weight = {}'.format(np.min(res),brand_weight[np.argmin(res)]))

plt.xlabel('brand_weight')
test = gatest.merge(phone[['device_id','brand','model']], how='left',on='device_id')

test.head(3)
clf = GenderAgeGroupProb().fit(train,['brand','model'],[15,15],[0.4,0.6])

pred = clf.predict_proba(test)
pd.DataFrame(pred, 

             index = test.device_id, 

             columns=clf.classes_).to_csv('pbm_subm.csv', index=True)