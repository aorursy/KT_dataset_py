import matplotlib.pyplot as plt

import numpy as np

import pandas

%matplotlib inline



from numpy import genfromtxt

from scipy.stats import multivariate_normal

from sklearn.metrics import f1_score
from sklearn.metrics import f1_score

y_true = [0, 0, 1, 0, 1, 1]

y_pred = [0, 1, 1, 1, 1, 0]

#f1_score(y_true, y_pred, average='binary')  

 



f1_score(y_true, y_pred, average=None)

#tr_data = read_dataset('tr_server_data.csv') 

#cv_data = read_dataset('cv_server_data.csv') 

#gt_data = read_dataset('gt_server_data.csv')



import csv

import numpy

filename = '../input/anomaly-detection-sample-dataset/tr_server_data.csv'

a2 ='../input/anomaly-detection-sample-dataset/cv_server_data.csv'

a3 = '../input/anomaly-detection-sample-dataset/gt_server_data.csv'





raw_data = open(filename, 'rt')

reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

x = list(reader)

tr_data = numpy.array(x).astype('float')



raw_data = open(a2, 'rt')

reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

x = list(reader)

cv_data = numpy.array(x).astype('float')



raw_data = open(a3, 'rt')

reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

x = list(reader)

gt_data = numpy.array(x).astype('float')





n_training_samples = tr_data.shape[0]

n_dim = tr_data.shape[1]



print('Number of datapoints in training set: %d' % n_training_samples)

print('Number of dimensions/features: %d' % n_dim)





print(tr_data[1:5,:])



plt.xlabel('Latency (ms)')

plt.ylabel('Throughput (mb/s)')

plt.plot(tr_data[:,0],tr_data[:,1],'bx')

plt.show()
def read_dataset(filePath,delimiter=','):

    return genfromtxt(filePath, delimiter=delimiter)



def estimateGaussian(dataset):

    mu = np.mean(dataset, axis=0)

    sigma = np.cov(dataset.T)

    return mu, sigma

    

def multivariateGaussian(dataset,mu,sigma):

    p = multivariate_normal(mean=mu, cov=sigma)

    return p.pdf(dataset)



def selectThresholdByCV(probs,gt):

    best_epsilon = 0

    best_f1 = 0

    f = 0

    stepsize = (max(probs) - min(probs)) / 1000

    epsilons = np.arange(min(probs),max(probs),stepsize)

    for epsilon in np.nditer(epsilons):



        predictions = (probs < epsilon) 

        #print(predictions)

        f = f1_score(gt, predictions,average='binary')

        #print(f)

        #print('------')

        if f > best_f1:

            #print('----')

            best_f1 = f

            best_epsilon = epsilon

    

    return best_f1, best_epsilon
mu, sigma = estimateGaussian(tr_data)

p = multivariateGaussian(tr_data,mu,sigma)
#selecting optimal value of epsilon using cross validation

p_cv = multivariateGaussian(cv_data,mu,sigma)

fscore, ep = selectThresholdByCV(p_cv,gt_data)

print(ep)
#selecting outlier datapoints 



outliers = np.asarray(np.where(p < ep))
plt.figure()

plt.xlabel('Latency (ms)')

plt.ylabel('Throughput (mb/s)')

plt.plot(tr_data[:,0],tr_data[:,1],'bx')

plt.plot(tr_data[outliers,0],tr_data[outliers,1],'ro')

plt.show()
from sklearn import svm
# use the same dataset

tr_data = read_dataset('../input/anomaly-detection-sample-dataset/tr_server_data.csv')

cv_data = read_dataset('../input/anomaly-detection-sample-dataset/cv_server_data.csv') 

gt_data = read_dataset('../input/anomaly-detection-sample-dataset/gt_server_data.csv')
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.02)

clf.fit(tr_data)
pred = clf.predict(tr_data)



# inliers are labeled 1, outliers are labeled -1

normal = tr_data[pred == 1]

abnormal = tr_data[pred == -1]
plt.figure()

plt.plot(normal[:,0],normal[:,1],'bx')

plt.plot(abnormal[:,0],abnormal[:,1],'ro')

plt.xlabel('Latency (ms)')

plt.ylabel('Throughput (mb/s)')

plt.show()