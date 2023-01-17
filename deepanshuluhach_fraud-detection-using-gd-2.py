import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import matplotlib.gridspec as gridspec

import seaborn as sns



from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score

from scipy.stats import norm, multivariate_normal

from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score,precision_recall_curve



plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12



import warnings

warnings.filterwarnings('ignore')



import random

random.seed(0)
def Print_Accuracy_Scores(y,y_pred):

    print("F1 Score: ", f1_score(y,y_pred))

    print("Precision Score: ", precision_score(y,y_pred))

    print("Recall Score: ", recall_score(y,y_pred))
#Loading Fraud dataset

dataset = pd.read_csv("../input//creditcard.csv")
v_features = dataset.columns

plt.figure(figsize=(12,31*4))

gs = gridspec.GridSpec(31,1)



for i, col in enumerate(v_features):

    ax = plt.subplot(gs[i])

    sns.distplot(dataset[col][dataset['Class']==0],color='g',label='Genuine Class')

    sns.distplot(dataset[col][dataset['Class']==1],color='r',label='Fraud Class')

    ax.legend()

plt.show()
dataset.drop(labels = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8','Time','V1','V2','V5','V6','V7','V21','Amount'], axis = 1, inplace = True)

dataset.columns
#Method for selecting epsilon with best F1-score

def SelectThresholdByCV_Anomaly(probs,y):

    best_epsilon = 0

    best_f1 = 0

    f = 0

    precision =0

    recall=0

    best_recall = 0

    best_precision = 0

    

    epsilons = sorted(np.unique(probs))

    #print(epsilons)

    

    precisions=[]

    recalls=[]

    for epsilon in epsilons:

        predictions = (probs < epsilon)

        f = f1_score(y, predictions)

        precision = precision_score(y, predictions)

        recall = recall_score(y, predictions)

        

        if f > best_f1:

            best_f1 = f

            best_precision = precision

            best_recall = recall

            best_epsilon = epsilon

        

        precisions.append(precision)

        recalls.append(recall)



    #Precision-Recall Trade-off

    plt.plot(epsilons,precisions,label='Precision')

    plt.plot(epsilons,recalls,label='Recall')

    plt.xlabel("Epsilon")

    plt.title('Precision Recall Trade Off')

    plt.legend()

    plt.show()



    print ('Best F1 Score %f' %best_f1)

    print ('Best Precision Score %f' %best_precision)

    print ('Best Recall Score %f' %best_recall)

    print ('Best Epsilon', best_epsilon)

    return best_epsilon
#Method for calculating parameters Mu & Co-variance

def estimateGaussian(data):

    mu = np.mean(data,axis=0)

    sigma = np.cov(data.T)

    return mu,sigma



#Method for implementing multivariate gaussian distribution function

def MultivariateGaussianDistribution(data,mu,sigma):

    p = multivariate_normal.pdf(data, mean=mu, cov=sigma)

    p_transformed = np.power(p,1/100) #transformed the probability scores by p^1/100 since the values are very low (up to e-150)

    return p_transformed
genuine_data = dataset[dataset['Class'] == 0]

fraud_data = dataset[dataset['Class'] == 1]

#Split Genuine records into train & test - 60:40 ratio

genuine_train, genuine_test = train_test_split(genuine_data, test_size=0.4, random_state=0)

# print(genuine_train.shape)

# print(genuine_test.shape)



#Split 40% of Genuine Test records into Cross Validation & Test again (50:50 ratio)

genuine_cv, genuine_test  = train_test_split(genuine_test, test_size=0.5, random_state=0)

# print(genuine_cv.shape)

# print(genuine_test.shape)



#Split Fraud records into Cross Validation & Test (50:50 ratio)

fraud_cv, fraud_test = train_test_split(fraud_data, test_size=0.5, random_state=0)

# print(fraud_cv.shape)

# print(fraud_test.shape)



#Drop Y-label from Train data

train_data = genuine_train.drop(labels='Class', axis=1)

# print(train_data.shape)



#Cross validation data

cv_data = pd.concat([genuine_cv, fraud_cv])

cv_data_y = cv_data['Class']

cv_data.drop(labels='Class', axis=1, inplace=True)

# print(cv_data.shape)



#Test data

test_data = pd.concat([genuine_test,fraud_test])

test_data_y = test_data['Class']

test_data.drop(labels='Class',axis=1,inplace=True)

# print(test_data.shape)
#Find out the parameters Mu and Covariance for passing to the probability density function

mu, sigma = estimateGaussian(train_data)
#Multivariate Gaussian distribution - This calculates the probability for each record.

p_train = MultivariateGaussianDistribution(train_data, mu, sigma)

print(p_train.mean())

print(p_train.std())

print(p_train.max())

print(p_train.min())
#Calculate the probabilities for cross validation and test records by passing the mean and co-variance matrix derived from train data

p_cv = MultivariateGaussianDistribution(cv_data, mu, sigma)

p_test = MultivariateGaussianDistribution(test_data, mu, sigma)
#Let us use cross validation to find the best threshold where the F1 -score is high

epsilon = SelectThresholdByCV_Anomaly(p_cv,cv_data_y)

# epsilon = 0.2425
from sklearn.externals import joblib



GDdata = {"epsilon" : epsilon, "mu" : mu, "sigma" : sigma}

joblib.dump(GDdata, 'GDdata.pkl')
data2 = joblib.load('GDdata.pkl')

# print(data2)

sig = data2["mu"]

print(sig)
#CV data - Predictions

pred_cv= (p_cv < epsilon)

Print_Accuracy_Scores(cv_data_y, pred_cv)
#Confusion matrix on CV

cnf_matrix = confusion_matrix(cv_data_y,pred_cv)

row_sum = cnf_matrix.sum(axis=1,keepdims=True)

cnf_matrix_norm =cnf_matrix / row_sum 

sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True)

plt.title("Normalized Confusion Matrix - Cross Validation")
#Test data - Check the F1-score by using the best threshold from cross validation

pred_test = (p_test < epsilon)

Print_Accuracy_Scores(test_data_y,pred_test)
cnf_matrix = confusion_matrix(test_data_y, pred_test)

row_sum = cnf_matrix.sum(axis=1,keepdims=True)

cnf_matrix_norm =cnf_matrix / row_sum 

sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True)

plt.title("Normalized Confusion Matrix - Test data")