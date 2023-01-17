import numpy as np 

import pandas as pd 

import sys

if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")

    

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
hist = df.hist(bins=100, figsize = (20,20))
df_distributed = df[["V11","V13","V15","V18","V19","V12","Class"]]
negative_df = df_distributed.loc[df_distributed['Class'] == 0]

positive_df = df_distributed.loc[df_distributed['Class'] == 1]
from sklearn.model_selection import train_test_split



y_negative = negative_df["Class"]

y_positive = positive_df["Class"]

negative_df.drop(["Class"], axis=1, inplace=True)

positive_df.drop(["Class"], axis=1, inplace=True)



# 90% of good data are training data to estimate Gaussian factors (mean, Standard deviation and variance)

negative_df_training, negative_df_testing, y_negative_training, y_negative_testing = train_test_split(negative_df,

                                                                                                      y_negative,

                                                                                                      test_size=0.1,

                                                                                                      random_state=0)

# 5% for CV dataset, 5% for testing dataset

negative_df_cv, negative_df_testing, y_negative_cv, y_negative_testing = train_test_split(negative_df_testing,

                                                                                          y_negative_testing,

                                                                                          test_size=0.5,

                                                                                          random_state=0)



# while 50% the anomalies data will be added to CV dataset and the other 50% will be added to Testing dataset

positive_df_cv, positive_df_testing, y_positive_cv, y_positive_testing = train_test_split(positive_df,

                                                                                          y_positive,

                                                                                          test_size=0.5,

                                                                                          random_state=0)



df_cv = pd.concat([positive_df_cv, negative_df_cv], ignore_index=True)

df_cv_y = pd.concat([y_positive_cv, y_negative_cv], ignore_index=True)

df_test = pd.concat([positive_df_testing, negative_df_testing], ignore_index=True)

df_test_y = pd.concat([y_positive_testing, y_negative_testing], ignore_index=True)



y_negative_training = y_negative_training.values.reshape(y_negative_training.shape[0], 1)

df_cv_y = df_cv_y.values.reshape(df_cv_y.shape[0], 1)

df_test_y = df_test_y.values.reshape(df_test_y.shape[0], 1)
def estimateGaussian(X):

    stds=[]

    mean = []

    variance =[]

    

    mean = X.mean(axis=0)

    stds =X.std(axis=0)

    variance = stds **2

    

    stds = stds.values.reshape(stds.shape[0], 1)

    mean = mean.values.reshape(mean.shape[0], 1)

    variance = variance.values.reshape(variance.shape[0], 1)

    return stds,mean,variance
stds,mean,variance = estimateGaussian(negative_df_training)
print(stds.shape)

print(stds.shape)

print(stds.shape)
def multivariateGaussian(stds, mean, variance, df_cv):

    probability = []

    for i in range(df_cv.shape[0]):

        result = 1

        for j in range(df_cv.shape[1]):

            var1 = 1/(np.sqrt(2* np.pi)* stds[j])

            var2 = (df_cv.iloc[i,j]-mean[j])**2

            var3 = 2*variance[j]



            result *= (var1) * np.exp(-(var2/var3))

        result = float(result)

        probability.append(result)

    return probability
def selectEpsilon(y_actual, y_probability):

    best_epi = 0

    best_F1 = 0

    best_rec = 0

    best_pre = 0

    

    stepsize = (max(y_probability) -min(y_probability))/1000000

    epi_range = np.arange(min(y_probability),max(y_probability),stepsize)

    for epi in epi_range:

        predictions = (y_probability<epi)[:,np.newaxis]

        tp = np.sum(predictions[y_actual==1]==1)

        fp = np.sum(predictions[y_actual==0]==1)

        fn = np.sum(predictions[y_actual==1]==0)

        

        prec = tp/(tp+fp)

        rec = tp/(tp+fn)

        

        if prec > best_pre:

            best_pre =prec

            best_epi_prec = epi

            

        if rec > best_rec:

            best_rec =rec

            best_epi_rec = epi

            

        F1 = (2*prec*rec)/(prec+rec)

        

        if F1 > best_F1:

            best_F1 =F1

            best_epi = epi

        

    return best_epi, best_F1,best_pre,best_epi_prec,best_rec,best_epi_rec
probability = multivariateGaussian(stds, mean, variance, df_cv)

best_epi, best_F1,best_pre,best_epi_prec,best_rec,best_epi_rec = selectEpsilon(df_cv_y, probability)

print("The best epsilon Threshold over the croos validation set is :",best_epi)

print("The best F1 score over the croos validation set is :",best_F1)

print("The best epsilon Threshold over the croos validation set is for recall :",best_epi_rec)

print("The best Recall score over the croos validation set is :",best_rec)

print("The best epsilon Threshold over the croos validation set is for precision:",best_epi_prec)

print("The best Precision score over the croos validation set is :",best_pre)
def prediction_scores(y_actual, y_probability, epsilon):

    predictions = (y_probability<epsilon)[:,np.newaxis]

    tp = np.sum(predictions[y_actual==1]==1)

    fp = np.sum(predictions[y_actual==0]==1)

    fn = np.sum(predictions[y_actual==1]==0)

        

    prec = tp/(tp+fp)

    rec = tp/(tp+fn) 

    F1 = (2*prec*rec)/(prec+rec)

        

    return prec,rec,F1
epsilon = best_epi

probability = multivariateGaussian(stds, mean, variance, df_test)

prec,rec,F1 = prediction_scores(df_test_y, probability,epsilon)

print("Percision on Testing Set:",prec)

print("Recall on Testing Set:",rec)

print("F1 on Testing Set:",F1)