# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import scipy

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import statsmodels.api as stats
df = pd.read_csv("/kaggle/input/policy_2001.csv")

df.head()
df.columns
data = df[["CREDIT_SCORE_BAND","BLUEBOOK_1000","CUST_LOYALTY","MVR_PTS","TIF","TRAVTIME"]]

data.head()
target = df["CLAIM_FLAG"]

target.head()
from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(data,target,stratify=target,random_state = 20191009,test_size=0.25)
Claim = labels_train.astype('category')

y = Claim

y_category = y.cat.categories

y_category
#Model with just the intercept

X = np.where(Claim.notnull(), 1, 0)

DF0 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



logit = stats.MNLogit(y, X)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

thisParameter = thisFit.params

LLK0 = logit.loglike(thisParameter.values)



print(thisFit.summary())

print("Model Parameter Estimates:\n", thisFit.params)

print("Model Log-Likelihood Value =", LLK0)

print("Number of Free Parameters =", DF0)
def one_predictor(predictor):

    X = features_train[[predictor]]

    X = stats.add_constant(X, prepend=True)

    DF1 = np.linalg.matrix_rank(X) * (len(y_category) - 1)





    logit = stats.MNLogit(y, X)

    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

    thisParameter = thisFit.params

    LLK1 = logit.loglike(thisParameter.values)



    Deviance = 2 * (LLK1 - LLK0)

    DF = DF1 - DF0

    pValue = scipy.stats.chi2.sf(Deviance, DF)



    #print(thisFit.summary())

    #print("Model Log-Likelihood Value =", LLK1)

    #print("Number of Free Parameters =", DF1)

    print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

    return (pValue,predictor)
list_of_pvalues = []
#Model with intercept + Blue Book Value

list_of_pvalues.append(one_predictor("BLUEBOOK_1000"))
#Model with intercept + Customer LOYALTY

list_of_pvalues.append(one_predictor('CUST_LOYALTY'))
#Model with intercept + Motor Vehicle Record Points

list_of_pvalues.append(one_predictor("MVR_PTS"))
#Model with intercept + Time in Force

list_of_pvalues.append(one_predictor("TIF"))
#Model with intercept + TRAVEL TIME

list_of_pvalues.append(one_predictor("TRAVTIME"))
#Model with intercept + Credit Score Band

cr_band = features_train[['CREDIT_SCORE_BAND']].astype('category')

X = pd.get_dummies(cr_band)

X = stats.add_constant(X, prepend=True)

DF1 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



logit = stats.MNLogit(y, X)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

thisParameter = thisFit.params

LLK1 = logit.loglike(thisParameter.values)



Deviance = 2 * (LLK1 - LLK0)

DF = DF1 - DF0

pValue = scipy.stats.chi2.sf(Deviance, DF)



#print(thisFit.summary())

#print("Model Log-Likelihood Value =", LLK1)

#print("Number of Free Parameters =", DF1)

print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

m = (pValue,'CREDIT_SCORE_BAND')

list_of_pvalues.append(m)
min(list_of_pvalues)

# So now here, we use the MVR_PTS in our logistic model

#Model will have Intercept + MVR_PTS. Now we check for each remaining predictor our pValue

list_of_pvalues
#Model will have Intercept + MVR_PTS. Now we check for each remaining predictor our pValue

X = features_train[["MVR_PTS"]]

X = stats.add_constant(X, prepend=True)

DF0 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



logit = stats.MNLogit(y, X)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

thisParameter = thisFit.params

LLK0 = logit.loglike(thisParameter.values)
def two_predictors(preds):

    mvrpts = features_train[['MVR_PTS']].astype('category')

    X = features_train[['MVR_PTS']]

    X = X.join(features_train[[preds]])

    X = stats.add_constant(X, prepend=True)

    #print(X)

    DF1 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



    logit = stats.MNLogit(y, X)

    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

    thisParameter = thisFit.params

    LLK1 = logit.loglike(thisParameter.values)



    Deviance = 2 * (LLK1 - LLK0)

    DF = DF1 - DF0

    pValue = scipy.stats.chi2.sf(Deviance, DF)

    return(pValue,preds)
lis = []



lis.append(two_predictors("BLUEBOOK_1000"))

lis.append(two_predictors("CUST_LOYALTY"))

lis.append(two_predictors("TIF"))

lis.append(two_predictors("TRAVTIME"))



cr_band = features_train[['CREDIT_SCORE_BAND']].astype('category')

X = pd.get_dummies(cr_band)

X = X.join(features_train[['MVR_PTS']])

X = stats.add_constant(X, prepend=True)

DF1 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



logit = stats.MNLogit(y, X)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

thisParameter = thisFit.params

LLK1 = logit.loglike(thisParameter.values)



Deviance = 2 * (LLK1 - LLK0)

DF = DF1 - DF0

pValue = scipy.stats.chi2.sf(Deviance, DF)

lis.append((pValue,"CREDIT_SCORE_BAND"))

min(lis)

#Hence, we add Travel TIme as a preddictor to our model as it have minimum pValue among all other predictors and also is less than 0.05
#Build a model with intercept + MVR_PTS + TRAVTIME

X = features_train[["MVR_PTS","TRAVTIME"]]

X = stats.add_constant(X, prepend=True)

DF0 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



logit = stats.MNLogit(y, X)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

thisParameter = thisFit.params

LLK0 = logit.loglike(thisParameter.values)
def three_predictors(pr):

    X = features_train[['MVR_PTS',"TRAVTIME"]]

    X = X.join(features_train[[pr]])

    X = stats.add_constant(X, prepend=True)

    #print(X)

    DF1 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



    logit = stats.MNLogit(y, X)

    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

    thisParameter = thisFit.params

    LLK1 = logit.loglike(thisParameter.values)



    Deviance = 2 * (LLK1 - LLK0)

    DF = DF1 - DF0

    pValue = scipy.stats.chi2.sf(Deviance, DF)

    return(pValue,pr)
listt = []

listt.append(three_predictors("BLUEBOOK_1000"))

listt.append(three_predictors("CUST_LOYALTY"))

listt.append(three_predictors("TIF"))



cr_band = features_train[['CREDIT_SCORE_BAND']].astype('category')

X = pd.get_dummies(cr_band)

X = X.join(features_train[['MVR_PTS',"TRAVTIME"]])

X = stats.add_constant(X, prepend=True)

DF1 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



logit = stats.MNLogit(y, X)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

thisParameter = thisFit.params

LLK1 = logit.loglike(thisParameter.values)



Deviance = 2 * (LLK1 - LLK0)

DF = DF1 - DF0

pValue = scipy.stats.chi2.sf(Deviance, DF)

listt.append((pValue,"CREDIT_SCORE_BAND"))

min(listt)

X = features_train[["MVR_PTS","TRAVTIME"]]

#X = stats.add_constant(X, prepend=True)

DF0 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



logit = stats.MNLogit(y, X)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

thisParameter = thisFit.params

LLK0 = logit.loglike(thisParameter.values)
cnt = 0

for i in labels_train:

    if i == 1:

        cnt+=1

threshold = cnt/len(labels_train)

print("Threshold probability of the event = ",threshold)
X = features_train[["MVR_PTS","TRAVTIME"]]

X = stats.add_constant(X, prepend=True)

DF0 = np.linalg.matrix_rank(X) * (len(y_category) - 1)



logit = stats.MNLogit(y, X)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

thisParameter = thisFit.params



W = features_test[["MVR_PTS","TRAVTIME"]]

W = stats.add_constant(W, prepend=True)

pred_proabs = thisFit.predict(W)
predictions = []

for j in pred_proabs[1]:

    if j >= threshold:

        predictions.append(1)

    else:

        predictions.append(0)



from sklearn.metrics import accuracy_score

print("The misclssification rate is",round(1-accuracy_score(labels_test,predictions),7))