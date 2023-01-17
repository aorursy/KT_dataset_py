!pip install regressors
import numpy as np

import pandas as pd 

import statsmodels.formula.api as sm

import statsmodels.api as sma



from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from sklearn.linear_model import LogisticRegression



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/cardio_train.csv',sep=';')

df = df.dropna()

df = df.drop(columns=['id'])

df.head()
df.corr()
f1 = 'cardio ~ age*cholesterol'

logitfit = sm.logit(formula = str(f1), data = df).fit()

print(logitfit.summary())
f2 = 'cardio ~ age*weight'

logitfit2 = sm.logit(formula = str(f2), data = df).fit()

print(logitfit2.summary())


f3 = 'cardio ~ weight*cholesterol*age'

logitfit3 = sm.logit(formula = str(f3), data = df).fit()

print(logitfit3.summary())

f4 = 'cardio ~ age*ap_hi'

logitfit4 = sm.logit(formula = str(f4), data = df).fit()

print(logitfit4.summary())
f5 = 'cardio ~ weight*active'

logitfit5 = sm.logit(formula = str(f5), data = df).fit()

print(logitfit5.summary())
f6 = 'cardio ~ age*cholesterol*weight*ap_hi'

logitfit6 = sm.logit(formula = str(f6), data = df).fit()

print(logitfit6.summary())
#Polynomial

fPolynomial = 'cardio ~ age+I(age*age)+I(age*age*age)'

logitfitPolynomial = sm.logit(formula = str(fPolynomial), data = df).fit()

print(logitfitPolynomial.summary())
#Logarithmic

fLog = 'cardio ~ np.log(age)+np.log(ap_hi)'

logitfitLog = sm.logit(formula = str(fLog), data = df).fit()

print(logitfitLog.summary())
def forward_selected(inputDF, outputDF, df):

    highestScore = 0.0

    selected = []

    outputColName = str(outputDF.columns[0])

    

    for x in range(2, len(inputDF.columns)+1):

        model = sfs(LinearRegression(),k_features=x,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')

        model.fit(inputDF,outputDF)

        predictors = model.k_feature_names_

        formula = "{} ~ {}".format(outputColName, ' + '.join(predictors))

        candidateModel = sm.logit(formula, df).fit()

        

        if(candidateModel.prsquared > highestScore):

            selected = predictors

            highestScore = candidateModel.prsquared

            print("score:", candidateModel.prsquared)

    

    formula = "{} ~ {}".format(outputColName,' + '.join(selected))

    model = sm.logit(formula, df).fit()

    return model

    
inputDF = df[["age", "gender", "height", "weight", "ap_hi", "ap_lo","cholesterol","gluc","smoke","alco","active"]]

outputDF = df[["cardio"]]

result = forward_selected(inputDF, outputDF, df)
print(result.model.formula)

print("Selected Model:", result.model.formula)

print("Selected Pseudo R-squqre:", result.prsquared)

print(result.summary())
def backward_selected(inputDF, outputDF, df):

    highestScore = 0.0

    selected = []

    outputColName = str(outputDF.columns[0])

    for x in range(2, len(inputDF.columns)+1):

        model = sfs(LinearRegression(),k_features=x,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')

        model.fit(inputDF,outputDF)

        predictors = model.k_feature_names_

        formula = "{} ~ {}".format(outputColName, ' + '.join(predictors))

        candidateModel = sm.logit(formula, df).fit()

        

        if(candidateModel.prsquared > highestScore):

            selected = predictors

            highestScore = candidateModel.prsquared

            print("score:", candidateModel.prsquared)

    

    formula = "{} ~ {}".format(outputColName,' + '.join(selected))

    model = sm.logit(formula, df).fit()

    return model

      
inputDF = df[["age", "gender", "height", "weight", "ap_hi", "ap_lo","cholesterol","gluc","smoke","alco","active"]]

outputDF = df[["cardio"]]

result = backward_selected(inputDF, outputDF, df)
print("Selected Model:", result.model.formula)

print("Selected Pseudo R-squqre:", result.prsquared)

print(result.summary())
#Model without Gender

formula = "cardio ~ age + height + weight + ap_hi + ap_lo + cholesterol + gluc + smoke + alco + active"

model = sm.logit(formula, df).fit()

print(model.summary())
!pip install pygam
!pip install pydotplus
from IPython.display import Image 

import pydotplus  
from sklearn.tree import DecisionTreeClassifier



inputDF_DC = df[["age", "gender", "height", "weight", "ap_hi", "ap_lo","cholesterol","gluc","smoke","alco","active"]]

outputDF_DC = df[["cardio"]]



X_train_DC, X_test_DC, y_train_DC, y_test_DC = train_test_split(inputDF_DC, outputDF_DC, test_size=0.30)



DTR = DecisionTreeClassifier()

clfDC = DTR.fit(X_train_DC, y_train_DC)

clfDC = DecisionTreeClassifier(random_state = 0)



clfDC = clfDC.fit(X_train_DC,y_train_DC)
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut

from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.preprocessing import PolynomialFeatures

from patsy import dmatrices
#Create Input

inputDF = df[["age", "gender", "height", "weight", "ap_hi", "ap_lo","cholesterol","gluc","smoke","alco","active"]]

outputDF = df[["cardio"]]
# create dummy variables, and their interactions

outputDFInteraction, inputDFInteraction = dmatrices('cardio ~ age*cholesterol*weight*ap_hi', df, return_type="dataframe")

# flatten y into a 1-D array so scikit-learn can understand it

outputDFInteraction = np.ravel(outputDFInteraction)
logisticRegr = LogisticRegression()

kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDFInteraction)

score = cross_val_score(logisticRegr, inputDFInteraction, outputDFInteraction, scoring="accuracy", cv = kf)

print("Score of cardio ~ age*cholesterol*weight*ap_hi:", score.mean())
#Create Input

inputDFLog = df[["age", "ap_hi","cardio"]]

inputDFLog["age"] = np.log(inputDFLog["age"])

inputDFLog["ap_hi"] = np.log(inputDFLog["ap_hi"])

inputDFLog = inputDFLog.dropna()



outputDFLog = inputDFLog["cardio"]

inputDFLog = inputDFLog.drop(columns=['cardio'])
logisticRegr = LogisticRegression()

kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDFLog)

score = cross_val_score(logisticRegr, inputDFLog, outputDFLog, scoring="accuracy", cv = kf)

print("Score of cardio ~ np.log(age)+np.log(ap_hi):", score.mean())
#Create Input

inputDF = df[["age", "gender", "height", "weight", "ap_hi", "ap_lo","cholesterol","gluc","smoke","alco","active"]]

outputDF = df[["cardio"]]

logisticRegr = LogisticRegression()

kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF)

score = cross_val_score(logisticRegr, inputDF, outputDF, scoring="accuracy", cv = kf)

print("Score of cardio ~ age + height + weight + ap_hi + ap_lo + cholesterol + gluc + smoke + alco + active:", score.mean())
y_pred_DC = clfDC.predict(X_test_DC)



print("Accuracy:",metrics.accuracy_score(y_test_DC, y_pred_DC))