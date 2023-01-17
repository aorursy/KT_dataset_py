#https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction/tasks?taskId=2055

#Task Details
#Your client is an Insurance company that has provided Health Insurance to its customers now they 
#need your help in building a model to predict whether the policyholders (customers) from past year will
#also be interested in Vehicle Insurance provided by the company.

#For example, you may pay a premium of Rs. 5000 each year for a health insurance cover of Rs. 200,000/- 
#so that if, God forbid, you fall ill and need to be hospitalised in that year, the insurance provider 
#company will bear the cost of hospitalisation etc. for upto Rs. 200,000. Now if you are wondering how 
#can company bear such high hospitalisation cost when it charges a premium of only Rs. 5000/-, that is 
#where the concept of probabilities comes in picture. For example, like you, there may be 100 customers 
#who would be paying a premium of Rs. 5000 every year, but only a few of them (say 2-3) would get 
#hospitalised that year and not everyone. This way everyone shares the risk of everyone else.

#Just like medical insurance, there is vehicle insurance where every year customer needs to pay a 
#premium of certain amount to insurance provider company so that in case of unfortunate accident by 
#the vehicle, the insurance provider company will provide a compensation (called ‘sum assured’) to the 
#customer.

#Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely 
#helpful for the company because it can then accordingly plan its communication strategy to reach out 
#to those customers and optimise its business model and revenue.

#Now, in order to predict, whether the customer would be interested in Vehicle insurance, you have 
#information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), 
#Policy (Premium, sourcing channel) etc.
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
train = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")
test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
train.head(10)
#map categorical data to numerical data
def numberise(series):
    if( series.infer_objects().dtype != np.dtype('O')):
        pass
    else:
        print("Updating: {}".format(series.name))
        labels = series.unique()
        dicmap = {}
        for level in labels:
            dicmap.update({level: np.where(labels==level)[0][0]})
        print("\t", dicmap)
        series.replace(dicmap, inplace=True)

for col in train.columns:
    numberise(train[col])
#Since Response is a binary categorical predictor, a logistic model is appropriate. It is assumed that
#all X have a linear relationship with logit(y), however in our case since our range is {0,1}
#linear relationship with y is sufficent
train.corr()
#Large correlation in Response given by X
X =['Age', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
reduc = train[X]
print("Covariance Matrix:\n", reduc.cov())
print('\n--------------------------------------------------------------')
print("Correlation Matrix:\n",reduc.corr())
#Analysing data imbalance
print(train['Response'].value_counts())
sns.countplot(train['Response'])
plt.show();
#data needs balancing for logistic training fit
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
y = train['Response']
X = train.drop(['Response'], axis=1)
print('Original dataset shape %s' % Counter(y))

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

#stitch back together for statsmodels
diction = {}
for col in X_res.columns:
    diction.update({col: X_res[col]})
diction.update({y_res.name: y_res})

data = pd.DataFrame(diction)

from statsmodels.formula.api import logit
from sklearn.metrics import roc_auc_score
# Logit Model
insurance_model = logit("Response ~ Age + Previously_Insured + Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel", data).fit()

y_actual = data['Response']
y_score = insurance_model.predict(data.drop(['Response'], axis=1))

print(insurance_model.summary())
print('ROC AUC:  ', roc_auc_score(y_actual, y_score))
print('Confusion Matrix:\n',insurance_model.pred_table())

#Match mapping as train data for test data
gender = {'Male': 0, 'Female': 1}
vehicle_age = {'> 2 Years': 0, '1-2 Year': 1, '< 1 Year': 2}
vehicle_damage = {'Yes': 0, 'No': 1}

test['Gender'].replace(gender, inplace=True)
test['Vehicle_Age'].replace(vehicle_age, inplace=True)
test['Vehicle_Damage'].replace(vehicle_damage, inplace=True)
#calculate results
predicted = insurance_model.predict(test)
predicted_choice = (predicted > 0.5).astype(int)
results = {'id': test['id'], 'Response': predicted_choice}
Results = pd.DataFrame(results)
Results.to_csv('results.csv', index=False)