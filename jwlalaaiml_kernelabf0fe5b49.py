# Python code using AUML classification - Binomial H2O

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import  os

import seaborn as sns
df=pd.read_csv('../input/KaggleV2-May-2016.csv')
#df.isnull().sum()
def get_day(x):

    return x.date()
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

df['DaysBeforeApp'] = ((df.AppointmentDay.apply(get_day) - df.ScheduledDay.apply(get_day)).astype('timedelta64[D]')).astype(int)

df.drop(df[df['DaysBeforeApp'] < 0].index, inplace=True)

df.drop(df[df['Age'] < 0].index, inplace=True)

df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
df.head(4)
df['PatientId'].astype('int64')

df['prehistory'] = df.sort_values(by = ['PatientId','ScheduledDay']).groupby(['PatientId']).cumcount()

df.loc[:, 'MissedAppointments'] = df.sort_values(['ScheduledDay']).groupby(['PatientId'])['No-show'].cumsum()

#df['Previousprob'] = (df[df['prehistory'] > 0].sort_values(['PatientId', 'ScheduledDay']).groupby(['PatientId'])['Noshow'].cumsum() / df[df['prehistory'] > 0]['prehistory'])
dfbin=df
dfbin.head(4)
dfbin['day_of_week'] = dfbin['AppointmentDay'].dt.day_name()

dfbin['Age_cat']=pd.cut(dfbin['Age'],bins=[0,3,12,18,31,51,80,120], labels=['kid','small','Teen','Mid age','above mid','old age','Rare age'])

dfbin['Day_cat']=pd.cut(dfbin['DaysBeforeApp'],bins=[-1,0,1,3,7,15,30,60,90,200], labels=['sameday','oneday','critcal','oneweek','2 weeks','1 month','2 months','3 months','3 months above'])

from sklearn.preprocessing import LabelEncoder

encodervar = LabelEncoder()

dfbin['Neighbourhood_enc'] = encodervar.fit_transform(dfbin['Neighbourhood'])
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000) 

dfbin.head(4)
dfdummy=dfbin
dfdummy = dfdummy.join(pd.get_dummies(dfdummy['day_of_week']))

dfdummy = dfdummy.join(pd.get_dummies(dfdummy['Age_cat']))

dfdummy = dfdummy.join(pd.get_dummies(dfdummy['Day_cat']))
dfdummy.head(4)
df2=dfdummy
df2=df2.drop(['PatientId','AppointmentID','ScheduledDay','AppointmentDay','Age','Neighbourhood','DaysBeforeApp','day_of_week','Age_cat','Day_cat'], axis=1)
df2=df2.drop(['Saturday','Rare age','3 months above'], axis=1)
df2.head(4)
from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import classification_report
Y=df2['No-show']

X=df2.drop(['No-show'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split( 

          X, Y, test_size = 0.3, random_state = 100)

import h2o
import h2o
h2o.init()
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
hf = h2o.H2OFrame(df)
hf.head()
hf.shape
X = hf.drop('No-show', axis = 1)
X.shape
X.head()
y = hf['No-show']
y.shape
y.head()
y['No-show'] = y['No-show'].asfactor()
y.head()
splits = hf.split_frame(ratios=[0.7, 0.15], seed=1)  



train = splits[0]

valid = splits[1]

test = splits[2]
print(train.nrow)

print(valid.nrow)

print(test.nrow)


y = 'No-show'

x = list(hf.columns)
x
param = {

      "ntrees" : 100

    , "max_depth" : 10

    , "learn_rate" : 0.02

    , "sample_rate" : 0.7

    , "col_sample_rate_per_tree" : 0.9

    , "min_rows" : 5

    , "seed": 4241

    , "score_tree_interval": 100

}
# Import H2O GBM:

from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm_fit1 = H2OGradientBoostingEstimator(model_id='gbm_fit1', seed=1)

gbm_fit1.train(x=x, y=y, training_frame=train)
gbm_perf1 = gbm_fit1.model_performance(test)
print(gbm_perf1.auc)
gbm_perf1
y_pred = gbm_fit1.predict(test)
y_pred
# Import H2O GLM:

from h2o.estimators.glm import H2OGeneralizedLinearEstimator
# Initialize the GLM estimator:

# Similar to R's glm() and H2O's R GLM, H2O's GLM has the "family" argument



glm_fitc = H2OGeneralizedLinearEstimator(family='binomial', model_id='glm_fitc')
glm_fitc.train(x=x, y=y, training_frame=train)
glm_perf1 = glm_fitc.model_performance(test)
print(glm_perf1)
glm_fit2 = H2OGeneralizedLinearEstimator(family='binomial', model_id='glm_fit2', lambda_search=True)

glm_fit2.train(x=x, y=y, training_frame=train, validation_frame=valid)
glm_perf2 = glm_fit2.model_performance(test)
print(glm_perf2)
glm_perf2.plot()
print(glm_perf2.precision())
test
y_pred = glm_fit2.predict(test)
print(y_pred.categories)