# import libraries
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('../input/finaldata1/train_final.csv')
test_df = pd.read_csv('../input/finaldata2/test_final.csv')

train_df.head()
days = pd.get_dummies(train_df.DayOfWeek)
district = pd.get_dummies(train_df.PdDistrict)
hours=pd.get_dummies(train_df.Hours)
years=pd.get_dummies(train_df.Years)
months=pd.get_dummies(train_df.Months)
house=pd.get_dummies(train_df.House)
#n1 is the final number of features
n1=70
n2=n1-1
features = [x for x in range(0,n1)]
len(features)
#creating a new feature from latitude and longitude by multiplying X and Y
train_df["XY"]=-train_df['X']*train_df['Y']
test_df["XY"]=-test_df['X']*test_df['Y']
train = pd.concat([hours, days, district,months, years,house, train_df['XY'], train_df['Category']], axis=1)
train.shape
train.columns=features
train.head()
train_sample, test_sample = train_test_split(train, train_size=.75)

X1=train_sample[[i for i in range(n2)]]
Y1=train_sample[n2]

X2=test_sample[[i for i in range(n2)]]
Y2=test_sample[n2]

Bernoulli = BernoulliNB()
Bernoulli.fit(X1,Y1)
predict_sample = np.array(Bernoulli.predict_proba(X2))
log_loss(Y2, predict_sample) 

features = [x for x in range(0,n2)]
len(features)
#vectorization of the features for the test set
days = pd.get_dummies(test_df.DayOfWeek)
district = pd.get_dummies(test_df.PdDistrict)
hours=pd.get_dummies(test_df.Hours)
years=pd.get_dummies(test_df.Years)
months=pd.get_dummies(test_df.Months)
house=pd.get_dummies(test_df.House)
#combining the features 
test = pd.concat([hours, days, district,months, years,house, test_df['XY']], axis=1)
test.shape
test.columns=features
X_train=train[[i for i in range (n2)]]
Y_train=train_df['Category']
X_test=test
X_train.head()
X_test.head()
Y_train.head()
X_test.shape
#prediction of probabilities for the test set
Bernoulli = BernoulliNB()
Bernoulli.fit(X_train, Y_train)
predictions = np.array(Bernoulli.predict_proba(X_test))

header = ["ARSON", "ASSAULT", "BAD CHECKS", "BRIBERY", "BURGLARY", "DISORDERLY CONDUCT",\
               "DRIVING UNDER THE INFLUENCE","DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION",\
               "FAMILY OFFENSES","FORGERY/COUNTERFEITING","FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT",\
               "LIQUOR LAWS","LOITERING","MISSING PERSON","NON-CRIMINAL","OTHER OFFENSES",\
               "PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE","ROBBERY","RUNAWAY",\
               "SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE","STOLEN PROPERTY","SUICIDE",\
               "SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT","WARRANTS","WEAPON LAWS"]

Y_pred_df=pd.DataFrame(data=predictions,columns=header)
Y_pred_df.head()
#submission file
#Y_pred_df.to_csv("submission_file3.csv",index_label = 'Id')


