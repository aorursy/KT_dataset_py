import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt
#train = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")

#train.head()
train_dir = "../input/predict-volcanic-eruptions-ingv-oe/train/"

test_dir = "../input/predict-volcanic-eruptions-ingv-oe/test/"    
def read_csv(index):

    train1 = pd.read_csv(train_dir + str(train.segment_id.iloc[index]) + ".csv")



    train1['timetoerupt'] = train.time_to_eruption.iloc[index]

    

    for feat in train1.drop('timetoerupt',1).columns:

        train1[feat] = train1[feat].mean()

    

    train1 = train1.sample(1)

           

    return (train1)
#data = pd.DataFrame()



#for idx in range(train.shape[0]):

#    df = read_csv(idx)

    

#    data=pd.concat([df,data])
# load training data

data = pd.read_csv("../input/volcano-eruption-data/data.csv")

data.head()
# this will confirm whether we have read all the files or not

data.shape
data.isnull().sum()
# replace null values with the mean value

for feat in data:

    data[feat] = data[feat].replace(np.nan, data[feat].mean())
data.isnull().sum()
from sklearn import linear_model

from sklearn.linear_model import LinearRegression



import statsmodels.api as sm



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, data.timetoerupt, test_size=0.2, random_state=42)
X_train.drop('timetoerupt',1,inplace = True)



# Add a constant to get an intercept

X_train_sm = sm.add_constant(X_train)



# train the model

lr = sm.OLS(y_train, X_train_sm).fit()
print(lr.summary())
X_test.drop('timetoerupt',1,inplace = True)



# Add a constant to get an intercept

X_test_sm = sm.add_constant(X_test)



# prediction on training dataset

y_test_pred = lr.predict(X_test_sm)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
r_squared = r2_score(y_test_pred, y_test)

r_squared
sub = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv")

sub.head()
def read_csv(index):

    

    test1 = pd.read_csv(test_dir + str(sub.segment_id.iloc[index]) + ".csv")



    for feat in test1.columns:

        test1[feat] = test1[feat].mean()

    

    test1 = test1.sample(1)

           

    return (test1)
#test = pd.DataFrame()



#for idx in range(sub.shape[0]):

#    df = read_csv(idx)

    

#    test = pd.concat([df,test])
# I have ran the steps mentioned above and saved the file, loading it now

test = pd.read_csv("../input/volcano-eruption-data/test.csv")

test.head()
# again verify whether all the files were read correctly or not

test.shape
test.isnull().sum()
# same as we did for the training data

for feat in test:

    test[feat] = test[feat].replace(np.nan, test[feat].mean())
#test.to_csv('test.csv',index=False)
# Add a constant to get an intercept

test_sm = sm.add_constant(test)



# prediction on test dataset

predictions = lr.predict(test_sm)



sub['time_to_eruption'] = predictions
sub.head()
# submission file

sub.to_csv('submission.csv',index=False)