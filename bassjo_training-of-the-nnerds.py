import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

%matplotlib inline
training_filepath='../input/train_V2.csv'
testing_filepath='../input/test_V2.csv'

full_training=pd.read_csv(training_filepath)
full_testing=pd.read_csv(testing_filepath)
def pre_process(df):
    
    df=df.dropna()
    match_types=pd.get_dummies(df['matchType'])
    temp=df.join(match_types)
    
    del[temp['matchType']]
    del[temp['Id']]
    del[temp['groupId']]
    del[temp['matchId']]
    
    return(temp)

training=pre_process(full_training)
testing=pre_process(full_testing)
chicken_dinner=training['winPlacePerc']
match_data=training.drop('winPlacePerc',axis=1)

var_list=list(match_data.columns)
from sklearn.model_selection import train_test_split
train_data, test_data, train_placing,test_placing = train_test_split(match_data,chicken_dinner,test_size=.25,random_state=69)
clf = MLPRegressor()
clf.fit(match_data,chicken_dinner)
#predict = clf.predict(test_data)

#errors = abs(predict-test_placing)

#print('Mean Absolute Error:',round(np.mean(errors),4))

predict = clf.predict(testing)

final_sub=full_testing[['Id']]

final_sub['winPlacePerc']=predict
final_sub.head()

final_sub.to_csv('submission.csv',index=False)
final_sub.head()
