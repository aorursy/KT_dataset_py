# import libraries

import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder



pd.set_option('display.max_column',None)

pd.set_option('display.max_row',None)
# import data - turn on 'Internet' option in sidebar

audit_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/credit_risk/training_set_labels.csv" )
oenc=OneHotEncoder(drop='first')# drop first is to avoid multicollinearity(It means repetition of data)

che_status_enc=oenc.fit_transform(audit_data[['checking_status']])
che_status_enc # the resulted matrix is sparse matrix so convert to numpy array 
che_status_enc=che_status_enc.toarray() # converting to numpy array
che_status_enc=pd.DataFrame(che_status_enc) # converting to dataframe

che_status_enc=che_status_enc.add_prefix('checking_status')
audit_data=pd.concat([audit_data,che_status_enc],axis=1) # append to original dataframe
audit_data.tail()#you can view new column at rightmost end 
audit_data=audit_data.drop('checking_status',axis=1)# drop the original column to avoid multicollinearity
oenc=OneHotEncoder()

multiple_enc=oenc.fit_transform(audit_data[['credit_history','purpose','employment']])# onehotencoding on 3 features



multiple_enc=multiple_enc.toarray()



#oenc.get_feature_names() use this method to get column names

multiple_enc=pd.DataFrame(multiple_enc,columns=oenc.get_feature_names())



audit_data=pd.concat([audit_data,multiple_enc],axis=1) # append to original dataframe
audit_data.head(5)