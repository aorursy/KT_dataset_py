import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test_bqCt9Pv.csv')

train.head()
from datetime import date,datetime



def calculate_age(born):

    

    dd,mm,yy = born.split('-')

    if int(yy)>19:

        yy = ''.join(['19',yy])

    else:

        yy = ''.join(['20',yy])

    

    born = "-".join([dd,mm,yy])

    born = datetime.strptime(born, '%d-%m-%Y').date()

    today = date.today()

    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
X = train.copy()

X_test = test.copy()



X['age'] = X['Date.of.Birth'].apply(lambda x: calculate_age(x))

X_test['age'] = X_test['Date.of.Birth'].apply(lambda x: calculate_age(x))
def get_months(val):

    yr,mon = val.split()

    yr = int(yr[:-3])*12

    mon = int(mon[:-3])

    tot = yr+mon

    return tot
X['AVERAGE.ACCT.AGE'] = X['AVERAGE.ACCT.AGE'].apply(lambda x: get_months(x))

X['CREDIT.HISTORY.LENGTH'] = X['CREDIT.HISTORY.LENGTH'].apply(lambda x: get_months(x))



X_test['AVERAGE.ACCT.AGE'] = X_test['AVERAGE.ACCT.AGE'].apply(lambda x: get_months(x))

X_test['CREDIT.HISTORY.LENGTH'] = X_test['CREDIT.HISTORY.LENGTH'].apply(lambda x: get_months(x))
labels = ['UniqueID','supplier_id','manufacturer_id','Current_pincode_ID',\

         'Date.of.Birth','Employment.Type','DisbursalDate','Employee_code_ID',\

         'MobileNo_Avl_Flag','Aadhar_flag','PAN_flag','VoterID_flag','Driving_flag',\

         'Passport_flag','PERFORM_CNS.SCORE.DESCRIPTION','branch_id']



X = X.drop(labels=labels,axis=1)

X_test = X_test.drop(labels=labels,axis=1)



X.head()
from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
y=X['loan_default'].values



X = X.drop(labels=['loan_default'],axis=1)



continuous_cols = ['disbursed_amount', 'asset_cost', 'ltv',

       'PERFORM_CNS.SCORE', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',

       'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',

       'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',

       'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',

       'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',

       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',

       'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES','age']



categorical_cols = ['State_ID']



mapper = DataFrameMapper(  

    [([continuous_col], StandardScaler()) for continuous_col in continuous_cols] +

    [([categorical_col], OneHotEncoder()) for categorical_col in categorical_cols])



pipe = Pipeline([('mapper',mapper)])



pipe.fit(X)
X = pipe.transform(X)

X_test = pipe.transform(X_test)
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.1,random_state=42)
# from sklearn.linear_model import LogisticRegression
# lrc = LogisticRegression(random_state = 42,n_jobs=-1,max_iter=1000)
# lrc.fit(X_train,y_train)
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
# y_pred = lrc.predict_proba(X_val)
# y_pred = y_pred[:,0]
# def best_score(y_val,y_pred):

#     th = []

#     scores =[]

#     for thresh in np.arange(0.1, 0.601, 0.01):

#         thresh = np.round(thresh,2)

#         th.append(thresh)

#         score = f1_score(y_val,(y_pred>thresh).astype(int))

#         scores.append(score)

#     return np.max(scores),th[(np.argmax(scores))]

        

    
# score,threshold = best_score(y_val,y_pred)



# print(score)

# print(threshold)
# from keras import backend as K



# def f1(y_true, y_pred):

#     def recall(y_true, y_pred):

#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

#         recall = true_positives / (possible_positives + K.epsilon())

#         return recall



#     def precision(y_true, y_pred):

#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

#         precision = true_positives / (predicted_positives + K.epsilon())

#         return precision

    

#     precision = precision(y_true, y_pred)

#     recall = recall(y_true, y_pred)

#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
# X.shape
# from keras.models import Sequential

# from keras.layers import Dense, Dropout
# model = Sequential()

# model.add(Dense(256,input_dim=128,activation='relu',))

# model.add((Dropout(0.4)))

# model.add(Dense(64,activation='relu'))

# model.add(Dense(32,activation='relu'))

# model.add(Dropout(rate=0.4))

# model.add(Dense(1,activation='sigmoid'))



# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', f1])

# model.summary()
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



# checkpoints = ModelCheckpoint('model.h5',monitor='val_f1',mode='max',save_best_only='True',verbose=True)

# reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)
# batch_size = 128

# epochs = 10
# history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 

#                     validation_data=[X_val, y_val], callbacks=[checkpoints, reduce_lr])
from sklearn.svm import LinearSVC
svc = LinearSVC(C=1.2,class_weight='balanced',random_state=42,max_iter=15000)
svc.fit(X_train,y_train)
pred = svc.predict(X_val)
print(f1_score(y_val,pred))

print(confusion_matrix(y_val,pred))
pred_test = svc.predict(X_test)
test1 = test.copy()



test1['loan_default'] = pred_test

test1 = test1.loc[:,['UniqueID','loan_default']]



test1.head()
test1.to_csv('submission.csv',index=False)