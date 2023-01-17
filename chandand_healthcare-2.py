import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import KFold,StratifiedKFold
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
from keras.layers import Dense,Dropout
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
path='../input/av-healthcare-analytics-ii/healthcare'
train_df=pd.read_csv(os.path.join(path,'train_data.csv'))
test_df=pd.read_csv(os.path.join(path,'test_data.csv'))
submission_df=pd.read_csv(os.path.join(path,'sample_sub.csv'))
train_df['Stay'].value_counts()
# train_df['City_Code_Patient'].fillna(-99,inplace=True)
# train_df['Bed Grade'].fillna(5,inplace=True)

train_df=train_df.drop_duplicates(subset=[ele for ele in list(train_df.columns) if ele not in ['case_id']])
#Adding more Features
combine_set=pd.concat([train_df,test_df],axis=0)
combine_set['City_Code_Patient'].fillna(-99,inplace=True)
combine_set['Bed Grade'].fillna(-99,inplace=True)
combine_set['Unique_Hospital_per_patient']=combine_set.groupby(['patientid'])['Hospital_code'].transform('nunique')
combine_set['Unique_patient_per_hospital']=combine_set.groupby(['Hospital_code'])['patientid'].transform('nunique')
combine_set['Unique_patient_per_Department']=combine_set.groupby(['Department'])['patientid'].transform('nunique')
combine_set['Unique_patient_per_Ward']=combine_set.groupby(['Ward_Type'])['patientid'].transform('nunique')
combine_set['Unique_Ward_per_patient']=combine_set.groupby(['patientid'])['Ward_Type'].transform('nunique')
combine_set['Unique_Hospital_per_ward']=combine_set.groupby(['Ward_Type'])['Hospital_code'].transform('nunique')
combine_set['Unique_Hospital_per_city']=combine_set.groupby(['City_Code_Hospital'])['Hospital_code'].transform('nunique')
combine_set['Unique_patients_per_city']=combine_set.groupby(['City_Code_Patient'])['patientid'].transform('nunique')

#creating Aggregate columns
combine_set['Total_available_rooms_per_hospital_per_department']=combine_set.groupby(['Hospital_code','Department'])['Available Extra Rooms in Hospital'].transform('sum')
combine_set['Total_deposit_paid_by_patient_in_each_hospital']=combine_set.groupby(['Hospital_code','patientid'])['Admission_Deposit'].transform('sum')
combine_set['Total_number_visitors_per_patient']=combine_set.groupby(['patientid'])['Visitors with Patient'].transform('sum')
combine_set['Total_Amount_paid_per_Bed_grade_during_stay']=combine_set.groupby(['patientid','Bed Grade'])['Admission_Deposit'].transform('sum')
combine_set['Total_number_of_visitors_per_ward_during_stay']=combine_set.groupby(['patientid','Ward_Type'])['Visitors with Patient'].transform('sum')
combine_set['Number_of_times_patient_joined_with_same_reason']=combine_set.groupby(['patientid','Type of Admission','Severity of Illness'])['Hospital_code'].transform('count')
combine_set.head(5)

#Encoding Categorical Columns
le=LabelEncoder()
for col in combine_set.select_dtypes(include='object').columns:
    if col not in ['Age','Stay']:
#         fe=combine_set.groupby([col]).size()/len(combine_set)
#         combine_set[col]=combine_set[col].apply(lambda x: fe[x])
        df=pd.get_dummies(combine_set[col],drop_first=True)
        combine_set=pd.concat([combine_set,df],axis=1).drop([col],axis=1)
          
    elif col!='Stay':
        combine_set[col]=le.fit_transform(combine_set[col].astype(str))
    else:
        pass
        
combine_set.head(5)        
X=combine_set[combine_set['Stay'].isnull()==False].drop(['case_id','Stay','patientid'],axis=1)
y=le.fit_transform(combine_set[combine_set['Stay'].isnull()==False]['Stay'])
y=pd.DataFrame(y,columns=['Stay'])
X_main_test=combine_set[combine_set['Stay'].isnull()==True].drop(['case_id','Stay','patientid'],axis=1)
y_hat=to_categorical(y)
y_hat=pd.DataFrame(y_hat)
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
X=pd.DataFrame(X)
sc_X_main=StandardScaler()
X_main_test=sc_X_main.fit_transform(X_main_test)
X.head(5)

# for col in X.select_dtypes(exclude='float64').columns:
#     X[col]=X[col].astype(int)
# y_total={}
# for i in range(0,11):
#     y_total[i+1]=y_hat[:, i:i+1]
#     y_total[i+1]=pd.DataFrame(y_total[i+1],columns=[i+1])
   
    
X_train,X_val,y_train,y_val=train_test_split(X,y_hat,test_size=0.2,random_state=294)
# classifier=Sequential()

# classifier.add(Dense(512,activation='relu', kernel_initializer='uniform',input_shape=(X_train.shape[1],)))
# classifier.add(Dropout(0.2))
# classifier.add(Dense(256,activation='relu',kernel_initializer='uniform'))
# # classifier.add(Dense(200,activation='relu',kernel_initializer='uniform'))
# classifier.add(Dense(128,activation='relu',kernel_initializer='uniform'))
# # classifier.add(Dense(64,activation='relu',kernel_initializer='uniform'))
# # classifier.add(Dense(32,activation='relu',kernel_initializer='uniform'))
# classifier.add(Dense(11,activation='softmax'))

# classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# callback_lr=ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.3,min_lr=0.00001)
# callback_mc=ModelCheckpoint(filepath='model_repli.hdf5',monitor='val_accuracy',save_best_only=True,mode='max')

# classifier.fit(X_train,y_train,epochs=50,batch_size=32,validation_data=(X_val,y_val),callbacks=[callback_lr,callback_mc])

# classifier=load_model('model_repli.hdf5')
# pred_val=classifier.predict(X_val)

# preds=classifier.predict(X_main_test)
# classifier=Sequential()

# classifier.add(Dense(512,activation='relu', kernel_initializer='uniform',input_shape=(X_train.shape[1],)))
# classifier.add(Dropout(0.2))
# classifier.add(Dense(256,activation='relu',kernel_initializer='uniform'))
# # classifier.add(Dense(200,activation='relu',kernel_initializer='uniform'))
# classifier.add(Dense(128,activation='relu',kernel_initializer='uniform'))
# classifier.add(Dense(128,activation='relu',kernel_initializer='uniform'))
# # classifier.add(Dense(32,activation='relu',kernel_initializer='uniform'))
# classifier.add(Dense(11,activation='softmax'))

# classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# callback_lr=ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.5,min_lr=0.00001)
# callback_mc=ModelCheckpoint(filepath='model_reli2.hdf5',monitor='val_accuracy',save_best_only=True,mode='max')

# classifier.fit(X_train,y_train,epochs=50,batch_size=32,validation_data=(X_val,y_val),callbacks=[callback_lr,callback_mc])

# classifier=load_model('model_reli2.hdf5')
# preds2_val=classifier.pedict(X_val)

# preds2=classifier.predict(X_main_test)


total_val_preds=pd.concat([pd.DataFrame(pred_val,columns=[col for col in range(0,11)]),pd.DataFrame(preds2_val,columns=[col for col in range(11,22)])],axis=1)
total_test_preds=pd.concat([pd.DataFrame(preds,columns=[col for col in range(0,11)]),pd.DataFrame(preds2,columns=[col for col in range(11,22)])],axis=1)


classifier=Sequential()

classifier.add(Dense(128,activation='relu', kernel_initializer='uniform',input_shape=(total_val_preds.shape[1],)))
classifier.add(Dropout(0.1))
classifier.add(Dense(64,activation='relu',kernel_initializer='uniform'))
# classifier.add(Dense(200,activation='relu',kernel_initializer='uniform'))
# classifier.add(Dense(128,activation='relu',kernel_initializer='uniform'))
# classifier.add(Dense(128,activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(64,activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(11,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
callback_lr=ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.00001)
callback_mc=ModelCheckpoint(filepath='model_test.hdf5',monitor='accuracy',save_best_only=True,mode='max')

classifier.fit(total_val_preds,y_val,epochs=50,batch_size=16,callbacks=[callback_lr,callback_mc])

classifier=load_model('model_test.hdf5')

final_preds=classifier.predict(total_test_preds)

final_preds=pd.DataFrame(final_preds).idxmax(axis=1)





perm = PermutationImportance(lg,random_state=294).fit(X_val, y_val)
eli5.show_weights(perm,feature_names=X_val.columns.tolist())

# class_weight=compute_class_weight('balanced',np.unique(y['Stay']), y['Stay'])
# class_weight=dict(zip(np.unique(y['Stay']),class_weight))

kf=KFold(n_splits=10,shuffle=True,random_state=2020)
# sc_X=StandardScaler()
# X=pd.DataFrame(sc_X.fit_transform(X))
preds={}
acc_score=0

    
for i,(train_idx,val_idx) in enumerate(kf.split(X)):    

    X_train, y_train = X.iloc[train_idx,:], y_hat.iloc[train_idx]

    X_val, y_val = X.iloc[val_idx, :], y_hat.iloc[val_idx]
    

    print('\nFold: {}\n'.format(i+1))
    #12,0.8,1000
    lg=LGBMClassifier(boosting_type='gbdt',learning_rate=0.08,depth=12,objective='multiclass',n_estimators=1000,num_class=11,
                     metric='multi_error',colsample_bytree=0.5,reg_alpha=2,reg_lambda=2,random_state=294,n_jobs=-1)

#     X_train,y_train=SMOTETomek(random_state=294).fit_resample(X_train,y_train)
    lg.fit(X_train,y_train)

    print(accuracy_score(y_val,lg.predict(X_val)))

    acc_score+=accuracy_score(y_val,lg.predict(X_val))
    
    preds[i+1]=lg.predict(X_main_test)
#     classifier=Sequential()

#     classifier.add(Dense(512,activation='relu', kernel_initializer='uniform',input_shape=(X_train.shape[1],)))
#     classifier.add(Dropout(0.2))
#     classifier.add(Dense(256,activation='relu',kernel_initializer='uniform'))
#     # classifier.add(Dense(200,activation='relu',kernel_initializer='uniform'))
#     classifier.add(Dense(128,activation='relu',kernel_initializer='uniform'))
#     # classifier.add(Dense(64,activation='relu',kernel_initializer='uniform'))
#     # classifier.add(Dense(32,activation='relu',kernel_initializer='uniform'))
#     classifier.add(Dense(11,activation='softmax'))

#     classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#     callback_lr=ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=0.00001)
#     callback_mc=ModelCheckpoint(filepath='model_'+str(i+1)+'.hdf5',monitor='val_accuracy',save_best_only=True,mode='max')

#     classifier.fit(X_train,y_train,epochs=30,batch_size=32,validation_data=(X_val,y_val),callbacks=[callback_lr,callback_mc])
    
#     classifier=load_model('model_'+str(i+1)+'.hdf5')

#     preds+=classifier.predict(X_main_test)
    

kf=KFold(n_splits=10,shuffle=True,random_state=2019)
# sc_X=StandardScaler()
# X=pd.DataFrame(sc_X.fit_transform(X))
preds=0
acc_score=0


    
for i,(train_idx,val_idx) in enumerate(kf.split(X)):    

    X_train, y_train = X.iloc[train_idx,:], y_hat.iloc[train_idx]

    X_val, y_val = X.iloc[val_idx, :], y_hat.iloc[val_idx]
    

    print('\nFold: {}\n'.format(i+1))
    #12,0.8,1000
#     lg=LGBMClassifier(boosting_type='gbdt',learning_rate=0.08,depth=12,objective='multiclass',n_estimators=1000,num_class=11,
#                      metric='multi_error',colsample_bytree=0.5,reg_alpha=2,reg_lambda=2,random_state=294,n_jobs=-1)

# #     X_train,y_train=SMOTETomek(random_state=294).fit_resample(X_train,y_train)
#     lg.fit(X_train,y_train)

#     print(accuracy_score(y_val,lg.predict(X_val)))

#     acc_score+=accuracy_score(y_val,lg.predict(X_val))
    classifier=Sequential()

    classifier.add(Dense(512,activation='relu', kernel_initializer='uniform',input_shape=(X_train.shape[1],)))
    classifier.add(Dropout(0.1))
#     classifier.add(Dense(256,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(200,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.05))
    classifier.add(Dense(128,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(64,activation='relu',kernel_initializer='uniform'))
#     classifier.add(Dense(32,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(11,activation='softmax'))

    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    callback_lr=ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.5,min_lr=0.00001)
    callback_mc=ModelCheckpoint(filepath='model_'+str(i+1)+'.hdf5',monitor='val_accuracy',save_best_only=True,mode='max')

    classifier.fit(X_train,y_train,epochs=30,batch_size=32,validation_data=(X_val,y_val),callbacks=[callback_lr,callback_mc])
    
    classifier=load_model('model_'+str(i+1)+'.hdf5')

    preds+=classifier.predict(X_main_test)
  
    
preds=preds/10
preds=pd.DataFrame(preds).idxmax(axis=1)

# d = pd.DataFrame()
# for i in range(1, 11):
#     d = pd.concat([d,pd.DataFrame(preds[i])],axis=1)
# d.columns=['1','2','3','4','5','6','7','8','9','10']
# re = d.mode(axis=1)[0]
submission_df['Stay']=le.inverse_transform(preds.astype(int))
submission_df.to_csv('/kaggle/working/main_test.csv',index=False)
submission_df.head(5)
# le.inverse_transform(re.astype(int))