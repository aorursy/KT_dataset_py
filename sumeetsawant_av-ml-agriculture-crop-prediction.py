import pandas as pd 

import numpy as np 



import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline 



from sklearn import ensemble

from sklearn import model_selection

from sklearn import metrics

from sklearn import preprocessing

from sklearn import linear_model





import warnings

warnings.filterwarnings('ignore')
df_train=pd.read_csv('/kaggle/input/av-jantahack-machine-learning-in-agriculture/train.csv')

df_test=pd.read_csv('/kaggle/input/av-jantahack-machine-learning-in-agriculture/test.csv')

df_sample=pd.read_csv('/kaggle/input/av-jantahack-machine-learning-in-agriculture/sample_submission.csv')
df_train.shape
df_train.head()
df_train['Crop_Type'].unique()
# Target Variable 

df_train['Crop_Damage'].unique()


df_train['Soil_Type'].unique()
df_train['Pesticide_Use_Category'].unique()
df_train['Number_Doses_Week'].unique()
df_train['Season'].unique()
color=sns.color_palette()[1]

sns.countplot(data=df_train,x='Crop_Damage',color=color)

sns.distplot(df_train['Estimated_Insects_Count'],bins=10,kde=False)
sns.boxplot(x=df_train['Crop_Damage'],y=df_train['Estimated_Insects_Count'])
df_train['Estimated_Insects_Count'].describe()
df_train.Number_Weeks_Used.unique()
df_train['Number_Weeks_Quit'].unique()
plt.figure(figsize=(12,12));

sns.heatmap(df_train.corr(),annot=True)
df_test['is_test']=1

df_train['is_test']=0



data=pd.concat([df_train,df_test]).reset_index(drop=True)

data.shape



data.isnull().sum()
# Lets one hot encode the categorical variable 



data=pd.get_dummies(data,columns=['Crop_Type','Soil_Type','Season','Pesticide_Use_Category'])

data.shape
# Let build a Simple Lasso Liner regression model to predict the missing values 



# Split Data into Train and Test sets 

null_train=data[data['Number_Weeks_Used'].notnull()]

null_test=data[data['Number_Weeks_Used'].isnull()]





X_train,X_val,y_train,y_val=model_selection.train_test_split(null_train.drop(columns=['ID','is_test','Crop_Damage','Number_Weeks_Used'],axis=1),

                                                                             null_train['Number_Weeks_Used'].values,random_state=7)



#Normalize the features 



for col in ['Estimated_Insects_Count','Number_Weeks_Quit', 'Number_Doses_Week']:

    scaler=preprocessing.StandardScaler()

    scaler.fit(X_train[col].values.reshape(-1,1))

    X_train.loc[:,col]=scaler.transform(X_train[col].values.reshape(-1,1))

    X_val.loc[:,col]=scaler.transform(X_val[col].values.reshape(-1,1))

    null_test.loc[:,col]=scaler.transform(null_test[col].values.reshape(-1,1))

    

# Normalize Y variable 



scaler=preprocessing.StandardScaler()

scaler.fit(y_train.reshape(-1,1))

y_train=scaler.transform(y_train.reshape(-1,1))

y_val=scaler.transform(y_val.reshape(-1,1))



#Define model 



lr=linear_model.LassoCV()

lr.fit(X_train,y_train)



print('The R2 score for Lasso model is {}'.format(lr.score(X_val,y_val)))



null_predict=lr.predict(null_test.drop(columns=['ID','is_test','Crop_Damage','Number_Weeks_Used'],axis=1))



null_test.loc[:,'Number_Weeks_Used']=scaler.inverse_transform(null_predict.reshape(-1,1))
null_train=null_train[['ID','Number_Weeks_Used']]

null_test=null_test[['ID','Number_Weeks_Used']]



data_lasso=pd.concat([null_train,null_test]).reset_index(drop=True)

data_lasso.shape
data=pd.merge(data,data_lasso,how='left',on='ID')

data.drop(axis=1,columns='Number_Weeks_Used_x',inplace=True)



data.loc[data['Number_Weeks_Used_y']<0,'Number_Weeks_Used_y']=0
# Now lets enchance the power of numerical columns using Skleanr polynomial 



#polynomial=preprocessing.PolynomialFeatures(degree=2,include_bias=False)



#polynomial.fit(data[['Estimated_Insects_Count','Number_Weeks_Quit','Number_Doses_Week','Number_Weeks_Used_y']])



#poly=polynomial.transform(data[['Estimated_Insects_Count','Number_Weeks_Quit','Number_Doses_Week','Number_Weeks_Used_y']])

#data=pd.concat([data,pd.DataFrame(poly)],axis=1)



#data.head()
b
# Creating some additional Features 



#data['crop_soil_pest']=data['Crop_Type']+data['Pesticide_Use_Category']+data['Soil_Type']

#data['crop_soil_pest_season']=data['Crop_Type']+data['Pesticide_Use_Category']+data['Soil_Type']+data['Season']



#data['crop_soil']=data['Crop_Type']+data['Soil_Type']

#data['soil_pest']=data['Soil_Type']+data['Pesticide_Use_Category']

#data['crop_pest']=data['Crop_Type']+data['Pesticide_Use_Category']

#data['Pest_season']=data['Pesticide_Use_Category']+data['Season']





#data['Total_pest_used']=data['Number_Doses_Week']*data['Number_Weeks_Used']

#data['Total_pest_quit']=data['Number_Doses_Week']*data['Number_Weeks_Quit']



#data['Estimated_Insects_weeks_Used']=data['Estimated_Insects_Count']*data['Number_Weeks_Used']

#data['Estimated_Insects_Used_1']=data['Estimated_Insects_Count']*data['Total_pest_used']

#data['Estimated_Insects_Used_2']=data['Estimated_Insects_Count']*data['Total_pest_quit']





#data['mean1']=data[['crop_soil_pest','crop_soil_pest_season','crop_soil','soil_pest','crop_pest','Pest_season']].mean(axis=1)

#data['sum1']=data[['crop_soil_pest','crop_soil_pest_season','crop_soil','soil_pest','crop_pest','Pest_season']].sum(axis=1)

#data['std1']=data[['crop_soil_pest','crop_soil_pest_season','crop_soil','soil_pest','crop_pest','Pest_season']].std(axis=1)

#data['kurt1']=data[['crop_soil_pest','crop_soil_pest_season','crop_soil','soil_pest','crop_pest','Pest_season']].kurtosis(axis=1)

#data['median1']=data[['crop_soil_pest','crop_soil_pest_season','crop_soil','soil_pest','crop_pest','Pest_season']].median(axis=1)



#data['mean2']=data[['Total_pest_used','Total_pest_quit','Estimated_Insects_weeks_Used','Estimated_Insects_Used_1','Estimated_Insects_Used_2']].mean(axis=1)

#data['sum2']=data[['Total_pest_used','Total_pest_quit','Estimated_Insects_weeks_Used','Estimated_Insects_Used_1','Estimated_Insects_Used_2']].sum(axis=1)

#data['std2']=data[['Total_pest_used','Total_pest_quit','Estimated_Insects_weeks_Used','Estimated_Insects_Used_1','Estimated_Insects_Used_2']].std(axis=1)

#data['kurt2']=data[['Total_pest_used','Total_pest_quit','Estimated_Insects_weeks_Used','Estimated_Insects_Used_1','Estimated_Insects_Used_2']].kurtosis(axis=1)

#data['median2']=data[['Total_pest_used','Total_pest_quit','Estimated_Insects_weeks_Used','Estimated_Insects_Used_1','Estimated_Insects_Used_2']].median(axis=1)



#data['Estimated_Insects_cut']=pd.cut(data['Estimated_Insects_Count'],bins=4,labels=[0,1,2,3])

#data['Estimated_Insects_cut']=data['Estimated_Insects_cut'].astype(int)









data['Estimated_Insects_Count_square']=data['Estimated_Insects_Count']*data['Estimated_Insects_Count']

data['Number_Weeks_Used_y_square']=data['Number_Weeks_Used_y']*data['Number_Weeks_Used_y']

#data['Number_Doses_Week_square']=data['Number_Doses_Week']*data['Number_Doses_Week']

data['Number_Weeks_Quit_square']=data['Number_Weeks_Quit']*data['Number_Weeks_Quit']



data['Estimated_Insects_doses']=data['Estimated_Insects_Count']*data['Number_Doses_Week']

data['Estimated_Insects_used']=data['Estimated_Insects_Count']*data['Number_Weeks_Used_y']

data['Estimated_Insects_quit']=data['Estimated_Insects_Count']*data['Number_Weeks_Quit']



data['Number_Weeks_Quit_Used']=data['Number_Weeks_Used_y']*data['Number_Weeks_Quit']
#Sepeate the data 



train=data[data['is_test']!=1]

train.drop('is_test',axis=1,inplace=True)



test=data[data['is_test']==1]

test.drop(columns=['Crop_Damage','is_test'],axis=1,inplace=True)



test.shape,train.shape
from sklearn import model_selection



X=train.drop(columns=['Crop_Damage','ID'],axis=1)

y=train['Crop_Damage']



X_train,X_val,y_train,y_val=model_selection.train_test_split(X,y,shuffle=True,stratify=y,random_state=101,test_size=0.1)
b
from sklearn.dummy import DummyClassifier



clf = DummyClassifier(strategy='stratified',random_state=101)



clf.fit(X_train,y_train)

y_pred = clf.predict(X_val)

print('Accuracy of a random classifier is: %.2f%%'%(metrics.accuracy_score(y_val,y_pred)*100))
from xgboost import XGBClassifier



clf = XGBClassifier(objective='multi:softmax',n_jobs=-1, max_depth=6,n_estimators=300,num_class=3)



XGB_prediction=[]



kfold=model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=101)



for train_idx,val_idx in kfold.split(X=X,y=y):

    clf.fit(X.loc[train_idx,:],y[train_idx])

    predict=clf.predict(X.loc[val_idx,:])

    XGB_prediction.append(metrics.accuracy_score(y[val_idx],predict))

    



print('Accuracy of XGBoost Baseline is {}'.format(np.mean(XGB_prediction)*100))



predict=clf.predict_proba(test.drop(columns='ID',axis=1))



XGB_baseline=pd.DataFrame(predict,columns=['XBG_baseline_0','XGB_baseline_1','XGB_baseline_2'])


rf_baseline=ensemble.RandomForestClassifier(n_estimators=300,random_state=101,class_weight='balanced')



rf_prediction_baseline=[]



kfold=model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=101)



for train_idx,val_idx in kfold.split(X=X,y=y):

    rf_baseline.fit(X.loc[train_idx,:],y[train_idx])

    predict=rf_baseline.predict(X.loc[val_idx,:])

    rf_prediction_baseline.append(metrics.accuracy_score(y[val_idx],predict))



print('Accuracy of Random Forest Baseline is {}'.format(np.mean(rf_prediction_baseline)))





predict=rf_baseline.predict_proba(test.drop(columns='ID',axis=1))



RF_baseline=pd.DataFrame(predict,columns=['RF_baseline_0','RF_baseline_1','RF_baseline_2'])
#classifier= ensemble.RandomForestClassifier(class_weight='balanced',random_state=101)



#param_grid={

#    'n_estimators':[50,100,150,200,250],

#    'criterion' : ['gini','entropy'],

#    'max_depth':[5,10,15,20,25],

#    'min_samples_split':[2,3,5],

    #'max_features':['auto', 'sqrt', 'log2']

     

#}



#model=model_selection.GridSearchCV(

#                        estimator=classifier,

#                        param_grid=param_grid,

#                        scoring='accuracy',

#                        cv=5,

#                        refit=True,

 #                       verbose=5,

 #                       n_jobs=-1

 #                       )



#model.fit(X,y)





#print('Best Scorer{}'.format(model.best_score_))



#print('/n')



#print('Best Parameters{}'.format(model.best_params_))



#predict=model.predict_proba(test.drop(columns='ID',axis=1))



#RF_Tuned=pd.DataFrame(predict,columns=['RF_Tuned_0','RF_Tuned_1','RF_Tuned_2'])
# Implementing the Grid Search Version 



rf_tuned=ensemble.RandomForestClassifier(n_estimators=250,random_state=101,criterion='entropy',max_depth=25,min_samples_split=2,

                                         class_weight='balanced')



rf_prediction_tuned=[]



kfold=model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=101)



for train_idx,val_idx in kfold.split(X=X,y=y):

    rf_tuned.fit(X.loc[train_idx,:],y[train_idx])

    predict=rf_tuned.predict(X.loc[val_idx,:])

    rf_prediction_tuned.append(metrics.accuracy_score(y[val_idx],predict))



print('Accuracy of Random Forest Tuned by Grid Search is {}'.format(np.mean(rf_prediction_tuned)))





predict=rf_tuned.predict_proba(test.drop(columns='ID',axis=1))



RF_tuned=pd.DataFrame(predict,columns=['RF_tuned_0','RF_tuned_1','RF_tuned_2'])
#import xgboost as xgb

#from sklearn.model_selection import RandomizedSearchCV



#params = {

#        'learning_rate': np.arange(0,1,0.1),

#       'n_estimators':np.arange(50,250,50),

#       'colsample_bytree': np.arange(0.4,1,0.2),

#        'max_depth': np.arange(5,15,5),

#        'reg_lambda':np.arange(0.5,1,0.5)

#        }



#xgb = xgb.XGBClassifier(objective='multi:softmax',\

 #                    nthread=1,num_class=3)



#folds = 5

#param_comb = 100





#kfold = model_selection.KFold(n_splits=folds, shuffle = True, random_state = 101)



#random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=-1, cv=kfold.split(X,y), verbose=5, random_state=101,refit=True )



#random_search.fit(X, y)





#print("The best score is {}".format(random_search.best_score_ ))



#print('/n')



#print ('The best paramerts are {}'.format(random_search.best_params_))





#predict=random_search.predict_proba(test.drop(columns='ID',axis=1))



#XGB_Tuned=pd.DataFrame(predict,columns=['XGB_Tuned_0','XGB_Tuned_1','XGB_Tuned_2'])
from xgboost import XGBClassifier



clf_tuned = XGBClassifier(objective='multi:softmax',n_jobs=-1,n_estimators=100,num_class=3,colsample_bytree= 0.6

                         ,gamma=2,max_depth=10,min_child_weight=11,reg_lambda=0.5,subsample=0.8)



XGB_prediction=[]



kfold=model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=101)



for train_idx,val_idx in kfold.split(X=X,y=y):

    clf_tuned.fit(X.loc[train_idx,:],y[train_idx])

    predict=clf_tuned.predict(X.loc[val_idx,:])

    XGB_prediction.append(metrics.accuracy_score(y[val_idx],predict))

    



print('Accuracy of XGBoost Tuned by Random Search is {}'.format(np.mean(XGB_prediction)))



predict=clf_tuned.predict_proba(test.drop(columns='ID',axis=1))



XGB_Tuned=pd.DataFrame(predict,columns=['XGB_Tuned_0','XGB_Tuned_1','XGB_Tuned_2'])


tree_classifier=ensemble.ExtraTreesClassifier(n_estimators=250,max_depth=5)



tree_classifier_prediction=[]



kfold=model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=101)



for train_idx,val_idx in kfold.split(X=X,y=y):

    tree_classifier.fit(X.loc[train_idx,:],y[train_idx])

    predict=tree_classifier.predict(X.loc[val_idx,:])

    tree_classifier_prediction.append(metrics.accuracy_score(y[val_idx],predict))

    



print('Accuracy of Extra Tree CLassifier is {}'.format(np.mean(tree_classifier_prediction)))



predict=tree_classifier.predict_proba(test.drop(columns='ID',axis=1))



ET_baseline=pd.DataFrame(predict,columns=['ET_baseline_0','ET_baseline_1','ET_baseline_2'])

# Data Preparation for NN 







# Making catergorical variables as one-hot encoding 



#data=pd.get_dummies(data,columns=['Pesticide_Use_Category','Season','crop_soil_pest','crop_soil_pest_season','crop_soil'\

                                  # ,'soil_pest','crop_pest','Pest_season'],drop_first=True)



Train=data[data['is_test']!=1]

Train.drop(columns=['is_test','ID'],axis=1,inplace=True)



X=Train.drop('Crop_Damage',axis=1)

y=Train['Crop_Damage']



Test=data[data['is_test']==1]

Test.drop(columns=['is_test','Crop_Damage'],axis=1,inplace=True)



# Using Oversampling using SMOTE 



X_train,X_val,y_train,y_val=model_selection.train_test_split(X,y,shuffle=True,random_state=101,test_size=0.20)



normalize_col=['Estimated_Insects_Count','Number_Doses_Week','Number_Weeks_Used_y','Number_Weeks_Quit',

              'Estimated_Insects_Count_square','Number_Weeks_Used_y_square','Number_Weeks_Quit_square','Estimated_Insects_doses',

       'Estimated_Insects_used', 'Estimated_Insects_quit','Number_Weeks_Quit_Used']



for col in normalize_col:

    scaler=preprocessing.MinMaxScaler()

    scaler.fit(X_train[col].values.reshape(-1,1))

    X_train.loc[:,col]=scaler.transform(X_train[col].values.reshape(-1,1))

    X_val.loc[:,col]=scaler.transform(X_val[col].values.reshape(-1,1))

    Test.loc[:,col]=scaler.transform(Test[col].values.reshape(-1,1))



from imblearn.over_sampling import SMOTE



smote=SMOTE('minority')

X_sm,y_sm=smote.fit_sample(X_train,y_train)   





y_train=pd.get_dummies(y_train)

y_val=pd.get_dummies(y_val)
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers

from keras.regularizers import l2,l1

from keras import layers

from keras.layers import BatchNormalization

from keras import optimizers



# Initialising the ANN

model = Sequential()



# Adding the input layer and the first hidden layer

model.add(Dense(units = 200, kernel_initializer = 'he_normal', activation = 'relu', input_dim = X_train.shape[1]))



model.add(layers.Dropout(0.05))



model.add(BatchNormalization())



# Adding second hidden layer

model.add(Dense(units = 200,kernel_initializer = 'he_normal', activation = 'relu'))



model.add(layers.Dropout(0.05))



model.add(BatchNormalization())



#Adding third hidden layer 

model.add(Dense(units = 100,kernel_initializer = 'he_normal', activation = 'relu'))



model.add(layers.Dropout(0.1))



model.add(BatchNormalization())





# Adding the output layer

model.add(Dense(units = 3, kernel_initializer = 'he_normal', activation = 'softmax'))



# Compiling the ANN



adam=optimizers.Adam(lr=0.0001)



model.compile(optimizer =adam, loss = 'categorical_crossentropy',metrics=['accuracy'])



model.fit(X_train,y_train,batch_size=128,epochs=110,validation_data=(X_val,y_val),verbose=2)
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()
predict=model.predict_proba(Test.drop('ID',axis=1))



NN=pd.DataFrame(predict,columns=['NN_0','NN_1','NN_2'])
#Test.drop(columns=['Crop_Damage_0','Crop_Damage_1','Crop_Damage_2'],axis=1,inplace=True)
# Ensemble of Various Model Public Leaderboard : 0.844



#Test['Crop_Damage_0']=(0.5*NN['NN_0']+0.0*RF_tuned2['RF_tuned2_0']+0.0*XGB_baseline['XBG_baseline_0']+0.0*RF_baseline['RF_baseline_0']+0*RF_tuned['RF_tuned_0']+0.45*XGB_Tuned['XGB_Tuned_0']+0.25*ET_baseline['ET_baseline_0'])

#Test['Crop_Damage_1']=(0.5*NN['NN_1']+0.0*RF_tuned2['RF_tuned2_1']+0.0*XGB_baseline['XGB_baseline_1']+0.0*RF_baseline['RF_baseline_1']+0*RF_tuned['RF_tuned_1']+0.45*XGB_Tuned['XGB_Tuned_1']+0.25*ET_baseline['ET_baseline_1'])

#Test['Crop_Damage_2']=(0.5*NN['NN_2']+0.0*RF_tuned2['RF_tuned2_2']+0.0*XGB_baseline['XGB_baseline_2']+0.0*RF_baseline['RF_baseline_2']+0*RF_tuned['RF_tuned_2']+0.45*XGB_Tuned['XGB_Tuned_2']+0.25*ET_baseline['ET_baseline_2'])



Test['Crop_Damage_0']=(0.4*NN['NN_0']+0.4*XGB_Tuned['XGB_Tuned_0']+0.2*ET_baseline['ET_baseline_0']).values.reshape(-1,1)

Test['Crop_Damage_1']=(0.4*NN['NN_1']+0.4*XGB_Tuned['XGB_Tuned_1']+0.2*ET_baseline['ET_baseline_1']).values.reshape(-1,1)

Test['Crop_Damage_2']=(0.4*NN['NN_2']+0.4*XGB_Tuned['XGB_Tuned_2']+0.2*ET_baseline['ET_baseline_2']).values.reshape(-1,1)
Test.sample(10)
Test['Crop_Damage']=Test[['Crop_Damage_0','Crop_Damage_1','Crop_Damage_2']].idxmax(axis=1)
Test['Crop_Damage'].replace({'Crop_Damage_0':'0','Crop_Damage_1':'1','Crop_Damage_2':'2'},inplace=True)


Test[['ID','Crop_Damage']].to_csv('/kaggle/working/Ensemble_weighted_NN+XGB+ET.csv',index=False)