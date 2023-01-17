import numpy as np

import pandas as pd

import matplotlib as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
application_train = pd.read_csv('../input/application_train.csv')

application_test = pd.read_csv('../input/application_test.csv')



bureau = pd.read_csv('../input/bureau.csv')

bureau_balance = pd.read_csv('../input/bureau_balance.csv')

credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')

installments_payments = pd.read_csv('../input/installments_payments.csv')

POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')

previous_application = pd.read_csv('../input/previous_application.csv')
def missing_table(df):

        mis_val = df.isnull().sum()       

        mis_val_percent = 100*df.isnull().sum()/len(df)      

        mis_val_table = pd.concat([mis_val,mis_val_percent],axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(columns ={0:'Missing Values',1:'% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

                                    mis_val_table_ren_columns.iloc[:,1]!=0].sort_values(

                                    '% of Total Values',ascending=False).round(1)

        return mis_val_table_ren_columns
application_train_missing = missing_table(application_train)

#print(application_train_missing)

drop_features_application_train = application_train_missing.loc[:'EXT_SOURCE_3',:].index.values

application_train = application_train.drop(columns=drop_features_application_train)

application_test = application_test.drop(columns=drop_features_application_train)



bureau_missing = missing_table(bureau)

#print(bureau_missing)

drop_features_bureau = bureau_missing.loc[:'AMT_CREDIT_SUM_LIMIT',:].index.values

bureau = bureau.drop(columns=drop_features_bureau)



bureau_balance_missing = missing_table(bureau_balance)

#print(bureau_balance_missing)



credit_card_balance_missing = missing_table(credit_card_balance)

#print(credit_card_balance_missing)

drop_features_credit_card_balance = credit_card_balance_missing.loc[:'CNT_DRAWINGS_POS_CURRENT',:].index.values

credit_card_balance = credit_card_balance.drop(columns=drop_features_credit_card_balance)



installments_payments_missing = missing_table(installments_payments)

#print(installments_payments_missing)



POS_CASH_balance_missing = missing_table(POS_CASH_balance)

#print(POS_CASH_balance_missing)



previous_application_missing = missing_table(previous_application)

#print(previous_application_missing)

drop_features_previous_application = previous_application_missing.loc[:'CNT_PAYMENT',:].index.values

previous_application = previous_application.drop(columns=drop_features_previous_application)
print(application_train.dtypes.value_counts())

#print(bureau.dtypes.value_counts())

#print(bureau_balance.dtypes.value_counts())

#print(credit_card_balance.dtypes.value_counts()) 

#print(installments_payments.dtypes.value_counts())

#print(POS_CASH_balance.dtypes.value_counts())

#print(previous_application.dtypes.value_counts()) 



bureau = pd.get_dummies(bureau)

bureau_balance = pd.get_dummies(bureau_balance)

credit_card_balance = pd.get_dummies(credit_card_balance)

installments_payments = pd.get_dummies(installments_payments)

POS_CASH_balance = pd.get_dummies(POS_CASH_balance)

previous_application = pd.get_dummies(previous_application)
right_1 = bureau_balance.groupby('SK_ID_BUREAU').agg(np.mean)

sub_dataset_1 = pd.merge(bureau,right_1,on='SK_ID_BUREAU',how='left')



main_dataset = application_train



sub_dataset_1 = sub_dataset_1.groupby('SK_ID_CURR').agg(np.mean)

sub_dataset_2 = credit_card_balance.groupby('SK_ID_CURR').agg(np.mean) 

sub_dataset_3 = installments_payments.groupby('SK_ID_CURR').agg(np.mean)

sub_dataset_4 = POS_CASH_balance.groupby('SK_ID_CURR').agg(np.mean) 

sub_dataset_5 = previous_application.groupby('SK_ID_CURR').agg(np.mean)



whole_dataset_train = pd.merge(main_dataset,sub_dataset_1,on='SK_ID_CURR',how='left')

whole_dataset_train = pd.merge(main_dataset,sub_dataset_2,on='SK_ID_CURR',how='left')

whole_dataset_train = pd.merge(main_dataset,sub_dataset_3,on='SK_ID_CURR',how='left')

whole_dataset_train = pd.merge(main_dataset,sub_dataset_4,on='SK_ID_CURR',how='left')

whole_dataset_train = pd.merge(main_dataset,sub_dataset_5,on='SK_ID_CURR',how='left')



main_dataset = application_test



whole_dataset_test = pd.merge(main_dataset,sub_dataset_1,on='SK_ID_CURR',how='left')

whole_dataset_test = pd.merge(main_dataset,sub_dataset_2,on='SK_ID_CURR',how='left')

whole_dataset_test = pd.merge(main_dataset,sub_dataset_3,on='SK_ID_CURR',how='left')

whole_dataset_test = pd.merge(main_dataset,sub_dataset_4,on='SK_ID_CURR',how='left')

whole_dataset_test = pd.merge(main_dataset,sub_dataset_5,on='SK_ID_CURR',how='left')
import missingno as msno

%matplotlib inline



#msno.matrix(whole_dataset_train.sample(100))

#msno.bar(whole_dataset_train.sample(1000))



whole_dataset_train = whole_dataset_train.dropna(axis=0,subset=whole_dataset_train.columns[-150:-1].values)



print(missing_table(whole_dataset_train))
whole_train_label = whole_dataset_train.loc[:,'TARGET']

whole_train_feature = whole_dataset_train.drop(columns = ['TARGET'])
from sklearn.model_selection import train_test_split



feature_train, feature_val, label_train, label_val = train_test_split(

                                                     whole_train_feature,whole_train_label, 

                                                     test_size=0.3,random_state=66,shuffle=True)



feature_names = whole_train_feature.columns.values

feature_train = pd.DataFrame(feature_train,columns=feature_names)

feature_val = pd.DataFrame(feature_val,columns=feature_names)



print(feature_train.shape)

print(feature_val.shape)

print(label_train.shape)

print(label_val.shape)

#feature_train.head()
print(label_train.value_counts())

label_train.astype(int).plot.hist()
cat_feature_indices = np.where(feature_train.dtypes == np.object)[0]



feature_train.iloc[:,cat_feature_indices] = feature_train.iloc[:,cat_feature_indices].fillna('missing')

feature_val.iloc[:,cat_feature_indices] = feature_val.iloc[:,cat_feature_indices].fillna('missing')





weight_0 = 1 / label_train.value_counts()[0] * len(label_train)

weight_1 = 1 / label_train.value_counts()[1] * len(label_train)



weights_list = []

for i in np.array(label_train):

    if i == 0:

        weights_list.append(weight_0)

    if i == 1:

        weights_list.append(weight_1)

    else:

        pass
from catboost import CatBoostClassifier,Pool



train_data_pool = Pool(data=feature_train,

                       label=label_train,

                       cat_features=cat_feature_indices,

                       weight=weights_list)



model_catboost = CatBoostClassifier(iterations=200,

                           learning_rate=0.5,

                           depth=6,

                           loss_function='Logloss')



model_catboost.fit(train_data_pool,

                   eval_set=(feature_val,label_val),

                   plot=True)
prediction_probality = model_catboost.predict_proba(feature_val)

prediction_class = model_catboost.predict(feature_val)



from sklearn.metrics import roc_auc_score

y_true = label_val

y_scores = prediction_probality[:,1]

roc_auc_score(y_true, y_scores)
feature_importance = model_catboost.feature_importances_

feature_names = model_catboost.feature_names_



import matplotlib.pyplot as plt

#plt.figure(figsize=(10, 15))

#plt.barh(feature_name,feature_importance,height =0.5)
importance_matrix = dict(zip(feature_names,feature_importance))



importance_sort = pd.DataFrame([importance_matrix]).T

importance_sort.columns = ['importance']

importance_sort = importance_sort.sort_values('importance',ascending=False)



useful_features = importance_sort[importance_sort.loc[:,'importance']!=0].index.values

useless_features = importance_sort[importance_sort.loc[:,'importance']==0].index.values
import missingno as msno

%matplotlib inline



#msno.matrix(whole_dataset_train.sample(100))

#msno.bar(whole_dataset_train.sample(1000))



cat_feature = np.where(whole_dataset_train.dtypes == np.object)[0]

num_feature = np.where(whole_dataset_train.dtypes != np.object)[0]



whole_dataset_train.iloc[:,cat_feature] = whole_dataset_train.iloc[:,cat_feature].fillna('missing')

whole_dataset_train.iloc[:,num_feature] = whole_dataset_train.iloc[:,num_feature].fillna(0)



cat_feature = np.where(whole_dataset_test.dtypes == np.object)[0]

num_feature = np.where(whole_dataset_test.dtypes != np.object)[0]



whole_dataset_test.iloc[:,cat_feature] = whole_dataset_test.iloc[:,cat_feature].fillna('missing')

whole_dataset_test.iloc[:,num_feature] = whole_dataset_test.iloc[:,num_feature].fillna(0)



#print(missing_table(whole_dataset_train))
train_label = whole_dataset_train.loc[:,'TARGET']



whole_train_feature = whole_dataset_train.drop(columns = ['TARGET'])

useful_train_feature = whole_train_feature.loc[:,useful_features]

useless_train_feature = whole_train_feature.loc[:,useless_features]



useful_test_feature = whole_dataset_test.loc[:,useful_features]

useless_test_feature = whole_dataset_test.loc[:,useless_features]
useful_train_feature = pd.get_dummies(useful_train_feature)

useless_train_feature = pd.get_dummies(useless_train_feature)



useful_test_feature = pd.get_dummies(useful_test_feature)

useless_test_feature = pd.get_dummies(useless_test_feature)



useful_train_feature,useful_test_feature = useful_train_feature.align(useful_test_feature,join='inner',axis=1)

useless_train_feature,useless_test_feature = useless_train_feature.align(useless_test_feature,join='inner',axis=1)

from sklearn import preprocessing



column_names = useful_train_feature.columns.values

standarder_useful = preprocessing.StandardScaler().fit(useful_train_feature)

useful_train_feature = standarder_useful.transform(useful_train_feature)

useful_train_feature = pd.DataFrame(useful_train_feature,columns=column_names)



column_names = useless_train_feature.columns.values

standarder_useless = preprocessing.StandardScaler().fit(useless_train_feature)

useless_train_feature = standarder_useless.transform(useless_train_feature)

useless_train_feature = pd.DataFrame(useless_train_feature,columns=column_names)



column_names = useful_test_feature.columns.values

useful_test_feature = standarder_useful.transform(useful_test_feature)

useful_test_feature = pd.DataFrame(useful_test_feature,columns=column_names)



column_names = useless_test_feature.columns.values

useless_test_feature = standarder_useless.transform(useless_test_feature)

useless_test_feature = pd.DataFrame(useless_test_feature,columns=column_names)
from sklearn.decomposition import PCA



pca = PCA(n_components=5)

pca.fit(useless_train_feature)

useless_train_feature = pca.transform(useless_train_feature)

useless_train_feature = pd.DataFrame(useless_train_feature)

useless_train_feature.columns = ['PCA_1','PCA_2','PCA_3','PCA_4','PCA_5']



useless_test_feature = pca.transform(useless_test_feature)

useless_test_feature = pd.DataFrame(useless_test_feature)

useless_test_feature.columns = ['PCA_1','PCA_2','PCA_3','PCA_4','PCA_5']
train_feature = pd.concat((useful_train_feature,useless_train_feature),axis=1)



test_feature = pd.concat((useful_test_feature,useless_test_feature),axis=1)
from sklearn.model_selection import train_test_split



feature_train, feature_val, label_train, label_val = train_test_split(

                                                     train_feature, train_label, 

                                                     test_size=0.33, random_state=88,shuffle=True)



feature_names = train_feature.columns.values

feature_train = pd.DataFrame(feature_train,columns=feature_names)

feature_val = pd.DataFrame(feature_val,columns=feature_names)



print(feature_train.shape)

print(feature_val.shape)

print(label_train.shape)

print(label_val.shape)
import tensorflow as tf

from tensorflow import keras



from keras import models

from keras import layers

from keras import regularizers



model_nn = models.Sequential()

model_nn.add(layers.Dense(32, activation='relu', input_shape=(feature_train.shape[1],)))

model_nn.add(layers.Dense(32, activation='relu'))

#model_nn.add(layers.Dropout(0.2))

model_nn.add(layers.Dense(16, activation='relu'))

model_nn.add(layers.Dense(16, activation='relu'))

model_nn.add(layers.Dense(8, activation='relu'))

model_nn.add(layers.Dense(8, activation='relu'))

#model_nn.add(layers.Dropout(0.2))

model_nn.add(layers.Dense(1, activation='sigmoid'))



model_nn.summary()



#from keras.utils.vis_utils import plot_model

#plot_model(model_nn,show_shapes=True,to_file='model.png',show_shapes=True)



#from IPython.display import SVG

#from keras.utils.vis_utils import model_to_dot

#SVG(model_to_dot(model_nn).create(prog='dot', format='svg'))
import keras.backend as K

import tensorflow as tf



def auc(y_true, y_pred):   #######bigger better

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc



from keras import callbacks



callbacks_list = [callbacks.ModelCheckpoint(monitor='val_auc',mode='max',save_best_only=True,filepath='best_model_nn.h5'),

                  callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=5),

                  callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min',factor=0.1,patience=3)]

#callbacks_tensorboard = [callbacks.TensorBoard(log_dir='C:/Users/chaok/Desktop/kaggle/Kaggle/tensorboard')]



model_nn.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy',auc])



history = model_nn.fit(x=feature_train,

                       y=label_train,

                       validation_data=(feature_val, label_val),

                       epochs=20,

                       batch_size=32*15,

                       class_weight='auto',

                       callbacks=callbacks_list)
history_dict = history.history
loss,accuracy,auc = model_nn.evaluate(feature_val,label_val)

print('validation loss: ',loss)

print('validation accuracy ',accuracy)

print('validation auc: ',auc)
import matplotlib.pyplot as plt



train_loss = history_dict['loss']

val_loss = history_dict['val_loss']

epochs = range(1, len(train_loss)+1)

plt.plot(epochs, train_loss, 'bo', label='Training loss') 

plt.plot(epochs, val_loss, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



train_acc = history_dict['acc'] 

val_acc = history_dict['val_acc']

epochs = range(1, len(train_acc)+1)

plt.plot(epochs, train_acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



train_auc = history_dict['auc']

val_auc = history_dict['val_auc']

epochs = range(1, len(train_auc)+1)

plt.plot(epochs, train_auc, 'bo', label='Training AUC') 

plt.plot(epochs, val_auc, 'b', label='Validation AUC') 

plt.title('Training and validation AUC')

plt.xlabel('Epochs')

plt.ylabel('AUC')

plt.legend()

plt.show()

#model_nn.save('best_model_nn.h5')
#from keras.models import load_model



#model_name = 'best_model_nn.h5'

#best_model_nn = load_model(model_name,custom_objects={'auc':auc})

#print(test_feature.shape)

#print(train_feature.shape)



#print(test_feature.columns == train_feature.columns)
prediction = model_nn.predict(test_feature)
SK_ID_CURR = pd.read_csv('../input/application_test.csv',usecols=['SK_ID_CURR'])



result = pd.concat([SK_ID_CURR,pd.DataFrame(prediction)],axis=1)

result.columns = ['SK_ID_CURR','TARGET']
result.to_csv('result_nn.csv',header=True,sep=',',index=False)
result