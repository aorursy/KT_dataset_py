import pandas as pd
import seaborn as sns
master=pd.read_csv("../input/meteorological-model-versus-real-data/vigo_model_vs_real.csv",index_col="datetime",parse_dates=True)
master_f=master[['HGT500_4K', 'HGT850_4K', 'HGTlev1_4K', 'HGTlev2_4K', 'HGTlev3_4K',
       'T500_4K', 'T850_4K', 'cape_4K', 'cfh_4K', 'cfl_4K', 'cfm_4K', 'cft_4K',
       'cin_4K', 'conv_prec_4K', 'dir_4K', 'lhflx_4K', 'lwflx_4K',
        'mod_4K', 'mslp_4K', 'pbl_height_4K', 'prec_4K',
       'rh_4K', 'shflx_4K', 'snow_prec_4K', 'snowlevel_4K', 'sst_4K',
       'swflx_4K', 'temp_4K', 'u_4K', 'ulev1_4K', 'ulev2_4K', 'ulev3_4K',
       'v_4K', 'visibility_4K', 'vlev1_4K', 'vlev2_4K', 'vlev3_4K',
       'wind_gust_4K', 'visibility_o']]

sns.set(rc={'figure.figsize':(20,20)})
g=sns.heatmap(master_f.corr().round(decimals=1),annot=True,linewidths=.5)

g=sns.heatmap(master_f[[ 'T850_4K', 'cape_4K', 'cfh_4K', 'cfl_4K', 'cfm_4K', 'cft_4K',
       'cin_4K', 'conv_prec_4K', 'dir_4K', 'lhflx_4K', 'lwflx_4K',
        'mod_4K', 'mslp_4K', 'pbl_height_4K', 'prec_4K',
       'rh_4K', 'shflx_4K', 'temp_4K', 'u_4K', 'v_4K', 'visibility_4K', 
       "visibility_o"]].corr().round(decimals=1),annot=True,linewidths=.5)
from sklearn.metrics import confusion_matrix,classification_report 
pd.set_option('mode.chained_assignment', None)
for threshold in [50,500,1000,5000]:
    master_f.loc[:,"visibility_o_"+str(threshold)]=[True if c<=threshold else False for c in master_f.visibility_o]
    master_f.loc[:,"visibility_4K_"+str(threshold)]=[True if c<=threshold else False for c in master_f.visibility_4K]
    print("**** Confusion matrix threshold:"+str(threshold)+"m"+" ****")
    print(confusion_matrix(master_f["visibility_o_"+str(threshold)],
                         master_f["visibility_4K_"+str(threshold)],
                         labels=[True,False]))
    print("***************")
    target_names = [">"+str(threshold)+"m","<="+str(threshold)+"m" ]
    print(classification_report(master_f["visibility_o_"+str(threshold)],
                              master_f["visibility_4K_"+str(threshold)],
                              target_names=target_names))
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
#ROC model one figure
ranges=[50,500,1000,5000]
fprl=[]
tprl=[]
thresholdsl=[]
roc_aucl=[]
for vis_range in ranges:
  y_data=[False if c<=vis_range else True for c in master_f.visibility_o]
  y_pred=master_f.visibility_4K
  fpr, tpr, thresholds = metrics.roc_curve(y_data,y_pred)
  roc_auc = auc(fpr, tpr)
  fprl.append(fpr)
  tprl.append(tpr)
  thresholdsl.append(thresholds)
  roc_aucl.append(roc_auc)
plt.figure(figsize=[12,10])
n_ranges=len(ranges)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue',"red"])
for i, color in zip(range(n_ranges), colors):
    plt.plot(fprl[i], tprl[i], color=color, lw=2,
             label='ROC curve of range {0} (area = {1:0.2f})'
             .format(ranges[i],roc_aucl[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC range:")
plt.legend(loc="lower right")
plt.show()  
#change label dependent variables
threshold1=1000
y_data=pd.DataFrame({"datetime":master_f.index,
                     "visibility_o":[1 if c<=threshold1 else 0 for c in 
                                     master_f["visibility_o"]]}).set_index("datetime")
#choosing independent variables
x_data=master_f[['dir_4K', 'lhflx_4K', 'mod_4K', 'prec_4K', 'rh_4K', 'visibility_4K',
        'mslp_4K', 'temp_4K', 'cape_4K', 'cfl_4K', 'cfm_4K', 'cin_4K',"wind_gust_4K",
       'conv_prec_4K']]

#neural network
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix ,classification_report 
from sklearn.model_selection import cross_val_score,cross_validate
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler =MinMaxScaler()
#transform x_data or pca_vectors
scaled_df = scaler.fit_transform(x_data)
x_train, x_test, y_train, y_test = train_test_split(scaled_df,y_data.visibility_o, test_size=0.2,)
class_weight = {0: (sum(y_train == 1)/len(y_train)), 1: (sum(y_train == 0)/len(y_train))}
mlp = Sequential()
mlp.add(Dense(48, input_dim=x_train.shape[1], activation='relu'))
mlp.add(Dropout(0.5))
mlp.add(Dense(48, activation='relu'))
mlp.add(Dropout(0.5))
mlp.add(Dense(1, activation='sigmoid'))
mlp.summary()
mlp.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss='binary_crossentropy',
            metrics=["binary_accuracy"]
           )

history = mlp.fit(x=x_train,
                  y=y_train,
                  batch_size=64,
                  epochs=80,
                  validation_data=(x_test, y_test),
                  class_weight=class_weight,
                  verbose=0).history
pd.DataFrame(history).plot(grid=True,figsize=(12,12),yticks=np.linspace(0.0, 1.0, num=11))
y_pred=mlp.predict(x_test)
result=pd.DataFrame({"y_test":y_test.values,"y_pred":y_pred.reshape(1,-1)[0],
                     "datatime":y_test.index}).set_index("datatime")
g=pd.DataFrame({"y_pred test==1":result["y_pred"][result.y_test==1],
              "y_pred test==0":result["y_pred"][result.y_test==0]}).plot(kind="box",figsize=(15,15))
#select threhold_nor
threshold_nor=0.8
y_pred_nor=[0 if c<=threshold_nor else 1 for c in result.y_pred]
target_names = [">"+str(threshold1)+"m","<="+str(threshold1)+"m" ]
print(classification_report(y_test.values,y_pred_nor , target_names=target_names))
print("**** Confusion matrix ****")
print(confusion_matrix(y_test,y_pred_nor,labels=[1,0]))

#ROC model 
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC ")
plt.legend(loc="lower right")
plt.show()
mlp.save("model.h5")