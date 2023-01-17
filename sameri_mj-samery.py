from sklearn import preprocessing,feature_selection
from sklearn.model_selection import KFold , train_test_split
from sklearn.svm import LinearSVC , SVC
import numpy as np
import pandas as pd
Data_train = pd.DataFrame.from_csv('../input/train.csv')
Data_train = pd.DataFrame.as_matrix(Data_train)
Data_train = np.asarray(Data_train[:,1:],dtype= np.float64)

Data_test = pd.DataFrame.from_csv('../input/test.csv')
Data_test = pd.DataFrame.as_matrix(Data_test)
Data_test = np.asarray(Data_test[:,1:],dtype= np.float64)


min_max_scaler = [ preprocessing.MinMaxScaler() for i in range(Data_train.shape[1]) ]

Data_train_Label  = Data_train[:,-1]
Data_train = Data_train[:,0:-1]

Data_train_scaled = np.zeros(Data_train.shape)
Data_test_scaled = np.zeros(Data_test.shape)

for i_feat in range(Data_train.shape[1]):
    Data_train[:,i_feat][np.isnan(Data_train[:,i_feat]) == True] = np.mean(Data_train[:,i_feat][np.isnan(Data_train[:,i_feat]) == False])
    Data_train_scaled[:,i_feat] = min_max_scaler[i_feat].fit_transform(Data_train[:,i_feat].reshape(-1,1)).reshape(Data_train_scaled[:,i_feat].shape)

for i_feat in range(Data_train.shape[1]):
    Data_test[:,i_feat][np.isnan(Data_test[:,i_feat]) == True] = np.mean(Data_test[:,i_feat][np.isnan(Data_test[:,i_feat]) == False])
    Data_test_scaled[:,i_feat] = min_max_scaler[i_feat].transform(Data_test[:,i_feat].reshape(-1,1)).reshape(Data_test_scaled[:,i_feat].shape)
    
MI = feature_selection.mutual_info_classif(Data_train_scaled,Data_train_Label)

sorted_index = np.argsort(-MI)

sorted_index = sorted_index[0:-5]
kf = KFold(n_splits=4)

selected_index = set([])

best_avg = 0
for i_ind in sorted_index :
    selected_index.add(i_ind)
    result = [0 for i in range(4)]
    counter = 0
    for train_index, test_index in kf.split(Data_train_Label):
        K_train_data  = Data_train_scaled[train_index,:]
        K_train_data  = np.take(K_train_data,list(selected_index),axis=1)
        k_train_label = Data_train_Label[train_index]
        K_test_data   = Data_train_scaled[test_index,:]
        K_test_data   = np.take(K_test_data,list(selected_index),axis=1)
        k_test_label  = Data_train_Label[test_index]
        clf = SVC(kernel='linear')
        clf.fit(K_train_data,k_train_label)
        result[counter] = clf.score(K_test_data,k_test_label)
        counter += 1
    if(np.mean(result) < best_avg):
        selected_index.remove(i_ind)
    else :
        best_avg = np.mean(result)
Data_train_selected = np.take(Data_train_scaled,list(selected_index),axis=1)

Data_test_selected = np.take(Data_test_scaled,list(selected_index),axis=1)
clf = SVC(kernel='linear')
clf.fit(Data_train_selected,Data_train_Label)

pred = clf.predict(Data_test_selected)

playerID = np.arange(901,1341,1)

submitPred = np.hstack((playerID.reshape(playerID.shape[0],1),pred.reshape(pred.shape[0],1)))

submitPred = pd.DataFrame({ 'PlayerID' : playerID ,'TARGET_5Yrs' : pred })

submitPred.to_csv("submission.csv")