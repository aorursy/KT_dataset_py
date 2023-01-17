import pandas as pd
import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
data_number=pd.read_csv('../input/train.csv')
data_train_train=np.array(data_number.iloc[:,1:])
data_train_train_lable=np.array(data_number.iloc[:,0])
data_train_test=np.array(data_number.iloc[int(len(data_number)*9/10):,1:])
data_train_test_label=np.array(data_number.iloc[int(len(data_number)*9/10):,0])
print(data_train_train_lable)
print(data_train_train_lable)
# neigh = RadiusNeighborsClassifier()
# %time neigh_fit=neigh.fit(data_train_train, data_train_train_lable) 
# neigh_predict=neigh.predict(data_train_test) 


neigh=KNeighborsClassifier(n_jobs=-1)
train_fit=neigh.fit(data_train_train,data_train_train_lable)

print('1')
#加载数据
data_test=pd.read_csv('../input/test.csv')
data_test_train=np.array(data_test)
train_predict=neigh.predict(data_test_train)
data_output=pd.DataFrame(columns=['ImageId','Label'])
data_output['ImageId']=list(range(1,int(len(train_predict)+1)))
data_output['Label']=train_predict
data_output.to_csv('output.csv')