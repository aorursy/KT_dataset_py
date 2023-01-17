!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c mlregression-cabbage-price
!unzip mlregression-cabbage-price.zip
import pandas as pd
import numpy as np
train=pd.read_csv('train_cabbage_price.csv')
test=pd.read_csv('test_cabbage_price.csv')
train=train.drop('year',axis=1)
test=test.drop('year',axis=1)
print(train)
print(test)
print(train.shape)
x=train.drop('avgPrice',axis=1)
print(x.shape)
y= train['avgPrice']


from sklearn.model_selection import train_test_split  #Scikit-Learn 의 model_selection library를 train_test_split로 명명
Y_train=y
X_train=x
X_test=test
print(Y_train.shape)
print(X_train.shape)
print(X_test.shape)

print(Y_train)
print(X_train)
print(X_test)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)    #data의 표준화
X_test_std=sc.transform(X_test)    
#표준화된 data의 확인
print(X_train.head())
X_train_std[1:5,] 
# KNN 의 적용
from sklearn.neighbors import KNeighborsRegressor  #KNN 불러오기
knn=KNeighborsRegressor(n_neighbors=15,weights="distance") #15개의 인접한이웃 거리
knn.fit(X_train_std,Y_train) #모델 fitting과정
y_test_pred=knn.predict(X_test_std)  #모델을 적용한 test data의 y값 예측치
print(y_test_pred)
submit=pd.read_csv('sample_submit.csv')
submit
for i in range(len(y_test_pred)):
  submit['Expected'][i]=int(y_test_pred[i])
submit['Expected']=submit['Expected'].astype(int)
submit.to_csv('submit.csv',index=False,header=True)
submit
!kaggle competitions submit -c mlregression-cabbage-price -f submit.csv -m "3"