!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c logistic-classification-diabetes-knn
!unzip logistic-classification-diabetes-knn.zip
import pandas as pd
import numpy as np
train=pd.read_csv('train.csv')
test=pd.read_csv('test_data.csv')
train=train.drop('Unnamed: 0',axis=1)
test=test.drop('Unnamed: 0',axis=1)
print(train)
print(test)
print(train.shape)
x=train.drop('8',axis=1)
print(x.shape)
y= train['8']


from sklearn.preprocessing import LabelEncoder # LabelEncoder() method를 불러옴
import numpy as np # numpy를 불러옴
classle=LabelEncoder() 
y=classle.fit_transform(train['8'].values) 
print('species labels:', np.unique(y)) # 중복되는 y 값을 하나로 정리하여 print
yo=classle.inverse_transform(y)
print('species:',np.unique(yo))
from sklearn.model_selection import train_test_split  #Scikit-Learn 의 model_selection library를 train_test_split로 명명
Y_train=y
X_train=x
X_test=test.drop('8',axis=1)
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
from sklearn.neighbors import KNeighborsClassifier  #KNN 불러오기
knn=KNeighborsClassifier(n_neighbors=7,p=2) #7개의 인접한이웃, 거리측정기준:유클리드 
knn.fit(X_train_std,Y_train) #모델 fitting과정
y_test_pred=knn.predict(X_test_std)  #모델을 적용한 test data의 y값 예측치
print(y_test_pred)
submit=pd.read_csv('submission_form.csv')
submit
for i in range(len(y_test_pred)):
  submit['Label'][i]=int(y_test_pred[i])
submit['Label']=submit['Label'].astype(int)
submit.to_csv('submit.csv',index=False,header=True)
submit
!kaggle competitions submit -c logistic-classification-diabetes-knn -f submit.csv -m "1"