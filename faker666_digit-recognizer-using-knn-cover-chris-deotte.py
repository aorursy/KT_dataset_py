import sys
!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
# LOAD LIBRARIES
import cudf, cuml
print('cuML version',cuml.__version__)
import pandas as pd
# LOAD TRAIN DATA
train = pd.read_csv('../input/digit-recognizer/train.csv').values
print('train',train)
print('train.shape',train.shape)
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator;
datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.10, width_shift_range=0.1, height_shift_range=0.1)
plt.figure(figsize=(15,5.5))

digit = train[0,1:].reshape(1,28,28,1)
for i in range(1,8):
    plt.subplot(3, 8, i+1)
    new_digit = datagen.flow(digit)
    plt.imshow(new_digit[0].reshape((28,28)),cmap=plt.cm.binary)
    
digit = train[1,1:].reshape(1,28,28,1)    
for i in range(9,16):
    plt.subplot(3, 8, i+1)
    new_digit = datagen.flow(digit)
    plt.imshow(new_digit[0].reshape((28,28)),cmap='binary')
    
digit = train[2,1:].reshape(1,28,28,1)    
for i in range(17,24):
    plt.subplot(3, 8, i+1)
    new_digit = datagen.flow(digit)
    plt.imshow(new_digit[0].reshape((28,28)),cmap='binary')
# LOAD TEST DATA
test = cudf.read_csv('../input/digit-recognizer/test.csv')
print('test shape =', test.shape )
from cuml.neighbors import KNeighborsClassifier
# CONVERT NumPy array to cuDF array
cudf_train = cudf.from_pandas(pd.DataFrame(train))
# FIT KNN MODEL
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit( cudf_train.iloc[:,1:], cudf_train.iloc[:,0] )
# PREDICT TEST DATA
y_hat_p = knn.predict(test)
y_hat = y_hat_p.to_pandas().values.argmax(axis=1)
# We could use knn.predict() but cuML v0.11.0 has bug
# y_hat = knn.predict(test)
# SAVE PREDICTIONS TO CSV
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sub.Label = y_hat
sub.to_csv('submission_cuML_DAx50.csv',index=False)
sub.head()