import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import glob
from sklearn.model_selection import train_test_split



from sklearn.model_selection import StratifiedShuffleSplit

# We will use the Seaborn library
import seaborn as sns
sns.set()

# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'svg' 
#df2=pd.read_csv("../input/noisefreedata/constantlen.csv",header=None)
df=df2
df=df.iloc[:,1:len(df)]
df=df.loc[~(df==0).all(axis=1)]
dfp=df
target=dfp.iloc[:,0]
print(dfp.shape)
data=dfp.iloc[:,1:len(df)]
print("different class data counts",target.value_counts())
target.value_counts().plot(kind='bar', title='Count (Unbalanced Classes)');
from sklearn.utils import resample
df_1=df[df.iloc[:,0]==1]
df_2=df[df.iloc[:,0]==2]
df_3=df[df.iloc[:,0]==3]
df_4=df[df.iloc[:,0]==4]
df_5=df[df.iloc[:,0]==5]
df_6=df[df.iloc[:,0]==6]
df_7=df[df.iloc[:,0]==7]
df_8=df[df.iloc[:,0]==8]
df_9=df[df.iloc[:,0]==9]
df_10=df[df.iloc[:,0]==10]
df_11=df[df.iloc[:,0]==11]
df_12=df[df.iloc[:,0]==12]
df_13=df[df.iloc[:,0]==13]
df_0=(df[df.iloc[:,0]==0]).sample(n=6000,random_state=42)

df_1_upsample=resample(df_1,replace=True,n_samples=6000,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=6000,random_state=124)
df_3_upsample=resample(df_3,replace=True,n_samples=6000,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=6000,random_state=126)
df_5_upsample=resample(df_5,replace=True,n_samples=6000,random_state=127)
df_6_upsample=resample(df_6,replace=True,n_samples=6000,random_state=128)
df_7_upsample=resample(df_7,replace=True,n_samples=6000,random_state=129)
df_8_upsample=resample(df_8,replace=True,n_samples=6000,random_state=130)
df_9_upsample=resample(df_9,replace=True,n_samples=6000,random_state=131)
df_10_upsample=resample(df_10,replace=True,n_samples=6000,random_state=132)
df_11_upsample=resample(df_11,replace=True,n_samples=6000,random_state=133)
df_12_upsample=resample(df_12,replace=True,n_samples=6000,random_state=134)
df_13_upsample=resample(df_13,replace=True,n_samples=6000,random_state=135)
                                                

train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample,df_5_upsample,df_6_upsample,df_7_upsample,df_8_upsample,df_9_upsample,df_10_upsample,df_11_upsample,df_12_upsample,df_13_upsample])
target=train_df.iloc[:,0]
print(train_df.shape)
data=train_df.iloc[:,1:len(train_df)]
print("Balanced Different Classes",target.value_counts())
target.value_counts().plot(kind='bar', title='Count (Balanced Classes)');




target=df.iloc[:,1]
print(df.shape)
data=df.iloc[:,2:701]
print(data.shape)


print("different class data counts",target.value_counts())
target.value_counts().plot(kind='bar', title='Count (Unbalanced Classes)');



from sklearn.utils import resample
df_1=df[df.iloc[:,1]==1]
df_2=df[df.iloc[:,1]==2]
df_3=df[df.iloc[:,1]==3]
df_4=df[df.iloc[:,1]==4]
df_5=df[df.iloc[:,1]==5]
df_6=df[df.iloc[:,1]==6]
df_7=df[df.iloc[:,1]==7]
df_8=df[df.iloc[:,1]==8]
df_9=df[df.iloc[:,1]==9]
df_10=df[df.iloc[:,1]==10]
df_11=df[df.iloc[:,1]==11]
df_12=df[df.iloc[:,1]==12]
df_13=df[df.iloc[:,1]==13]
df_0=(df[df.iloc[:,1]==0]).sample(n=5000,random_state=42)

df_1_upsample=resample(df_1,replace=True,n_samples=5000,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=5000,random_state=124)
df_3_upsample=resample(df_3,replace=True,n_samples=5000,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=5000,random_state=126)
df_5_upsample=resample(df_5,replace=True,n_samples=5000,random_state=127)
df_6_upsample=resample(df_6,replace=True,n_samples=5000,random_state=128)
df_7_upsample=resample(df_7,replace=True,n_samples=5000,random_state=129)
df_8_upsample=resample(df_8,replace=True,n_samples=5000,random_state=130)
df_9_upsample=resample(df_9,replace=True,n_samples=5000,random_state=131)
df_10_upsample=resample(df_10,replace=True,n_samples=5000,random_state=132)
df_11_upsample=resample(df_11,replace=True,n_samples=5000,random_state=133)
df_12_upsample=resample(df_12,replace=True,n_samples=5000,random_state=134)
df_13_upsample=resample(df_13,replace=True,n_samples=5000,random_state=135)
                                                

train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample,df_5_upsample,df_6_upsample,df_7_upsample,df_8_upsample,df_9_upsample,df_10_upsample,df_11_upsample,df_12_upsample,df_13_upsample])
target=train_df.iloc[:,1]
print(train_df.shape)
data=train_df.iloc[:,2:701]
print(data.shape)




print("Balanced Different Classes",target.value_counts())
target.value_counts().plot(kind='bar', title='Count (Balanced Classes)');

[X_train, X_test, y_train, y_test] = train_test_split(data, target, test_size=0.1, random_state=10, stratify=target)

from keras.utils import to_categorical

print("--- X ---")
X = pd.DataFrame(X_train)
X_train=X
print(X.head())
print(X.info())

print("--- Y ---")
y = pd.DataFrame(y_train)
y = to_categorical(y)
print(y.shape)
y_train=y

print("--- testX ---")
testX = pd.DataFrame(X_test)
print(testX.head())
print(testX.info())
X_test=testX

print("--- testy ---")
testy = pd.DataFrame(y_test)
testy = to_categorical(testy)
y_test=testy
print(X.shape)
print(y.shape)
print(testX.shape)
print(testy.shape)

from keras import backend as K
    
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten, SeparableConv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler

X = np.expand_dims(X,2)
testX = np.expand_dims(testX,2)
n_obs, feature, depth = X.shape
batch_size = 1024
def build_model():
    input_img = Input(shape=(feature, depth), name='ImageInput')
    x = Conv1D(64, 3, activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv1D(64, 3, activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling1D(2, name='pool1')(x)
    
    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling1D(2, name='pool2')(x)
    
    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    
    x = SeparableConv1D(256, 3, activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling1D(2, name='pool3')(x)
    x = Dropout(0.6, name='dropout0')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.6, name='dropout1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(14, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model


model =  build_model()

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc',f1])
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)

history = model.fit(X, y, validation_split=0.2,epochs=60,batch_size=batch_size,shuffle=True,class_weight='auto',callbacks=[checkpointer])
print("Evaluation: ")
mse, acc, F1 = model.evaluate(testX, testy)
print('mean_squared_error :', mse)
print('accuracy:', acc)
print('F1:', F1)
#model.save('ConstantR.h5')
model.save_weights("Contant6000weight.h5")
print("Saved model01 to disk")

model_json = model.to_json()
with open("Constant6000.json", "w") as json_file:
    json_file.write(model_json)
y_pred = model.predict(testX, batch_size=1000)
y_pred
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 

print(classification_report(testy.argmax(axis=1), y_pred.argmax(axis=1)))
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
dfx=pd.read_csv('../input/noiseconstant/NoiseConstant.csv')
dfx
dfx=pd.DataFrame(dfx)

target=dfx.iloc[:,1]

data=dfx.iloc[:,2:701]
print("different class data counts",target.value_counts())
target.value_counts().plot(kind='bar', title='Count (Unbalanced Classes)');


  
from keras.utils import to_categorical
print("--- testX ---")
testX = pd.DataFrame(data)
print(testX.head())
print(testX.info())
X_test=testX


print("--- testy ---")
testy = pd.DataFrame(target)
testy = to_categorical(testy,14)
y_test=testy
print(y_test.shape)


testX = np.expand_dims(testX,2)
print(testX.shape)




# Compute confusion matrix
cnf_matrix = confusion_matrix(testy.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'L','R', 'A','a','J', 'V', 'F','!', 'j', 'E', '/', 'f', '|' ],
                     title='Confusion matrix, with normalization')
plt.show()
sens = []

spec = []
acc = []

for each in range(0,14):
    match=sum(testy.argmax(axis=1)== y_pred.argmax(axis=1))
   
    sens.append( cnf_matrix[each, each] / sum( cnf_matrix[each, :]))
    spec.append((match -  cnf_matrix[each, each]) / ((match - cnf_matrix[each, each] + sum( cnf_matrix[:, each]) -  cnf_matrix[each, each])))
    
speci = pd.DataFrame(spec)
spec=speci[0]


macc=(sens+spec)/2

sens= np.array(sens)
sens=np.transpose(sens)

print('sensitivity',sens)
print(spec)
print(macc)  
import pandas as pd 
  
# intialise data of lists. 
data = {'Sensitivity':sens, 'Specificty': spec} 
  
# Create DataFrame 
df = pd.DataFrame(data) 
  
# Print the output. 
df 
import pandas as pd 
  
# intialise data of lists. 
data = {'Sensitivity':sens, 'Specificty':spec,'MAcc':(sens+spec)/2} 
  
# Create DataFrame 
result = pd.DataFrame(data) 
  
# Print the output. 
result
#data finder 
import scipy
from scipy import signal
df=pd.read_csv("../input/datafinder/222.csv",header=None)

df1=df[1]

tetX=df1.iloc[2:701]
print(tetX)

plt.plot(tetX)
testArr=tetX.to_numpy()

plt.plot(testArr)

p=testArr.reshape((-1,1))
testX=p.transpose()
print(testX.shape)
#plt.plot(testX)

#scipy.signal.filtfilt
#scipy.signal.lfilter
#b, a = scipy.signal.butter(1, 0.5, 'low')
#output_signal = scipy.signal.filtfilt(b, a, testX)
#plt.plot(testX)
#from sklearn import preprocessing
#testX= preprocessing.normalize(testX)


testX = np.expand_dims(testX,2)
print(testX.shape)




from keras.models import model_from_json
# load json and create model
json_file = open('../input/modelloadconstant/ConstantR.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../input/modelloadconstant/ContantR.weight.h5")












   





