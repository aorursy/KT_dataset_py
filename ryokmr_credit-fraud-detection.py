import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
data=pd.read_csv("../input/creditcard.csv")
data.head(10)
print(data.shape)
print(data.Time.max())
print(data.Time.min())
print(data.Amount.max())
print(data.Amount.min())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data.iloc[:,[0,29]]=scaler.fit_transform(data.iloc[:,[0,29]])
data.head()
print(data.Amount.max())
print(data.Amount.min())
class_count=pd.value_counts(data.Class,sort=True)
class_count.plot(kind='bar')
plt.xlabel("Class")
plt.ylabel("Freq")
plt.title("Class vs Freq")
X=data.ix[:,data.columns!="Class"]
Y=data.ix[:,data.columns=="Class"]
print(X.shape)
print(Y.shape)
number_fraud=len(data[data.Class==1])
number_fraud_index=np.array(data[data.Class==1].index)
number_normal=np.array(data[data.Class==0].index)

number_normal_index=np.random.choice(number_normal,number_fraud,replace=False)
assert(number_normal_index.shape==number_fraud_index.shape)
print(len(number_normal_index))
undersample_index=np.concatenate([number_fraud_index,number_normal_index])
print(undersample_index.shape)
undersample_data=data.iloc[undersample_index,:]
X_sampling=undersample_data.iloc[:,:30]
Y_sampling=undersample_data.iloc[:,30:]
print(X.shape)
print(Y.shape)
print(len(undersample_data[undersample_data.Class==0]))
from sklearn.model_selection import train_test_split
x_sampling_train,x_sampling_test,y_sampling_train,y_sampling_test=train_test_split(X_sampling,Y_sampling,test_size=0.3)
print(x_sampling_train.shape)
print(x_sampling_test.shape)
from sklearn.svm import SVC
classifier=SVC(kernel="rbf",probability=True)
classifier.fit(x_sampling_train,y_sampling_train)
y_sampling_pred=classifier.predict(x_sampling_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_sampling_test,y_sampling_pred)
from sklearn.metrics import recall_score
recall_score(y_sampling_test,y_sampling_pred)
Y_pred=classifier.predict(X)
print(confusion_matrix(Y,Y_pred))
print(recall_score(Y,Y_pred))
threshold=[0.1,0.2,0.4,0.5]
for i in threshold:
    Y_pred_temp=classifier.predict_proba(X)
    Y_pred_temp=Y_pred_temp[:,1]>i
    print("-------------------------------------------------------------")
    print("Threshold : "+ str(i)+"\n")
    print(confusion_matrix(Y,Y_pred_temp))
    print("\n")
    print(recall_score(Y,Y_pred_temp))