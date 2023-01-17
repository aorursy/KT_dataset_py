# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df1=pd.read_csv("/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv",index_col="job_id")
df1["location"]=df1["location"].fillna("LOC")
df1["department"]=df1["department"].fillna("DEPART")
df1["salary_range"]=df1["salary_range"].fillna("0-0")
df1["company_profile"]=df1["company_profile"].fillna("No Description")
df1["description"]=df1["description"].fillna("No Description")
df1["requirements"]=df1["requirements"].fillna("No Description")
df1["benefits"]=df1["benefits"].fillna("Adequete benefits")
df1["employment_type"]=df1["employment_type"].fillna("Other")
df1["required_experience"]=df1["required_experience"].fillna("Not Applicable")
df1["required_education"]=df1["required_education"].fillna("Bachelor's Degree")
df1["industry"]=df1["industry"].fillna("None")
df1["function"]=df1["function"].fillna("None")
df1.head()
# df1.industry.value_counts()
df1.info()
from gensim.models import Doc2Vec
model=Doc2Vec.load("/kaggle/input/doc2vec-english-binary-file/doc2vec.bin")
#=========[CLEAN DATA]==============#
# !pip install textcleaner==0.4.26
import string

#=========[CLEAN PANDAS]==============#
# employment_type	required_experience	required_education	industry	function
from sklearn.preprocessing import LabelEncoder
df1["location"]=LabelEncoder().fit_transform(df1["location"])
df1["department"]=LabelEncoder().fit_transform(df1["department"])
df1["salary_range"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["employment_type"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["required_experience"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["required_education"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["industry"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["function"]=LabelEncoder().fit_transform(df1["salary_range"])
df1.head()
!pip install clean-text
from cleantext import clean

print("#==============[BEFORE]======================#")
print(df1["company_profile"].iloc[0])
print("#==============[AFTER]======================#")
text=clean(df1["company_profile"].iloc[0],no_punct=True)
print(text)
def convert_to_embeddings(text):
    try:
        text=clean(text,no_punct=True)
    except:
        text=" "
    return model.infer_vector(text.split())



#==========[IVED SAVED THIS PORTION IN .NPY FILE]=======================#
# df1["title"]=df1["title"].apply(convert_to_embeddings)
# df1["company_profile"]=df1["company_profile"].apply(convert_to_embeddings)
# df1["description"]=df1["description"].apply(convert_to_embeddings)
# df1["requirements"]=df1["requirements"].apply(convert_to_embeddings)
# df1["benefits"]=df1["benefits"].apply(convert_to_embeddings)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

swag=np.load("/kaggle/input/df1tonpy1/data.npy",allow_pickle=True)

training_data_text=np.hstack([np.vstack(swag[:,0]),np.vstack(swag[:,4]),np.vstack(swag[:,5]),np.vstack(swag[:,6]),np.vstack(swag[:,7])])
training_data_text.shape

training_data=np.hstack([training_data_text,swag[:,1:3],swag[:,8:]])


training_data=scaler.fit_transform(training_data)

X=training_data[:,:-1]
Y=training_data[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
from tensorflow.keras.layers import Dense,Input,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential

model2=Sequential()
model2.add(Input(shape=(X.shape[1])))
model2.add(BatchNormalization())
model2.add(Dense(128,activation=tf.nn.selu))
model2.add(Dropout(0.5))
model2.add(Dense(64,activation=tf.nn.selu))
model2.add(Dropout(0.2))
model2.add(Dense(32,activation=tf.nn.selu))
model2.add(Dense(1,activation=tf.nn.sigmoid))


model2.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

model2.summary()
history=model2.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=80)
from sklearn.metrics import classification_report,confusion_matrix

pred=model2.predict(X_test)
pred=np.array([1 if row>=0.5 else 0 for row in pred])
print(classification_report(y_test,pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)
plt.show()

pred=model2.predict(X_train)
pred=np.array([1 if row>=0.5 else 0 for row in pred])
print(classification_report(y_train,pred))
sns.heatmap(confusion_matrix(y_train,pred),annot=True)
plt.plot(history.history["val_loss"])
plt.plot(history.history["loss"])

