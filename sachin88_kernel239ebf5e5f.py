from keras import losses

from keras.utils import to_categorical

from keras.layers import Input, Dense, Dropout

from keras.models import Model, Sequential 

from keras.optimizers import Adam

from keras import optimizers

from keras import backend as K

from keras.callbacks import Callback

from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/prudential-life-insurance-assessment/train.csv')
test=pd.read_csv('/kaggle/input/prudential-life-insurance-assessment/test.csv')
submission=pd.read_csv('/kaggle/input/prudential-life-insurance-assessment/sample_submission.csv')
submission.index
train.columns
train.describe()
train.dropna()
train_missing = train.isna()
train_num_missing = train_missing.sum()
a=pd.DataFrame(train_num_missing,columns=['null_count'])

a['total']=len(train)

a['percentage_missing']=a['null_count']/a['total']
a[a.null_count>0]
unique_count=pd.DataFrame(train.nunique(),columns=['unique'])
unique_count
unique_count[unique_count.unique>100]
train_new=train.drop(['Medical_History_32','Medical_History_24','Medical_History_15','Medical_History_10',

                      'Family_Hist_5','Family_Hist_3','Family_Hist_2','Insurance_History_5'], axis=1)
test_new=test.drop(['Medical_History_32','Medical_History_24','Medical_History_15','Medical_History_10',

                      'Family_Hist_5','Family_Hist_3','Family_Hist_2','Insurance_History_5'], axis=1)
train_new.shape,test_new.shape
import seaborn as sn
x=pd.Series(train_new['Employment_Info_1'])
sn.distplot(x,bins=300)
train_new['Employment_Info_1'].median()
test_new['Employment_Info_1'].median()
train_new['Employment_Info_1'] = train_new['Employment_Info_1'].fillna(train_new['Employment_Info_1'].median())
test_new['Employment_Info_1'] = test_new['Employment_Info_1'].fillna(train_new['Employment_Info_1'].median())
train_new[['Employment_Info_1']].isna().sum()
x1=pd.Series(train_new['Employment_Info_4'])
sn.distplot(x1,bins=1,)
train_new['Employment_Info_4'] = train_new['Employment_Info_4'].fillna(train_new['Employment_Info_4'].median())
test_new['Employment_Info_4'] = test_new['Employment_Info_4'].fillna(train_new['Employment_Info_4'].median())
train_new['Employment_Info_4'].isna().sum()
x2=pd.Series(train_new['Employment_Info_6'])
sn.distplot(x2,bins=10)
train_new['Employment_Info_6'].mean()
train_new['Employment_Info_6'] = train_new['Employment_Info_6'].fillna(train_new['Employment_Info_6'].mean())
test_new['Employment_Info_6'] = test_new['Employment_Info_6'].fillna(train_new['Employment_Info_6'].mean())
train_new['Employment_Info_6'].isna().sum()
train_new['Family_Hist_4']
x3=pd.Series(train_new['Family_Hist_4'])
sn.distplot(x3,bins=10)
train_new['Family_Hist_4'].mean()
train_new['Family_Hist_4'] = train_new['Family_Hist_4'].fillna(train_new['Family_Hist_4'].mean())
test_new['Family_Hist_4'] = test_new['Family_Hist_4'].fillna(train_new['Family_Hist_4'].mean())
train_new['Family_Hist_4'].isna().sum()
x4=pd.Series(train_new['Medical_History_1'])
train_new['Medical_History_1'].median()
train_new['Medical_History_1'] = train_new['Medical_History_1'].fillna(train_new['Medical_History_1'].median())
test_new['Medical_History_1'] = test_new['Medical_History_1'].fillna(train_new['Medical_History_1'].median())
train_new['Medical_History_1'].isna().sum()
train_new.head()
test_new.head()
train_new['Product_Info_2'] = train_new['Product_Info_2'].astype('category').cat.codes
test_new['Product_Info_2'] = test_new['Product_Info_2'].astype('category').cat.codes
sn.scatterplot(x="Ins_Age", y="Wt", data=train_new)
train_new['Ht'].corr(train_new['Wt'])
train_new['Ht_Wt']=train_new['Ht']*train_new['Wt']
test_new['Ht_Wt']=test_new['Ht']*test_new['Wt']
train_x = train_new.drop(['Id', 'Response'], axis=1)

train_y = train_new['Response']



print(train_x.shape)

print(train_y.shape)
test_x=test_new.drop(['Id'],axis=1)

print(test_x.shape)
train_y = train_y-1

train_y = to_categorical(train_y, num_classes= 8)
train_x.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_x_scaled=pd.DataFrame(scaler.fit_transform(train_x),columns=train_x.columns)
test_x_scaled=pd.DataFrame(scaler.transform(test_x),columns=test_x.columns)
train_x_scaled.head()
test_x_scaled.head()
# create model

model = Sequential()

model.add(Dense(400, input_dim=119, activation='relu'))

model.add(Dense(400, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(300, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(8, activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x_scaled,train_y,batch_size=32, epochs=100)
pred = model.predict(test_x_scaled)

y = np.argmax(pred, axis=1)+1
y
submission.head()
submission.Response=y
submission.head()
submission.columns
submission = submission.set_index('Id')
submission.head()
submission.to_csv('sub_8.csv')
submission.head()