import numpy as np

import pandas as pd 
train = pd.read_csv('../input/train_file.csv')

test = pd.read_csv('../input/test_file.csv')
train.head()
train.isna().sum()
train.dtypes
train['YEAR'].value_counts().plot.bar()
train['LocationDesc'].value_counts()
train['Subtopic'].value_counts(normalize = True).plot.bar()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
plt.figure(1,figsize=(16,6))



plt.subplot(121)

sns.distplot(train['Sample_Size'])



plt.subplot(122)

sns.boxplot(y=train['Sample_Size'])



plt.show()
plt.figure(1,figsize=(16,6))



plt.subplot(121)



sns.distplot(np.log(train['Sample_Size']))



plt.subplot(122)

sns.boxplot(y=np.log(train['Sample_Size']))



plt.show()
train.Sex.value_counts().plot.bar()
train.loc[train['Sex']=='Total'].head()
train['Race'].value_counts().plot.bar(figsize = (16,6))
train['Grade'].value_counts().plot.bar(figsize = (10,6))
train['StratificationType'].value_counts().plot.bar()
def make_plot(df,col_name,figsize=(16,6)):

    

    plt.figure(1,figsize=figsize)

    dic = {}

    for val in df[col_name].value_counts().keys():

        dic[val] = np.mean(df['Greater_Risk_Probability'].loc[df[col_name]==val])

    

    plt.bar(range(len(dic)),dic.values(),align='center')

    plt.xticks(range(len(dic)),dic.keys())

    plt.show()
make_plot(train,'YEAR')
dic = {}

for val in train['LocationDesc'].value_counts().keys():

    dic[val] = np.mean(train['Greater_Risk_Probability'].loc[train['LocationDesc']==val])



dic = sorted(dic.items(),key=lambda x: x[1])[::-1]



dic
make_plot(train,col_name='Subtopic',figsize=(8,4))
plt.figure(1,(16,8))

plt.scatter(x=train['Greater_Risk_Probability'],y=train['Sample_Size'])
make_plot(train,'Sex',(8,4))
make_plot(train,'Race',(16,4))
make_plot(train,'Grade',(8,4))
make_plot(train,'StratificationType',(8,4))
X = train.copy()

X_test = test.copy()



y= X['Greater_Risk_Probability']

X = X.drop(labels = 'Greater_Risk_Probability',axis=1)



from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

X['LocationDesc'] = label.fit_transform(X['LocationDesc'])

X_test['LocationDesc'] = label.fit_transform(X_test['LocationDesc'])



label = LabelEncoder()

X['Sex'] = label.fit_transform(X['Sex'])

X_test['Sex'] = label.fit_transform(X_test['Sex'])



label = LabelEncoder()

X['Race'] = label.fit_transform(X['Race'])

X_test['Race'] = label.fit_transform(X_test['Race'])



label = LabelEncoder()

X['StratificationType'] = label.fit_transform(X['StratificationType'])

X_test['StratificationType'] = label.fit_transform(X_test['StratificationType'])



label = LabelEncoder()

X['QuestionCode'] = label.fit_transform(X['QuestionCode'])

X_test['QuestionCode'] = label.fit_transform(X_test['QuestionCode'])



X['Sample_Size'] = np.log(X['Sample_Size'])

X_test['Sample_Size'] = np.log(X_test['Sample_Size'])



# drop = ['Patient_ID','Greater_Risk_Question','Description','GeoLocation','QuestionCode']

drop = ['Patient_ID','Greater_Risk_Question','Description','GeoLocation']

X = X.drop(labels = drop,axis=1)

X_test = X_test.drop(labels = drop,axis=1)
from sklearn.feature_selection import f_classif
fval,p_val = f_classif(X,y)



print('F-values for different features')

print(fval)



print('P-values for different features')

print(p_val)
X = X.drop(labels = ['StratificationType','Race','Grade'],axis=1)

X_test = X_test.drop(labels = ['StratificationType','Race','Grade'],axis=1)



fval,p_val = f_classif(X,y)



print('F-values for different features')

print(fval)



print('P-values for different features')

print(p_val)
from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

import warnings

warnings.filterwarnings('ignore')
continuous_cols = ['YEAR','Sample_Size']



categorical_cols = ['LocationDesc','Sex','StratID1','StratID2','StratID3','QuestionCode']



mapper = DataFrameMapper(  

    [([continuous_col], StandardScaler()) for continuous_col in continuous_cols] +

    [([categorical_col], OneHotEncoder()) for categorical_col in categorical_cols])



pipe = Pipeline([('mapper',mapper)])



pipe.fit(X)
X = pipe.transform(X)

X_test = pipe.transform(X_test)
X.shape
y = (y/100).values
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss,mean_squared_error
kf = KFold(n_splits=10,shuffle=True,random_state=42)

cv_scores = []



for train_index, val_index in kf.split(X):

    

    X_train, X_val = X[train_index],X[val_index]

    y_train, y_val = y[train_index],y[val_index]

    

    regressor = LinearRegression(n_jobs=-1)

    regressor.fit(X_train,y_train)



    pred = regressor.predict(X_val)

    mean = mean_squared_error(y_val,pred)

    cv_scores.append(mean)

    print(mean)

print(f"Mean Score {np.mean(cv_scores)}")
from keras.layers import Dense, Dropout

from keras.models import Sequential
model = Sequential()

model.add(Dense(256,input_dim=132,activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(64,activation='relu'))

model.add(Dropout(rate=0.7))

model.add(Dense(1,activation='relu'))



model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_squared_error'])
model.summary()
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
checkpoints = ModelCheckpoint('model.h5',monitor='val_mean_squared_error',mode='min',save_best_only='True',verbose=True)

reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.1, patience=2, verbose=1, min_lr=0.000001)
epochs = 50

batch_size = 64
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 

                    validation_data=[X_val, y_val], callbacks=[checkpoints, reduce_lr])
import matplotlib.pyplot as plt



plt.figure(figsize=(12,8))

plt.plot(history.history['mean_squared_error'], label='Train MSE')

plt.plot(history.history['val_mean_squared_error'], label='Test MSE')

plt.legend(('Train MSE', 'Val MSE'))

plt.show()
model.load_weights('model.h5')
pred_test = (model.predict(X_test)*100).round(4)

test['Greater_Risk_Probability'] = pred_test

test.head()
df_sub = test.loc[:,['Patient_ID','Greater_Risk_Probability']]

df_sub.head()
df_sub.to_csv(path_or_buf = 'submission.csv',index=False)