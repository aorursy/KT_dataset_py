import numpy as np 

import pandas as pd 



dftrain_raw = pd.read_csv('../input/train.csv')

dftest_raw = pd.read_csv('../input/test.csv')



dftrain_raw.head(10)
%matplotlib inline

%config InlineBackend.figure_format = 'png'

ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar',

     figsize = (12,8),fontsize=15,rot = 0)

ax.set_ylabel('Counts',fontsize = 15)

ax.set_xlabel('Survived',fontsize = 15)
%matplotlib inline

%config InlineBackend.figure_format = 'png'

ax = dftrain_raw['Age'].plot(kind = 'hist',bins = 20,color= 'purple',

                    figsize = (12,8),fontsize=15)



ax.set_ylabel('Frequency',fontsize = 15)

ax.set_xlabel('Age',fontsize = 15)
%matplotlib inline

%config InlineBackend.figure_format = 'svg'

ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind = 'density',

                      fontsize=10)

dftrain_raw.query('Survived == 1')['Age'].plot(kind = 'density',

                      fontsize=10)

ax.legend(['Survived==0','Survived==1'],fontsize = 10)

ax.set_ylabel('Density',fontsize = 10)

ax.set_xlabel('Age',fontsize = 10)
%matplotlib inline

%config InlineBackend.figure_format = 'svg'

ax = dftrain_raw['Sex'].value_counts(dropna= False).plot(kind = 'bar',

     fontsize=15,rot = 0)



ax.set_ylabel('Count',fontsize = 15)

ax.set_xlabel('Sex',fontsize = 15)

# 数据预处理

def preprocessing(dfdata):



    dfresult= pd.DataFrame()



    #Pclass

    dfPclass = pd.get_dummies(dfdata['Pclass'])

    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]

    dfresult = pd.concat([dfresult,dfPclass],axis = 1)



    #Sex

    dfSex = pd.get_dummies(dfdata['Sex'])

    dfresult = pd.concat([dfresult,dfSex],axis = 1)



    #Age

    dfresult['Age'] = dfdata['Age'].fillna(dfdata['Age'].mean())

    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')



    #SibSp,Parch,Fare

    dfresult['SibSp'] = dfdata['SibSp']

    dfresult['Parch'] = dfdata['Parch']

    dfresult['Fare'] = dfdata['Fare']



    #Carbin

    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')



    #Embarked

    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)

    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]

    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    

    return(dfresult)



x_train = preprocessing(dftrain_raw)

y_train = dftrain_raw['Survived'].values



x_test = preprocessing(dftest_raw)
print('x_train.shape:',x_train.shape)

print('x_test.shape:',x_test.shape)
from keras import models,layers

from keras import backend as K

K.clear_session()



model = models.Sequential()

model.add(layers.Dense(64,activation = 'relu',input_shape=(15,)))

model.add(layers.Dense(64,activation = 'relu' ))

model.add(layers.Dense(64,activation = 'relu' ))

model.add(layers.Dense(64,activation = 'relu' ))

model.add(layers.Dense(1,activation = 'sigmoid' ))



# 二分类问题选择二元交叉熵损失函数

model.compile(optimizer='adam',

            loss='binary_crossentropy',

            metrics=['accuracy'])

model.summary()
history = model.fit(x_train,y_train,

                    batch_size= 64,

                    epochs= 150,

                    validation_split=0.2 #分割一部分训练数据用于验证

                   )



import pandas as pd 

dfhistory = pd.DataFrame(history.history)

dfhistory.index = range(1,len(dfhistory) + 1)

dfhistory.index.name = 'epoch'

dfhistory 
history = model.fit(x_train,y_train,

                    batch_size= 64,

                    epochs= 150,

                    validation_data =(x_train,y_train) #重新用全部样本进行训练

                   )
import pandas as pd 

dfhistory = pd.DataFrame(history.history)

dfhistory.index = range(1,len(dfhistory) + 1)

dfhistory.index.name = 'epoch'
dfhistory
import matplotlib.pyplot as plt



%matplotlib inline

%config InlineBackend.figure_format = 'svg'



loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
import matplotlib.pyplot as plt



%matplotlib inline

%config InlineBackend.figure_format = 'svg'



acc = history.history['acc']

val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
# 输出概率

model.predict(x_test[0:5])
# 输出类别

model.predict_classes(x_train[0:5])
y_test_pred = model.predict_classes(x_test)

y_test_pred
dfresult = pd.DataFrame()

dfresult['PassengerId'] = dftest_raw['PassengerId']

dfresult['Survived'] = y_test_pred



dfresult.head(10)

dfresult.index = range(1,len(dfresult) + 1)



dfresult.to_csv('submission.csv',index = None)
!head submission.csv
# 保存模型结构

json_str = model.to_json()

yaml_str = model.to_yaml()



#保存模型权重

#model.save_weights('model_weights.h5')



# 恢复模型结构

#model_json = models.model_from_json(json_str)

#model_json.compile(optimizer='adam',

#            loss='binary_crossentropy',

#            metrics=['accuracy'])

# 加载权重

#model_json.load_weights('model_weights.h5')



#model_json.evaluate(x_train,y_train)
!pip install kaggle
!pip install --upgrade pip 
import json

j = {"username":"lyhue1991","key":"783423432ed3418189b6c"}
with open('/tmp/.kaggle/kaggle.json','w') as f:

    json.dump(j,f)
!kaggle competitions submit -c titanic -f submission.csv -m "give me seven"
!kaggle config view
!kaggle kernels list 