import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
%matplotlib inline
import pdb
!pip install GPyOpt
import GPyOpt
#データの読み込み
path = '/kaggle/input/titanic/'
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
submit_df = pd.read_csv(path + 'gender_submission.csv')
train_df.head()
len(train_df)
#trainデータの欠損値確認
train_df.isnull().sum()
#testデータの欠損値確認
test_df.isnull().sum()
#train_dfの"Name"をいじる
title = [name.split(",")[1].split(".")[0] for name in train_df["Name"].tolist()]
plt.figure(figsize=(18,6))
plt.hist(title,align="left",bins=15)
set(title)
#各データセットで名前から特徴をとる
for i in range(len(title)):
    if title[i] in [' Don', ' Sir', ' Jonkheer',' the Countess', ' Lady', ' Dona']:
        title[i] = ' Noble'
    elif title[i] in [' Mlle', ' Ms']:
        title[i] = ' Miss'
    elif title[i] == ' Mme':
        title[i] = ' Mrs'
    elif title[i] in [' Capt', ' Col', ' Dr', ' Major', ' Rev']:
        title[i] = 'other'
train_df["title"] = title
title = [name.split(",")[1].split(".")[0] for name in test_df["Name"].tolist()]
for i in range(len(title)):
    if title[i] in [' Don', ' Sir', ' Jonkheer',' the Countess', ' Lady', ' Dona']:
        title[i] = ' Noble'
    elif title[i] in [' Mlle', ' Ms']:
        title[i] = ' Miss'
    elif title[i] == ' Mme':
        title[i] = ' Mrs'
    elif title[i] in [' Capt', ' Col', ' Dr', ' Major', ' Rev']:
        title[i] = 'other'
test_df["title"] = title
#それぞれの平均年齢を算出
train_df.groupby("title")["Age"].mean()
#年齢のnull値を先ほど算出した年齢で補間
train_df['Age'].fillna(train_df.groupby('title')['Age'].transform("mean"), inplace=True)
test_df['Age'].fillna(train_df.groupby('title')['Age'].transform("mean"), inplace=True)
train_df.loc[pd.isnull(train_df['Embarked'])]
#Embarkedのnull値補間（2つだけ）
train_df.loc[61,'Embarked'] = 'S'
train_df.loc[829,'Embarked'] = 'S'
#Fareのnull値の補間（1つだけ）
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace = True)
train_df.head()
#質的変数のダミー変数化
cols = ["Survived","Age","SibSp","Parch","Fare","Pclass_2","Pclass_3","Sex_male","Embarked_Q","Embarked_S"]
X = pd.get_dummies(train_df,drop_first=True,columns=["Pclass","Sex","Embarked"])[cols]

cols_test = ["Age","SibSp","Parch","Fare","Pclass_2","Pclass_3","Sex_male","Embarked_Q","Embarked_S"]
test = pd.get_dummies(test_df,drop_first=True,columns=["Pclass","Sex","Embarked"])[cols_test]
X.head()
test.head()
#"Age","SibSp","Parch","Fare"の数値を正規化に変換。
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X[["Age","Fare"]])
X[["Age","Fare"]] = scaler.transform(X[["Age","Fare"]])

scaler = preprocessing.StandardScaler()
scaler.fit(test[["Age","Fare"]])
test[["Age","Fare"]] = scaler.transform(test[["Age","Fare"]])
X.head()
test.head()
data = X[["Age","SibSp","Parch","Fare","Pclass_2","Pclass_3","Sex_male","Embarked_Q","Embarked_S"]].values
target = X["Survived"].values
#中間層を１層としてノードの個数でfor文を回して変化をみる
x_train,y_train,x_test,y_test = model_selection.train_test_split(data,target,test_size=0.1,random_state=0)
for i in range(10):
    unit = i+1
    epoch = 50
    batch=100
    model = keras.Sequential([
        keras.layers.Dense(unit, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],)
    history = model.fit(x_train,x_test,batch_size=batch,epochs=epoch,validation_split = 0.1, verbose=0)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    print("unit={}".format(unit))
    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('accuracy')
    ax1.plot(hist['epoch'], hist['accuracy'],label='Train Error')
    ax1.plot(hist['epoch'], hist['val_accuracy'],label = 'Val Error')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('loss')
    ax2.plot(hist["epoch"],hist["loss"],label="loss")
    plt.show()
model.evaluate(y_train,y_test,batch_size=batch,verbose=0)
pred = model.predict(y_train).argmax(axis=1)
accuracy_score(y_test, pred)
#中間層を２層にしてノード数 9,5とした。（なんとなく）
epoch = 500
model = keras.Sequential([
    keras.layers.Dense(9, activation='relu'),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],)
history = model.fit(x_train,x_test,batch_size=batch,epochs=epoch,validation_split = 0.1, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

#print("unit={}".format(i+1))
fig = plt.figure(figsize=(18,8))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('accuracy')
ax1.plot(hist['epoch'], hist['accuracy'],label='Train Error')
ax1.plot(hist['epoch'], hist['val_accuracy'],label = 'Val Error')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('loss')
ax2.plot(hist["epoch"],hist["loss"],label="loss")
plt.show()
pred = model.predict(y_train).argmax(axis=1)
accuracy_score(y_test, pred)
#中間層を３層にしてノード数も大きくしてみた
x_train,y_train,x_test,y_test = model_selection.train_test_split(data,target,test_size=0.1,random_state=0)
epoch = 50
model = keras.Sequential([
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],)
history = model.fit(x_train,x_test,batch_size=batch,epochs=epoch,validation_split = 0.1, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

#print("unit={}".format(i+1))
fig = plt.figure(figsize=(18,8))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('accuracy')
ax1.plot(hist['epoch'], hist['accuracy'],label='Train Error')
ax1.plot(hist['epoch'], hist['val_accuracy'],label = 'Val Error')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('loss')
ax2.plot(hist["epoch"],hist["loss"],label="loss")
plt.show()
pred = model.predict(y_train).argmax(axis=1)
accuracy_score(y_test, pred)
#dataの準備
data = X[["Age","SibSp","Parch","Fare","Pclass_2","Pclass_3","Sex_male","Embarked_Q","Embarked_S"]].values
target = X["Survived"].values
x_train,y_train,x_test,y_test = model_selection.train_test_split(data,target,test_size=0.1,random_state=0)
#ニューラルネットワークの構築と実装する関数
def neural(unit_1=10,unit_2=10,unit_3=10,batch_size=100,epoch=10):
    model = keras.Sequential([
        keras.layers.Dense(unit_1, activation='relu'),
        keras.layers.Dense(unit_2, activation='relu'),
        keras.layers.Dense(unit_3, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train,x_test,batch_size=batch_size,epochs=epoch,validation_split = 0.1, verbose=0)
    evaluation = model.evaluate(y_train,y_test,batch_size=batch,verbose=0)
    return evaluation
#ベイズ最適化用のパラメータの選択肢を作成
bounds = [{"name":"unit_1","type":"discrete","domain":(4,6,8,10,12,14,16,18,20)},
          {"name":"unit_2","type":"discrete","domain":(4,6,8,10,12,14,16,18,20)},
          {"name":"unit_2","type":"discrete","domain":(4,6,8,10,12,14,16,18,20)},
          {"name":"batch_size","type":"discrete","domain":(50,100,200)},
          {"name":"epoch","type":"discrete","domain":(5,10,20,30,40,50)}]
def f(x):
    evaluation = neural(unit_1=int(x[:,0]),unit_2=int(x[:,1]),unit_3=int(x[:,2]),batch_size=int(x[:,3]),epoch=int(x[:,4]))
    print("loss:{0} \t\t accuracy:{1}".format(evaluation[0], evaluation[1]))
    return evaluation[0]
opt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
opt.run_optimization(max_iter=30)
print("optimized parameters: {0}".format(opt.x_opt))
print("optimized loss: {0}".format(opt.fx_opt))
op_unit_1 = int(opt.x_opt[0])
op_unit_2 = int(opt.x_opt[1])
op_unit_3 = int(opt.x_opt[2])
op_batch_size = int(opt.x_opt[3])
op_epoch = int(opt.x_opt[4])
model = keras.Sequential([
        keras.layers.Dense(op_unit_1, activation='relu'),
        keras.layers.Dense(op_unit_2, activation='relu'),
        keras.layers.Dense(op_unit_3, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
history = model.fit(x_train,x_test,batch_size=op_batch_size,epochs=op_epoch,validation_split = 0.1, verbose=0)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

fig = plt.figure(figsize=(18,8))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('accuracy')
ax1.plot(hist['epoch'], hist['accuracy'],label='Train Error')
ax1.plot(hist['epoch'], hist['val_accuracy'],label = 'Val Error')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('loss')
ax2.plot(hist["epoch"],hist["loss"],label="loss")
plt.show()
pred = model.predict(y_train).argmax(axis=1)
accuracy_score(y_test, pred)
sub_pred = model.predict(test).argmax(axis=1)
sub_df["Survived"] = sub_pred
sub_df.to_csv("../submit_csv/titanic_neural_network_2.csv",index=False)
