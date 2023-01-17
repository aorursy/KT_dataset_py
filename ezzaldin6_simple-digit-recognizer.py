import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold,train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

%matplotlib inline

plt.style.use('fivethirtyeight')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print('train shape',train_df.shape)

print('test shape',test_df.shape)
fig,ax=plt.subplots(2,4)

ax[0, 0].imshow(train_df.drop('label',axis=1).iloc[0].values.reshape(28,28), cmap='gray_r')

ax[0, 1].imshow(train_df.drop('label',axis=1).iloc[99].values.reshape(28,28), cmap='gray_r')

ax[0, 2].imshow(train_df.drop('label',axis=1).iloc[199].values.reshape(28,28), cmap='gray_r')

ax[0, 3].imshow(train_df.drop('label',axis=1).iloc[299].values.reshape(28,28), cmap='gray_r')



ax[1, 0].imshow(train_df.drop('label',axis=1).iloc[999].values.reshape(28,28), cmap='gray_r')

ax[1, 1].imshow(train_df.drop('label',axis=1).iloc[1099].values.reshape(28,28), cmap='gray_r')

ax[1, 2].imshow(train_df.drop('label',axis=1).iloc[1199].values.reshape(28,28), cmap='gray_r')

ax[1, 3].imshow(train_df.drop('label',axis=1).iloc[1299].values.reshape(28,28), cmap='gray_r')
target=train_df['label']

features=train_df.drop('label',axis=1)
def train(neuron_arch,features,target):

    mlp=MLPClassifier(hidden_layer_sizes=neuron_arch)

    mlp.fit(features,target)

    return mlp

def test(model, test_features, test_target):

    predictions = model.predict(test_features)

    score=accuracy_score(test_target,predictions)

    return score

def cross_validate(n):

    fold_acc=[]

    kf=KFold(n_splits=6,random_state=2)

    for train_index, test_index in kf.split(features):

        train_features, test_features = features.loc[train_index], features.loc[test_index]

        train_labels, test_labels = target.loc[train_index], target.loc[test_index]

        model=train(n,train_features,train_labels)

        acc=test(model,test_features,test_labels)

        fold_acc.append(acc)

    fold_acc_mean=np.mean(fold_acc)

    return fold_acc_mean
nn_one_neurons = [

    (64,),

    (128,),

    (256,)

]

nn_one_accuracies = []

for n in nn_one_neurons:

    nn_accuracies = cross_validate(n)

    nn_one_accuracies.append(nn_accuracies)

x=[i[0] for i in nn_one_neurons]

plt.plot(x,nn_one_accuracies)

plt.scatter(x,nn_one_accuracies)

plt.xlabel('Neuron Architecture')

plt.ylabel('Accuracy')

plt.show()
nn_one_accuracies
nn_two_neurons = [

    (64,64),

    (128,128),

    (256,256)

]

nn_two_accuracies = []



for n in nn_two_neurons:

    nn_accuracies = cross_validate(n)

    nn_two_accuracies.append(nn_accuracies)

x=[i[0] for i in nn_two_neurons]

plt.plot(x,nn_two_accuracies)

plt.scatter(x,nn_two_accuracies)

plt.xlabel('Neuron Architecture')

plt.ylabel('Accuracy')

plt.show()
nn_two_accuracies
mlp=MLPClassifier(hidden_layer_sizes=(256,256))

mlp.fit(features,target)

pred=mlp.predict(test_df)

pred[:15]
pred=pd.Series(pred,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)

submission.to_csv('mlp_digit_recognizer.csv',index=False)