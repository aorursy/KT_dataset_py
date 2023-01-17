import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten

from keras.models import Sequential, save_model

from keras.utils import np_utils
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
train_features.drop(['sig_id','cp_type','cp_dose'], axis='columns', inplace=True)

train_features.head(n=5)
train_targets_scored.drop(['sig_id'], axis='columns', inplace=True)

train_targets_scored.head(n=5)
x=train_features.values

y=train_targets_scored.values

print(x.shape)

print(y.shape)

num_classes=y.shape[1]

print(num_classes)

input_dim=x.shape[1]

print(input_dim)
model = Sequential()

model.add( Dense(32, input_dim = input_dim ) )

model.add( Activation('relu') )



model.add( Dense(64 ) )

model.add( Activation('relu') )



# model.add( Flatten() )



model.add( Dropout(0.25) )

model.add( Dense(num_classes))

model.add( Activation('softmax') )
model.summary()
model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )

train_features=x

train_labels=y
model.fit( train_features, train_labels, batch_size=128, epochs=3 )

sample_submission.head(n=5)
col_names=sample_submission.columns

# print(col_names)
test_id=x_test=test_features['sig_id'].values

x_test=test_features.drop(['sig_id','cp_type','cp_dose'], axis='columns', inplace=False)

# x_test.head(n=5)

# test_id=x_test=test_features['sig_id'].values

print(test_id.shape)
y_pred=model.predict(x_test)

print(y_pred.shape)
id_list=[]

y_pred_list=[]

for i in range(len(y_pred)):

#     print(i)

    id_list.append(test_id[i])

    y_pred_list.append(y_pred[i])

#     break
print(len(id_list))

print(len(y_pred_list))
y_pred_list[0].argmax()
sub_dict={}

for i in col_names:

    sub_dict[i]=[]

# print(sub_dict.keys())
for i in range(len(y_pred)):

    sub_dict['sig_id'].append(test_id[i])

    ans=y_pred[i].argmax()+1

#     print(ans)

    for j in range(1,207):

        if j==ans:

            sub_dict[col_names[j]].append(1)

        else:

            sub_dict[col_names[j]].append(0)

#     break

#     for j
df=pd.DataFrame(sub_dict)
df.head()
df.to_csv('./submission.csv',index=False)