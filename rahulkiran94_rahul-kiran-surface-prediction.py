# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



X_train=pd.read_csv('../input/X_train.csv')

y_train=pd.read_csv('../input/y_train.csv')





X_train.head(10)
y_train.head(10)
X_train.shape
y_train.shape
X_test=pd.read_csv('../input/X_test.csv')

X_test.head(10)
X_test.shape
for i in X_train.columns:

    print(X_train[i].value_counts())
def eulerAnglesToRotationMatrix(theta) :

     

    R_x = np.array([[1,         0,                  0                   ],

                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],

                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]

                    ])

         

         

                     

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],

                    [0,                     1,      0                   ],

                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]

                    ])

                 

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],

                    [math.sin(theta[2]),    math.cos(theta[2]),     0],

                    [0,                     0,                      1]

                    ])

                     

                     

    R = np.dot(R_z, np.dot( R_y, R_x ))

 

    return R



df=X_train.copy()

df.head(10)

df['T_acceleration']=np.sqrt(np.square(df['linear_acceleration_X'])+np.square(df['linear_acceleration_Y'])+np.square(df['linear_acceleration_Z']))

df['T_angular_v']=np.sqrt(np.square(df['angular_velocity_X'])+np.square(df['angular_velocity_Y'])+np.square(df['angular_velocity_Z']))
df2=df.join(y_train.set_index('series_id'), on='series_id')
df2.head(10)
# a quaternion is a complex number with w as the real part and x, y, z as imaginary parts.

#q=s+xi+yj+zk
df2.drop('row_id',axis=1,inplace=True)
df2.head(10)
#w = cos(theta / 2)



df2['theta']=2*np.arccos(df2['orientation_W'])
df2.head(10)
df2['x'] = (df2['angular_velocity_X']*np.sin(df2['theta'])/ 2)

df2['y'] = (df2['angular_velocity_Y']*np.sin(df2['theta'])/ 2)

df2['z'] = (df2['angular_velocity_Z']*np.sin(df2['theta'])/ 2)
#df2['theta'].value_counts()
df2['theta2']=df2['theta'].round(3)
#features_selected=['series_id','T_acceleration','group_id','x','y','z','theta2','surface']

df2['RotZ90']=df2['z']*np.sin(0.5*90)





#features_selected=['series_id','T_acceleration','theta2','RotZ90','surface']

features_selected=['series_id','T_acceleration','x','y','z','theta2','RotZ90','angular_velocity_X','angular_velocity_Y','angular_velocity_Z','T_angular_v']



colList=df2.columns

colList=colList.drop('surface')

colList=colList.drop('theta')

colList=colList.drop('measurement_number')

colList=colList.drop('group_id')

normalize =df2[features_selected].copy()

normalize=(np.min(normalize)-normalize)/(np.max(normalize)-np.min(normalize))

normalize['series_id']=df2['series_id']

labels = pd.Series(df2['surface'])

features["bias"] = 1



shuffled_index = np.random.permutation(df2.index)

shuffled_data = features.loc[shuffled_index]

shuffled_labels = labels.loc[shuffled_index]

mid_length = int(len(shuffled_data)/2)



train_features = shuffled_data.iloc[0:mid_length]

test_features = shuffled_data.iloc[mid_length:len(shuffled_data)]

train_labels = shuffled_labels.iloc[0:mid_length]

test_labels = shuffled_labels.iloc[mid_length: len(labels)]
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score



mlp = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=1000,activation='tanh',random_state=1,alpha=0.1,solver='adam')

#mlp = MLPClassifier(hidden_layer_sizes=(7,3), max_iter=1000,activation='tanh',random_state=1,alpha=0.1,solver='adam')

mlp.fit(train_features, train_labels)

nn_predictions = mlp.predict(test_features)



nn_accuracy = accuracy_score(test_labels, nn_predictions)



print("NN Model Accuracy: ", nn_accuracy)

X_test2=X_test.copy()

X_test2['T_acceleration']=np.sqrt(np.square(X_test['linear_acceleration_X'])+np.square(X_test['linear_acceleration_Y'])+np.square(X_test['linear_acceleration_Z']))

X_test2['T_angular_v']=np.sqrt(np.square(X_test['angular_velocity_X'])+np.square(X_test['angular_velocity_Y'])+np.square(X_test['angular_velocity_Z']))

X_test2['theta']=2*np.arccos(X_test['orientation_W'])

X_test2['theta2']=X_test2['theta'].round(3)
X_test2['x'] = (X_test['angular_velocity_X']*np.sin(X_test2['theta'])/ 2)

X_test2['y'] = (X_test['angular_velocity_Y']*np.sin(X_test2['theta'])/ 2)

X_test2['z'] = (X_test['angular_velocity_Z']*np.sin(X_test2['theta'])/ 2)

X_test2['RotZ90']=X_test2['z']*np.sin(0.5*90)
colList2=X_test2.columns

colList2=colList2.drop('theta')

colList2=colList2.drop('row_id')

colList2=colList2.drop('measurement_number')

normalize2 =X_test2[colList2].copy()

normalize2.isnull().sum()
normalize2=(np.min(normalize2)-normalize2)/(np.max(normalize2)-np.min(normalize2))
normalize2['series_id']=X_test['series_id']
normalize.columns

#normalize2["bias"] = 1



normalize2.drop('bias',axis=1,inplace=True)



normalize2.columns=normalize.columns
mlp.fit(shuffled_data, shuffled_labels)

nn_predictions2 = mlp.predict(normalize2)
result=normalize[['series_id']]

result['surface']=nn_predictions2
result.head(500)
result.to_csv('submisson.csv',index = False)