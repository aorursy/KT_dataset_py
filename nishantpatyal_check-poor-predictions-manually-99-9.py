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
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = pd.read_csv('/kaggle/input/digit-recognizer/train.csv').drop('label',1)
y = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')['label'].values

pred = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



#normalizing pixels
X = X/255
pred = pred/255

#changing datframes to arrays
X = X.values
pred = pred.values

# you know what is this for
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

#reshaping for Conv2d ayers
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
#nueral Networks
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
model = Sequential()
model.add(Conv2D(15, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))

model.add(Conv2D(55, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(784, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
#segmoid for binary classification 
#softmax for multiclass classification

model.add(Dense(10, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=8,
    verbose=1,
    mode='min',
    baseline=None,
    restore_best_weights=False)

model.fit(x=X_train,y=y_train ,epochs=500,verbose=1,validation_data=(X_test, y_test), callbacks=[es])
#model Evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print('Loss & Accuracy',model.evaluate(X_test, y_test, verbose=0))
pred = pred.reshape(pred.shape[0],28,28,1)


#writing Submision File with custom name
(pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
 .join(pd.DataFrame(model.predict_classes(pred)))
 .drop('Label',1)
 .rename(columns = {0:'Label'})
 .to_csv(str(input()),index = None))
sub = pd.read_csv('/kaggle/input/finaln/ffdffdfd.csv')
for x,i in enumerate(model.predict(pred[0:])):
    if i.max() <= 0.9 and i.max() >= 0.8:
        plt.imshow(pred[x].reshape(28,28), cmap = 'Greys')
        plt.show()
        print('Image Id:',x+1)
        print('Predicted Class:', list(i).index(i.max()))
        print('Change the value from',sub.iloc[x]['Label'],'to')
        jk = input()
        if pd.isnull(jk):
            sub.iloc[x]['Label']=sub.iloc[2]['Label']
        else:
            sub.iloc[x]['Label']=jk
            
print('\n')
print('Enter the name for your output file (add .csv at the end):')
sub.to_csv(input(), index = False)
