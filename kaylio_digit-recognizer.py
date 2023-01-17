import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

%matplotlib inline
def create_model(num_layers, num_nodes):
    
    # Initialise sequential model
    model = Sequential()
    
    early_stopping = EarlyStopping
    
    for i in list(range(num_layers)):
        if i == 0:
            model.add(Dense(num_nodes, activation='relu', input_shape=(X.shape[1], )))
        else:
            model.add(Dense(num_nodes, activation='relu'))
    model.add(Dense(10, activation='softmax'))
            
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
# Let's now load in the full dataset and train our model on it
df = pd.read_csv('../input/train.csv')

df.shape
X = df.drop('label', axis=1).values
y = to_categorical(df.label)
model = create_model(5, 50)

model.fit(X, y, epochs=20)
X_test = pd.read_csv('../input/test.csv').values

predictions = model.predict(X_test)
preds_values = np.argmax(predictions, axis=1)
image_id = list(range(1, len(preds_values)+1))

my_submission = pd.DataFrame({'ImageId': image_id, 'Label': preds_values})
my_submission.set_index('ImageId')

my_submission.to_csv('submission.csv', index=False)