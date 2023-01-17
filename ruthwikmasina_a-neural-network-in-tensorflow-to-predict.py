import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
import tensorflow as tf
from tensorflow.keras.layers import Dense
df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df.head()
plt.hist(df['Target'].astype(str))
plt.title('Target histogram')
plt.ylabel('Count')
plt.xlabel('Class')
plt.show()
for feature in df.columns:
    print (feature)
# sns.barplot(x=train_null_non_zero, y=train_null_non_zero.index)
# _ = plt.title('Fraction of NaN values, %')
train_null = df.isnull().sum()
train_null_non_zero = train_null[train_null>0] / df.shape[0]
train_null_non_zero
df = df.fillna(df.mean())
test = test.fillna(test.mean())
y = df['Target']
X = df.drop(['Target', 'Id'], axis=1)
test_id = test['Id']
test.drop('Id', axis=1, inplace=True)
train_test_df = pd.concat([X, test], axis=0)
cols = [col for col in train_test_df.columns if train_test_df[col].dtype == 'object']

le = LabelEncoder()
for col in cols:
    le.fit(train_test_df[col])
    X[col] = le.transform(X[col])
    test[col] = le.transform(test[col])
y.unique()
y_test_classes = pd.get_dummies(y,prefix=['Target'])
from xgboost import XGBClassifier
from xgboost import plot_importance

# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance

plt.rcParams["figure.figsize"] = (15,20)
plot_importance(model)
plt.show()
# Y = df[['Target']].values
X_train,X_test,Y_train,Y_test = train_test_split(X,y_test_classes,test_size=0.2)
print(X_train.shape,Y_train.shape)
def get_model(n_x, n_h1, n_h2,n_h3):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_h1, input_dim=n_x, activation='relu'))
    model.add(tf.keras.layers.Dense(n_h2, activation='relu'))
    model.add(tf.keras.layers.Dense(n_h3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    return model
(m,n_x) = X.shape
n_h1 = 256
n_h2 = 1024
n_h3 = 512
batch_size = 128
epochs = 150

print(n_x)
model = get_model(n_x, n_h1, n_h2, n_h3)
# Set a learning rate annealer
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.6, 
                                            min_lr=0.0001)

# Set EarlyStopping
EStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=20, verbose=2,
                      mode='auto')

history = model.fit(X,y_test_classes,validation_data=(X_test,Y_test),
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=[learning_rate_reduction, EStop],verbose=2)
y_predict = model.predict(test)
y_predict= pd.Series([np.argmax(x)+1 for x in y_predict])
pred = pd.DataFrame({"Id": test_id, "Target": y_predict})
pred.to_csv('submission.csv', index=False)
pred.head()