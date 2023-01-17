import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

pd.set_option('display.max_columns', None)
np.random.seed(42)
tf.random.set_seed(42)
XX = pd.read_csv("../input/nnfl-demo-lab-1/test-2.csv")
df = pd.read_csv("../input/nnfl-demo-lab-1/train-2.csv")
df.head(7)
#correlation matrix
k = 10
corrmat = df.corr()
cols = corrmat.nlargest(k, 'Type')['Type'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
df.columns
df.groupby('Type').mean()
df.groupby('Type').std()
X = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'Ba']]
XX = XX[['RI', 'Na', 'Mg', 'Al', 'Si', 'Ba']]
y = df['Type']
scalar = StandardScaler().fit(X)
X_scaled = scalar.transform(X)
XX_scaled = scalar.transform(XX)

XX_scaled = pd.DataFrame(XX_scaled, columns = XX.columns, index=XX.index)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns, index=X.index)
X_scaled
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)
tf.keras.backend.set_floatx('float64')
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(x=X_train, y=y_train, batch_size = 18, validation_split=0.2, epochs = 500, verbose = 1)
model.evaluate(X_test, y_test, verbose=1)
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'cyan', label='Validation Loss')
plt.plot(epochs, val_acc, 'green', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()
final_outputs = np.argmax(model.predict(XX_scaled), axis=-1)
sample = pd.read_csv("../input/nnfl-demo-lab-1/sample_submission-2.csv")
sample.loc[:, "Type"] = final_outputs
sample.to_csv("submission.csv", index=False)
model.save_weights('model.h5')