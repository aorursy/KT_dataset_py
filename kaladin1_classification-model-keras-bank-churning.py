# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def load_dataset():
    return pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv', index_col=0)
df = load_dataset()
df.head()
df.nunique()
# droping CustomerId column 
df.drop(columns=['CustomerId', 'Surname'], inplace=True)
# feature_set 
X = df.drop(columns=['Exited'])
# target_set 
y = df.pop('Exited')
X.info()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

num_attribs = df.select_dtypes(exclude='object').columns.values
cat_attribs = df.select_dtypes(include='object').columns.values
num_pipeline = Pipeline([
    ("std_scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("cat_encoder", OneHotEncoder(drop="first"))
])
ct = ColumnTransformer([
    ("cat", cat_pipeline, cat_attribs),
    ("num", num_pipeline, num_attribs)
])
X = ct.fit_transform(X)
# The classes are imbalanced 
sns.countplot(y)
from imblearn.over_sampling import SMOTE

# using smote for oversampling
smote = SMOTE()
X, y = smote.fit_resample(X, y)
# distribution of classes after oversampling using smote
sns.countplot(y)
from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.2)
print(f'training shapes: {X_train.shape}, {y_train.shape}')
print(f'testing shapes: {X_val.shape}, {y_val.shape}')
import tensorflow as tf 
from tensorflow import keras 
model = keras.models.Sequential([
    keras.layers.Input(shape=(X_train.shape[1], )),
    keras.layers.Dense(units=300, activation='relu'),
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, callbacks=[early_stopping_cb], validation_data=(X_val, y_val))
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.xlabel('epochs')
plt.ylabel('scores')
plt.show()
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)
print(y_pred[:10])
from sklearn.metrics import accuracy_score, confusion_matrix 
import seaborn as sns 

print(f'accuracy_score={accuracy_score(y_val, y_pred)}')
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d',
            cmap='binary', cbar=False)
