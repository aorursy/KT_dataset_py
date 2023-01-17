import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers.core import Dense



TARGET_VARIABLE = 'Class';

TRAIN_TEST_SPLIT=0.5;

HIDDEN_LAYER_SIZE=30;

df = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv', na_values='?');
df.info()
df.head()
df.tail()
df.isnull().sum()

#df.dropna()
int_cols = df.columns[df.dtypes==float];

for x in int_cols:

    df[x].fillna(df[x].mean(), inplace=True);

df.isnull().sum()
df = df.drop_duplicates()

df.info()

df['Size'].isna().sum()
df = df[~df['Size'].isna()]

df.info()

df['Size'].unique()
df_onehot = df.copy()

df_onehot = pd.get_dummies(df_onehot, columns=['Size'], prefix = ['Size'])

df_onehot.head()
corr = df_onehot.corr();

import matplotlib.pyplot as plt

import seaborn as sns

fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), square=True, ax=ax, annot = False, cmap = 'Spectral')
df_drop_corr = df_onehot.drop(columns = ['Number of Insignificant Quantities', 'Number of Sentences', 'Total Number of Characters', 'ID'])

df_drop_corr.info()
corr = df_drop_corr.corr();

import matplotlib.pyplot as plt

import seaborn as sns

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), square=True, ax=ax, annot = False, cmap = 'Spectral')
dff = df_drop_corr;

dff.describe()
dff2 = dff.copy();

dff2 = pd.get_dummies(dff2, columns = ['Class'], prefix = ['Class']);

X = dff2.drop(columns = ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']);

y = dff2[['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']].copy(deep = True);

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler();

scaled_data=scaler.fit(X).transform(X);

scaled_df=pd.DataFrame(scaled_data,columns=X.columns);

scaled_df.tail()


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(X,y)
# model = Sequential()

# model.add(Dense(16,input_dim=23, activation='relu'))

# model.add(Dropout(rate=0.2))

# model.add(Dense(8, activation='relu'))

# model.add(Dropout(rate=0.2))

# model.add(Dense(1,activation='softmax'))

print(len(X.columns))

from keras.layers import Dense, Dropout

from keras.models import Sequential



from sklearn.metrics import mean_absolute_error
import keras

model = Sequential();

model.add(Dense(32, activation='tanh', input_dim=10));

model.add(Dropout(0.25));

model.add(Dense(16, activation='tanh'));

model.add(Dropout(0.25));

model.add(Dense(6, activation='softmax'));

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']);



ckpt = keras.callbacks.ModelCheckpoint("model1.h5",monitor="val_loss",mode="min");



history = model.fit(x_train, y_train, validation_split=0.2, epochs=700, batch_size=20);

#score = model.evaluate(x_test, y_test, batch_size=20);
tf = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv', na_values='?');

tf_onehot = tf.copy()

tf_onehot = pd.get_dummies(tf_onehot, columns=['Size'], prefix = ['Size'])

tf_onehot.head()

tf_drop_corr = tf_onehot.drop(columns = ['Number of Insignificant Quantities', 'Total Number of Characters', 'Number of Sentences', 'ID'])

tf_drop_corr.info()

tff = tf_drop_corr;

tff.describe()
#tff2 = tff.copy();

#tff2 = pd.get_dummies(tff2, columns = ['Class'], prefix = ['Class']);

X = tff.copy(deep = True);
y = model.predict(X);
a = y.argmax(-1);
a
A = np.array([range(371, 371+len(a)), a]);

data = pd.DataFrame(A.T, columns = ['ID', 'Class']);
data
data.to_csv('praty_test_2.csv', index = False);
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df2, title = "Download CSV file", filename = "data.csv"):

    csv = data.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(data)