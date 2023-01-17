import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/nslkdd/kdd_train.csv')
df.head()
df.info()
print('number of classes:', df['labels'].nunique())
print('')
label_counts = df['labels'].value_counts()
plt.figure(figsize=(18,6));
sns.barplot(y=label_counts.index, x=label_counts.values, color='Grey');
plt.title('values per class');
display(label_counts)

#binary traffic proportions
binary_class = []
for label in df['labels']:
    if label !='normal':
        binary_class.append('malicious')
    else:
        binary_class.append('normal')
binary_class = pd.Series(binary_class)
plt.figure()
binary_class.value_counts().plot(kind='pie', label='traffic proportions', autopct='%.2f%%' );
df['protocol_type'].value_counts()
df['service'].value_counts()
#how many different categories in column 'service'
print('number of categories in column \'service\':', df['service'].nunique())
df['flag'].value_counts()
#summary statistics
display(df.iloc[:,:10].describe())
display(df.iloc[:,10:17].describe())
display(df.iloc[:,17:24].describe())
display(df.iloc[:,24:31].describe())
display(df.iloc[:,31:36].describe())
display(df.iloc[:,36:].describe())
numeric_columns = []
categorical_columns = []
for column in df.columns:
    if df[column].dtype != 'object':
        numeric_columns.append(column)
    else:
        categorical_columns.append(column)

categorical_columns = categorical_columns[:-1]       
labels=df['labels'].unique()
#distribution boxenplots (per class)
for column in numeric_columns:
    plt.figure(figsize=(18,7))
    sns.boxenplot(x='labels', y=df[column], data=df);
    plt.title(column);
    #for label in labels:
    #    plt.figure(figsize=(18,7))
    #    sns.kdeplot(df[column][df['labels']==label]);
    #    plt.title(label);
#feature means (per class)
group_mean = df.groupby(by='labels').mean()
for column in numeric_columns:
    plt.figure(figsize=(16,5));
    sns.barplot(y=group_mean[column].squeeze().index, x=group_mean[column].squeeze().values, 
                color='Gray');
    plt.title(column);
#correlation heatmap
plt.figure(figsize=(18,12));
sns.heatmap(df.corr(), annot=True, fmt='1.1f');
#column 'num_outbound_cmds' is zero everywhere, we will delete it
df.drop(columns='num_outbound_cmds', inplace=True)

#remove from list of numeric columns
numeric_columns.remove('num_outbound_cmds')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

#standardize data
df_num = df[numeric_columns].copy()
sc = StandardScaler()
df_num = sc.fit_transform(df_num)

#ordinalize labels
ordinal = OrdinalEncoder()
ord_labels = ordinal.fit_transform(df['labels'].values[:,np.newaxis])
ord_labels = np.squeeze(ord_labels.astype(int))

#PCA
pca = PCA()
df_pca = pca.fit_transform(df_num)

#PCA 3d-scatterplot
plt.figure(figsize=(12,6));
ax=plt.axes(projection='3d')
ax.scatter(df_pca[:,0], df_pca[:,1], df_pca[:,2], 
           c=ord_labels, cmap='winter');
plt.title('3D PCA Visualization');

#explained variance
var_index = np.arange(pca.explained_variance_.shape[0])+1
plt.figure(figsize=(14,6));
sns.barplot(x=var_index, y=pca.explained_variance_ratio_, color='gray');
plt.title('Explained Variance Ratio');
plt.figure(figsize=(14,6));
sns.lineplot(x=var_index, y=pca.explained_variance_ratio_.cumsum());
plt.title('Cumulative Explained Variance Ratio');
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
y = enc.fit_transform(df['labels'])
x = np.arange(y.shape[0])

plt.figure(figsize=(18,8));
sns.lineplot(x=x[:800], y=y[:800]); #for visual clarity, only a small slice is selected
plt.title('traffic in time')

class_labels = pd.DataFrame(data=enc.classes_,columns=['class'])
class_labels['label'] = np.unique(y)
display(class_labels)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, BatchNormalization, Dropout
#from tensorflow.keras.layers.core import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score
#from tensorflow.keras.metrics import 
from sklearn.preprocessing import StandardScaler

dummies = pd.get_dummies(df[categorical_columns])
x = pd.concat((df[numeric_columns], dummies), axis=1).values

enc_bin = LabelEncoder()
y_bin = enc_bin.fit_transform(binary_class)
enc_multi = LabelEncoder()
y_multi = enc_multi.fit_transform(df['labels'].values)

# for manual train_test_split, splitting indices instead of actual values
np.random.RandomState(seed=0)
train_indexes = np.random.choice(np.arange(x.shape[0]), size=x.shape[0]*8//10, replace=False)
test_indexes = np.delete(np.arange(x.shape[0]), np.arange(x.shape[0])[train_indexes])
print('train size:', train_indexes.shape[0])
print('test size:  ', test_indexes.shape[0])

x_tr = x[train_indexes]
x_ts = x[test_indexes]
y_bin_tr =y_bin[train_indexes]
y_bin_ts =y_bin[test_indexes]
y_multi_tr = y_multi[train_indexes]
y_multi_ts = y_multi[test_indexes]

# scale x
sc=StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_ts = sc.transform(x_ts)

#make x 3-dimensional for the CNN to process
x_tr = x_tr[:,:,np.newaxis]
x_ts = x_ts[:,:,np.newaxis]
model=Sequential()
model.add(Conv1D(128,2, activation='relu',input_shape=x_tr[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.2))

model.add(Conv1D(256,2, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(patience=5,verbose=1)
model.fit(x_tr, y_bin_tr, epochs=50, validation_split=0.1, batch_size=16, callbacks=[early_stop])
pred = model.predict(x_ts)
pred_d = []
for prediction in pred:
    if prediction <0.5:
        pred_d.append(0)
    else:
        pred_d.append(1)
        
pred = np.array(pred_d)
print('accuracy:', accuracy_score(y_bin_ts, pred))
print('f1-score:', f1_score(y_bin_ts, pred, average='macro'))
model1=Sequential()
model1.add(Conv1D(180,2, activation='relu',input_shape=x_tr[0].shape))
model1.add(BatchNormalization())
model1.add(MaxPool1D(2))
model1.add(Dropout(0.2))

model1.add(Conv1D(300,2, activation='relu'))
model1.add(BatchNormalization())
model1.add(MaxPool1D(2))
model1.add(Dropout(0.4))

model1.add(Flatten())
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.4))

model1.add(Dense(len(np.unique(enc.classes_)),activation='softmax'))

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.summary()
early_stop = EarlyStopping(patience=4,verbose=1)
model1.fit(x_tr, y_multi_tr, epochs=30, validation_split=0.1, batch_size=16, callbacks=[early_stop])
pred = model1.predict(x_ts)
pred = np.argmax(pred,axis=1)

print('accuracy:', accuracy_score(y_multi_ts, pred))
print('f1-score:', f1_score(y_multi_ts, pred, average='weighted'))