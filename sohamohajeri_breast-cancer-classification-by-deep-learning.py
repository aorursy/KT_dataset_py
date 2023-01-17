import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df.columns
df.shape
df.info()
df.describe()
plt.figure(figsize=(8,5))
sns.countplot(x='diagnosis',data=df, palette='PuBuGn')
plt.title('Number of Different Types of Tumors', fontsize=15)
plt.show()
diagnosis_type=pd.get_dummies(df['diagnosis'], drop_first=True)
df=pd.concat([df,diagnosis_type], axis=1)
df.drop(['diagnosis'], axis=1, inplace=True)
df.columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'malignant_1_benign_0']
plt.figure(figsize=(8,6))
df.corr()['malignant_1_benign_0'].sort_values(ascending=False).plot(kind='line', color='c')
plt.xticks(rotation=15)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.title('Correlation of Features with Cancer Type', fontsize=15)
plt.show()
plt.figure(figsize=(8,6))
df.corr()['malignant_1_benign_0'].sort_values(ascending=False).drop('malignant_1_benign_0').plot(kind='bar', color='g')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.xticks(rotation=90)
plt.title('Correlation with cancer type', fontsize=15)
plt.show()
plt.figure(figsize=(20,20))
corr=df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask,square=True, annot=True, cmap='winter')
X=df.drop(['malignant_1_benign_0'], axis=1).values
y=df['malignant_1_benign_0'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
mms=MinMaxScaler()
mms.fit(X_train)
X_train=mms.transform(X_train)
X_test=mms.transform(X_test)
model= Sequential()

model.add(Dense(30,activation='relu'))
          
model.add(Dense(30,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')
model.fit(X_train,y_train,epochs=600, validation_data=(X_test,y_test),batch_size=128)
first_loss_df=pd.DataFrame(model.history.history)
first_loss_df
first_loss_df.plot(figsize=(7,6), colormap='RdBu_r')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss per Epoch\nNo Early Stopping, No Droupout Layer', fontsize=15)
plt.show()
predictions=model.predict_classes(X_test)
print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test,predictions))
model= Sequential()

model.add(Dense(30,activation='relu'))
          
model.add(Dense(30,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')
es=EarlyStopping(monitor='val_loss', mode='min',patience=25,verbose=1)
model.fit(X_train,y_train,epochs=600, validation_data=(X_test,y_test),batch_size=128, callbacks=[es])
second_loss_df=pd.DataFrame(model.history.history)
second_loss_df
second_loss_df.plot(figsize=(7,6), colormap='RdBu_r')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss per Epoch\nEarly Stopping, No Droupout Layer', fontsize=15)
plt.show()
predictions=model.predict_classes(X_test)
print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test,predictions))
model= Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))
          
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')
es=EarlyStopping(monitor='val_loss', mode='min',patience=25,verbose=1)
model.fit(X_train,y_train,epochs=600, validation_data=(X_test,y_test),batch_size=128, callbacks=[es])
third_loss_df=pd.DataFrame(model.history.history)
third_loss_df
third_loss_df.plot(figsize=(7,6), colormap='RdBu_r')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss per Epoch\n Early Stopping, Droupout Layers', fontsize=15)
plt.show()
predictions=model.predict_classes(X_test)
print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test,predictions))