import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
import os
%matplotlib inline
print(os.listdir("../input"))
original_data = pd.read_csv('../input/BlackFriday.csv')
original_data.head()
original_data.columns
original_data.describe()
original_data.info()
sns.set_style('whitegrid')
plt.figure(figsize=(16,9))
sns.heatmap(original_data.isnull(),cmap="viridis",cbar=False,yticklabels=False)
plt.figure(figsize=(16,9))
sns.distplot(original_data['Purchase'],bins=80,kde=False)
sns.countplot(x='Gender',data=original_data,hue='Marital_Status')
plt.figure(figsize=(16,9))
sns.countplot(x='Age',data=original_data,hue='Gender')
plt.figure(figsize=(16,9))
sns.countplot(x='Occupation',data=original_data,hue='Gender')
plt.figure(figsize=(16,9))
sns.boxplot(x='Age',y='Purchase',data=original_data)
plt.figure(figsize=(16,9))
sns.violinplot(x='Occupation',y='Purchase',data=original_data)
sns.countplot(x='City_Category',data=original_data)
sns.violinplot(x='City_Category',y='Purchase',data=original_data)
plt.figure(figsize=(16,9))
sns.countplot(x='Stay_In_Current_City_Years',data=original_data)
plt.figure(figsize=(16,9))
sns.boxplot(x='Stay_In_Current_City_Years',y='Purchase',data=original_data)
plt.figure(figsize=(16,9))
data = original_data['Product_ID'].value_counts().sort_values(ascending=False)[:10]
sns.barplot(x=data.index,y=data.values)
plt.figure(figsize=(16,9))
data = original_data['User_ID'].value_counts().sort_values(ascending=False)[:10]
sns.barplot(x=data.index,y=data.values)
original_data.columns
original_data['Product_Category_2'].fillna(0, inplace=True)
original_data['Product_Category_3'].fillna(0, inplace=True)
original_data['Product_Category_2'] = original_data['Product_Category_2'].astype(int)
original_data['Product_Category_3'] = original_data['Product_Category_3'].astype(int)
original_data.Stay_In_Current_City_Years = original_data.Stay_In_Current_City_Years.replace('4+',4)
original_data['Stay_In_Current_City_Years'] = original_data['Stay_In_Current_City_Years'].astype(int)
original_data.head()
X = original_data.iloc[:,2:11].values
y = original_data.iloc[:,11].values
lb_x_1 = LabelEncoder()
X[:,0] = lb_x_1.fit_transform(X[:,0])
lb_x_2 = LabelEncoder()
X[:,1] = lb_x_2.fit_transform(X[:,1])
lb_x_4 = LabelEncoder()
X[:,3] = lb_x_2.fit_transform(X[:,3])
lb_x_3 = LabelEncoder()
X[:,2] = lb_x_3.fit_transform(X[:,2])
onh = OneHotEncoder(categorical_features=[1,2,3])
X = onh.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
model = Sequential()

# The Input Layer :
model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()
checkpoint_name = '{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
def get_best_weight(num_ext, ext):
    tmp_file_name = os.listdir('.')
    test = []
    num_element = -num_ext
    all_weights_file = [k for k in tmp_file_name if '.hdf5' in k]
    for x in range(0, len(all_weights_file)):
        test.append(all_weights_file[x][:num_element])
        float(test[x])

    lowest = min(test)
    return str(lowest) + ext
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
weights_file = get_best_weight(5, ".hdf5")
model.load_weights(weights_file)
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
predictions = model.predict(X_test)
plt.figure(figsize=(16,9))
plt.scatter(y_test[:500],predictions[:500]) #Taken only 500 points here for better visualisation
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))