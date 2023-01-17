from google.colab import drive
drive.mount('/content/drive')
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 100)
fifa19 = pd.read_csv(r'./drive/My Drive/data.csv')
fifa19.head()
fifa19.shape
print('Jumlah baris dan kolom dari dataset FIFA 19 adalah :', fifa19.shape)
print()
fifa19.columns
fifa19.describe()
fifa19.describe(include='object')
pd.set_option('display.max_rows', 100)
fifa19.isnull().sum().sort_values()
fifa19.isnull().sum().sum()
fifa19['ShortPassing'].fillna(fifa19['ShortPassing'].mean(), inplace=True)
fifa19['Volleys'].fillna(fifa19['Volleys'].mean(), inplace=True)
fifa19['Dribbling'].fillna(fifa19['Dribbling'].mean(), inplace=True)
fifa19['Curve'].fillna(fifa19['Curve'].mean(), inplace=True)
fifa19['FKAccuracy'].fillna(fifa19['FKAccuracy'].mean(), inplace=True)
fifa19['LongPassing'].fillna(fifa19['LongPassing'].mean(), inplace=True)
fifa19['BallControl'].fillna(fifa19['BallControl'].mean(), inplace=True)
fifa19['HeadingAccuracy'].fillna(fifa19['HeadingAccuracy'].mean(), inplace=True)
fifa19['Finishing'].fillna(fifa19['Finishing'].mean(), inplace=True)
fifa19['Crossing'].fillna(fifa19['Crossing'].mean(), inplace=True)
fifa19['Weight'].fillna('200 lbs', inplace=True)
fifa19['Contract Valid Until'].fillna(2019, inplace=True)
fifa19['Height'].fillna("5'11", inplace=True)
fifa19['Loaned From'].fillna('None', inplace=True)
fifa19['Joined'].fillna('Jul 1, 2018', inplace=True)
fifa19['Jersey Number'].fillna(8, inplace=True)
fifa19['Body Type'].fillna('Normal', inplace=True)
fifa19['Position'].fillna('ST', inplace=True)
fifa19['Club'].fillna('No Club', inplace=True)
fifa19['Work Rate'].fillna('Medium/ Medium', inplace=True)
fifa19['Skill Moves'].fillna(fifa19['Skill Moves'].median(), inplace=True)
fifa19['Weak Foot'].fillna(3, inplace=True)
fifa19['Preferred Foot'].fillna('Right', inplace=True)
fifa19['International Reputation'].fillna(1, inplace=True)
fifa19['Wage'].fillna('€200K', inplace=True)
fifa19['Release Clause'].fillna('0', inplace=True)
fifa19.isnull().sum().sort_values()
fifa19.fillna(0, inplace=True)
fifa19.isnull().sum().sort_values()
fifa19.isnull().sum().sum()
fifa19.drop(['Unnamed: 0', 'Photo', 'Club Logo', 'Flag'], axis=1, inplace=True)
fifa19.shape
def pace(fifa):
  return int(round((fifa[['SprintSpeed', 'Acceleration']].mean()).mean()))

def passing(fifa):
  return int(round((fifa[['ShortPassing', 'Crossing', 'Vision', 'LongPassing', 'Curve', 'FKAccuracy']].mean()).mean()))

def defending(fifa):
  return int(round((fifa[['StandingTackle', 'Marking', 'Interceptions', 'HeadingAccuracy', 'SlidingTackle']].mean()).mean()))

def shooting(fifa):
  return int(round((fifa[['Finishing', 'LongShots', 'ShotPower', 'Volleys', 'Positioning', 'Penalties']].mean()).mean()))

def dribbling(fifa):
  return int(round((fifa[['Dribbling', 'BallControl', 'Agility', 'Balance']].mean()).mean()))

def physical(fifa):
  return int(round((fifa[['Strength', 'Stamina', 'Aggression', 'Jumping']].mean()).mean()))
#menambahkan kategori-kategori berikut ke dalam data
fifa19['Pace'] = fifa19.apply(pace, axis=1)
fifa19['Passing'] = fifa19.apply(passing, axis=1)
fifa19['Defending'] = fifa19.apply(defending, axis=1)
fifa19['Shooting'] = fifa19.apply(shooting, axis=1)
fifa19['Dribbling'] = fifa19.apply(dribbling, axis=1)
fifa19['Physical'] = fifa19.apply(physical, axis=1)
fifa19.drop(['SprintSpeed', 'Acceleration', 'ShortPassing', 'Crossing', 'Vision', 'LongPassing', 'Curve', 'FKAccuracy', 'StandingTackle', 'Marking', 'Interceptions', 
             'HeadingAccuracy', 'SlidingTackle', 'Finishing', 'LongShots', 'ShotPower', 'Volleys', 'Positioning', 'Penalties', 'BallControl', 'Agility', 'Balance', 
             'Strength', 'Stamina', 'Aggression', 'Jumping'], axis=1, inplace=True)
fifa19.shape
rosters = fifa19[['Name', 'Club', 'Age', 'Nationality', 'Pace', 'Passing', 'Defending', 'Shooting', 'Dribbling', 'Physical', 'Potential', 'Overall', 'International Reputation']]

rosters.head()
def weight_cleaning(weight):
  out = weight.replace('lbs', '')
  return float(out)
fifa19['Weight'] = fifa19['Weight'].apply(lambda x : weight_cleaning(x))
fifa19['Weight'].head()
print(fifa19[['Value']].head())
print()
print(fifa19[['Wage']].head())
print()
print(fifa19[['Release Clause']].head())
def wage_cleaning(wage):
  out = wage.replace('€', '')
  if 'M' in out:
    out = float(out.replace('M', ''))*1000000
  elif 'K' in out:
    out = float(out.replace('K', ''))*1000
  return float(out)
fifa19['Value'] = fifa19['Value'].apply(lambda x : wage_cleaning(x))
fifa19['Wage'] = fifa19['Wage'].apply(lambda x : wage_cleaning(x))
fifa19['Release Clause'] = fifa19['Release Clause'].apply(lambda x : wage_cleaning(x))
fifa19.dtypes
fifa19.shape
fifa19.to_csv('FIFA 19 Clean for Visualize.csv')
def country(x):
  return fifa19[fifa19['Nationality'] == x][['Name', 'Overall', 'Potential', 'Position']]
country('Italy')
def club(x):
  return fifa19[fifa19['Club'] == x][['Name', 'Jersey Number', 'Position', 'Overall', 'Nationality', 'Age', 'Wage', 'Value', 'Contract Valid Until']]
club('Milan')
plt.figure(figsize=(8, 6))
sns.countplot(fifa19['Preferred Foot'], palette='winter')
plt.title('Preferred Foot para pesepakbola di FIFA 19', fontsize=15)
plt.savefig('preferred foot.jpg')
plt.show()
fifa19['International Reputation'].value_counts()
labels = ['1', '2', '3', '4', '5']
sizes = fifa19['International Reputation'].value_counts()
colors = plt.cm.winter_r(np.linspace(0, 1, 5))
explode = [0.1, 0.1, 0.2, 0.5, 0.9]

plt.figure(figsize=(9, 9))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, shadow=True)
plt.title('International Reputation dari Para Pesepakbola di FIFA 19', fontsize=15)
plt.savefig('intrepu.jpg')
plt.legend()
plt.show()
labels = ['3', '2', '4', '5', '1']
size = fifa19['Weak Foot'].value_counts()
colors = plt.cm.Blues(np.linspace(0, 1, 5))
explode = [0, 0, 0, 0, 0.1]

plt.figure(figsize=(9, 9))
plt.pie(size, labels=labels, colors=colors, explode=explode, shadow=True, startangle=90)
plt.title('Distribusi Weak Foot di FIFA 19', fontsize=15)
plt.legend()
plt.savefig('weekfoot.jpg')
plt.show()
plt.figure(figsize=(18 ,8))
sns.countplot(fifa19['Position'], palette='winter')
plt.xlabel('Posisi Pemain di FIFA 19')
plt.ylabel('Jumlah Pemain per tiap posisi')
plt.title('Distribusi Pemain ditinjau berdasarkan Posisinya', fontsize=15)
plt.savefig('position.jpg')
plt.show()
plt.plot(fifa19["Wage"], np.zeros_like(fifa19["Wage"]), "o", color='blue')
plt.title('Persebaran Gaji Para Pesepakbola di FIFA 19', fontsize=15)
plt.savefig('wage.jpg')
plt.show()
plt.figure(figsize=(13, 6))
sns.countplot(fifa19['Work Rate'], palette='winter')
plt.title('Tipe-tipe Work Rate dari Para Pesepakbola di FIFA 19', fontsize=15)
plt.xlabel('Work Rates')
plt.savefig('work rates.jpg')
plt.show()
fifa19['Nationality'].value_counts().head(25)
negara = fifa19['Nationality'].value_counts().head(25)
plt.figure(figsize=(15, 6))
sns.barplot(x=negara[:].index, y=negara[:].values, palette='winter')
plt.title('Top 25 Negara dengan Pemain Terbanyak di FIFA 19', fontsize=15)
plt.xticks(rotation=90)
plt.xlabel('Negara')
plt.savefig('national.jpg')
plt.show()
kebangsaan = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Colombia', 'Japan', 'Netherlands')
data_negara = fifa19.loc[fifa19['Nationality'].isin(kebangsaan) & fifa19['Overall']]
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Nationality', y='Overall', data=data_negara, palette='winter')
ax.set_xlabel(xlabel='Negara')
ax.set_ylabel(ylabel='Overall')
ax.set_title(label='Overall Para Pemain berdasarkan Asal Negaranya', fontsize=15)
plt.savefig('ovr country.jpg')
plt.show()
kebangsaan = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Colombia', 'Japan', 'Netherlands')
data_negara = fifa19.loc[fifa19['Nationality'].isin(kebangsaan) & fifa19['Wage']]
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Nationality', y='Wage', data=data_negara, palette='coolwarm')
ax.set_xlabel(xlabel='Negara')
ax.set_ylabel(ylabel='Gaji')
ax.set_title(label='Gaji Pemain dilihat dari Asal Negaranya', fontsize=15)
plt.savefig('wage based on nationality.jpg')
plt.show()
pd.set_option('display.max_rows', 100)
fifa19.groupby(fifa19['Club'])['Wage'].mean().sort_values(ascending=False).head(10)
klub = ('Real Madrid', 'FC Barcelona', 'Juventus', 'Manchester City', 'Manchester United', 'Chelsea', 'Liverpool', 'Tottenham Hotspur', 'FC Bayern München', 'Arsenal')
data_klub = fifa19.loc[fifa19['Club'].isin(klub) & fifa19['Wage']]
plt.figure(figsize=(16, 8))
ax = sns.barplot(x='Club', y='Wage', data=data_klub, palette='dark')
ax.set_xlabel(xlabel='Klub', fontsize=10)
ax.set_ylabel(ylabel='Gaji', fontsize=10)
ax.set_title(label='Persebaran Gaji di Beberapa Klub di FIFA 19', fontsize=20)
plt.xticks(rotation=90)
plt.savefig('wage based on club.jpg')
plt.show()
fifa19.iloc[fifa19.groupby(fifa19['Position'])['Potential'].idxmax()][['Position', 'Name', 'Club', 'Age', 'Nationality', 'Potential']]
fifa19.iloc[fifa19.groupby(fifa19['Position'])['Overall'].idxmax()][['Position', 'Name', 'Club', 'Age', 'Nationality', 'Overall']]
fifa19.iloc[fifa19.groupby(fifa19['Position'])['Potential'].idxmin()][['Position', 'Name', 'Age', 'Club', 'Nationality', 'Potential']]
fifa19.iloc[fifa19.groupby(fifa19['Position'])['Overall'].idxmin()][['Position', 'Name', 'Age', 'Club', 'Nationality', 'Overall']]
termuda = fifa19.sort_values('Age', ascending=True)[['Name', 'Age', 'Club', 'Nationality']].head(10)
termuda
tertua = fifa19.sort_values('Age', ascending=False)[['Name', 'Age', 'Club', 'Nationality']].head(10)
tertua
print('Null Values dari Kolom Joined : ', fifa19['Joined'].isnull().sum())
print()
fifa19['Joined'].head()
import datetime
now = datetime.datetime.now()
fifa19['Join_year'] = fifa19['Joined'].dropna().map(lambda x : x.split(',')[1].split(' ')[1]) #kita akan ambil tahunnya saja
fifa19['Years_of_member'] = (fifa19['Join_year'].dropna().map(lambda x : now.year - int(x))).astype('int')
masa_bakti_panjang = fifa19[['Name', 'Club', 'Years_of_member']].sort_values(by='Years_of_member', ascending=False).head(10)
#masa_bakti_panjang.set_index('Name', inplace=True)
masa_bakti_panjang
masa_bakti_pendek = fifa19[['Name', 'Club', 'Years_of_member']].sort_values(by='Years_of_member', ascending=True).head(10)
masa_bakti_pendek
fifa19[fifa19['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)
fifa19[fifa19['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)
fifa19.groupby(fifa19['Club'])['Nationality'].nunique().sort_values(ascending=False).head(11)
fifa19.groupby(fifa19['Club'])['Nationality'].nunique().sort_values(ascending=True).head(10)
fifa19.head()
def face_to_num(data):
    if (data['Real Face'] == 'Yes'):
        return 1
    else:
        return 0
    
def preferred_foot(data):
    if (data['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0
fifa19['Real Face'] = fifa19.apply(face_to_num, axis=1)
fifa19['Preferred Foot'] = fifa19.apply(preferred_foot, axis=1)
fifa19.head()
df = fifa19[['Age', 'Wage', 'Value', 'Special', 'Dribbling', 'Pace', 'Defending', 'Shooting', 'Passing', 'Physical', 'Potential', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Reactions', 'Composure', 'GKDiving', 'GKHandling', 'GKKicking',	'GKPositioning', 'GKReflexes', 'Overall']]
df.head()
df[['Age', 'Dribbling', 'Pace', 'Defending', 'Shooting', 'Passing', 'Physical', 'Potential', 'Overall']] = df[['Age', 'Dribbling', 'Pace', 'Defending', 'Shooting', 'Passing', 'Physical', 'Potential', 'Overall']].astype('float64')
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
X = df.drop(['Overall'], axis=1)
y = df['Overall']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
pipeline = Pipeline([('scaler', RobustScaler()), ('model', LinearRegression())])
pipeline.fit(X_train, y_train)
#Terhadap data training
prediksi_linreg_train = pipeline.predict(X_train)
mse_train = mean_squared_error(y_train, prediksi_linreg_train)
r2_train = r2_score(y_train, prediksi_linreg_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
prediksi_linreg = pipeline.predict(X_test)
#Terhadap data testing
mse = mean_squared_error(y_test, prediksi_linreg)
mae = mean_absolute_error(y_test, prediksi_linreg)
r2 = r2_score(y_test, prediksi_linreg)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
from sklearn.svm import SVR
svr = SVR()
svr.get_params().keys()
C = [0.001, 0.01, 1.0, 10.0, 100.0, 1000.0]
gamma = [1, 0.1, 0.01, 0.001]
svm_grid = {'model__C' : [1.0, 10.0, 100.0, 1000.0],
            'model__gamma' : [1, 0.1, 0.01, 0.001]}

print(svm_grid)
pipeline_svm = Pipeline([('scaler', RobustScaler()), ('model', SVR(kernel='rbf'))])
svm_random = RandomizedSearchCV(pipeline_svm, svm_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
svm_random.fit(X_train, y_train)
svm_random.best_params_
pipeline_svm = Pipeline([('scaler', RobustScaler()), ('model', SVR(kernel='rbf', C=1000, gamma=0.01))])
pipeline_svm.fit(X_train, y_train)
#Terhadap data training
prediksi_svm_train = pipeline_svm.predict(X_train)
r2_train = r2_score(y_train, prediksi_svm_train)
mse_train = mean_squared_error(y_train, prediksi_svm_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
#Terhadap data testing
prediksi_svm = pipeline_svm.predict(X_test)

mse = mean_squared_error(y_test, prediksi_svm)
mae = mean_absolute_error(y_test, prediksi_svm)
r2 = r2_score(y_test, prediksi_svm)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.get_params()
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
pipeline_rf = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor())])
random_grid = {'model__n_estimators' : n_estimators,
               'model__max_features' : max_features, 
               'model__max_depth' : max_depth,
               'model__min_samples_split' : min_samples_split,
               'model__min_samples_leaf' : min_samples_leaf,
               'model__bootstrap' : bootstrap}

print(random_grid)
rf_random = RandomizedSearchCV(pipeline_rf, random_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
rf_random.fit(X_train, y_train)
rf_random.best_params_
pipeline_rf = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor(bootstrap=True, max_depth=20, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=1800))])
pipeline_rf.fit(X_train, y_train)
#Terhadap data training
prediksi_rf_train = pipeline_rf.predict(X_train)
r2_train = r2_score(y_train, prediksi_rf_train)
mse_train = mean_squared_error(y_train, prediksi_rf_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
#Terhadap data testing
prediksi_rf = pipeline_rf.predict(X_test)

mse = mean_squared_error(y_test, prediksi_rf)
mae = mean_absolute_error(y_test, prediksi_rf)
r2 = r2_score(y_test, prediksi_rf)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.get_params()
n_neighbors = [int(x) for x in np.linspace(start=1, stop=25, num=13)]
pipeline_knn = Pipeline([('scaler', RobustScaler()), ('model', KNeighborsRegressor())])
knn_grid = {'model__n_neighbors' : n_neighbors}
print(knn_grid)
knn_random = RandomizedSearchCV(pipeline_knn, knn_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
knn_random.fit(X_train, y_train)
knn_random.best_params_
pipeline_knn = Pipeline([('scaler', RobustScaler()), ('model', KNeighborsRegressor(n_neighbors=15))])
pipeline_knn.fit(X_train, y_train)
#Terhadap data training
prediksi_knn_train = pipeline_knn.predict(X_train)
r2_train = r2_score(y_train, prediksi_knn_train)
mse_train = mean_squared_error(y_train, prediksi_knn_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
#Terhadap data testing
prediksi_knn = pipeline_knn.predict(X_test)

mse = mean_squared_error(y_test, prediksi_knn)
mae = mean_absolute_error(y_test, prediksi_knn)
r2 = r2_score(y_test, prediksi_knn)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
#Save model Machine Learning
import pickle
filename = 'best_model.pkl' #Nama filenya
pickle.dump(pipeline_rf, open(filename, 'wb')) #Membuat file model
df.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime
df.head()
X_dl = df.drop(['Overall'], axis=1)
y_dl = df['Overall']
rbst1 = RobustScaler()
rbst2 = RobustScaler()
rbst1 = rbst1.fit(X_dl)
rbst2 = rbst2.fit(df['Overall'].values.reshape(-1, 1))
X_dl = rbst1.transform(X_dl)
y_dl = rbst2.transform(df['Overall'].values.reshape(-1, 1)).flatten()
import pickle
scalername = 'scaler_feature.pkl' #Nama filenya
pickle.dump(rbst1, open(scalername, 'wb')) 
scalername2 = 'scaler_label.pkl' #Nama filenya
pickle.dump(rbst2, open(scalername2, 'wb')) 
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_dl, test_size=0.2, random_state=10)
model = Sequential()
model.add(Dense(21, input_dim=21, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform'))

opt = SGD(learning_rate=0.001, momentum=0.9)

model.summary()

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['CosineSimilarity'])
filepath="weights_best_only.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Tempat dimana log tensorboard akan di
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))
history = model.fit(X_train_dl, y_train_dl, batch_size=32, validation_data=(X_test_dl, y_test_dl), epochs=50, callbacks=callbacks_list, verbose=0)
predict_dl = model.predict(X_test_dl)

predict_dl = predict_dl.flatten()
mse = mean_squared_error(y_test_dl, predict_dl)
mae = mean_absolute_error(y_test_dl, predict_dl)
r2 = r2_score(y_test_dl, predict_dl)
print("MSE (Mean Squared Error)       :", mse)
print("MAE (Mean Absolute Error)      :", mae)
print("r^2 score                      :", r2)
print('RMSE (Root Mean Squared Error) :', np.sqrt(mse))
#load extension jupiter notebook
%reload_ext tensorboard

#load tenserboard
%tensorboard --logdir logs
def plot_loss_new(history):
  history_df = pd.DataFrame(history.history)

  min_loss_index = history_df[history_df['loss']==min(history_df['loss'])].index.values
  min_loss = history_df.loc[min_loss_index]['loss']
  min_val_loss_index = history_df[history_df['val_loss']==min(history_df['val_loss'])].index.values
  min_val_loss = history_df.loc[min_val_loss_index]['val_loss']

  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.plot(min_loss_index, min_loss, 'o', c='k', ms=4, label='min loss')
  plt.plot(min_val_loss_index, min_val_loss, 'o', c='k', ms=4, label='min val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.savefig('plot loss.jpg')
  plt.show()

  print('Minimum Loss             :', min_loss)
  print()
  print('Minimum Validation Loss  :', min_val_loss)
plot_loss_new(history)
X_test.head()
from tensorflow.keras.models import load_model
loaded_model = load_model('/content/weights_best_only.h5')
scaler_feature = pickle.load(open(scalername, 'rb'))
scaler_label = pickle.load(open(scalername2, 'rb'))
#Ambil data pada index 3207
y[3207]
#Testing data dengan model terbaik ML yaitu Random Forest
test_data = [[31.0,	7000.0,	2700000.0,	1957,	73.0,	69.0,	65.0,	62.0,	70.0,	72.0,	73.0,	1.0,	3.0,	3.0,	68.0,	72.0,	10.0,	12.0,	8.0,	7.0,	14.0]]

test_data1 = scaler_feature.transform(test_data)
predict_model = loaded_model.predict(test_data1)
inv_pred = scaler_label.inverse_transform(predict_model)

print('Overall dari pemain tersebut adalah {}. Keren!'.format(inv_pred[0]))
!pip install flask-ngrok
from flask_ngrok import run_with_ngrok
from flask import Flask
# Membuat sebuah object Flask dan homepage
app = Flask(__name__) 

@app.route("/home")
def home():
    return """<h1>Running Flask on Google Colab!</h1>
              <h2>This is home page!</h2>"""
from flask import jsonify, request
#Regresi dengan halaman Machine Learning

@app.route('/regression', methods=['POST'])
def regression():
  age = float(request.json['Age'])
  wage = float(request.json['Wage'])
  value = float(request.json['Value'])
  special = float(request.json['Special'])
  dribbl_ing = float(request.json['Dribbling'])
  pa_ce = float(request.json['Pace'])
  defend_ing = float(request.json['Defending'])
  shoot_ing = float(request.json['Shooting'])
  pass_ing = float(request.json['Passing'])
  physic_al = float(request.json['Physical'])
  potential = float(request.json['Potential'])
  inter_repu = float(request.json['International Reputation'])
  weak_foot = float(request.json['Weak Foot'])
  skill_move = float(request.json['Skill Moves'])
  reactions = float(request.json['Reactions'])
  composure = float(request.json['Composure'])
  gkd = float(request.json['GKDiving'])
  gkh = float(request.json['GKHandling'])
  gkk = float(request.json['GKKicking'])
  gkp = float(request.json['GKPositioning'])
  gkr = float(request.json['GKReflexes'])

  #Load model
  loaded_model = load_model('/content/weights_best_only.h5')
  scaler_feature = pickle.load(open('scaler_feature.pkl', 'rb'))
  scaler_label = pickle.load(open('scaler_label.pkl', 'rb'))

  data = [[age, wage, value, special, dribbl_ing, pa_ce, defend_ing, shoot_ing, pass_ing, physic_al, potential, inter_repu, weak_foot, skill_move, 
           reactions, composure, gkd, gkh, gkk, gkp, gkr]]
  data1 = scaler_feature.transform(data)
  predict_model = loaded_model.predict(data1)
  inv_pred = scaler_label.inverse_transform(predict_model)

  return jsonify({
      "Player's Overall": str(inv_pred[0][0])
  })
#Jalankan flask di localhost lewat Insomnia
run_with_ngrok(app)

app.run()