# for reading data
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stat
import matplotlib.pyplot as plt 
%matplotlib inline
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# For data splitting
from sklearn.model_selection import train_test_split

# import other functions we'll need for regression modeling
from sklearn.linear_model import LogisticRegression # LR
from sklearn.tree import DecisionTreeRegressor # DTR
from sklearn.ensemble import RandomForestRegressor # RFR
from sklearn.ensemble import GradientBoostingRegressor #GBR

# regression error metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# classification error metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# for modeling
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers
df = pd.read_csv("../input/the-movies-dataset/movies_metadata.csv")
#Summary of original dataset
df.info()
#Shape of original dataset
df.shape
#Let's look into first few rows of the dataset to check the content of each column
df.head()
# As I will be concentrating on nuerical columns let's delete the text columns, which I donot need for solving the business problem
df.drop(['id','belongs_to_collection', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 'production_companies', 'production_countries', 'runtime', 'release_date', 'spoken_languages', 'status', 'tagline', 'title', 'video'], axis=1, inplace=True)
df.info()
df[df['revenue'] == 0].shape
df['revenue'] = df['revenue'].replace(0, np.nan)
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)
df[df['budget'].isnull()].shape
df['return'] = df['revenue'] / df['budget']
df[df['return'].isnull()].shape
df['adult'].value_counts()
df = df.drop('adult', axis=1)
def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan
df['popularity'] = df['popularity'].apply(clean_numeric).astype('float')
df["vote_count"] = df["vote_count"].apply(clean_numeric).astype('float')
df['vote_average'] = df['vote_average'].apply(clean_numeric).astype('float')
#Summary statistics of each feature
df['popularity'].describe()
sns.distplot(df['popularity'].fillna(df['popularity'].median()))
plt.show()
df['popularity'].plot(logy=True, kind='hist')
df['vote_count'].describe()
df['vote_average'] = df['vote_average'].replace(0, np.nan)
df['vote_average'].describe()
sns.distplot(df['vote_average'].fillna(df['vote_average'].median()))
df['budget'].describe()
sns.distplot(df[df['budget'].notnull()]['budget'])
df['budget'].plot(logy=True, kind='hist')
df['revenue'].describe()
sns.distplot(df[df['revenue'].notnull()]['revenue'])
#Let's check the number of missing values in the dataset
df.isnull().sum()
#Dropping rows with missing values
df.dropna(inplace=True)
#Check the number of missing values to ensure we have none
df.isnull().sum()
df["popularity"] = np.round(pd.to_numeric(df.popularity, errors='coerce'),2)
df.info()
df.shape
#converting 'genre' column into panda series and extract the type of the genre only from the column
s = pd.Series(df['genres'], dtype= str)
s1=s.str.split(pat="'",expand=True)
df['genre_ed']=s1[5]
#count of each genre in the dataset
df['genre_ed'].value_counts()
#Remove rows for genres with count less than 100
df=df[~df['genre_ed'].isin(['Mystery', 'Family', 'Documentary', 'War', 'Music', 'Western', 'History', 'Foreign', 'TV Movie'])]
df.drop(df[df['budget'] < 1000000].index, inplace=True)
#df.drop(df[df['vote_count'] < 100].index, inplace=True)
df.drop(df[df['revenue'] < 2000000].index, inplace=True)
#df.drop(df[df['vote_average'] == 0].index, inplace=True)
df.shape
#Drop original column from dataset
df.drop(['genres'], axis=1, inplace=True)
#get dummy columns for genre
df= pd.get_dummies(df, columns=["genre_ed"])
df.info()
df.head()
#Summary statistics of dataset 
df.describe()
#Let's check for outliers
df.boxplot(column=['budget', 'revenue'])
df.boxplot(column=['popularity', 'vote_average', 'vote_count'])
df_o= df[(np.nan_to_num(np.abs(stats.zscore(df,nan_policy='omit')),0) < 3).all(axis=1)]
df.shape
print ('Shape of original input dataset:', df.shape)
#After removing outliers
df_o.shape
print ('Shape of input dataset after removing outliers:', df_o.shape)
Q1 = df_o.quantile(0.25)
Q3 = df_o.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df_out = df_o[~((df_o < (Q1 - 1.5 * IQR)) |(df_o > (Q3 + 1.5 * IQR))).any(axis=1)]
print ('Shape of input dataset after managing misspread:', df_out.shape)
df_out.boxplot(column=['budget', 'revenue'])
df_out_high = df_out.apply(lambda x : True
            if x['revenue'] > 13500000 or x['budget'] > 30000000 else False, axis = 1) 
num_rows = len(df_out_high[df_out_high == True].index) 
  
print('Number of Rows in dataframe with revenue more than 13.5 million dollars or budget more than 30 million: ', 
      num_rows ) 
df_out.drop(df_out[df_out['revenue'] > 13500000].index, inplace=True)
df_out.drop(df_out[df_out['budget'] > 30000000].index, inplace=True)
df_out.boxplot(column=['budget', 'revenue'])
df_out['revenue'].describe()
# recode the revenue into high and low
df_out['Revenue'] = 0
df_out.loc[df_out['revenue'] <7017731,'Revenue'] = 0
df_out.loc[df_out['revenue'] >=7017731,'Revenue'] = 1
df_out.head()
# check distribution of target variable
df_out['Revenue'].value_counts()
df_out.drop(['revenue'],axis=1, inplace=True)
df_out
# Assign X and Y
X = df_out.drop(['Revenue'], axis=1)
y = df_out['Revenue']

print(X.shape)
print(y.shape)
import imblearn.under_sampling as u
# example of random oversampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

X,y = make_classification(n_features = 16, n_samples=312) 
# Make classification default = 20, so I need to set the number
ros = u.RandomUnderSampler(sampling_strategy='majority')
X_resampled, Y_resampled = ros.fit_resample(X, y)
X_resampled = pd.DataFrame(X_resampled)
print(X_resampled.shape)

Y_resampled = pd.DataFrame(Y_resampled)
print(Y_resampled.shape)
X_resampled.columns = ['budget', 'popularity', 'vote_average', 'vote_count','return',
       'genre_ed_Action', 'genre_ed_Adventure', 'genre_ed_Animation',
       'genre_ed_Comedy', 'genre_ed_Crime', 'genre_ed_Drama',
       'genre_ed_Fantasy', 'genre_ed_Horror', 'genre_ed_Romance',
       'genre_ed_Science Fiction', 'genre_ed_Thriller']
X_resampled
# I will be performing 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled,
                                                    test_size = 0.20,
                                                    shuffle = True,
                                                    random_state = 42)
X_train
# check your work - does the shape match what you think it should be?
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Fit the model
# make a variable to store the general model
LR = LogisticRegression()
# fit the model - one line of code
LR = LR.fit(X_train, y_train)
# store the predictions
train_preds_LR = LR.predict(X_train) 
test_preds_LR = LR.predict(X_test) 
#Evaluate the model
# train confusion matrix
confusion_matrix(y_train, train_preds_LR)
# test confusion matrix
confusion_matrix(y_test, test_preds_LR)
# extract TP, TN, FP, FN
tn, fp, fn, tp = confusion_matrix(y_test, test_preds_LR).ravel()
(tn, fp, fn, tp)
print(classification_report(y_test, test_preds_LR))
plt.savefig('baseline_accuracy.png') 
cm = confusion_matrix(y_test, test_preds_LR)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
classNames = ['False','True']
plt.title('Logistic Regression accuracy')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

# plt.savefig("Logistic_regression_accuracy.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/Logistic_regression_accuracy.png")
# plt.show()
# Setting up the model
model1 = Sequential()
# this is hidden layer 1
model1.add(Dense(50,activation='relu', input_shape=(X.shape[1],))) # input shape is = (features,)
# this is hidden layer 2
model1.add(Dense(25, activation='relu'))
# this is hidden layer 3
model1.add(Dense(15, activation='relu'))
# this is hidden layer 4
model1.add(Dense(10, activation='relu'))
# this is the output node
model1.add(Dense(1, activation='sigmoid')) # the activation function here is 'linear' by default
model1.summary()
#  this compiles the model, specifies model evaluation metrics
model1.compile(optimizer='Adam', loss='binary_crossentropy',
              metrics=['accuracy'])
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, 
                   verbose=1,
                   restore_best_weights=True)
# fit model
history = model1.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=4000, 
                    batch_size = 44,
                    verbose=1, 
                    callbacks=[es]) #notice we won't have to manually watch it
history_dict = history.history
history_dict.keys() 

# out of all of these, let's plot the val_mean_absolute_error
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig("Test_valid_loss_DDN_withoutdropout.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/Test_valid_loss_DDN_withoutdropout.png")
# plt.show()
plt.clf()   # clear figure
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig("Accuracy_DDN_withoutdropout.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/Accuracy_DDN_withoutdropout.png")
# plt.show()
# see how the model did!
# if you don't round to a whole number (0 or 1), the confusion matrix won't work!
preds = np.round(model1.predict(X_test),0)

# confusion matrix
confusion_matrix(y_test, preds) # order matters! (actual, predicted)

# TP is bottom right
# TN is top left
# FP is top right
# FN is bottom left

# look at documentation for conf matrix on sklearn if you have questions!
print(classification_report(y_test, preds))
cm = confusion_matrix(y_test, preds)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
classNames = ['False','True']
plt.title('DNN without callback')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

# plt.savefig("DNN_withoutdropout_confusion_matrix.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/DNN_withoutdropout_confusion_matrix.png")
# plt.show()
# Setting up the model
model2 = Sequential()
# this is hidden layer 1
model2.add(Dense(50,activation='relu', input_shape=(X.shape[1],))) # input shape is = (features,)
model2.add(Dropout(0.4)) # Dropout Layer 1
# this is hidden layer 2
model2.add(Dense(25, activation='relu'))
model2.add(Dropout(0.3)) #Dropout Layer 2
# this is hidden layer 3
model2.add(Dense(10, activation='relu'))
model2.add(Dropout(0.2)) #Dropout Layer 3
# this is the output node
model2.add(Dense(1, activation='sigmoid')) # the activation function here is 'linear' by default
model2.summary()
#  this compiles the model, specifies model evaluation metrics
model2.compile(optimizer='Adam', loss='binary_crossentropy',
              metrics=['accuracy'])
from keras.callbacks import EarlyStopping
es2 = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, 
                   verbose=1,
                   restore_best_weights=True)
# fit model
history2 = model2.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=4000, 
                    batch_size = 44,
                    verbose=1, 
                    callbacks=[es2]) #notice we won't have to manually watch it
history_dict2 = history2.history
history_dict2.keys() 

# out of all of these, let's plot the val_mean_absolute_error
acc2 = history2.history['accuracy']
val_acc2 = history2.history['val_accuracy']
loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']

epochs2 = range(1, len(acc2) + 1)

# "bo" is for "blue dot"
plt.plot(epochs2, loss2, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs2, val_loss2, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig("trial1_Test_valid_loss_DDN_withdropout.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/trial1_Test_valid_loss_DDN_withdropout.png")

plt.clf()   # clear figure
acc_values2 = history_dict2['accuracy']
val_acc_values2 = history_dict2['val_accuracy']

plt.plot(epochs2, acc2, 'bo', label='Training acc')
plt.plot(epochs2, val_acc2, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig("trial1_accuracy_DDN_withdropout.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/trial1_accuracy_DDN_withdropout.png")
# see how the model did!
# if you don't round to a whole number (0 or 1), the confusion matrix won't work!
preds2 = np.round(model2.predict(X_test),0)

# confusion matrix
confusion_matrix(y_test, preds2) # order matters! (actual, predicted)

# TP is bottom right
# TN is top left
# FP is top right
# FN is bottom left

# look at documentation for conf matrix on sklearn if you have questions!
print(classification_report(y_test, preds2))
cm = confusion_matrix(y_test, preds2)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
classNames = ['False','True']
plt.title('DNN without callback')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

# plt.savefig("Trail1_DNN_withdropout_confusion_matrix.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/Trail1_DNN_withdropout_confusion_matrix.png")
# Setting up the model
model3 = Sequential()
# this is hidden layer 1
model3.add(Dense(20,activation='relu', input_shape=(X.shape[1],))) # input shape is = (features,)
# this is hidden layer 2
model3.add(Dense(50, activation='relu'))
# this is hidden layer 3
model3.add(Dense(50, activation='relu'))
# this is hidden layer 4
model3.add(Dense(20, activation='relu'))
# this is the output node
model3.add(Dense(1, activation='sigmoid')) # the activation function here is 'linear' by default
model3.summary()
#  this compiles the model, specifies model evaluation metrics
model3.compile(optimizer='Adam', loss='binary_crossentropy',
              metrics=['accuracy'])
from keras.callbacks import EarlyStopping
es3 = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, 
                   verbose=1,
                   restore_best_weights=True)
# fit model
history3 = model3.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=4000, 
                    batch_size = 44,
                    verbose=1, 
                    callbacks=[es3]) #notice we won't have to manually watch it
history_dict3 = history3.history
history_dict3.keys() 

# out of all of these, let's plot the val_mean_absolute_error
acc3 = history3.history['accuracy']
val_acc3 = history3.history['val_accuracy']
loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']

epochs3 = range(1, len(acc3) + 1)

# "bo" is for "blue dot"
plt.plot(epochs3, loss3, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs3, val_loss3, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig("trial2_train_valid__loss.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/trial2_train_valid__loss.png")

plt.clf()   # clear figure
acc_values3 = history_dict3['accuracy']
val_acc_values3 = history_dict3['val_accuracy']

plt.plot(epochs3, acc3, 'bo', label='Training acc')
plt.plot(epochs3, val_acc3, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig("trial2_accuracy.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/trial2_accuracy.png")
# see how the model did!
# if you don't round to a whole number (0 or 1), the confusion matrix won't work!
preds3 = np.round(model3.predict(X_test),0)

# confusion matrix
confusion_matrix(y_test, preds3) # order matters! (actual, predicted)

# TP is bottom right
# TN is top left
# FP is top right
# FN is bottom left

# look at documentation for conf matrix on sklearn if you have questions!
print(classification_report(y_test, preds3))
cm = confusion_matrix(y_test, preds3)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
classNames = ['False','True']
plt.title('DNN without callback')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

# plt.savefig("Trail2_confusion_matrix.png")
# images_dir = '/content/drive/Shared drives/Deep Learning Group Projects/Project #1 /Work'
# plt.savefig(f"{images_dir}/Trail2_confusion_matrix.png")
plt.show()
#Feature Importance for NN

from sklearn.inspection import permutation_importance

results = permutation_importance(model2, X_train, y_train, scoring='neg_root_mean_squared_error')

plt.figure(figsize=(10,10))

#get importance
importance_nn = results.importances_mean
sorted_idx = np.argsort(importance_nn)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, importance_nn[sorted_idx],align='center')

plt.yticks(pos, X_train.columns[sorted_idx],fontsize=25)
plt.xlabel('Permutation Feature Importance Scores', fontsize=25)
#plt.xticks(fontsize=100)
plt.title('Permutation Feature Importance for Neural Network', fontsize=30)

plt.tight_layout()

plt.show()