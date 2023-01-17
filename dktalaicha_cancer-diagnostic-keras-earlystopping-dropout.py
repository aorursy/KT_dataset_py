# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math
import matplotlib
import tensorflow as tf

# Print versions of libraries
print(f"Numpy version : Numpy {np.__version__}")
print(f"Pandas version : Pandas {pd.__version__}")
print(f"Matplotlib version : Matplotlib {matplotlib.__version__}")
print(f"Seaborn version : Seaborn {sns.__version__}")
print(f"Tensorflow version : Tensorflow {tf.__version__}")

#Magic function to display In-Notebook display
%matplotlib inline

# Setting seabon style
sns.set(style='darkgrid', palette='Set2')
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv', encoding = 'latin-1')
df.head(10).T
df.columns
df.info()
df.drop(['id','Unnamed: 32'],axis=1, inplace=True)
df.head().T
df.describe().T
df["diagnosis"].value_counts().plot(kind = 'pie',explode=[0, 0.1],figsize=(6, 6),autopct='%1.1f%%',shadow=True)
plt.title("Malignant and Benign Distribution",fontsize=20)
plt.legend(["Benign", "Malignant"])
plt.show()
print(df['diagnosis'].value_counts())
print('\n')
print(df['diagnosis'].value_counts(normalize=True))
plt.figure(figsize=(12,10))

sns.distplot(df[df['diagnosis'] == 'M']["radius_mean"], color='g', label = "Bening") 
sns.distplot(df[df['diagnosis'] == 'B']["radius_mean"], color='r', label = "Malignant") 

plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors", fontsize=14)
plt.legend()

plt.show()
# most_frequent_bening_radius_mean
df[df["diagnosis"] == 'B']['radius_mean'].value_counts().idxmax()
# most_frequent_malignant_radius_mean
df[df["diagnosis"] == 'M']['radius_mean'].value_counts().idxmax()
features_mean=list(df.columns[1:11])
# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] == 'M']
dfB=df[df['diagnosis'] == 'B']

#Stack the data
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14,16))
axes = axes.ravel()

for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], 
            bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , 
            alpha=0.5,
            stacked=True, 
            density = True, 
            label=['M','B'],
            color=['r','g'])
    
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()
melted_data = pd.melt(df,id_vars = "diagnosis",value_vars = ['radius_mean', 'texture_mean'])

plt.figure(figsize = (14,8))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data, fliersize=0)

# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
# plt.legend(["Benign", "Malignant"])
plt.show()
# Also we can look relationship between more than 2 distribution
# sns.set(style = "white")

sns.pairplot(df, vars=["radius_mean","area_mean","texture_mean",'smoothness_mean',"fractal_dimension_se"], hue='diagnosis')
plt.suptitle('Relations ship between features');
plt.show()
plt.figure(figsize = (15,10))
sns.jointplot(df['radius_mean'],df['area_mean'],kind="reg")
plt.show()
plt.figure(figsize=(18,18))
plt.title('Pearson Correlation Matrix')
# Generating correlation
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr,mask = mask,linewidths=0.25,vmax=0.7,square=True,cmap="viridis",linecolor='w',annot=True,cbar_kws={"shrink": .7});
plt.show()
df.reset_index(inplace = True , drop = True)
df['diagnosis'].value_counts()
df['diagnosis'] = df['diagnosis'].map({'M': 1,'B': 0})
df['diagnosis'].value_counts()
X = df.drop('diagnosis',axis=1).values
y = df['diagnosis'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
# Quick sanity check with the shapes of Training and testing datasets
print("X_train - ",X_train.shape)
print("y_train - ",y_train.shape)
print("X_test - ",X_test.shape)
print("y_test - ",y_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')
# https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
# https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1
          )
model_loss = pd.DataFrame(model.history.history)
plt.figure(figsize=(12,8))
model_loss.plot()
plt.show()
model = Sequential()

model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
from tensorflow.keras.layers import Dropout
model = Sequential()

model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)
from sklearn import metrics
# https://en.wikipedia.org/wiki/Precision_and_recall
print(metrics.classification_report(y_test, y_test_pred))
y_test_pred = y_test_pred.flatten()
# Heatmap for Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test,y_test_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 25}, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1, fontsize = 22)
plt.ylabel('Actual',fontsize = 18)
plt.xlabel('Predicted',fontsize = 18)
plt.show()
# Printing the Overall Accuracy of the model
print("Accuracy of the model : {0:0.3f}".format(metrics.accuracy_score(y_test, y_test_pred)))
print("Count of Actual values of Test data :")
print(pd.Series(y_test).value_counts())

print("\n")

print("Count of Predected values of Test data :")
print(pd.Series(y_test_pred).value_counts())
54/55
cnf_matrix[1][1]/pd.Series(y_test).value_counts()[1]
