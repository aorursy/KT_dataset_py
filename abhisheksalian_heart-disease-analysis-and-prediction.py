

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

style.use('ggplot')

df = pd.read_csv("../input/heart-disease-uci//heart.csv") # data loading into dataframe
df.head()
df.isna().sum() # output of this  indicate that there is no missing values in any column
cols = df.columns

cols
max_age = max(df['age'])

min_age = min(df['age'])

print(f"Max age {max_age}, min age {min_age}")
plt.figure(figsize = (12,6))

age_count = dict(df['age'].value_counts())

plt.bar(age_count.keys(),age_count.values(),width = 0.5)

plt.xticks(ticks=[i for i in range(min_age-1,max_age+1)])

plt.title("Age frequency plot")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.show()

plt.figure(figsize = (5,5))

sex_count = dict(df['sex'].value_counts())

plt.bar(sex_count.keys(),sex_count.values(),width = 0.5)

plt.xticks([0,1],("Female(0)","Male(1)"))

plt.xlabel("Sex")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (5,5))

count = dict(df['cp'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.5)

plt.xticks([0,1,2,3],("typical angina","Atypical angina","Non-anginal pain","Asymtomatic"),rotation=45)

plt.title("Chest pain frequency plot")

plt.xlabel("Chest pain type")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (10,8))

count = dict(df['trestbps'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.7)

#plt.xticks(ticks=[i for i in range(min(df['trestbps'])-1,max(df['trestbps'])+1)])

plt.title("resting blood pressure frequency plot")

plt.xlabel("resting blood pressure in mm")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (8,5))

count = dict(df['chol'].value_counts())

plt.bar(count.keys(),count.values(),width = 1,color='red')

#plt.xticks(ticks=[i for i in range(min(df['trestbps'])-1,max(df['trestbps'])+1)])

plt.title("serum cholestoral in mg/dl plot")

plt.xlabel("serum cholestoral in mg/dl")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (5,5))

count = dict(df['fbs'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.6)

plt.xticks([0,1],("False(0)","True(1)"))

plt.title("Fasting blood sugar plot")

plt.xlabel("Fasting blood sugar > 120 mg/dl")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (5,5))

count = dict(df['restecg'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.6)

plt.xticks([0,1,2])

plt.title("Resting electrocardiographic results plot")

plt.xlabel("Resting electrocardiographic value")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (32,10))

count = dict(df['thalach'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.6)

plt.xticks(ticks= list(count.keys()),rotation=90)

plt.title("Maximum heart rate achieved plot")

plt.xlabel("Maximum heart rate achieved")

plt.ylabel("Frequency")

plt.show()
sns.distplot(df['thalach'])

plt.show()
plt.figure(figsize = (5,5))

count = dict(df['exang'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.6)

plt.xticks(ticks= list(count.keys()),labels=["NO(0)","YES(1)"])

plt.title("Exercise induced angina plot")

plt.xlabel("Exercise induced angina")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (20,8))

count = dict(df['oldpeak'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.08)

plt.xticks(ticks= list(count.keys()))

plt.title("ST depression induced by exercise plot")

plt.xlabel("ST depression induced by exercise relative to rest")

plt.ylabel("Frequency")

plt.show()
sns.distplot(df['oldpeak'])

plt.show()
plt.figure(figsize = (5,5))

count = dict(df['slope'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.8)

plt.xticks(ticks= list(count.keys()))

plt.title("The slope of the peak exercise plot")

plt.xlabel("The slope of the peak exercise ST segment")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (5,5))

count = dict(df['ca'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.8)

plt.xticks(ticks= list(count.keys()))

plt.title("Number of major vessels frequency plot")

plt.xlabel("Number of major vessels (0-3) colored by flourosopy")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (5,5))

count = dict(df['thal'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.8)

plt.xticks(ticks= list(count.keys()))

plt.title("Thal plot")

plt.xlabel("Thal")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (5,5))

count = dict(df['target'].value_counts())

plt.bar(count.keys(),count.values(),width = 0.8)

plt.xticks(ticks= list(count.keys()))

plt.title("Target plot")

plt.xlabel("Target")

plt.ylabel("Frequency")

plt.show()
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
tar1_df = df[df['target']==1]

tar0_df = df[df['target']==0]
print("No. of samples where target==1 (Yes heart disease):",len(tar1_df))

print("No. of samples where target==0 (No heart disease):",len(tar0_df))
total1 = sum(list(tar1_df['age'].value_counts())) # target == 1 

total0 = sum(list(tar0_df['age'].value_counts())) # target == 0



print(f"Target 1: {total1} | Target 0: {total0}")



d = dict(tar1_df['age'].value_counts())

plt.figure(figsize=(13,7))

plt.bar(x =list(d.keys()),height = list(d.values()))

plt.xticks(list(d.keys()))

plt.xlabel("Age")

plt.ylabel("#people")

plt.title("Target=1 vs Age")

plt.show()
d = dict(tar0_df['age'].value_counts())

plt.figure(figsize=(13,7))

plt.bar(x =list(d.keys()),height = list(d.values()))

plt.xticks(list(d.keys()))

plt.xlabel("Age")

plt.ylabel("#people")

plt.title("Target=0 vs Age")

plt.show()
tar1_df['sex'].value_counts() # target == 1 , male = 1 and female = 0
tar0_df['sex'].value_counts() # target == 0  , male = 1 and female = 0
male = tar1_df['sex'].value_counts()[1]

female = tar1_df['sex'].value_counts()[0]



print(f'No. of male with heart diease:{male}\nNo. of with female heart disease:{female}\n')



print(f'% of male with heart disease = {round(male/(male+female)*100,2)} %')



print(f'% of female with heart disease = {round(female/(male+female)*100,2)} %')

male = tar0_df['sex'].value_counts()[1]

female = tar0_df['sex'].value_counts()[0]



print(f'No. of male with no heart diease:{male}\nNo. of with female no heart disease:{female}\n')



print(f'% of male with no heart disease = {round(male/(male+female)*100,2)} %')



print(f'% of female with no heart disease = {round(female/(male+female)*100,2)} %')
d1 = dict(tar1_df["cp"].value_counts().sort_index())

d0 = dict(tar0_df["cp"].value_counts().sort_index())

plt.figure(1)

plt.subplot(121)

plt.bar(np.array(list(d1.keys())), np.array(list(d1.values())))

plt.xlabel("Chest pain type")

plt.title("Target=1")

plt.subplot(122)

plt.bar(np.array(list(d0.keys())), np.array(list(d0.values())))

plt.xlabel("Chest pain type")

plt.title("Target=0")

plt.show()

for key in d1:

    print(f"Chest pain of type {key} for target 1 is {round((d1[key]/sum(list(d1.values())))*100,2)} %")

    print(f"Chest pain of type {key} for target 0 is {round((d1[key]/sum(list(d0.values())))*100,2)} %")

    print("--------------------------------------------------------------------------------------------------------------------")
from sklearn.model_selection import train_test_split
random_state = 42

test_size = 0.2

shuffle = True
X = df[cols[:-1]].values

Y = df[cols[-1]].values



X_train, X_test,y_train, y_test = train_test_split(X, Y,

                                                   test_size = test_size,

                                                   shuffle = shuffle,

                                                   random_state = random_state)



#y_train = y_train.reshape((y_train.shape[0],1))

#y_test = y_test.reshape((y_test.shape[0],1))



print(f"Training X shape : {X_train.shape}")

print(f"Training y shape : {y_train.shape} \n")

print(f"Testing X shape : {X_test.shape}")

print(f"Testing y shape : {y_test.shape}")
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression 

max_iter =100
logReg_clf = LogisticRegression(tol = 0.0001,

                                C = 1.0,

                                random_state = random_state,

                                solver = 'lbfgs',

                                max_iter = max_iter,

                                verbose = 0,

                                n_jobs = 5)



# training

logReg_clf.fit(X_train, y_train)



#testing

y_pred = logReg_clf.predict(X_test)



score = accuracy_score(y_test, y_pred)

print("Accuracy of logistic regression classifier is",score*100,"%")

from sklearn import svm
# defining parameter range 

param_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}  

  

grid1 = GridSearchCV(svm.SVC(random_state = random_state),

                    param_grid,

                    refit = True,

                    verbose = 3) 

  

# fitting the model for grid

grid1.fit(X_train, y_train)
# print best parameter after tuning 

print(grid1.best_params_) 

  

# print how our model looks after hyper-parameter tuning 

print(grid1.best_estimator_) 
#testing

y_pred = grid1.predict(X_test)



score = accuracy_score(y_test, y_pred)

print("Accuracy of SVM classifier is",score*100,"%")
from xgboost import XGBClassifier
param_grid = {'n_estimator': [10, 100, 1000, 2000, 3000],  

              'gamma': [50, 25, 12, 10, 5, 1], 

              'max_depth': [4,5,6,7,8,9,10]} 
 

xgb_clf = XGBClassifier(scale_pos_weight=1,

                          learning_rate=0.001,  

                          colsample_bytree = 0.4,

                          subsample = 0.2,

                          objective='binary:logistic', 

                          n_estimators=1500, 

                          reg_alpha = 0.3,

                          max_depth=5, 

                          gamma= 5,

                          random_state = random_state,

                       )

xgb_clf.fit(X_train, y_train)
#testing

y_pred = xgb_clf.predict(X_test)



score = accuracy_score(y_test, y_pred)

print("Accuracy of XGBOOST classifier is",score*100,"%")
param_grid = {'n_estimator': [10, 100, 1000, 2000, 3000],  

              'gamma': [50, 25, 12, 10, 5, 1],

              'learning_rate':[0.1,0.01,0.001,0.0001]}

              #'max_depth': [4,5,6,7,8,9,10]} 



grid2 = GridSearchCV(XGBClassifier( max_depth = 4,

                                    random_state = random_state),

                    param_grid,

                    refit = True,

                    verbose = 3) 

  

# fitting the model for grid

grid2.fit(X_train, y_train)
# print best parameter after tuning 

print(grid2.best_params_) 

  

# print how our model looks after hyper-parameter tuning 

print(grid2.best_estimator_) 
#testing

y_pred = grid2.predict(X_test)



score = accuracy_score(y_test, y_pred)

print("Accuracy of XGBOOST classifier is",score*100,"%")
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical

from keras.callbacks.callbacks import ModelCheckpoint

from keras.models import Sequential,load_model

from keras.layers import Dense

scaler = StandardScaler()



scaler = scaler.fit(X_train)



X_trainS = scaler.transform(X_train)

X_testS = scaler.transform(X_test)



y_train = to_categorical(y_train, num_classes = 2)


# Neural network

model = Sequential()

model.add(Dense(80, input_dim = X_train.shape[1], activation = "relu"))

model.add(Dense(50, activation = "relu"))

model.add(Dense(2, activation = "softmax"))

model.compile(loss ='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])





# best model file path

filepath = "../working/nn.pkl"



# callback 

Callback = [ModelCheckpoint(filepath = filepath,

                          monitor = 'val_loss',

                          verbose=1,

                          save_best_only = True,

                          save_weights_only=False,

                          mode='auto', period=1)]

## 

epochs = 120

batch_size = 64





# training model

history = model.fit(X_trainS, y_train,

                    epochs = epochs,

                    batch_size = batch_size,

                    validation_split=0.1,

                    callbacks = Callback)





# plotting  accuracy and losses

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = [*range(epochs)]



plt.plot(epochs,acc, label = "train_acc")

plt.plot(epochs,val_acc, label = "val_acc")

plt.legend()



plt.show()

plt.plot(epochs,loss, label = "train_loss")

plt.plot(epochs,val_loss, label = "test_loss")

plt.legend()

plt.show()

model = load_model(filepath)



y_pred = np.argmax(model.predict(X_testS),axis=1)



print(f"Accuracy of nn is: {accuracy_score(y_pred, y_test)*100}%")