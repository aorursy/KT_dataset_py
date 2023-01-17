import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sns.set()
#loading data
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv", index_col="PassengerId")
train = pd.read_csv("../input/titanic/train.csv", index_col="PassengerId")
dfs = [train, test]
train.head()
total_passangers = train.shape[0] + test.shape[0]
print(f' Total Passangers = {total_passangers} \n')
print(f' Train Dataset \n Rows:{train.shape[0]}  Features: {train.shape[1]} ({train.shape[0]/total_passangers * 100}) \n')
print(f' Test Dataset \n Rows:{test.shape[0]}  Features: {test.shape[1]} ({test.shape[0]/total_passangers * 100}) ')
print(f'Train: {dfs[0].isnull().sum()} \n')
print(f'Test: {dfs[1].isnull().sum()}')
fig = plt.figure(figsize=(25,5))
ax1 = fig.add_subplot(151)
ax2 = fig.add_subplot(152)
ax3 = fig.add_subplot(153)
ax4 = fig.add_subplot(154)
ax5 = fig.add_subplot(155)


age_cut = pd.cut(train['Age'], 5)

sns.countplot(x=train['Survived'], ax=ax1)
sns.countplot(x=train['Survived'], hue=train['Sex'], ax=ax2)
sns.countplot(x=train['Survived'], hue=train['Pclass'], ax=ax3)
sns.countplot(x=train['Survived'], hue=train['Parch'], ax=ax4)
sns.countplot(x=train['Survived'], hue=age_cut, ax=ax5)
# Checkpoint
df = train.copy()
df_test = test.copy()
dfs = [df, df_test]
# Update embarked

def update_embarked(data):
  highest = data['Embarked'].value_counts().index[0]
  data['Embarked'] = data['Embarked'].fillna(highest)
  return data

df = update_embarked(df)
df_test = update_embarked(df_test)
# Check if passanger had a Cabin

def get_cabin(data):
  data['HadCabin'] = data['Cabin'].notna()
  data['HadCabin'] = data['HadCabin'].astype(int)  
  return data

df = get_cabin(df)
df_test = get_cabin(df_test)

sns.countplot(x=df['Survived'], hue=df['HadCabin'])
# FARE

def classify_fare(fare):
  if fare <= 7.91:
    return 0
  elif fare > 7.91 and fare <= 14.454:
    return 1
  elif fare > 14.454 and fare <= 31:
    return 2
  elif fare > 31:
    return 3
  else:
    return 4

def update_fare(df):  
  df['Fare'] = df['Fare'].fillna(df['Fare'].median())
  df['FareCat'] = df['Fare'].apply(classify_fare)
  df['Fare'] = df['Fare'].astype(int)
  return df

df = update_fare(df)
df_test = update_fare(df_test)
# AGE
def classify_age(age):    
    if age > 0 and age < 17:
      return 0        
    elif age >= 17 and age < 32:
      return 1
    elif age >= 32 and age < 42:
      return 2        
    elif age >= 42 and age < 64:
      return 3        
    elif age >= 64:
      return 4
    else:
      return 2        

def update_age(df):
  df['Age'] = df['Age'].fillna(df['Age'].median())
  df['AgeCat'] = df['Age'].apply(classify_age)
  return df

df = update_age(df)
df_test = update_age(df_test)

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.countplot(x=df['Survived'], hue=df['AgeCat'], ax=ax1)
sns.countplot(x=df['AgeCat'], ax=ax2)
sns.countplot(x=df['AgeCat'], hue=df['Sex'], ax=ax3)
# Check if is a Male Adult

def get_male_adult(df):
  df['IsMaleAdult'] = 0  
  df['IsMaleAdult'] = np.where( (df['Sex'] == 'male') & (df['Age'] >= 17), 1, 0)
  return df

df = get_male_adult(df)
df_test = get_male_adult(df_test)
sns.countplot(x=df['Survived'], hue=df['IsMaleAdult'])
def update_family(df):
  df['Family'] = df['SibSp'] + df['Parch'] + 1
  df['IsAlone'] = 0
  df['IsAlone'] = df['Family'] < 2
  df['IsAlone'] = df['IsAlone'].astype(int)
  return df

df = update_family(df)
df_test = update_family(df_test)
sns.countplot(x=df['Survived'], hue=df['IsAlone'])
# CheckPoint
processed_df = df.copy()
processed_df_test = df_test.copy()
processed_df
def get_categoricals(df, columns):
  return pd.get_dummies(df, columns=columns, drop_first=True)

columns = processed_df.columns
print(columns)

columns = ['Pclass', 'Embarked', 'AgeCat', 'FareCat']

processed_df = get_categoricals(processed_df, columns)
processed_df_test = get_categoricals(processed_df_test,columns)
def drop_columns(df, columns):
  for col in columns:
    df = df.drop(col, axis=1)
  return df

columns =  ['Sex', 'Parch','Fare','Age', 'Ticket', 'Cabin', 'Name', 'SibSp']

processed_df = drop_columns(processed_df, columns) 
processed_df_test = drop_columns(processed_df_test, columns) 

# Final Table

processed_df
# Final correlation
plt.figure(figsize=(30,20))
sns.heatmap(processed_df.corr(), vmax=0.6, square=True, annot=True, cmap="coolwarm")
processed_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X = processed_df.iloc[:, 1:].values
y = processed_df.iloc[:, 0].values
X_test_pred = processed_df_test.iloc[:].values
processed_df
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.85)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_test_pred = scaler.transform(X_test_pred)
X_test.shape
# !pip install tensorflow==2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_test_pred.shape)
model = Sequential()

model.add(Dense(X_train.shape[1] + 2 , activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()
es = EarlyStopping(patience=2)
epochs = 300
batch_size = X_train.shape[0] // 12

model.fit(X_train, 
          y_train, 
          batch_size = batch_size,
          validation_split=0.2, 
          epochs=epochs, callbacks=[es], 
          verbose=0)

loss = pd.DataFrame(model.history.history)
loss.plot()

plt.show()
pred = model.predict_classes(X_test)
rep = classification_report(y_test, pred)
mx = confusion_matrix(y_true=y_test, y_pred=pred)
sns.heatmap(mx, annot=True, xticklabels=False, yticklabels=False)
print(rep)
pred_test = model.predict_classes(X_test_pred)
d = pd.DataFrame(pred_test, columns=['Survived'])
d.index = test.index
sns.countplot(x='Survived', data=d)
d.to_csv('Survived.csv')
