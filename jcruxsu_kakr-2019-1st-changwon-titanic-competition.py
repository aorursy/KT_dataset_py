import os 
import pandas as pd
import numpy as np
data_dir = '../input'

train_path = os.path.join(data_dir,'train.csv') 
test_path = os.path.join(data_dir,'test.csv') 

raw_train_data = pd.read_csv(train_path)
raw_test_data = pd.read_csv(test_path)

raw_train_data.head()
#위에서부터 5개의 raw데이터만 뽑는 것입니다.
raw_test_data.head()
view_train_data = raw_train_data.copy()
view_test_data = raw_test_data.copy()
import matplotlib.pyplot as plt
#시각화를 하기위한 파이썬 라이브러리 matplotlib를 import 합니다.

view_train_data.describe()
#각 데이터가 가진 통계치입니다. 계산가능한 데이터를 출력하기떄문에 스트링값들은 제외됩니다. 

view_test_data.describe()
view_train_data.isnull().sum()
#각 에리어에 몇개의 Null 데이터가 있는지 합산해서 보여줍니다.
for col in view_train_data.columns:
    null_count = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (view_train_data[col].isnull().sum() / view_train_data[col].shape[0]))
    print(null_count)
import missingno as msno
import seaborn as sns
msno.matrix(view_train_data, figsize=(7, 7),color=(0.2,0.4,0.8))
msno.matrix(view_test_data, figsize=(7, 7),color=(0.2,0.8,0.2))
msno.bar(view_train_data, figsize=(7, 7),color=(0.2,0.4,0.8))
msno.bar(view_test_data, figsize=(7, 7),color=(0.2,0.8,0.2))
#print(train_data.isnull().sum())

data_cleaner = [view_train_data,view_test_data]        

for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)
    
print('정리된 후 Null값 확인')
print('Train_data')
print('_______________________________')
print(view_train_data.isnull().sum())
print('\n Test data')
print('_______________________________')
print(view_test_data.isnull().sum())

f, ax = plt.subplots(1, 2, figsize=(18, 8))

view_train_data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=view_train_data, ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()
view_train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
view_train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
pd.crosstab(view_train_data['Pclass'], view_train_data['Survived'], margins=True).style.background_gradient(cmap='summer_r')
view_train_data[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
#각 pclass의 테이블의 속성에 대한 평균을 구한후 그래프로 구현함(ex.1의 경우 136/216으로 계산)
sns.countplot('Pclass', hue='Survived', data=view_train_data)
view_train_data[['Sex','Survived']].groupby(['Sex']).mean()
f,ax = plt.subplots(1,2,figsize=(17,7))
view_train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
sns.countplot('Sex', hue='Survived', data=view_train_data,ax=ax[1])
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(view_train_data['Age'].max()))
print('제일 어린 탑승객 : {:.1f} Years'.format(view_train_data['Age'].min()))
print('탑승객 평균 나이 : {:.1f} Years'.format(view_train_data['Age'].mean()))
#view_train_data[view_train_data['Survived']==1]['Age']
fig,ax = plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(view_train_data[view_train_data['Survived']==0]['Age'],ax = ax)
sns.kdeplot(view_train_data[view_train_data['Survived']==1]['Age'],ax=ax)
plt.legend(['Survived==0','Survived==1'])
plt.show()
fig,ax = plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(view_train_data[view_train_data['Pclass']==1]['Age'])
sns.kdeplot(view_train_data[view_train_data['Pclass']==2]['Age'])
sns.kdeplot(view_train_data[view_train_data['Pclass']==3]['Age'])
plt.title('Age distribution with plcass')
plt.legend(['pclass==1','pclass==2','pclass==3'])
plt.show()
fig,ax = plt.subplots(1,2,figsize=(16,7))
sns.violinplot('Pclass','Age',hue='Survived',data=view_train_data,scale='count',split=True,ax=ax[0])
ax[0].set_title('Pclass and Age relation with Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot('Sex','Age',hue='Survived',data=view_train_data,scale='count',split=True,ax=ax[1])
ax[1].set_yticks(range(0,110,10))
ax[1].set_title('Sex and Age relation with Survived')
plt.show()
view_train_data[['Embarked','Survived']].groupby(['Embarked']).mean()
fig,ax = plt.subplots(1,1,figsize=(8,8))
view_train_data[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar(ax=ax)
ax.set_title('Relationship Embarked  and Survived')
fig,ax =plt.subplots(2,2, figsize=(20,10))
sns.countplot('Embarked',data=view_train_data,ax=ax[0][0])
ax[0][0].set_title('The number of Passengers who take a ship by embarked')
sns.countplot('Embarked',hue='Survived',data=view_train_data,ax=ax[0][1])
ax[0][1].set_title('The number of Survived Passengers ')
sns.countplot('Embarked',hue='Pclass',data=view_train_data,ax=ax[1][0])
ax[1][0].set_title('The number of  Passengers  relation with embarked which is classified by pclass ')
sns.countplot('Embarked',hue='Sex',data=view_train_data,ax=ax[1][1])
ax[1][1].set_title('The number of  Passengers  relation with embarked which is classified by Sex ')
view_train_data.isnull().sum()
view_test_data.isnull().sum()
drop_column = ['PassengerId','Cabin', 'Ticket']
view_test_data.drop(drop_column, axis=1, inplace = True)
view_train_data.drop(drop_column, axis=1, inplace = True)

view_test_data.info()
print('________________________________________________________')
print("PassengerId,Cabin, Ticket의 피쳐가 각 테이블 셋으로부터 제거되었습니다.")
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1 
    
    dataset['Alone'] = 1
    
    dataset['Alone'].loc[dataset['FamilySize'] > 1] = 0
        
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".",expand=True)[0]
    
title_names = (view_train_data['Title'].value_counts() < 10)

view_train_data['Title'] = view_train_data['Title'].apply(lambda x:'Misc' if title_names.loc[x] == True else x)

view_train_data[['Title','Alone','FamilySize']].head()
pd.crosstab(view_train_data['Title'], view_train_data['Survived'], margins=True).style.background_gradient(cmap='summer_r')
title_names = (view_train_data['Title'].value_counts() < 10)

view_train_data['Title'] = view_train_data['Title'].apply(lambda x:'Misc' if title_names.loc[x] == True else x)

pd.crosstab(view_train_data['Title'], view_train_data['Survived'], margins=True).style.background_gradient(cmap='summer_r')
view_train_data.drop('Name', axis=1, inplace = True)
view_test_data.drop('Name', axis=1, inplace = True)
view_test_data.head()
sns.distplot(view_train_data['Fare'])
def data_regularization(data):
    cols = list(data.columns)
    for col in cols:
        std = data[col].std()
        mean = data[col].mean()
        data[col] = np.abs(data[col] - mean)
        data[col] /= std
    return data


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for dataset in data_cleaner:
    dataset['Title'] =  label_encoder.fit_transform(dataset['Title'])
    dataset['Sex'] =  label_encoder.fit_transform(dataset['Sex'])
    dataset['Embarked'] =  label_encoder.fit_transform(dataset['Embarked'])
    dataset['Embarked'] =  label_encoder.fit_transform(dataset['Embarked'])
train_label =np.array(view_train_data.pop('Survived').tolist())
view_train_data = data_regularization(view_train_data)
view_test_data = data_regularization(view_test_data)
view_train_data.info()
view_test_data.info()
view_train_data.head()
sns.distplot(view_train_data['Fare'])
from keras import *
from keras.models import *

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_tr, X_vld, y_tr, y_vld = train_test_split(view_train_data, train_label, test_size=0.25, random_state=2019)
##models
def Vanila_fully_connected_layer_model():
    model = Sequential()
    model.add(layers.Dense(256,activation='relu',input_shape=(10,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(2048,activation='relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(2048,activation='relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['acc'])
    history= model.fit(x=X_tr,y=y_tr,epochs=20,validation_data=(X_vld,y_vld))
    return model,history


def Adaboost_classifier():
    model=AdaBoostClassifier()
    model.fit(X_tr, y_tr)
    prediction = model.predict(X_vld)
    return model.score(X_vld, y_vld)

def svc_model():
    model =LinearSVC()
    model.fit(X_tr, y_tr)
    prediction = model.predict(X_vld)
    return model.score(X_vld, y_vld)

def random_forest_classfier():
    model = RandomForestClassifier()
    model.fit(X_tr, y_tr)
    prediction = model.predict(X_vld)
    return model.score(X_vld, y_vld)


vanila_model,vanila_history = Vanila_fully_connected_layer_model()
acc = vanila_history.history['acc']
val_acc = vanila_history.history['val_acc']
loss= vanila_history.history['loss']
val_loss = vanila_history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,"bo")
plt.plot(epochs,val_acc,"b")
plt.legend(["train_accuracy","validation_accuracy"])
plt.show()
plt.plot(epochs,loss,"bo")
plt.plot(epochs,val_loss,"b")
plt.legend(["train_loss","validation_loss"])
plt.show()
others_models_results = [svc_model(),Adaboost_classifier(),random_forest_classfier(),vanila_model.evaluate(X_vld,y_vld)[1]]
plt.plot(range(1,len(others_models_results)+1),others_models_results,"bo")
plt.show()
print("순서대로 svc,Adaboost,random_forest,Fc_layer 입니다.")
print(others_models_results)
print("84프로면 나쁘지않은 결과이군요 ")
from pandas import Series

skmodel = RandomForestClassifier()
skmodel.fit(X_tr, y_tr)

feature_importance = skmodel.feature_importances_
Series_feat_imp = Series(feature_importance, index=view_test_data.columns)
plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
prediction = vanila_model.predict(view_test_data)
submission = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))
submission['Survived'] =prediction.round()

submission.to_csv("./submission.csv", header=True, index=False)
