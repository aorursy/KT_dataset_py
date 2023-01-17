#Supporting libs

%matplotlib notebook

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

sns.set()
#Load both trainand test datasets

train_data= pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')
#Add one column names 'dataType' to mark data types

train_data['dataType']='train'

test_data['dataType']='test'
#Combine train and test dataset to do feature engineering all together

all_data=pd.concat([train_data,test_data],axis=0)

all_data.set_index('PassengerId',inplace=True)

all_data.info()
# Get a summary of missing value for all fields

all_data.isnull().sum()
#Use 'U' to fill missing value in Cabin

all_data['Cabin'].fillna('U',inplace =True)

#Using mean value to fill missing value in Fare

all_data['Fare'].fillna(all_data['Fare'].mean(),inplace =True)
all_data['Embarked'].value_counts()
all_data['Embarked'].fillna('S',inplace = True)
#Process Age, while there are 263 missing value, this should impact the the probility of survival

#Let's use histogram to check how the Age distributed 

all_data['Age'].hist()
avg_age=all_data['Age'].mean()

std_age=all_data['Age'].std()

no_nan=all_data['Age'].isnull().sum()

rand=np.random.randint(avg_age-std_age,avg_age+std_age,size=no_nan)

all_data['Age'][all_data.Age.isnull()]=rand

all_data['Age'].hist()
all_data.info()
all_data['Pclass']=all_data['Pclass'].astype(str)
tobe_dummied_cols = ['Pclass', 'Sex', 'Cabin', 'Embarked']

obj_df = all_data[tobe_dummied_cols]

obj_df_dummy = pd.get_dummies(obj_df)
obj_df_dummy.shape
#Procced column Name, transform it into title

titles = set()

for name in all_data['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())



def status(feature):

    print('Processing', feature, ': Done')

    

Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

}



# Extract the title from each name, and map names to titles

def get_titles(dataset):

    dataset['Title'] = dataset['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    dataset['Title'] = dataset.Title.map(Title_Dictionary)

    status('Title')

    return dataset
all_data = get_titles(all_data)
all_data['Family']=all_data['Parch']+all_data['SibSp']

all_data.drop(['Parch','SibSp'],inplace = True, axis=1)
all_data.head()
all_data.drop(['Cabin','Embarked','Name','Pclass','Sex','Ticket'],inplace=True,axis=1)
dummy_title=pd.get_dummies(all_data['Title'],prefix='Title')

all_data.drop('Title',inplace=True,axis=1)
all_data=pd.concat((all_data,dummy_title,obj_df_dummy),axis=1)
from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()

all_data['Age']=scaler.fit_transform(all_data.filter(['Age']))

all_data['Fare']=scaler.fit_transform(all_data.filter(['Fare']))
# all_data['Family']=scaler.fit_transform(all_data.filter(['Family']))
all_data.head()
from sklearn.decomposition import PCA
new_train_data=all_data[all_data['dataType']=='train']

new_test_data=all_data[all_data['dataType']=='test']
new_train_data.drop(['dataType','Survived'],inplace = True,axis=1)

new_test_data.drop(['dataType','Survived'],inplace = True,axis=1)
x_train_reduced = PCA(n_components=0.98).fit_transform(new_train_data)

x_test_reduced = PCA(n_components=66).fit_transform(new_test_data)
x_test_reduced.shape
y_label = train_data['Survived']
y_label.shape
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(0.0001)

loss_function = "sparse_categorical_crossentropy"
model = tf.keras.Sequential([

    tf.keras.layers.Dense(units=64, input_dim=66,

                          activation='relu'),  #input dims = number of fields

    tf.keras.layers.Dense(units=32, activation="relu"),

    tf.keras.layers.Dense(units=2, activation='softmax')

])

model.compile(optimizer = optimizer, loss = loss_function, metrics=['accuracy'])
model.summary()
history=model.fit(x_train_reduced,y_label,epochs=170,validation_split = 0.2)
def v_train_history(trainhist, train_metrics, valid_metrics):

    plt.plot(trainhist.history[train_metrics])

    plt.plot(trainhist.history[valid_metrics])

    plt.title('Training metrics')

    plt.ylabel(train_metrics)

    plt.xlabel('Epochs')

    plt.legend(['train','validation'],loc='upper left')

    plt.show()
v_train_history(history,'loss','val_loss')
v_train_history(history,'acc','val_acc')
x_test_reduced.shape
y_pred=model.predict_classes(x_test_reduced)
pred_survied_pd = pd.DataFrame(y_pred,

                               index=new_test_data.index,

                               columns=['Survived'])

pred_survied_pd.reset_index()

pred_survied_pd.to_csv('submission.csv')