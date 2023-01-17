#import libraries and datasets



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense

from keras import backend as K

from keras.utils import get_custom_objects

from keras.layers import Activation,Dropout

from keras.optimizers import Adam

sns.set_style('darkgrid')

plt.rcParams['figure.figsize']=(16, 8.27) #set graphs size to A4 dimensions

sns.set(font_scale = 1.4)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))







train_set = pd.read_csv('/kaggle/input/titanic/train.csv')

test_set = pd.read_csv('/kaggle/input/titanic/test.csv')

train_set.info()
test_set.info()
train_set.head(10)
test_set.head(10)
for feature in train_set.columns:

    print(feature, round((train_set[feature].isnull().mean()*100),2),'% missing')
for feature in test_set.columns:

    print(feature, round((test_set[feature].isnull().mean()*100),2),'% missing')
sns.countplot(train_set['Sex'],edgecolor="k", palette="Set1")
sns.countplot(train_set['Pclass'],hue=train_set['Sex'],edgecolor="k", palette="Set1")
sns.countplot(train_set['Pclass'],hue=train_set['Survived'],edgecolor="k", palette="Set1")
sns.countplot(train_set['Embarked'],hue=train_set['Survived'],edgecolor="k", palette="Set1")
sns.countplot(train_set['Sex'],hue=train_set['Survived'],edgecolor="k", palette="Set1")
sns.distplot(train_set['Age'],hist_kws=dict(edgecolor="k", linewidth=2),kde_kws={"color": "#ce0d55", "lw": 2})
sns.distplot(train_set['Fare'],hist_kws=dict(edgecolor="k", linewidth=2),kde_kws={"color": "#ce0d55", "lw": 2})
# Combine train and test set to achieve a better accuracy score 

# !!This is NOT the proper way in a real-world use case scenario!!



dataset=pd.concat([train_set,test_set])    
dataset.info() 
#sex mapping 

dataset['Sex']=np.where(dataset['Sex']=='male',1,0) #convert Male to 1 and Female to 0
#Title mapping

dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



#transform all titles to 6 unique categories

Title_Dict = { "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Jonkheer": "Royalty", "Don": "Royalty",

                    "Sir" : "Royalty", "Dr": "Officer", "Rev": "Officer", "the Countess":"Royalty", "Mme": "Mrs",

                    "Mlle": "Miss", "Ms": "Mrs", "Mr" : "Mr", "Mrs" : "Mrs", "Miss" : "Miss", "Master" : "Master",

                    "Lady" : "Royalty" }



dataset['Title'] = dataset['Title'].map(Title_Dict) #apply transformation to the column
dataset[dataset['Title'].isnull()] #check if mapping worked properly
dataset['Title'].fillna('Mrs',inplace=True) #filling NaN values with Mrs because both passengers are ladies at 30's 
dataset['Title'].value_counts() #check frequency of each category
#encoding categories according to the frequency in dataset (0 most common, 5 rarest) and apply them to dataset

title_labels=dataset['Title'].value_counts().index

title_labels={k:i for i,k in enumerate(title_labels,0)}

dataset['Title']=dataset['Title'].map(title_labels)
dataset['Title'].value_counts() #chech if encoding worked properly
cabin_mapping={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7} #find cabins unique starting letter and encode them



dataset['Cabin']=dataset['Cabin'].str[:1]

dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)



#fill nan values of cabin feature, according to the median cabin value for each class.  

dataset['Cabin'].fillna(dataset.groupby('Pclass')['Cabin'].transform('median'),inplace=True) 
#fill nan values for age, according to the median value of each category title.

dataset["Age"].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
#fill nan values of Embarked with 'S' because is the mode value for embarked feature

dataset['Embarked'] = dataset['Embarked'].fillna('S')



#encode the embarked feature according to the frequency of each class.(2 most common, 0 rarest)

embarked_mapping = {"S": 2, "C": 1, "Q": 0}

dataset['Embarked']=dataset['Embarked'].map(embarked_mapping)# apply encoding to dataset
#fill nan values for Fare feature, according to median of fare for each cabin unique value.

dataset["Fare"].fillna(dataset.groupby("Cabin")["Fare"].transform("median"), inplace=True)
#Create a new feature FamilySize that includes all members of Parch and SibSp for every passenger 

#(including the passenger too)

dataset['FamilySize']=dataset['Parch']+dataset['SibSp']+1

dataset.drop(['SibSp','Parch'],axis=1,inplace=True)
dataset['FamilySize'].value_counts()
#apply sqrt transformation to handle outliers.

dataset['Age']=np.sqrt(dataset['Age'])

dataset['Fare']=np.sqrt(dataset['Fare'])

dataset['FamilySize']=np.sqrt(dataset['FamilySize'])
#drop unuseful features

dataset.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)



#split features (X) from target (y)

X=dataset.drop('Survived',axis=1)

y=dataset['Survived']



#scaling the X dataframe (features)

scaler=MinMaxScaler()

scaled_features=scaler.fit_transform(X)

scaled_features_X=pd.DataFrame(scaled_features,columns=X.columns)

scaled_features_X.index=X.index



#concat scaled X features with y target 

final_dataset=pd.concat([scaled_features_X,y],axis=1)
X=final_dataset.drop('Survived',axis=1)

y=final_dataset['Survived']



#split to train and test set

X_train=X[:891]

X_test=X[891:]



y_train=y[:891]

y_test=y[891:]
class Swish(Activation):

    

    def __init__(self, activation, **kwargs):

        super(Swish, self).__init__(activation, **kwargs)

        self.__name__ = 'swish'



def swish(x):

    return (K.sigmoid(x) * x)



get_custom_objects().update({'swish': Swish(swish)})





#build custom Adam optimizer 

optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)





# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 64, kernel_initializer = 'normal',activation='swish', input_dim = 8))



#Adding the second hidden layer

classifier.add(Dense(units =64, kernel_initializer = 'normal', activation='swish'))



classifier.add(Dropout(0.4))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history=classifier.fit(X_train, y_train, batch_size = 32, epochs = 200, validation_split=0.1, shuffle=True)



# Predicting the Test set results

y_pred = classifier.predict(X_test)

# each prediction above 0.5 is classified as 1 and the rest as 0

y_pred = (y_pred > 0.5)

y_pred= y_pred*1

y_pred =y_pred.reshape(418)

y_pred=pd.Series(y_pred).rename('Survived')

plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train set')

plt.plot(history.history['val_accuracy'], label='validation set')

plt.legend()

plt.show()
users_id=test_set['PassengerId']



#concat users id and y_pred to create final submission DataFramet

submission=pd.concat([users_id,y_pred],axis=1)



#export to csv

submission.to_csv('submission.csv',index=False)