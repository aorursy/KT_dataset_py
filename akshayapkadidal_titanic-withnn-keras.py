# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install distance
!pip install stringdist
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib notebook

from tensorflow.keras.layers import Dense, BatchNormalization

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor



from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error



from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

from sklearn import metrics 

from scipy.spatial.distance import cdist

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as shc

import sklearn.cluster

import distance

import stringdist
d1 = pd.read_csv('../input/titanic/test.csv')

d2 = pd.read_csv('../input/titanic/train.csv')
df = pd.concat([d1,d2], axis=0)
df.describe(include='all')
df['GroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')
#df[df['Fare'].isna()] # there is just one record

impute_fare = df[(df['Pclass']==3) & (df['GroupSize']==1) & (df['Age']>21) & (df['Embarked']=='S')].dropna().Fare.mode()[0]

# we are looking all people who boarded from S who are adults and singletons who traevled in 3rd class.
df['Fare'].fillna(impute_fare, inplace = True)
#df[(df['Fare'] < 800) & (df['GroupSize']==2) & (df['Cabin'].str.contains('B2'))]

df['PersonPerCabin'] = df.groupby('Cabin')['Cabin'].transform('count')
df['NoOfCabinsBooked'] = df["Cabin"].str.split().str.len()

df['PersonPerCabin'] = df['PersonPerCabin'] / df['NoOfCabinsBooked']
#df[-df.PersonPerCabin.isna()]

#df[(df['Cabin']=='B51 B53 B55') | (df['Ticket']=='PC 17755')]

#df[(df['GroupSize']!=df['NoOfCabinsBooked'] ) & (-df['NoOfCabinsBooked'].isna())]

#df['Cabin'].value_counts()

#df[df['Embarked'].isna()]

df[(df['Fare'] < 800) & (df['GroupSize']==2) & (df['Cabin'].str.contains('B')) &  (df['PersonPerCabin']==2) & (df['NoOfCabinsBooked']==1)].groupby('Embarked').aggregate(np.mean)

# the fare 80 is closest fare is for those embarked in C
df['Embarked'].fillna('C', inplace=True)
df['Cabin'].fillna('Mis', inplace=True)

df['NoOfCabinsBooked'].fillna(11111, inplace=True)

df['PersonPerCabin'].fillna(11111, inplace=True)
df['Istoddler'] = (df['Age']<4)*1

df['IsChild'] = ((df['Age']>=4) & (df['Age']< 13))*1

df['IsYoung'] = ((df['Age']>=13) & (df['Age']< 18))*1

df['ToddlersPergroup'] = df.groupby('Ticket')['Istoddler'].transform('sum')

df['ChildrenPergroup'] = df.groupby('Ticket')['IsChild'].transform('sum')

df['Youngpergroup'] = df.groupby('Ticket')['IsYoung'].transform('sum')

df['FamilySize'] =  df['Parch']+df['SibSp']
x1 = pd.to_numeric(df[df['Ticket'].str.isnumeric()].Ticket).unique()

x2 = x1.reshape(-1, 1)
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
distortions = [] 

inertias = [] 

mapping1 = {} 

mapping2 = {} 

K = range(1,10) 

  

for k in K: 

    #Building and fitting the model 

    kmeanModel = KMeans(n_clusters=k).fit(x2) 

    kmeanModel.fit(X)     

      

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                      'euclidean'),axis=1)) / X.shape[0]) 

    inertias.append(kmeanModel.inertia_) 

  

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                 'euclidean'),axis=1)) / X.shape[0] 

    mapping2[k] = kmeanModel.inertia_ 
plt.plot(K, distortions) 

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show() 

# using 5 clusters based on this image
kmeanModel = KMeans(n_clusters=5, random_state=23).fit(x2) 

kmeanModel.fit(x2) 
clstr = {'T_Cluster':kmeanModel.labels_, 'Ticket':x2.reshape(705,)}
cluster1 = pd.DataFrame(clstr)

cluster1.T_Cluster = cluster1.T_Cluster.astype('category')
cluster1.groupby('T_Cluster').describe()
x1 =df[-df['Ticket'].str.isnumeric()].Ticket.unique()

x = df[-df['Ticket'].str.isnumeric()].Ticket.str.replace(r'\W+','').unique()
lev_similarity = -1*np.array([[distance.levenshtein(x1,x2) for x1 in x] for x2 in x]) #dendo gramss wants a negetive matrix The symmetric non-negative hollow observation matrix looks suspiciously like
plt.title("Dendograms")

dend = shc.dendrogram(shc.linkage(lev_similarity, method='ward'))

# creating 5 clusters based on dendogram
model = AgglomerativeClustering(distance_threshold=None, n_clusters=5)

model = model.fit(lev_similarity)
clstr = {'T_Cluster':model.labels_, 'Ticket':x1.reshape(224 ,)}

cluster2 = pd.DataFrame(clstr)

cluster2.T_Cluster = cluster2.T_Cluster + 4 #previously we created 5 clusters

cluster2.T_Cluster = cluster2.T_Cluster.astype('category')
T_Cluster = pd.concat([cluster1,cluster2], axis=0)

T_Cluster.Ticket = T_Cluster.Ticket.astype('str')
df = df.merge(T_Cluster, on= 'Ticket', how='left')
df.Pclass = df.Pclass.astype('category')

df.Sex = df.Sex.astype('category')

df.Cabin = df.Cabin.astype('category')

df.Embarked = df.Embarked.astype('category')

df.T_Cluster = df.T_Cluster.astype('category')
def get_model(activation,nn,batch_size, optimizer):

    model = Sequential()

    model.add(Dense(nn, input_shape = (i_shp,), activation=activation))

    model.add(BatchNormalization())

    model.add(Dense(nn, activation = activation))

    #model.add(Dense(nn, activation = activation))

    model.add(BatchNormalization())

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = optimizer, loss = 'mean_absolute_error', metrics=['mae','mse'])

    return(model)
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256], 

          'epochs': [50, 100, 200], 'nn': [28, 32, 64], 'optimizer' : ['adam','sgd']}
df.head(2)
#i_shp 

X_train = df[-df['Age'].isna()][['Age','Cabin','Embarked','Fare','Parch','Pclass','SibSp','Sex','GroupSize','NoOfCabinsBooked','PersonPerCabin','T_Cluster','FamilySize']]
model = KerasRegressor(build_fn=get_model, batch_size=16)

random_search = RandomizedSearchCV(model, param_distributions = params, cv = 3)
X = pd.get_dummies(X_train.drop('Age', axis=1), drop_first=True)

Y = X_train['Age']
min_max_scaler = MinMaxScaler(feature_range=(0, 1))

y = min_max_scaler.fit_transform(np.float32(Y[:, np.newaxis]))
i_shp =  len(X.columns)
#random_search.fit(X,y)

#y_pred = random_search.best_estimator_.predict(X)

#y_pred
#test = pd.DataFrame(min_max_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(1,-1)[0]-Y)

#test.hist()

#test.plot()

#abs(min_max_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(1,-1)[0]-Y).sum()
#test = pd.DataFrame(min_max_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(1,-1)[0])

#test.hist()
model = Sequential()

model.add(Dense (i_shp, input_shape=(i_shp,), activation = 'relu'))

model.add(Dense(32, activation='tanh'))

model.add(BatchNormalization())

model.add(Dense(16, activation ='relu'))

model.add(BatchNormalization())

model.add(Dense(2, activation='softmax'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mae','mse'])
hist = model.fit(X,y, batch_size=20, epochs=50)
%matplotlib inline

plt.plot(hist.history['mae'])

plt.show()
y_pred = model.predict(X)
test = pd.DataFrame(min_max_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(1,-1)[0]-Y)

test.hist()

test.plot()

abs(min_max_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(1,-1)[0]-Y).sum()
#predict and impute age



X_test = df[['Age','Cabin','Embarked','Fare','Parch','Pclass','SibSp','Sex','GroupSize','NoOfCabinsBooked','PersonPerCabin','T_Cluster','FamilySize']]
X = pd.get_dummies(X_test.drop('Age', axis=1), drop_first=True)
y = min_max_scaler.inverse_transform(model.predict(X))
df['predictedAge'] = y

df['predictedAge'].hist()
df['Age'].fillna(df['predictedAge'], inplace=True)
df['Cabintype'] = df['Cabin'].str[0]
df['Title']= df['Name'].str.split(r'[A-Z]*\.',expand=True)[0].str.split(',',expand=True)[1]
df['FamilyName'] = df['Name'].str.split(', ', expand = True)[0]

df['FNameCount'] = df.groupby('FamilyName')['FamilyName'].transform('count')
master_age_impute = df[(df['Title']==' Master') & (df['Age']<=18)].Age.mean()

master_age_impute
df.loc[(df['Title']==' Master') & (df['Age']>18),'Age'] = master_age_impute
df['Training'] = -df['Survived'].isna()*1
df.info()
family_survival = df[df['Training']==True].groupby('FamilyName')['Survived'].sum()/df[df['Training']==True].groupby('FamilyName')['Survived'].count()

group_survival = df[df['Training']==True].groupby('Ticket')['Survived'].sum()/df[df['Training']==True].groupby('Ticket')['Survived'].count()

df['Istoddler'] = (df['Age']<4)*1

df['IsChild'] = ((df['Age']>=4) & (df['Age']< 13))*1

df['IsYoung'] = ((df['Age']>=13) & (df['Age']< 18))*1

df['ToddlersPergroup'] = df.groupby('Ticket')['Istoddler'].transform('sum')

df['ChildrenPergroup'] = df.groupby('Ticket')['IsChild'].transform('sum')

df['Youngpergroup'] = df.groupby('Ticket')['IsYoung'].transform('sum')

# new additions

df['ToddlersPerFamily'] = df.groupby('FamilyName')['Istoddler'].transform('sum')

df['ChildrenPerFamily'] = df.groupby('FamilyName')['IsChild'].transform('sum')

df['YoungperFamily'] = df.groupby('FamilyName')['IsYoung'].transform('sum')
df['Sex'] = (df['Sex']=='female')*1
df['Womenpergroup'] = df.groupby('Ticket')['Sex'].transform('sum')

df['WomenperFamily'] = df.groupby('FamilyName')['Sex'].transform('sum')
df.drop('Survived', axis = 1).isna().sum() # we have all the columns populated.
X_train = df[['Age',

'Embarked',

'Fare',

'Parch',

'Pclass',

'Sex',

'SibSp','Survived',                                      

'GroupSize',

'PersonPerCabin',

'NoOfCabinsBooked',

'Istoddler',

'IsChild',

'IsYoung',

'ToddlersPergroup',

'ChildrenPergroup',

'Youngpergroup',

'FamilySize',

'Cabintype',

'FNameCount',

'ToddlersPerFamily',

'ChildrenPerFamily',

'YoungperFamily',

'Womenpergroup',

'WomenperFamily','Training']]
X = pd.get_dummies(X_train, drop_first=True)
X = X[-X['Survived'].isna()]

X = X.drop('Training', axis =1)

X = X.drop('Survived', axis =1)

Y = X_train[-df['Survived'].isna()]['Survived']

i_shp =  len(X.columns)
#def get_survival_model():

model = Sequential()

model.add(Dense (128, input_shape=(i_shp,), activation = 'relu'))

model.add(Dense(64, activation='tanh'))

model.add(BatchNormalization())

model.add(Dense(128, activation ='tanh'))

model.add(BatchNormalization())

model.add(Dense(64, activation ='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
hist = model.fit(X, Y, epochs=500)
%matplotlib inline

plt.plot(hist.history['accuracy'])

plt.show()
X = pd.get_dummies(X_train, drop_first=True)

X = X[X['Survived'].isna()]

X = X.drop('Training', axis =1)

X = X.drop('Survived', axis =1)
results = {'PassengerId':df[df['Survived'].isna()].PassengerId.values ,'Survived':((model.predict(X)>.5 )*1).T[0]}
result = pd.DataFrame(results)
result.to_csv('submission.csv', index=False)