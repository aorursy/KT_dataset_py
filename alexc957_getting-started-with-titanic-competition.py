# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import seaborn as sns 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv',index_col='PassengerId')

test_df = pd.read_csv('../input/test.csv',index_col='PassengerId')

test_df['Survived'] = -1 

df = pd.concat([train_df,test_df],axis=0)

df.head() 
df.info()
df.describe(include='all')
columns = ['Pclass','SibSp','Parch','Sex','Embarked','Survived']

for column in columns:

    print(f'unique elements of {column}: {df[column].unique()}')

    print(f'value count of {column} :\n {df[column].value_counts()}')

    
df[df.Age.isnull()==True].head()
age_median_Pclass = df.groupby('Pclass').Age.transform('median')

df.Age.fillna(age_median_Pclass,inplace=True)
df[df.Cabin.isnull()==True].head() 
df.Cabin.unique() 
df.loc[df.Cabin=='T','Cabin']= np.NaN

# extract first character of cabin string to the deck 

def get_deck(cabin):

    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

df['Deck'] = df.Cabin.map(lambda x : get_deck(x))

df.Deck.value_counts()
df[df.Embarked.isnull()==True].head() 
# we can see the value counts for embarked 



df.Embarked.value_counts() 
# we can replace nan with S 

df.Embarked.fillna('S',inplace=True)
mean_fare = df.Fare.mean() 

df.Fare.fillna(mean_fare,inplace=True)
df.info() 
import matplotlib.pyplot as plt

%matplotlib inline

df.Age.plot(kind='hist',bins=5); 
df.Age.plot(kind='box'); 
train_df[df.Age==df.Age.max()].head() 
df[df.Age>df.Age.quantile(0.75)].head() 
df['Child_Adult'] = np.where(df['Age']>18,'Adult','Child')
df.Fare.plot(kind='hist',bins=5)
df.Fare.plot(kind='box'); 
df[df.Fare==df.Fare.max()].head()
df['Fare_binning'] = pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high'])

#df['Fare_binning'] = pd.qcut(df.Fare,3,labels=['low','medium','high'])
# now we see 

df.Fare_binning.value_counts().plot(kind='bar',rot=0); 
df.Name.head() 
df.Name.tail() 
titles = [name.split(',')[1].split('.')[0] for name in df.Name]

print(f"number of titles {len(titles)} ")

names = pd.DataFrame(titles,columns=['Title'])

print("unique values: ",names.Title.unique())
# lets create the new feature 

# but we have to create new values for title that are relate to others

def get_title(name):

    title_group = {

        'mr':'Mr',

        'mrs':'Mrs',

        'miss':'Miss',

        'master':'Master',

        'don':'Sir',

        'rev':'Sir',

        'dr':'Officer',  

        'mme':'Mrs',

        'ms':'Mrs',

        'major':'Officer',

        'lady':'Lady',

        'sir':'Sir',

        'mlle':'Miss',

        'col':'Officer',

        'capt':'Officer',

        'the countess':'Lady',

        'jonkheer':'Sir',

        'dona':'Lady'

    }

    first_name_with_title = name.split(',')[1]

    title = first_name_with_title.split('.')[0]

    title = title.strip().lower()

    return title_group[title]
df['Title'] = df.Name.map(lambda x : get_title(x))
pd.crosstab(df[df.Survived!=-1].Pclass,df[df.Survived!=-1].Survived)
pd.crosstab(df[df.Survived!=-1].Pclass,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0);    
pd.crosstab(df[df.Survived!=-1].Sex,df[df.Survived!=-1].Survived).plot(kind='bar');
pd.crosstab(df[df.Survived!=-1].SibSp,df[df.Survived!=-1].Survived).plot(kind='bar'); 
pd.crosstab(df[df.Survived!=-1].Parch,df[df.Survived!=-1].Survived).plot(kind='bar'); 
df['Travel_alone'] = np.where((df.Parch + df.SibSp)==0,1,0)
pd.crosstab(df[df.Survived!=-1].Travel_alone,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 
pd.crosstab(df[df.Survived!=-1].Embarked,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 
pd.crosstab(df[df.Survived!=-1].Child_Adult,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 
pd.crosstab(df[df.Survived!=-1].Fare_binning,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 
pd.crosstab(df[df.Survived!=-1].Deck,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 
has_cabin = np.where(df.Deck[df.Survived!=-1]=='Z',0,1)

pd.crosstab(has_cabin,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 
df['has_cabin_information'] = has_cabin = np.where(df.Deck=='Z',0,1)
pd.crosstab(df[df.Survived!=-1].Title,df[df.Survived!=-1].Survived).plot(kind='bar',rot=0); 
columns = ['Survived','Sex','Pclass','Travel_alone','Embarked','Fare_binning','has_cabin_information','Title']  

proccesed_df = df[columns].copy()    

proccesed_df.head() 

proccesed_df.shape
from sklearn.preprocessing import LabelEncoder

columns_to_transform = ['Sex','Embarked','Fare_binning','Title']

for column in columns_to_transform:

    encoder = LabelEncoder()

    proccesed_df[column] = encoder.fit_transform( proccesed_df[column])

proccesed_df.head() 

        

        

proccesed_df.info() 
model_train_df = proccesed_df[proccesed_df.Survived!=-1]

model_test_df = proccesed_df[proccesed_df.Survived==-1]

model_train_df.head() 
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split 



columns = [column for column in model_train_df.columns if column!='Survived']

X = model_train_df[columns].values 

y = model_train_df.Survived.values 

X_train,x_validation,y_train, y_validation = train_test_split(X,y,test_size=0.2,random_state=25)
logistic = LogisticRegression(solver='liblinear',multi_class='ovr')

logistic.fit(X_train,y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score



print("logistic validation score ",logistic.score(x_validation,y_validation))

print("logistic train score ",logistic.score(X_train,y_train))

print("logistic accuracy score ",accuracy_score(y_validation,logistic.predict(x_validation)))

print("logistic precision score ",precision_score(y_validation,logistic.predict(x_validation)))

print("logistic recall socre ",recall_score(y_validation,logistic.predict(x_validation)))
confusion_matrix(y_validation,logistic.predict(x_validation))
x_test = model_test_df[columns].values 

y_hat = logistic.predict(x_test) 

y_hat[:10]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train,y_train)
knn.score(X_train,y_train)
knn.score(x_validation,y_validation)
score_train  = []

score_validation = [] 

for n in range(2,9):

    knn1 = KNeighborsClassifier(n_neighbors=n)

    knn1.fit(X_train,y_train)

    score_train.append(knn1.score(X_train,y_train))

    score_validation.append(knn1.score(x_validation,y_validation))

    

import matplotlib.pyplot as plt 

neighbours = np.array(range(2,9))

plt.plot(neighbours,np.array(score_train),'bo',label='Training score')

plt.plot(neighbours,np.array(score_validation),'b',label='Validation score')

plt.xlabel('neighbours')

plt.ylabel('score')

plt.legend()

plt.show() 
# the prediction 

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

y_hat_knn = knn.predict(x_test)

y_hat_knn[:10]
print("knn validation score ",logistic.score(x_validation,y_validation))

print("knn train score ",logistic.score(X_train,y_train))

print("knn accuracy score ",accuracy_score(y_validation,knn.predict(x_validation)))

print("knn precision score ",precision_score(y_validation,knn.predict(x_validation)))

print("knn recall socre ",recall_score(y_validation,knn.predict(x_validation)))
from sklearn.svm import SVC

model = SVC() 

model.fit(X_train,y_train)

y_hat_SVC =  model.predict(x_test)

print(y_hat_SVC[:10])
print("SVC validation score ",model.score(x_validation,y_validation))

print("SVC train score ",model.score(X_train,y_train))

print("SVC accuracy score ",accuracy_score(y_validation,model.predict(x_validation)))

print("SVC precision score ",precision_score(y_validation,model.predict(x_validation)))

print("SVC recall socre ",recall_score(y_validation,model.predict(x_validation)))
from sklearn.neural_network import MLPClassifier

network = MLPClassifier(solver='sgd',learning_rate_init=0.15)

network.fit(X_train,y_train)
learning_rates = [0.0015,0.015,0.1,0.15,0.20,0.25,0.35,0.45,0.55,0.66,0.75,0.85]

training_score = []

validation_score = []

loss = []

for learnin_rate in learning_rates:

    net = MLPClassifier(solver='sgd',learning_rate_init=learnin_rate)

    net.fit(X_train,y_train)

    training_score.append(net.score(X_train,y_train))

    validation_score.append(net.score(x_validation,y_validation))

    loss.append(net.loss_)



plt.plot(np.array(learning_rates),np.array(training_score),'b',label='training score')

plt.plot(np.array(learning_rates),np.array(validation_score),'bo',label='validation score')

plt.xlabel('learning rate')

plt.ylabel('score')

plt.title('Neural network with many learning rate values')

plt.legend()

plt.plot() 
plt.plot(np.array(learning_rates),np.array(loss),'b',label='training loss')

plt.xlabel('learning rate')

plt.ylabel('loss')

plt.title('Neural network with many learning rate values')

plt.legend()

plt.plot() 
print("network validation score ",network.score(x_validation,y_validation))

print("network train score ",network.score(X_train,y_train))

print("network accuracy score ",accuracy_score(y_validation,network.predict(x_validation)))

print("network precision score ",precision_score(y_validation,network.predict(x_validation)))

print("network recall socre ",recall_score(y_validation,network.predict(x_validation)))
from keras import models

from keras import layers

keras_model = models.Sequential()

keras_model.add(layers.Dense(64,activation='relu',input_shape=(7,)))

keras_model.add(layers.Dense(32,activation='relu'))

keras_model.add(layers.Dense(16,activation='relu'))

keras_model.add(layers.Dense(1,activation='sigmoid'))

keras_model.compile(optimizer='rmsprop',

             loss='binary_crossentropy',

             metrics=['accuracy'])

history = keras_model.fit(X_train,

                   y_train,

                   epochs=28,

                   batch_size=100,

                   validation_data=[x_validation,y_validation])


print("keras_model accuracy score ",accuracy_score(y_validation,keras_model.predict_classes(x_validation)))

print("keras_model precision score ",precision_score(y_validation,keras_model.predict_classes(x_validation)))

print("keras_model recall socre ",recall_score(y_validation,network.predict(x_validation)))
from sklearn.tree import DecisionTreeClassifier

titanic_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

titanic_tree.fit(X_train,y_train)
score_train  = []

score_validation = [] 

for n in range(2,9):

    tree = DecisionTreeClassifier(criterion="entropy", max_depth = n)

    tree.fit(X_train,y_train)

    score_train.append(tree.score(X_train,y_train))

    score_validation.append(tree.score(x_validation,y_validation))

    

import matplotlib.pyplot as plt 

neighbours = np.array(range(2,9))

plt.plot(neighbours,np.array(score_train),'bo',label='Training score')

plt.plot(neighbours,np.array(score_validation),'b',label='Validation score')

plt.xlabel('max depth')

plt.ylabel('score')

plt.legend()

plt.show() 
print("tree validation score ",titanic_tree.score(x_validation,y_validation))

print("tree train score ",titanic_tree.score(X_train,y_train))

print("tree accuracy score ",accuracy_score(y_validation,titanic_tree.predict(x_validation)))

print("tree precision score ",precision_score(y_validation,titanic_tree.predict(x_validation)))

print("tree recall socre ",recall_score(y_validation,titanic_tree.predict(x_validation)))
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=50, max_depth=4,

                                  random_state=0,criterion='entropy') 



random_forest.fit(X_train,y_train)
print("random_forest validation score ",random_forest.score(x_validation,y_validation))

print("random_forest train score ",random_forest.score(X_train,y_train))

print("random_forest accuracy score ",accuracy_score(y_validation,random_forest.predict(x_validation)))

print("random_forest precision score ",precision_score(y_validation,random_forest.predict(x_validation)))

print("random_forest recall socre ",recall_score(y_validation,random_forest.predict(x_validation)))
score_train  = []

score_validation = [] 

for n in range(2,9):

    forest = RandomForestClassifier(n_estimators=50, max_depth=n,

                                  random_state=0) 

    forest.fit(X_train,y_train)

    score_train.append(forest.score(X_train,y_train))

    score_validation.append(forest.score(x_validation,y_validation))

    

import matplotlib.pyplot as plt 

neighbours = np.array(range(2,9))

plt.plot(neighbours,np.array(score_train),'bo',label='Training score')

plt.plot(neighbours,np.array(score_validation),'b',label='Validation score')

plt.xlabel('max depth')

plt.ylabel('score')

plt.legend()

plt.show() 