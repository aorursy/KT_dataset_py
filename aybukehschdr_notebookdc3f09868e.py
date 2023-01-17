from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""")


import warnings

warnings.filterwarnings('ignore')

# ---



%matplotlib inline

import pandas as pd

pd.options.display.max_columns = 100

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import numpy as np

pd.options.display.max_rows = 100



import csv





train_df = pd.read_csv('../input/train.csv' ,header=0)

test_df    = pd.read_csv('../input/test.csv' ,header=0)
survived_sex = train_df[train_df['Survived']==1]['Sex'].value_counts()

dead_sex = train_df[train_df['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(15,8))
figure = plt.figure(figsize=(15,8))

plt.hist([train_df[train_df['Survived']==1]['Fare'],train_df[train_df['Survived']==0]['Fare']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:

     train_df.Embarked[train_df.Embarked.isnull()]=train_df.Embarked.dropna().mode().values

   

    
#enumerate içerden dönen değerleri yazar.

#np.unique ise arraydeki elemanlar için unique bir değer döndürür. astype(int) ile de bu değerler sayısallaştırılır

Ports = list(enumerate(np.unique(train_df['Embarked'])))

Ports_dict = { name : i for i, name in Ports } 

#Burada dataframemiz içindeji gelen her elemean sırası ile unique sayısal değerler ile değiştirilmiş dizi elemanları ile yer değiştirir

train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


median_age = train_df['Age'].dropna().median()

if len(train_df.Age[ train_df.Age.isnull() ]) > 0:

    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:

     test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)
median_age = test_df['Age'].dropna().median()

if len(test_df.Age[ test_df.Age.isnull() ]) > 0:

    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

    
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):

          median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()

                                              

      

    for f in range(0,3):                                              

        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]
ids = test_df['PassengerId'].values



test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
print ("train dataset.keys(): {}".format(train_df.keys()))

print ("test dataset.keys(): {}".format(test_df.keys()))
X=train_df.ix[:,1:8]

y=train_df.Survived

print("my_train shape:{}".format(train_df.shape))

print("X_train shape: {}".format(X.shape))

print("y_train shape: {}".format(y.shape))
X_test=test_df.ix[:,0:7]



print("test_df shape:{}".format(test_df.shape))

print("Xrest shape: {}".format(X_test.shape))
X, Xtest, y, ytest = cross_validation.train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(random_state=0)

gbrt.fit(X,y)

prediction_y=gbrt.predict(X_test)

output = prediction_y

print("Default değerler ile  train : {:.3f}".format(gbrt.score(X,y)))

for i in [0.001,0.01,0.1,1,3]:

    from sklearn.ensemble import GradientBoostingClassifier

    gbrt = GradientBoostingClassifier(random_state=0 , learning_rate=i)

    gbrt.fit(X,y)

    prediction_y=gbrt.predict(X_test)

    print("Öğrenme katsayısı "+ str(i) + " train: {:.3f}".format(gbrt.score(X,y)))

 

    print("---------------------------------")

  
for i in [1,5,10,40,80]:

    gbrt = GradientBoostingClassifier(random_state=0, max_depth=i)

    gbrt.fit(X, y)

    prediction_y=gbrt.predict(X_test)

    print("Ağaç derinliği "+ str(i) + " train: {:.3f}".format(gbrt.score(X,y)))

 

    print("---------------------------------")

   
gbrt = GradientBoostingClassifier(random_state=0, max_depth=5,learning_rate=0.1)

gbrt.fit(X, y)

prediction_y=gbrt.predict(X_test)

print("Ağaç derinliği "+ str(5) +  "öğrenme katsayısı "+ str(0.1) + ": {:.3f}".format(gbrt.score(X,y)))

test_df['Survived'] = pd.DataFrame(prediction_y)

X_test=test_df.ix[:,0:7]

y_test=test_df.Survived
gbrt = GradientBoostingClassifier(random_state=0, max_depth=5,learning_rate=0.1)

gbrt.fit(X, y)



print("Ağaç derinliği "+ str(5) +  "öğrenme katsayısı "+ str(0.1) + ": {:.3f}".format(gbrt.score(X,y)))

print("Ağaç derinliği "+ str(5) +  "öğrenme katsayısı "+ str(0.1) + ": {:.3f}".format(gbrt.score(X_test,y_test)))
from sklearn.ensemble import RandomForestClassifier
print("agac sayisinin sirasi ile 5,20,100 olmasi ile alinan sonuclar:")

for i in [5,20,100]:

    forest = RandomForestClassifier(n_estimators=i, random_state=2)

    forest.fit(X, y)

    prediction_y=forest.predict(X_test)

    print("Ağaç sayısının" + str(i)+ "olması ile sonuc train"+ ": {:.3f}".format(forest.score(X, y)))

   

    print("---------------------------------")

   
for i in [1,5,10,40,80]:

    forest = RandomForestClassifier(random_state=0, max_depth=i)

    forest.fit(X, y)

    prediction_y=forest.predict(X_test)

    print("Ağaç derinliği "+ str(i) + ": {:.3f}".format(forest.score(X,y)))

    

    print("--------------------------------")

   
forest = RandomForestClassifier(random_state=0, max_depth=5,n_estimators=5)

gbrt.fit(X, y)

prediction_y=gbrt.predict(X_test)

print("Ağaç derinliği "+ str(5) +  "ağaç sayısı "+ str(5) + ": {:.3f}".format(gbrt.score(X,y)))

test_df['Survived'] = pd.DataFrame(prediction_y)

X_test=test_df.ix[:,0:7]

y_test=test_df.Survived
forest = RandomForestClassifier(random_state=0, max_depth=5,n_estimators=5)

gbrt.fit(X, y)



print("Ağaç derinliği "+ str(40) +  "ağaç sayısı "+ str(20) + ": {:.3f}".format(gbrt.score(X,y)))

print("Ağaç derinliği "+ str(40) +  "ağaç sayısı "+ str(20) + ": {:.3f}".format(gbrt.score(X_test,y_test)))
test_df   = pd.read_csv('../input/test.csv' ,header=0)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": prediction_y

    })

submission.to_csv('titanic.csv', index=False)