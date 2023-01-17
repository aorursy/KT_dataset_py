# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
y_train = train['species']#[x[:x.find('_')] for x in train['species']]

X_train = train.drop(['species','id'],axis=1)

Z_test = test.drop('id',axis=1)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaler.fit(X_train.values)

X_train = scaler.transform(X_train.values)

Z_test = scaler.transform(Z_test.values)
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

enc.fit(y_train)

y_train = enc.transform(y_train)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,stratify=y_train)
from sklearn.model_selection import GridSearchCV

import xgboost
#clf = xgboost.XGBClassifier()

#params = {'max_depth':[2,3,4,5],'learning_rate':[.01,.1,.5],'n_estimators':[10,100,400]}



#grid = xgboost.XGBClassifier()#GridSearchCV(clf,params)



#from sklearn.linear_model import LinearRegression

#grid = LinearRegression()

#grid.fit(X_train,y_train)

#print(grid.score(X_test,y_test))



from sklearn.neighbors import KNeighborsClassifier

grid = KNeighborsClassifier()

grid.fit(X_train,y_train)

print(grid.score(X_test,y_test))



#from sklearn.tree import DecisionTreeClassifier

#grid = DecisionTreeClassifier(max_depth=None)

#grid.fit(X_train,y_train)

#print(grid.score(X_test.values,y_test))



#from sklearn.naive_bayes import GaussianNB

#grid = GaussianNB()

#grid.fit(X_train,y_train)

#print(grid.score(X_test,y_test))
#print(grid.best_estimator_)

guesses = grid.predict(Z_test)
submission = pd.DataFrame({

        "id": test['id']

    })

print(submission.head())

pred = enc.inverse_transform(guesses)

#print(pred)



for species in sample.columns:

    mask = []

    for x in pred:

        if x==species:

            mask.append(1)

        else:

            mask.append(0)

    submission[species]=mask



submission["id"]=test['id']

#for i in range(len(enc.classes_)):

#    genus = enc.classes_[i]

#    print(genus)

#    species_names = []

#    for species in sample.columns:

#        if species[:len(genus)]==genus:

#            species_names.append(species)

#    mask = []

#    for x in pred:

#        if x==genus:

#            mask.append(1)

#        else:

#            mask.append(0)

#    mask = np.array(mask)

#    for species in species_names:

#       

#        #print(pred[genus])

#        #print(pred[genus]/len(species_names))

#        submission[species]= mask/len(species_names)
submission.to_csv('Leaf.csv', index=False)
print(submission.head())