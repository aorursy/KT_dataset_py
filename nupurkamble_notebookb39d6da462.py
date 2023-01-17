import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold 

import matplotlib as plt

from sklearn.model_selection import ShuffleSplit

train =  pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('Test and Train files loaded')

print("train shape: %s", str(train.shape))

print("test shape: %s", str(test.shape))
target = train.target.values

train = train.drop('target',axis=1)

train = train.drop('id',axis=1)

test = test.drop('id',axis=1)

print('train '+str(train.shape))

print('test '+str(test.shape))

print(train.columns.values == test.columns.values)

print(train.columns.values)

print(test.columns.values)

print(target.shape)


train_df = train

train_df['label'] = 'train'

score_df = test

score_df['label'] = 'score'

concat_df = pd.concat([train_df , score_df])

l=[]

for i in concat_df.values:

    if 'cat' in i:

        l.append(i)

for i in l:

    concat_df[i] = concat_df[i].astype(object)



# Create your dummies

features_df = pd.get_dummies(concat_df, columns=l)

# Split your data

train_df = features_df[features_df['label'] == 'train']

test_df = features_df[features_df['label'] == 'score']

features_df.shape

print(train_df.shape)

print(score_df.shape)

train_df.head(3)
train = train_df.drop('label',axis=1)

test = score_df.drop('label',axis=1)

print(train.shape)

print(test.shape)

test.head(4)
#python3.5

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold 

import matplotlib as plt

from sklearn.model_selection import ShuffleSplit

y=target

X=train.values

test = test.values

print(X.shape)

print(target.shape)

n_folds = 4                                    

skf = StratifiedKFold(y, n_folds, shuffle = False, random_state = 14)

print(test.shape)

print(target)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

#We pass X and y as numpyarray

def stacking(X,y,clfs,test,clf_names):

    '''Inputs: X:numpy array of features of training set excluding label or response or target

               y:numpy array of labels or response or target

               clfs: classifier list 

               test: test set without labels and only features set

       Output:List of [blend_train:metafeatures, blend_test:meta predictions of test, y:label, test:test'''

    

    clf_length = (len(clfs))

    x_rows = (X.shape[0])

    test_rows = int(test.shape[0])

    blend_train = np.zeros((x_rows,clf_length))#construct a 2d list containing rows=len of X and no of columns=len(classifier list) 

    blend_test = np.zeros((test_rows,clf_length),dtype = float)#construct a 2d list containing rows = len of test and no of columns = len(classifier list)

    

    a=clf_names

    for i, clf in enumerate(clfs):#iterate over classifiers from list

        blend_test_j = np.zeros((test.shape[0], len(skf)))#we take mean of all entries in each row blend_test_j and store it in blend_test

        print('classifier: %s'%(a[i]))

        for j, (train,cv) in enumerate(skf):

            xtrain = X[train]

            ytrain = y[train]

            xtest = X[cv]

            ytest = y[cv]

            clf.fit(xtrain,ytrain)

            accuracy = accuracy_score(ytest,clf.predict(xtest))

            logloss = log_loss(ytest,clf.predict_proba(xtest))

            blend_train[cv,i] = clf.predict_proba(xtest)[:,1]#collect meta features(predictions over cross validation indices)

            blend_test_j[:,j] =clf.predict_proba(test)[:,1]#predict the test set and take mean every time once for loop exits.  

            print('fold= %s and logloss is %s and accuracy is %s'%(str(j),str(logloss),str(accuracy))) 

        #print blend_test_j

        blend_test[:,i] = blend_test_j.mean(1)#Calculate mean of each row of predictions. 

    return [blend_train,blend_test,y,test]
from sklearn.metrics import log_loss

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

clfs = [RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='gini'),

            RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='entropy'),

            ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='gini'),

            ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='entropy'),

            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=300)]

stack = stacking(X,y,clfs,test,['rf1','rf2','et1','et2','gb'])
#So, far we have blend_train,y and blend_test

#Now, training is done on bend_train,y and predictions are made on blend_test

blend_train = stack[0]

print(blend_train)

print('Shape of blend_train is: %s'%(str(blend_train.shape)))



blend_test = stack[1]

print(blend_test)

print('Shape of blend_test is: %s'%(str(blend_test.shape)))



#Now, we train on blend_train as meta features on LogisticRegression.You can use SVM too

print('Blending Procedure')

clf = LogisticRegression()

clf.fit(blend_train, y)
y_submission = clf.predict_proba(blend_test)[:,1]#save first column or column of your choice in submission

print("Linear stretch of predictions to [0,1]")# we scale the probabilities between 0 to 1 

y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())#we can also use X-mean/Xdev to normalize

print("Saving Results.")
idc = pd.read_csv('test.csv')

idc = idc.iloc[:,0].values

idc
tmp = np.vstack([idc, y_submission]).T#transpose the horizontal to vertical

np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',

               header='id,target', comments='')