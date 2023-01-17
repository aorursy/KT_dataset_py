# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import string

from sklearn.feature_extraction import FeatureHasher

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc



# Any results you write to the current directory are saved as output.
basepath="/kaggle/input/cat-in-the-dat"

print("List of files in the directory %s" %(os.listdir(basepath)))

traindata=os.path.join(basepath, 'train.csv')

testdata=os.path.join(basepath, 'test.csv')
train=pd.read_csv(traindata)

test=pd.read_csv(testdata)



train_X=train.drop(['id','target'],axis=1)

train_y=train['target']

test_X=test.drop('id',axis=1)



train_X.head()
print("Shape of train and test data is %s %s" %(train_X.shape, test_X.shape))
traintestencode=pd.concat([train_X,test_X])



print("Various ordinal types %s" %(traintestencode['ord_1'].unique()))

print("Various ordinal types %s" %(traintestencode['ord_2'].unique()))

print("Various ordinal types %s" %(traintestencode['ord_3'].unique()))

print("Various ordinal types %s" %(traintestencode['ord_4'].unique()))

print("Various ordinal types %s" %(traintestencode['ord_5'].unique()))



ord_1_map = {'Grandmaster': 5, 'Master': 4, 'Expert': 3,'Contributor': 2, 'Novice': 1}



ord_2map = {'Lava Hot':6,'Boiling Hot': 5, 'Hot': 4, 'Warm': 3, 

               'Freezing': 2, 'Cold': 1}



traintestencode['ord_1']=traintestencode['ord_1'].replace(ord_1_map)

traintestencode['ord_2']=traintestencode['ord_2'].replace(ord_2map)
print("No of unique categories in the ord_5 ->" ,traintestencode['ord_5'].nunique())

print("No of unique categories in the ord_4 ->" ,traintestencode['ord_4'].nunique())

print("No of unique categories in the ord_3 ->" ,traintestencode['ord_3'].nunique())
ord345=traintestencode[['ord_3','ord_4','ord_5']]



n_orig_features = ord345.shape[1]

hash_vector_size = 10

ct = ColumnTransformer([(f't_{i}', FeatureHasher(n_features=hash_vector_size, 

                        input_type='string'), i) for i in range(n_orig_features)])



ordencoded = ct.fit_transform(ord345)  # n_orig_features * hash_vector_size

hashedfeatures=pd.DataFrame(data=ordencoded.toarray())

hashedfeatures.columns=['hash_ord_'+str(f) for f in hashedfeatures.columns]



traintestencode.reset_index(inplace=True)



traintestencode=pd.concat([traintestencode,hashedfeatures],axis=1)

traintestencode.drop(['ord_3','ord_4','ord_5','index'],axis=1,inplace=True)
traintestencode.shape
nominalfeatures=[ feat for feat in traintestencode.columns if feat.split('_')[0]=='nom' ]
for nominal in nominalfeatures:

        print("No of unique values for the column %s is %s" %(nominal,traintestencode[nominal].nunique()))
def nominalohencoding(features):

    featurelist=[]

    hashnominallist=[]

    for feature in features:

            if traintestencode[feature].nunique()<=10:

                featurelist.append(feature)

            else:

                hashnominallist.append(feature)

    dummies = pd.get_dummies(traintestencode[featurelist], drop_first=True, sparse=True)

    return dummies,featurelist,hashnominallist
nominals,nominallist,hashnominallist=nominalohencoding(nominalfeatures)

traintestencode=pd.concat([traintestencode,nominals],axis=1)

traintestencode.drop(nominallist,axis=1,inplace=True)
traintestencode.columns
hashnominal=traintestencode[hashnominallist]

hash_vector_size=20



ct = ColumnTransformer([(f't_{i}', FeatureHasher(n_features=hash_vector_size, 

                        input_type='string'), i) for i in range(len(hashnominallist))])



hashnominalencoded = ct.fit_transform(hashnominal)  # n_orig_features * hash_vector_size



hashedfeatures=pd.DataFrame(data=hashnominalencoded)

hashedfeatures.columns=['hash_nom_'+str(f) for f in hashedfeatures.columns]



#traintestencode.reset_index(inplace=True)



traintestencode=pd.concat([traintestencode,hashedfeatures],axis=1)

traintestencode.drop(hashnominallist,axis=1,inplace=True)
traintestencode.head()
binlist=['bin_3','bin_4']

bindummies = pd.get_dummies(traintestencode[binlist], drop_first=True, sparse=True)

traintestencode=pd.concat([traintestencode,bindummies],axis=1)

traintestencode.drop(binlist,axis=1,inplace=True)
def encode(data, col, max_val):

    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)

    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)

    return data



traintestencode = encode(traintestencode, 'day', 365)

traintestencode = encode(traintestencode, 'month', 12)



traintestencode.drop(['day','month'],axis=1,inplace=True)
X_train=traintestencode.loc[:train_X.shape[0]-1,:]

X_test=traintestencode.loc[train_X.shape[0]:,:]
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)

pred_test_full =0

cv_score =[]

i=1



for train_index,test_index in kf.split(X_train,train_y):

    print('{} of KFold {}'.format(i,kf.n_splits))

    #print("dfdf",X_train[train_index])

    xtr,xvl= X_train.loc[train_index],X_train.loc[test_index]

    ytr,yvl = train_y.loc[train_index],train_y.loc[test_index]

    

    #model

    lr = LogisticRegression()

    lr.fit(xtr,ytr)

    score = roc_auc_score(yvl,lr.predict(xvl))

    print('ROC AUC score:',score)

    cv_score.append(score)    

    pred_test = lr.predict_proba(X_test)[:,1]

    pred_test_full +=pred_test

    i+=1
# Make submission

y_pred = pred_test_full/5

submission = pd.DataFrame({'id': test['id'].values.tolist(), 'target': y_pred})

submission.to_csv('submission.csv', index=False)