#Importing all essential libraries 

import numpy as np

import pandas as pd, os





import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score
#Setting the directory path

print(os.listdir("../input"))



#Assigning training and testing dataset to dataframes

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.head()
test.head()
train.describe()
cols = [c for c in train.columns if c not in ['id', 'target']]

for i in range(255):

    histogram = train.hist(cols[i],bins = 10, figsize=(2,2))

    

#Notice the distribution of "wheezy-copper-turtle-magic" column
#Further checking the number of unique values in "wheezy-..-magic" 

train['wheezy-copper-turtle-magic'].nunique()
#Here it can be seen that the dataset consists of 512 mini datasets 

len(train.index) / train['wheezy-copper-turtle-magic'].nunique()
#Create arrays with zeros

ans = np.zeros(len(train))

predictions = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
for i in range(512):



    #Extracting subset of dataset where wheezy-copper-turtle-magic equals i 

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    #Applying Principal Components Analysis to deduct dimensionality 

    dim_red = PCA(n_components=40).fit(train2[cols])

    train3 = dim_red.transform(train2[cols])

    test3 = dim_red.transform(test2[cols])

    

    

    #Using Stratified K-fold cross-validation 

    skf = StratifiedKFold(n_splits=25, random_state=42)

    for train_index, test_index in skf.split(train3, train2['target']):



        

        #Data modelling using Quadratic Discriminant Analysis (here I did use SVM, but QDA gave a better score)

        classification = QuadraticDiscriminantAnalysis()

        classification.fit(train3[train_index,:],train2.loc[train_index]['target'])

        ans[idx1[test_index]] = classification.predict_proba(train3[test_index,:])[:,1]

        predictions[idx2] += classification.predict_proba(test3)[:,1] / skf.n_splits 





        

    if i==512: print(i)



    #Printing the validation cross-validation area under the curve (The larger the better.)

    auc = roc_auc_score(train['target'],ans)

    print('CV score =',round(auc,5))    

    
#Creating a submission file and filling the target column with predicted probabilities for respective ids

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = predictions

sub.to_csv('submission.csv', index=False)
#The CV score is 0.96238 