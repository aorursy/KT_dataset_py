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
import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import rcParams



# figure size in inches

rcParams['figure.figsize'] = 18,9
traindf=pd.read_csv("../input/predict-the-income-bi-hack/train.csv")

testdf=pd.read_csv("../input/predict-the-income-bi-hack/test.csv")
traindf.head()
testdf.head()
traindf.describe()
testdf.describe()
traindf.drop(['ID'],axis=1,inplace=True)

testdf.drop(['ID'],axis=1,inplace=True)
traindf.columns
testdf.columns
for i in traindf.columns:

    print(i)

    print(traindf[i].unique())
traindf.loc[traindf['Income']=='<=50K','Income']=0

traindf.loc[traindf['Income']=='>50K','Income']=1

traindf.loc[traindf['Nationality']=='?','Nationality']='unknown'

traindf.loc[traindf['Occupation']=='?','Occupation']='unknown'

traindf.loc[traindf['Work']=='?','Work']='unknown'

testdf.loc[testdf['Nationality']=='?','Nationality']='unknown'

testdf.loc[testdf['Occupation']=='?','Occupation']='unknown'

testdf.loc[testdf['Work']=='?','Work']='unknown'

traindf['Income']=pd.to_numeric(traindf['Income'])
#reducing the categories

traindf[['Hours_per_Week']]=traindf[['Hours_per_Week']]/10

traindf[['Hours_per_Week']]=traindf['Hours_per_Week'].astype(int)

testdf[['Hours_per_Week']]=testdf[['Hours_per_Week']]/10

testdf[['Hours_per_Week']]=testdf['Hours_per_Week'].astype(int)



traindf[['Age']]=traindf[['Age']]/10

traindf[['Age']]=traindf['Age'].astype(int)

testdf[['Age']]=testdf[['Age']]/10

testdf[['Age']]=testdf['Age'].astype(int)

traindf.head()
#checking distribution

for i in['Age','Work', 'Education', 'Education_Num', 'Marital_Status',

       'Occupation', 'Relationship', 'Race', 'Gender', 'Hours_per_Week',

       'Nationality']:

    print(i)

    ax=traindf[i].value_counts().plot(kind='bar')

    plt.show()
#plt fig !width not changing in kaggle kernel

rcParams['figure.figsize'] = 30,9


from statsmodels.graphics.mosaicplot import mosaic

plt.rcParams['font.size'] = 15.0

mosaic(traindf, ['Work', 'Income'])

plt.show()
for i in [ 'Education', 'Education_Num', 'Marital_Status',

       'Occupation', 'Relationship', 'Race', 'Gender', 'Hours_per_Week',

       'Nationality']:

    print(i)

    ax=mosaic(traindf, [i, 'Income'])

    plt.show()
#string to numeric categorical

map_dict={}

for i in ['Age','Work', 'Education', 'Education_Num', 'Marital_Status',

       'Occupation', 'Relationship', 'Race', 'Gender', 'Hours_per_Week',

       'Nationality']:

        name_dict={}

        ct=1

        for j in traindf[i].unique():

            name_dict[j]=ct 

            ct+=1



        map_dict[i]=name_dict
map_dict['Work']['Local-gov']
#copy of dataset for EDA

edf=traindf.copy()
#encoding

for i in ['Age','Work', 'Education', 'Education_Num', 'Marital_Status',

       'Occupation', 'Relationship', 'Race', 'Gender', 'Hours_per_Week',

       'Nationality']:

    for j in edf[i].unique():

        edf.loc[edf[i]==j,i]=map_dict[i][j]

    edf[i]=pd.to_numeric(edf[i])

edf.head()


sns.heatmap(edf.corr(),cmap="YlGnBu")
st= ['Age', 'Work', 'Education', 'Education_Num', 'Marital_Status',

       'Occupation', 'Relationship', 'Race', 'Gender', 'Hours_per_Week',

       'Nationality']

stt=st+['Income']

trdf= traindf[stt] #traindf.drop(['Gender','Occupation','Relationship','Marital_Status','Race'],axis=1)

tstdf=testdf[st]
edf.head()
stt
dst=[]

for i in stt:

    if i != 'Income' and i != 'Education_Num': #and i!='Hours_per_Week' and i!='Age':

        dst.append(i)
trdf.columns
tstdf.columns
dst


trdf=pd.get_dummies(trdf,columns = dst)
tstdf=pd.get_dummies(tstdf,columns = dst)
print(len(trdf.columns))

len(tstdf.columns)
dummydf=trdf.drop('Income',axis=1)
set( dummydf.columns ) - set( tstdf.columns )
missing_cols = set( dummydf.columns ) - set( tstdf.columns )





for c in missing_cols:

    tstdf[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

tstdf = tstdf[dummydf.columns]

tstdf.shape

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

import xgboost as xgb







#trdf['Income'] = LabelEncoder().fit_transform(trdf['Income'])

X= trdf.drop('Income',axis=1)

y=trdf[['Income']]

X_train, X_val, y_train, y_val = train_test_split( X , y, test_size=0.3, random_state=42)

#clf=XGBClassifier(n_estimators=1000, learning_rate=0.05)





import warnings

warnings.filterwarnings("ignore")



#clf=XGBClassifier()





from sklearn.model_selection import GridSearchCV



param_test = {

    

    'gamma': [0.5, 1, 1.5, 2, 5],

    'max_depth': [3, 4, 5]

  

}



clf = GridSearchCV(estimator = 

XGBClassifier(learning_rate =0.1,

              objective= 'binary:logistic',

              nthread=4,

              seed=27), 

              param_grid = param_test,

              scoring= 'accuracy',

              n_jobs=4,

              iid=False,

              verbose=10)



#clf=XGBClassifier()

clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, average_precision_score
y_pred= clf.predict(X_val)

print(y_pred)

accuracy_score(y_val,y_pred)


y_sol=clf.predict( tstdf)



print(y_sol)
with open('sol.csv','w') as fw:

    fw.write('ID,Income\n')

    ct=24001

    for i in y_sol:

        s=""

        if i==0:

            s="<=50K"

        else:

            s=">50K"

        fw.write(str(ct)+','+str(s)+'\n')

        ct+=1
#downlo

from IPython.display import FileLink

FileLink(r'sol.csv')