import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        datafile =(os.path.join(dirname, filename))
df  = pd.read_csv(datafile)

df.head(5)
df['InvestmentnType'].unique()

#There are some values which means the same eg Pre-Series A and pre-Series A , Pre Series A,pre-series A

# eg. Seed / Angel Funding ,Seed / Angle Funding , Seed/Angel Funding, Seed/ Angel Funding

# Angel Funding and Angel

#Seed\\\\nFunding, Seed,Seed funding,Seed Funding Round,Seed Funding

#Private Equity Round,PrivateEquity,Private Equity,Private,Private\\\\nEquity,Equity,Equity Based Funding

#

# We can now label data



dict1 = {'Seed':'Seed','Seed\\\\nFunding':'Seed',

        'Seed funding':'Seed','Seed Funding Round':'Seed',

        'Seed Funding':'Seed','Seed / Angel Funding':'seed/angel',

             'Seed / Angle Funding':'seed/angel',

            'Seed/Angel Funding':'seed/angel',

            'Seed/ Angel Funding':'seed/angel',

            'Angel Funding':'angel','Angel':'angel',

            'Private Equity Round':'pr/eq','PrivateEquity':'pr/eq',

            'Private Equity':'pr/eq',

            'Private\\\\nEquity':'pr/eq','Equity':'pr/eq',

            'Equity Based Funding':'pr/eq',

            'Series A':'series','pre-series A':'series','Series C':'series','Series D':'series',

            'Series B':'series','Series J':'series','Series F':'series','Pre-Series A':'series',

            'pre-Series A':'series','Series E':'series','Pre Series A':'series',

            'Corporate Round':'others','Venture Round':'others','Single Venture':'others','Bridge Round':'others',

             'Mezzanine':'others','Inhouse Funding':'others','Crowd Funding':'others','Crowd funding':'others',

              'Structured Debt':'others','Term Loan':'others','Debt':'others','Maiden Round':'others'}

    
df['lables']=df['InvestmentnType'].map(dict1)

#We will create a dataframe with relevant coloumns only

# We will try to classify what kind of funding does a startup is more likely

#to get based on the industry vertical and sub vertical

X = df[['Industry Vertical']]

Y=df[['lables']]

X.head(5)



#X['categorical'] = pd.Categorical(X['Industry Vertical'])

#XDummies = pd.get_dummies(X['categorical'], prefix = 'category')

#XDummies.head(4)

miss_values = X.isnull().sum()/len(X)*100

miss_values.sort_values()
import re

X.head(4)

#['Industry Vertical'] =X['Industry Vertical'].apply(lambda x : re.sub(r'[^A-Za-z]','',x)) 

X['Industry Vertical'].value_counts()

X =X.apply(lambda col:pd.factorize(col,sort=True)[0])

X['Industry Vertical'] = X['Industry Vertical'].fillna((X['Industry Vertical'].mode()))

X.head(5)

YY = Y.apply(lambda col:pd.factorize(col,sort=True)[0])

YY['lables'] =YY['lables'].fillna((YY['lables'].mode()))

YY.head(4)

YY['lables'].value_counts()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from imblearn.over_sampling import SMOTE

smt = SMOTE()

#X_train, y_train = smt.fit_sample(X, YY)

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, YY, test_size=0.5) # 70% training and 30% test

clf=RandomForestClassifier(n_estimators=100,random_state=2,max_depth=2,criterion='entropy',n_jobs=-1)

#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)



import numpy

Pred_Y = numpy.ravel(y_pred,order='A')

Test_Y = numpy.ravel(y_test,order='A')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



from sklearn.metrics import f1_score

print(f1_score(Pred_Y, Test_Y,average='micro'))

print(f1_score(Pred_Y, Test_Y, average='macro'))




