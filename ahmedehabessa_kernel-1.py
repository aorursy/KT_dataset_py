# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)`



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")



test_identity = pd.read_csv("../input/input/test_identity.csv" , index_col = 'TransactionID')

# test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv" , index_col='TransactionID')

test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv",index_col='TransactionID')

train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv",index_col='TransactionID')

train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv",index_col='TransactionID')



train_data = train_transaction.merge(train_identity, how='left' ,left_index=True , right_index=True)

test_data  = test_transaction.merge(test_identity,how='left' , left_index=True, right_index=True)





test_data.to_csv('merged_test_data.csv')

train_data.to_csv('merged_train_data.csv')

del train_identity,train_transaction,test_identity, test_transaction , test_data , train_data
import pandas as pd 

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    return summary

    



# THIS FUNCTION WILL PLOT A CORRELATION HEATMAP WITH A SET THRESHOLD OF 0.9 CORRELATION.

def corrfunc(df , col):

    color = plt.get_cmap('RdYlGn') 

    color.set_bad('green') 

    correalation =df[col].corr()

    correalation[np.abs(correalation)<.9] = 0 # This will set all correlations less than 0.9 to 0

    plt.figure(figsize= (len(col),len(col)))

    sns.heatmap(correalation, yticklabels= True, annot = True, vmin=-1, vmax=1,cmap = color)



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: 

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df





data = pd.read_csv("../input/kernel-1/merged_train_data.csv" , index_col='TransactionID')

print(data.shape)



train = reduce_mem_usage(data)





data = pd.read_csv("../input/kernel-1/merged_test_data.csv" , index_col='TransactionID')



X_test = reduce_mem_usage(data)

with pd.option_context('display.max_columns', 433):

    print(train.describe(include='all'))

resumetable(train)[:25]
print(train.shape , X_test.shape)

train.head()
plt.figure(figsize=(9,6))

g =sns.countplot(x='isFraud' , data= train )

g.set_title("Fraud Transactions Distribution \n# 0: No Fraud | 1: Fraud #", fontsize=22)

g.set_xlabel("Is fraud?", fontsize=18)

g.set_ylabel('Count', fontsize=18)

plt.show()
print("Transaction Amounts Quantiles:")

print(train['TransactionAmt'].quantile([.01, .025, .1, .25, .5, .75, .9, .975, .99]))
plt.figure(figsize=(16,12))

plt.suptitle('Transaction Values Distribution', fontsize=22)

plt.subplot(221)

g = sns.distplot(train['TransactionAmt'])

g.set_title("Transaction Amount", fontsize=18)

g.set_xlabel("")

g.set_ylabel("Probability", fontsize=15)



plt.subplot(222)

g1 = sns.distplot(np.log(train['TransactionAmt']))

g1.set_title("Transaction Amount (Log) Distribuition", fontsize=18)

g1.set_xlabel("")

g1.set_ylabel("Probability", fontsize=15)



plt.show()


plt.figure(figsize=(14,10))

plt.title('ProductCD Distributions', fontsize=22)

plt.subplot(221)

g = sns.countplot(x='ProductCD', data=train)

g.set_title("ProductCD Distribution", fontsize=19)

g.set_xlabel("ProductCD Name", fontsize=17)

g.set_ylabel("Count", fontsize=17)



plt.subplot(222)

g1 = sns.countplot(x='ProductCD', hue='isFraud', data=train)

plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])



g1.set_title("Product CD by Target(isFraud)", fontsize=19)

g1.set_xlabel("ProductCD Name", fontsize=17)

g1.set_ylabel("Count", fontsize=17)



plt.show()
resumetable(train[['card1', 'card2', 'card3','card4', 'card5', 'card6']])
corrfunc(train,['card1','card2','card3','card5'])
plt.figure(figsize=(14,22))

plt.subplot(411)

g = sns.distplot(train[train['isFraud'] == 1]['card1'], label='Fraud')  

g = sns.distplot(train[train['isFraud'] == 0]['card1'], label='NoFraud')

g.legend()

g.set_title("Card 1 Values Distribution by Target", fontsize=16)

g.set_xlabel("Card 1 Values", fontsize=12)

g.set_ylabel("Probability", fontsize=18)



plt.subplot(412)

g1 = sns.distplot(train[train['isFraud'] == 1]['card2'].dropna(), label='Fraud')

g1 = sns.distplot(train[train['isFraud'] == 0]['card2'].dropna(), label='NoFraud')

g1.legend()

g1.set_title("Card 2 Values Distribution by Target", fontsize=18)

g1.set_xlabel("Card 2 Values", fontsize=12)

g1.set_ylabel("Probability", fontsize=18)



plt.subplot(413)

g3 = sns.distplot(train[train['isFraud']==1]['card3'].dropna(),label='Fraud')

g3 = sns.distplot(train[train['isFraud']==0]['card3'].dropna(),label='NotFraud')

g3.legend()

g3.set_title('Card3 values Distibution by Target' , fontsize = 18)

g3.set_xlabel('Card3 Values' ,fontsize=12)

g3.set_ylabel('Probability' ,fontsize=18)



plt.subplot(414)

g4=sns.distplot(train[train['isFraud']==1]['card5'].dropna() , label='Fraud' )

g4=sns.distplot(train[train['isFraud']==1]['card5'].dropna() , label='Fraud' )

g4.legend()

g4.set_title('Card5 values Distibution by Target' , fontsize = 18)

g4.set_xlabel('Card5 Values' ,fontsize=12)

g4.set_ylabel('Probability' ,fontsize=18)





plt.show()



plt.figure(figsize=(14,12))

plt.subplot(211)

g=sns.countplot(x='card4' , data = train)

g.set_title("Card4 Distribution", fontsize=19)

g.set_xlabel("Card4 Category Names", fontsize=17)

g.set_ylabel("Count", fontsize=17)



plt.subplot(212)

g2=sns.countplot(x='card6' , data = train)

g2.set_title("Card6 Distribution", fontsize=19)

g2.set_xlabel("Card6 Category Names", fontsize=17)

g2.set_ylabel("Count", fontsize=17)

plt.show()
for col in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:   

    plt.figure(figsize=(5,5))

    g =sns.countplot(x=col , data = train)

    g.set_title(col+ " Distribution", fontsize=19)

    g.set_xlabel(col+ " Category Names", fontsize=17)

    g.set_ylabel("Count", fontsize=17)

    plt.show()
data_null = train.isnull().sum()/len(train) * 100

data_null = data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)[:500]



missing_data = pd.DataFrame({'Missing Ratio': data_null})

print(missing_data.shape)

missing_data.head()

# find attributes with more than 90 percent missing vaules 

def get_useless_columns(data):

    

    too_many_null = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.80]

    print("More than 80% null columns: " + str(len(too_many_null)))

    

#     too_many_rpeated_values = [col for col in data.columns if data[col].value_counts(dropna=False 

#                 ,normalize =True).values[0] >0.90]

    

#     print("More than 90% repeated value columns: " + str(len(too_many_rpeated_values)))

    

    cols_to_drop = list(set(too_many_null))# + too_many_rpeated_values))

   # cols_to_drop.remove('isFraud')

    return cols_to_drop







cols_to_drop = get_useless_columns(train)

print(cols_to_drop)
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 

          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',

          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 

          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',

          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',

          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 

          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 

          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',

          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',

          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 

          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 

          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 

          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 

          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',

          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}



us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    X_test[c + '_bin'] = X_test[c].map(emails)

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    X_test[c + '_suffix'] = X_test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    X_test[c + '_suffix'] = X_test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')



Y =train['isFraud'].copy()



train =train.drop(['isFraud'],axis=1)





x_train_reduced = train.drop(cols_to_drop , axis=1)

# x_vaild_reduced = x_valid_full.drop(cols_to_drop, axis=1)



X_test_reduced = X_test.drop(cols_to_drop , axis = 1)



# # del x_train_full





print(x_train_reduced.shape , X_test_reduced.shape)    

x_train_reduced.head()

#  separete categorical values from numerical values



categorical_col = [cname for cname in x_train_reduced.columns if   #x_train_reduced[cname].nunique()<15 and 

                   x_train_reduced[cname].dtype=='object']



print(categorical_col)





numerical_col = [cname for cname in x_train_reduced.columns 

                 

                 if x_train_reduced[cname].dtype!='object']

# print(numerical_col) 

print(len(numerical_col)+len(categorical_col))





my_col=categorical_col+numerical_col



#  now we have total cols 

X_train = x_train_reduced[my_col].copy()

# X_vaild=x_vaild_reduced[my_col].copy()

X_test = X_test_reduced[my_col].copy()
print(X_train.shape , X_test.shape)


numircal_imputer = SimpleImputer(strategy='mean')



cat_encoder= LabelEncoder()

cat_imputer = SimpleImputer(strategy='most_frequent')





#impute numerical values  



x_train_imputed_numerical = pd.DataFrame(numircal_imputer.fit_transform(X_train[numerical_col]))

# x_vaild_imputed_numerical = pd.DataFrame(numircal_imputer.transform(X_vaild[numerical_col]))

x_test_imputed_numerical = pd.DataFrame(numircal_imputer.transform(X_test[numerical_col]))



x_train_imputed_numerical.columns= X_train[numerical_col].columns

# x_vaild_imputed_numerical.columns=X_vaild[numerical_col].columns

x_test_imputed_numerical.columns=X_test[numerical_col].columns





# impute cat values 



x_train_imputed_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[categorical_col]))

# x_vaild_imputed_cat = pd.DataFrame(cat_imputer.transform(X_vaild[categorical_col]))

x_test_imputed_cat = pd.DataFrame(cat_imputer.transform(X_test[categorical_col]))



x_train_imputed_cat.columns = X_train[categorical_col].columns

# x_vaild_imputed_cat.columns = X_vaild[categorical_col].columns

x_test_imputed_cat.columns = X_test[categorical_col].columns



# encode categical variables 



my_encoder = LabelEncoder()





for col in categorical_col:

    my_encoder.fit(list(x_train_imputed_cat[col].values) + list(x_test_imputed_cat[col].values))

    x_train_imputed_cat[col] = my_encoder.transform(x_train_imputed_cat[col])

#     x_vaild_imputed_cat[col] = my_encoder.transform(x_vaild_imputed_cat[col])

    x_test_imputed_cat[col]  = my_encoder.transform(x_test_imputed_cat[col])

del X_train ,X_test

X_train = pd.concat([x_train_imputed_numerical , x_train_imputed_cat] ,axis=1)



X_test  = pd.concat([x_test_imputed_numerical , x_test_imputed_cat] ,axis=1)



print(X_train.shape , X_test.shape)
del X_test_reduced , c , cat_encoder ,cat_imputer , categorical_col , col , cols_to_drop , corrfunc ,data 

del emails , g , g1 , g2 , g3 ,g4 , get_useless_columns ,gridspec , missing_data

del my_encoder , my_col ,numerical_col , numircal_imputer , train

del x_test_imputed_cat ,x_test_imputed_numerical ,x_train_imputed_cat ,x_train_reduced

del data_null
whos
X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
X1_train , X1_valid ,y_train , y_vaild = train_test_split(X_train , Y ,train_size = 0.8 ,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100,verbose=1)

model.fit(X1_train , y_train)



pred=model.predict(X1_valid)

print(accuracy_score(y_vaild, pred))

sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv",index_col='TransactionID')

sample_submission['isFraud']=model.predict(X_test)

sample_submission.to_csv('randomForest.csv')

from keras.regularizers import l2

import tensorflow as tf
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid , activity_regularizer=l2(0.1)))

model.add(tf.keras.layers.Dense(4, activation=tf.nn.sigmoid , activity_regularizer = l2(0.01)))

model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))



model.compile(optimizer="adam", loss="binary_crossentropy"  , metrics=['accuracy'])



model.fit(X1_train.values ,y_train.values, epochs=10 , batch_size=100)

print(model.evaluate(X1_valid, y_vaild))
sample_submission['isFraud']=model.predict(X_test)

sample_submission.to_csv('MLP no PCA with regularization L2-2.csv')

                         

whos
del LabelEncoder ,RandomForestClassifier ,tf ,model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis()



lda.fit(X1_train ,y_train)



pred=lda.predict(X1_valid)

print(accuracy_score(y_vaild, pred))



sample_submission['isFraud']=lda.predict(X_test)

sample_submission.to_csv('LDA.csv')

                         



X1_train = lda.transform(X1_train)

X1_test = lda.transform(X_test)
X1_test.shape
from sklearn import svm



model = svm.SVC(kernel='linear' , C=0.3 ,verbose=True)

model.fit(X1_train, y_train)

# preds = model.predict(X1_valid)

# print(model.score(y_valid, preds))

sample_submission['isFraud']=model.predict(X1_test)

sample_submission.to_csv('SVM.csv')
del lda , X1_test , X1_train ,model
from sklearn.decomposition import PCA

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()    # normalize data before PCA

# train_scaled = scaler.fit_transform(X_train)  





pca = PCA(n_components=260)

X_train = pca.fit_transform(X_train)

print(pca.n_components_ )



X_train = pd.DataFrame(X_train)

X_train=reduce_mem_usage(X_train)



X_test = pca.transform(X_test)

X_test = pd.DataFrame(X_test)

X_test = reduce_mem_usage(X_test)
X1_train , X1_valid ,y_train , y_vaild = train_test_split(X_train , Y ,train_size = 0.8 ,test_size=0.2)
%%time

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier()

model.fit(X1_train, y_train)

preds = model.predict(X1_valid)

print(accuracy_score(y_vaild, preds))



# print(model.score(y_vaild, preds))

# cross_val_score(model, X_train ,Y, cv=10)

print(accuracy_score(y_vaild, preds))
sample_submission['isFraud']=model.predict(X_test)

sample_submission.to_csv('decisionTree with PCA 260.csv')