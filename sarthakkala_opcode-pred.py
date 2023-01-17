# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
#import matplotlib
#mpl.use("Qt4Agg")
import matplotlib.pyplot as plt # data visualisation
import seaborn as sns # data visualisation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print("===================================loading dataset============================")
df = pd.read_csv("../input/ga_map_compact_all_58K.CSV",header=1)
print("\t\t\t dataset shape: %s" %(str(df.shape)))
print("===============================================================================")
df.head()
dataHasNaN = False
if df.count().min()==df.shape[0]:
    print("We don't have to take care of NaN.")
else:
    dataHasNaN = True
    print("We have to take care of NaN.")
df.describe(include = 'all')
df.drop(['End time'], axis = 1, inplace = True)  #useless value
df.drop(['Begin time'], axis = 1, inplace = True)  #useless value

df.drop(['Matching Id'], axis = 1, inplace = True)   #all values are 0
df.drop(['Nb missing units'], axis = 1, inplace = True)  #all values are 0

df.drop(['Provider cause'], axis = 1, inplace = True)  #mostly empty
df.drop(['TCAP Error'], axis = 1, inplace = True)   #mostly empty
df.drop(['SCCP Cause'], axis = 1, inplace = True)   #mostly empty

df.drop(['CmlPhs'], axis = 1, inplace = True)

df.drop(['ITC'], axis = 1, inplace = True)  #out of two categories, all but 2 values are one category
df.drop(['SMS TP-MTI'], axis = 1, inplace = True)  #mostly empty

#no idea on how to fill missing values since addresses are not linear but unique IDs
df.drop(["Cd Sccp Add"], axis = 1, inplace = True)
df.drop(["Cg Sccp Add"], axis = 1, inplace = True)
####################################################################################

df.drop(['MSISDN'], axis = 1, inplace = True)  #no idea on how to fill NaN Values
df.drop(['B MSISDN'], axis = 1, inplace = True)  #mostly empty

df.drop(['OTID'], axis = 1, inplace = True)  #~48k outta 58k unique values
df.drop(['DTID'], axis = 1, inplace = True)  #~41k outta 58k unique values

df.drop(['TMSI'], axis = 1, inplace = True) #mostly empty
df.drop(['MCC'], axis = 1, inplace = True)  #mostly empty
df.drop(['MNC'], axis = 1, inplace = True)  #mostly empty
df.drop(['TP-OA'], axis = 1, inplace = True) #mostly empty

df.drop(['IMSI'], axis = 1, inplace = True)   #no idea on how to fill empty values since most are empty
df.drop(['LAC'], axis = 1, inplace = True)  #~54k outta 58k values are same
df.drop(['Cell ident'], axis = 1, inplace = True)  #54k outta 58k values are same
df.drop(['IMEI'], axis = 1, inplace = True)  #no idea on how to fill missing values since most are empty
df.drop(['VLAN Id'], axis = 1, inplace = True)   #all values are NaN

df.describe(include = 'all') #'User error' 'Released by'  'OpCode' 
df['User error'].fillna('No error', inplace = True)
df['Released by'].value_counts()        #called party has only 1 prediction, 
                                        #hence it is not a good dataset to predict 'called party' value in 'Released by'
df['OpCode'].value_counts()
print("=======removed undefined values=========")
df.dropna(inplace = True) #remove nan values
df.drop(df[df["Cd-SSN"]=="241"].index, axis = 0, inplace = True)  #remove undefined values
df.drop(df[df["Cd-SSN"]=="253"].index, axis = 0, inplace = True)
df.drop(df[df["Cg-SSN"]=="253"].index, axis = 0, inplace = True)
df.drop(df[df["Cd-SSN"]=="222"].index, axis = 0, inplace = True)
df.drop(df[df["OpCode"]=="Send parameters"].index, axis = 0, inplace = True)
df.drop(df[df["OpCode"]=="Reset"].index, axis = 0, inplace = True)
df.drop(df[df["OpCode"]=="Unstructured SS notify"].index, axis = 0, inplace = True)
df.drop(df[df["OpCode"]=="Interrogate SS"].index, axis = 0, inplace = True)
df.drop(df[df["OpCode"]=="Unstructured SS request"].index, axis = 0, inplace = True)
df.drop(df[df["OpCode"]=="Authentication failure report"].index, axis = 0, inplace = True)
df.head()
X = df.drop(['OpCode'], axis = 1)  #independent variable
y = pd.DataFrame(df['OpCode'])   #dependent variable
X.describe(include = 'all')
X_DPC = pd.get_dummies(X['DPC'], prefix = 'DPC')
X_DPC.drop(['DPC_$1-60-2'], axis = 1, inplace = True)
X = X.join(X_DPC)
X.drop(['DPC'], axis = 1, inplace = True)
X_OPC = pd.get_dummies(X['OPC'], prefix = 'OPC')
X_OPC.drop(['OPC_$5-12-3'], axis = 1, inplace = True)
X = X.join(X_OPC)
X.drop(['OPC'], axis = 1, inplace = True)
X_Link = pd.get_dummies(X['Link'], prefix = 'Link')
X_Link.drop(['Link_SCTP213'], axis = 1, inplace = True)
X = X.join(X_Link)
X.drop(['Link'], axis = 1, inplace = True)
X_SIO = pd.get_dummies(X['SIO'], prefix = 'SIO')
X_SIO.drop(['SIO_$83'], axis = 1, inplace = True)
X = X.join(X_SIO)
X.drop(['SIO'], axis = 1, inplace = True)
X_Protocol = pd.get_dummies(X['Protocol'], prefix = 'Protocol')
X_Protocol.drop(['Protocol_MAP V3 layer application'], axis = 1, inplace = True)
X = X.join(X_Protocol)
X.drop(['Protocol'], axis = 1, inplace = True)
X_DR_Status = pd.get_dummies(X['DR Status'], prefix = 'DR Status')
X_DR_Status.drop(['DR Status_OK (no problem)'], axis = 1, inplace = True)
X = X.join(X_DR_Status)
X.drop(['DR Status'], axis = 1, inplace = True)
X_Way = pd.get_dummies(X['Way'], prefix = 'Way')
X_Way.drop(['Way_Incoming'], axis = 1, inplace = True)
X = X.join(X_Way)
X.drop(['Way'], axis = 1, inplace = True)
X_Successful = pd.get_dummies(X['Successful'], prefix = 'Successful')
X_Successful.drop(['Successful_No'], axis = 1, inplace = True)
X = X.join(X_Successful)
X.drop(['Successful'], axis = 1, inplace = True)
X_User_error = pd.get_dummies(X['User error'], prefix = 'User_error')
X_User_error.drop(['User_error_No error'], axis = 1, inplace = True)
X = X.join(X_User_error)
X.drop(['User error'], axis = 1, inplace = True)
X_Released_by = pd.get_dummies(X['Released by'], prefix = 'Released_by')
X_Released_by.drop(['Released_by_Calling party'],axis = 1, inplace = True)
X = X.join(X_Released_by)
X.drop(['Released by'], axis = 1, inplace = True)
X_DR_Type = pd.get_dummies(X['DR Type'], prefix = 'DR_Type')
X_DR_Type.drop(['DR_Type_Complete TDR'],axis = 1, inplace = True)
X = X.join(X_DR_Type)
X.drop(['DR Type'], axis = 1, inplace = True)
X_Cg_SSN = pd.get_dummies(X['Cg-SSN'], prefix = 'Cg_SSN')
X_Cg_SSN.drop(['Cg_SSN_Mobile switching center'], axis = 1, inplace = True)
X = X.join(X_Cg_SSN)
X.drop(['Cg-SSN'], axis = 1, inplace = True)
X_Cd_SSN = pd.get_dummies(X['Cd-SSN'], prefix = 'Cd_SSN')
X_Cd_SSN.drop(['Cd_SSN_Home location register'],axis = 1, inplace = True)
X = X.join(X_Cd_SSN)
X.drop(['Cd-SSN'], axis = 1, inplace = True)
'''
X_OTID = pd.get_dummies(X['OTID'], prefix = 'OTID')
X_OTID.drop(['OTID_$FFFFFFFF'],axis = 1, inplace = True)
X = X.join(X_OTID)
X.drop(['OTID'], axis = 1, inplace = True)
'''
'''
X_DTID = pd.get_dummies(X['DTID'], prefix = 'DTID')
X_DTID.drop(['DTID_$FFFFFFFF'],axis = 1, inplace = True)
X = X.join(X_DTID)
X.drop(['DTID'], axis = 1, inplace = True)
'''
y_opcode = pd.get_dummies(y['OpCode'], prefix = 'OpCode')
y = y.join(y_opcode)
y.drop(['OpCode'], axis = 1, inplace = True)
y = np.array(y).argmax( axis = 1)
y = y.reshape(-1,1)
y = pd.DataFrame(y, columns=['OpCode'])
X.reset_index(drop = True, inplace = True)
X.head()
df = y.join(X)
correlation = abs(df.corr())
correlation.head()
df.head()
correlation.drop(correlation[correlation['OpCode'] < 0.4].index, axis = 1, inplace = True)
#correlation.drop(correlation[correlation["OpCode"] < 0.4].index, axis = 0, inplace = True)
correlation.head()
X = df.drop( ['OpCode'], axis = 1 ) 

'''
X = df.loc[:,['DPC_$4B-20-2',
             'OPC_$2B-A0-1',
             'Link_SCTP224',
             'Protocol_MAP V2 layer application',
             'Way_Outgoing',
             "Released_by_Tekelec's Timer",
             'Cd_SSN_Equipment identifier center']]
'''
y = pd.DataFrame(df.loc[:,'OpCode'])
#Splitting data into test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train.head()
df.shape

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.loc[:,['MS','Units size','Nb units']] = sc.fit_transform(X_train.loc[:,['MS','Units size','Nb units']])
X_test.loc[:,['MS','Units size','Nb units']] = sc.transform(X_test.loc[:,['MS','Units size','Nb units']])

X_train.shape, y_train.shape
X_train = np.array(X_train)
y_train = np.array(y_train)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 'auto')
parameters = [{'kernel' : ['linear']},
              {'kernel' : ['rbf']}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train.ravel())
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
best_parameters
#Training our model with the classifiers

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train.ravel())

X_test = np.array(X_test)
#Predicting values with our classifier
y_pred = classifier.predict(X_test)
y_pred
y_test = np.array(y_test).ravel()
y_test
#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
diag_sum = 0
tot_sum = 0
row,col = cm.shape
for i in np.arange(row):
    for j in np.arange(col):
        tot_sum = tot_sum + cm[i,j]
        if i==j:
            diag_sum = cm[i,j] + diag_sum
acc = (diag_sum*100)/tot_sum
acc
# Applying k-fold cross-validation
from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(classifier, X_train, y_train, cv=10).mean()

print('validation score = %s' %(cross_val))
