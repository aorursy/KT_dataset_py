# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/network-anamoly-detection/"))



# Any results you write to the current directory are saved as output.
# import relevant modules

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import sklearn

import imblearn

# Ignore warnings

import warnings

warnings.filterwarnings('ignore')
# Dataset field names

datacols = ["duration","protocol_type","service","flag","src_bytes",

    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",

    "logged_in","num_compromised","root_shell","su_attempted","num_root",

    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",

    "is_host_login","is_guest_login","count","srv_count","serror_rate",

    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",

    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",

    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",

    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",

    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]
# Load NSL_KDD train dataset

df_train = pd.read_table("../input/network-anamoly-detection/Train.txt", sep=",", names=datacols) # change path to where the dataset is located.

df_train = df_train.iloc[:,:-1] # removes an unwanted extra field



# Load NSL_KDD test dataset

df_test = pd.read_table("../input/network-anamoly-detection/Test.txt", sep=",", names=datacols)

df_test = df_test.iloc[:,:-1]
# View train data

df_train.head(3)
# View test data

df_test.head(3)
# train set dimension

print('Train set dimension: {} rows, {} columns'.format(df_train.shape[0], df_train.shape[1]))



# train set dimension

print('Test set dimension: {} rows, {} columns'.format(df_test.shape[0], df_test.shape[1]))
mapping = {'ipsweep': 'Probe','satan': 'Probe','nmap': 'Probe','portsweep': 'Probe','saint': 'Probe','mscan': 'Probe',

           'teardrop': 'DoS','pod': 'DoS','land': 'DoS','back': 'DoS','neptune': 'DoS','smurf': 'DoS','mailbomb': 'DoS',

           'udpstorm': 'DoS','apache2': 'DoS','processtable': 'DoS',

           'perl': 'U2R','loadmodule': 'U2R','rootkit': 'U2R','buffer_overflow': 'U2R','xterm': 'U2R','ps': 'U2R',

           'sqlattack': 'U2R','httptunnel': 'U2R',

           'ftp_write': 'R2L','phf': 'R2L','guess_passwd': 'R2L','warezmaster': 'R2L','warezclient': 'R2L','imap': 'R2L',

           'spy': 'R2L','multihop': 'R2L','named': 'R2L','snmpguess': 'R2L','worm': 'R2L','snmpgetattack': 'R2L',

           'xsnoop': 'R2L','xlock': 'R2L','sendmail': 'R2L',

           'normal': 'Normal'

        }
# Apply attack class mappings to the dataset

df_train['attack_class'] = df_train['attack'].apply(lambda v: mapping[v])

df_test['attack_class'] = df_test['attack'].apply(lambda v: mapping[v])
# Drop attack field from both train and test data

df_train.drop(['attack'], axis=1, inplace=True)

df_test.drop(['attack'], axis=1, inplace=True)
# View top 3 train data 

df_train.head(3)
# Descriptive statistics

df_train.describe()
print(df_train['num_outbound_cmds'].value_counts())

print(df_test['num_outbound_cmds'].value_counts())
# 'num_outbound_cmds' field has all 0 values. Hence, it will be removed from both train and test dataset since it is a redundant field.

df_train.drop(['num_outbound_cmds'], axis=1, inplace=True)

df_test.drop(['num_outbound_cmds'], axis=1, inplace=True)
# Attack Class Distribution

attack_class_freq_train = df_train[['attack_class']].apply(lambda x: x.value_counts())

attack_class_freq_test = df_test[['attack_class']].apply(lambda x: x.value_counts())

attack_class_freq_train['frequency_percent_train'] = round((100 * attack_class_freq_train / attack_class_freq_train.sum()),2)

attack_class_freq_test['frequency_percent_test'] = round((100 * attack_class_freq_test / attack_class_freq_test.sum()),2)



attack_class_dist = pd.concat([attack_class_freq_train,attack_class_freq_test], axis=1) 

attack_class_dist
# Attack class bar plot

plot = attack_class_dist[['frequency_percent_train', 'frequency_percent_test']].plot(kind="bar");

plot.set_title("Attack Class Distribution", fontsize=20);

plot.grid(color='lightgray', alpha=0.5);
df_train.head(5)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



# extract numerical attributes and scale it to have zero mean and unit variance  

cols = df_train.select_dtypes(include=['float64','int64']).columns



sc_train = scaler.fit_transform(df_train.select_dtypes(include=['float64','int64']))

sc_test = scaler.fit_transform(df_test.select_dtypes(include=['float64','int64']))



# turn the result back to a dataframe

sc_traindf = pd.DataFrame(sc_train, columns = cols)

sc_testdf = pd.DataFrame(sc_test, columns = cols)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()



# extract categorical attributes from both training and test sets 

cattrain = df_train.select_dtypes(include=['object']).copy()

cattest = df_test.select_dtypes(include=['object']).copy()



# encode the categorical attributes

traincat = cattrain.apply(encoder.fit_transform)

testcat = cattest.apply(encoder.fit_transform)



# separate target column from encoded data 

enctrain = traincat.drop(['attack_class'], axis=1)

enctest = testcat.drop(['attack_class'], axis=1)



cat_Ytrain = traincat[['attack_class']].copy()

cat_Ytest = testcat[['attack_class']].copy()
from imblearn.over_sampling import RandomOverSampler 

from collections import Counter



# define columns and extract encoded train set for sampling 

sc_traindf = df_train.select_dtypes(include=['float64','int64'])

refclasscol = pd.concat([sc_traindf, enctrain], axis=1).columns

refclass = np.concatenate((sc_train, enctrain.values), axis=1)

X = refclass



# reshape target column to 1D array shape  

c, r = cat_Ytest.values.shape

y_test = cat_Ytest.values.reshape(c,)



c, r = cat_Ytrain.values.shape

y = cat_Ytrain.values.reshape(c,)



# apply the random over-sampling

ros = RandomOverSampler(random_state=42)

X_res, y_res = ros.fit_sample(X, y)

print('Original dataset shape {}'.format(Counter(y)))

print('Resampled dataset shape {}'.format(Counter(y_res)))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier();



# fit random forest classifier on the training set

rfc.fit(X_res, y_res);

# extract important features

score = np.round(rfc.feature_importances_,3)

importances = pd.DataFrame({'feature':refclasscol,'importance':score})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

# plot importances

plt.rcParams['figure.figsize'] = (11, 4)

importances.plot.bar()
from sklearn.feature_selection import RFE

import itertools

rfc = RandomForestClassifier()



# create the RFE model and select 10 attributes

rfe = RFE(rfc, n_features_to_select=10)

rfe = rfe.fit(X_res, y_res)



# summarize the selection of the attributes

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), refclasscol)]

selected_features = [v for i, v in feature_map if i==True]
selected_features
# define columns to new dataframe

newcol = list(refclasscol)

newcol.append('attack_class')



# add a dimension to target

new_y_res = y_res[:, np.newaxis]



# create a dataframe from sampled data

res_arr = np.concatenate((X_res, new_y_res), axis=1)

res_df = pd.DataFrame(res_arr, columns = newcol) 



# create test dataframe

reftest = pd.concat([sc_testdf, testcat], axis=1)

reftest['attack_class'] = reftest['attack_class'].astype(np.float64)

reftest['protocol_type'] = reftest['protocol_type'].astype(np.float64)

reftest['flag'] = reftest['flag'].astype(np.float64)

reftest['service'] = reftest['service'].astype(np.float64)



res_df.shape

reftest.shape
from collections import defaultdict

classdict = defaultdict(list)



# create two-target classes (normal class and an attack class)  

attacklist = [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)]

normalclass = [('Normal', 1.0)]



def create_classdict():

    '''This function subdivides train and test dataset into two-class attack labels''' 

    for j, k in normalclass: 

        for i, v in attacklist: 

            restrain_set = res_df.loc[(res_df['attack_class'] == k) | (res_df['attack_class'] == v)]

            classdict[j +'_' + i].append(restrain_set)

            # test labels

            reftest_set = reftest.loc[(reftest['attack_class'] == k) | (reftest['attack_class'] == v)]

            classdict[j +'_' + i].append(reftest_set)

        

create_classdict()
for k, v in classdict.items():

    print(k)
pretrain = classdict['Normal_DoS'][0]

pretest = classdict['Normal_DoS'][1]

grpclass = 'Normal_DoS'
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()



Xresdf = pretrain 

newtest = pretest



Xresdfnew = Xresdf[selected_features]

Xresdfnum = Xresdfnew.drop(['service'], axis=1)

Xresdfcat = Xresdfnew[['service']].copy()



Xtest_features = newtest[selected_features]

Xtestdfnum = Xtest_features.drop(['service'], axis=1)

Xtestcat = Xtest_features[['service']].copy()





# Fit train data

enc.fit(Xresdfcat)



# Transform train data

X_train_1hotenc = enc.transform(Xresdfcat).toarray()

       

# Transform test data

X_test_1hotenc = enc.transform(Xtestcat).toarray()



X_train = np.concatenate((Xresdfnum.values, X_train_1hotenc), axis=1)

X_test = np.concatenate((Xtestdfnum.values, X_test_1hotenc), axis=1) 



y_train = Xresdf[['attack_class']].copy()

c, r = y_train.values.shape

Y_train = y_train.values.reshape(c,)



y_test = newtest[['attack_class']].copy()

c, r = y_test.values.shape

Y_test = y_test.values.reshape(c,)
# import pickle 

  

# # Save the trained model as a pickle string. 

# saved_model = pickle.dumps(knn) 

  

# # Load the pickled model 

# knn_from_pickle = pickle.loads(saved_model) 

  

# # Use the loaded pickled model to make predictions 

# knn_from_pickle.predict(X_test)
from sklearn.svm import SVC 

from sklearn.naive_bayes import BernoulliNB 

from sklearn import tree

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier

from sklearn import metrics

import pickle


# Train KNeighborsClassifier Model

KNN_Classifier = KNeighborsClassifier(n_jobs=-1)

KNN_Classifier.fit(X_train, Y_train)

with open('KNN_Classifier.pkl', 'wb') as pkl:

        pickle.dump(KNN_Classifier, pkl) 

        
# Train LogisticRegression Model

LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)

LGR_Classifier.fit(X_train, Y_train)

with open('LGR_Classifier.pkl', 'wb') as pkl:

        pickle.dump(LGR_Classifier, pkl) 
# Train Gaussian Naive Baye Model

BNB_Classifier = BernoulliNB()

BNB_Classifier.fit(X_train, Y_train)

with open('BNB_Classifier.pkl', 'wb') as pkl:

        pickle.dump(BNB_Classifier, pkl) 
# Train Decision Tree Model

DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)

DTC_Classifier.fit(X_train, Y_train)

with open('DTC_Classifier.pkl', 'wb') as pkl:

        pickle.dump(DTC_Classifier, pkl) 
            

# Train RandomForestClassifier Model

RF_Classifier = RandomForestClassifier(criterion='entropy', n_jobs=-1, random_state=0)

RF_Classifier.fit(X_train, Y_train)

DTC_Classifier.fit(X_train, Y_train)

with open('RF_Classifier.pkl', 'wb') as pkl:

        pickle.dump(RF_Classifier, pkl) 
# # Train SVM Model

# SVC_Classifier = SVC(random_state=0)

# SVC_Classifier.fit(X_train, Y_train)

# with open('SVC_Classifier.pkl', 'wb') as pkl:

#         pickle.dump(SVC_Classifier, pkl) 
## Train Ensemble Model (This method combines all the individual models above except RandomForest)

combined_model = [('Naive Baye Classifier', BNB_Classifier), 

                  ('Decision Tree Classifier', DTC_Classifier), 

                  ('KNeighborsClassifier', KNN_Classifier), 

                  ('LogisticRegression', LGR_Classifier)

                 ]

VotingClassifier =  VotingClassifier(estimators = combined_model,voting = 'soft', n_jobs=-1)

VotingClassifier.fit(X_train, Y_train)

with open('VotingClassifier.pkl', 'wb') as pkl:

        pickle.dump(VotingClassifier, pkl) 

models = []

# models.append(('SVM Classifier', SVC_Classifier))

models.append(('Naive Baye Classifier', BNB_Classifier))

models.append(('LogisticRegression', LGR_Classifier))

models.append(('KNeighborsClassifier', KNN_Classifier))

models.append(('Decision Tree Classifier', DTC_Classifier))

models.append(('RandomForest Classifier', RF_Classifier))

# models.append(('VotingClassifier', VotingClassifier))
for i, v in models:

#     v = RF_Classifier

    scores = cross_val_score(v, X_train, Y_train, cv=10)

    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))

    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))

    classification = metrics.classification_report(Y_train, v.predict(X_train))

    print()

    print('============================== {} {} Model Evaluation =============================='.format(grpclass, i))

    print()

    print ("Cross Validation Mean Score:" "\n", scores.mean())

    print()

    print ("Model Accuracy:" "\n", accuracy)

    print()

    print("Confusion matrix:" "\n", confusion_matrix)

    print()

    print("Classification report:" "\n", classification) 

    print()