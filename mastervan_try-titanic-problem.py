import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

import re as re



import warnings

warnings.filterwarnings("ignore")
# load data

train_df = pd.read_csv( '../input/train.csv')

test_df = pd.read_csv( '../input/test.csv')
train_df.shape
# Store our passenger ID 

PassengerId = test_df['PassengerId']

PassengerId.shape
#creat a total data

train_df['source']= 'train'

test_df['source'] = 'test'

full_df=pd.concat([train_df, test_df],ignore_index=True)
col_names = full_df.columns.values

col_names
counts=[]

for name in col_names:

    count = full_df[name].isnull().sum()

    counts.append(count)

counts
import warnings

warnings.filterwarnings("ignore")



# nulls in 'Age', 'Fare', 'Cabin', 'Embarked'



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



full_df['Title'] = full_df['Name'].apply(get_title)



full_df['Title'] = full_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



full_df['Title'] = full_df['Title'].replace('Mlle', 'Miss')

full_df['Title'] = full_df['Title'].replace('Ms', 'Miss')

full_df['Title'] = full_df['Title'].replace('Mme', 'Mrs')



# remove all nulls in 'Age'

age_avg = full_df['Age'].mean()

age_std = full_df['Age'].std()

age_null_count = full_df['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

full_df['Age'][np.isnan(full_df['Age'])] = age_null_random_list

full_df['Age'] = full_df['Age'].astype(int)



# remove all nulls in 'Fare'

full_df['Fare'] = full_df['Fare'].fillna(full_df['Fare'].median())

full_df['Fare'] = full_df['Fare'].astype(int)



# remove all nulls in 'Cabin'

full_df.loc[ full_df.Cabin.isnull()==True,'Cabin'] = 0

full_df.loc[ full_df.Cabin.isnull()==False,'Cabin'] = 1

# full_df = full_df.drop('Cabin',axis=1)

    

# remove all nulls in 'Embarked'

full_df['Embarked'] = full_df['Embarked'].fillna('S')



# Family

full_df['Family'] = full_df['Parch'] + full_df['SibSp']

full_df['Family'].loc[full_df['Family'] > 0] = 1

full_df['Family'].loc[full_df['Family'] == 0] = 0



# drop Parch & SibSp

full_df = full_df.drop(['SibSp','Parch'], axis=1)



# Sex

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

full_df['Person'] = full_df[['Age','Sex']].apply(get_person,axis=1)

full_df = full_df.drop(['Sex'], axis=1)



# Age cutting

full_df['Age_bins'] = pd.qcut(full_df['Age'],4)



# Fare cutting

full_df['Fare_bins'] = pd.qcut(full_df['Fare'],4)



# remove useless features

full_df = full_df.drop(['Name','PassengerId','Ticket'], axis=1) #remove 'Name', 'PassengerId', 'Ticket'

full_df.head()
# One-Hot encoding

var_to_encode = ['Embarked','Pclass','Title','Person','Age_bins','Fare_bins'] # need encoding feature list

full_df = pd.get_dummies(full_df, columns=var_to_encode)

full_df.head()
# split the full data

train = full_df.loc[full_df['source']=='train']

test = full_df.loc[full_df['source']=='test']



train_y = train['Survived'].astype(int)

train_X = train.drop(['source','Survived'],axis=1)

test_X = test.drop(['source','Survived'],axis=1)
# Standardize features

from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()

train_X = X_scaler.fit_transform(train_X)

test_X = X_scaler.transform(test_X)
# classifier comparison

import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

acc_dict = {}



for train_index, test_index in sss.split(train_X, train_y):

#     X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]

#     y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

    X_train, X_test = train_X[train_index], train_X[test_index]

    y_train, y_test = train_y[train_index], train_y[test_index]



    for clf in classifiers:

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



acc_df = log.sort_values(by='Accuracy',ascending = False)

acc_df
plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=acc_df, color="b")
# Create Numpy arrays of train, test and label

y_train = train_y # Creates an array of label

x_train = train_X # Creates an array of the train data

x_test = test_X # Creats an array of the test data
# K-Folds cross-validation

from sklearn.model_selection import KFold



ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 15 # set folds for out-of-fold prediction

# kf = KFold(ntrain, n_splits= NFOLDS, random_state=SEED)

kf = KFold(n_splits=NFOLDS, random_state=SEED)



def kfolds_pre(clf, x_train, y_train, x_test):

    kfolds_train = np.zeros((ntrain,))

    kfolds_test = np.zeros((ntest,))

    kfolds_test_all = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        

        clf.fit(x_tr, y_tr)



        kfolds_train[test_index] = clf.predict(x_te)

        kfolds_test_all[i, :] = clf.predict(x_test)



    kfolds_test[:] = kfolds_test_all.mean(axis=0)

    return kfolds_train.reshape(-1, 1), kfolds_test.reshape(-1, 1)
kn = KNeighborsClassifier()

svc = SVC()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()

ab = AdaBoostClassifier()

gb = GradientBoostingClassifier()

gnb = GaussianNB()

ld = LinearDiscriminantAnalysis()

qd = QuadraticDiscriminantAnalysis()

lr = LogisticRegression()
# Create train and test predictions. These base results will be used as new features

gb_kfolds_train, gb_kfolds_test = kfolds_pre(gb, x_train, y_train, x_test) # GradientBoostingClassifier

rf_kfolds_train, rf_kfolds_test = kfolds_pre(rf,x_train, y_train, x_test) # RandomForestClassifier

svc_kfolds_train, svc_kfolds_test = kfolds_pre(svc, x_train, y_train, x_test) # SVC

lr_kfolds_train, lr_kfolds_test = kfolds_pre(lr, x_train, y_train, x_test) # LogisticRegression

ab_kfolds_train, ab_kfolds_test = kfolds_pre(ab, x_train, y_train, x_test) # AdaBoostClassifier

kn_kfolds_train, kn_kfolds_test = kfolds_pre(kn, x_train, y_train, x_test) # KNeighborsClassifier

ld_kfolds_train, ld_kfolds_test = kfolds_pre(kn, x_train, y_train, x_test) # LinearDiscriminantAnalysis

gnb_kfolds_train, gnb_kfolds_test = kfolds_pre(kn, x_train, y_train, x_test) # GaussianNB
# Concatenate the output of the classifiers on the first level



x_train_1 = np.concatenate(( svc_kfolds_train, ld_kfolds_train, lr_kfolds_train, gnb_kfolds_train, gb_kfolds_train), axis=1)

x_test_1 = np.concatenate(( svc_kfolds_test, ld_kfolds_test, lr_kfolds_test, gnb_kfolds_test, gb_kfolds_test), axis=1)



# x_train_1 = np.concatenate(( svc_kfolds_train, gb_kfolds_train, ab_kfolds_train, lr_kfolds_train, rf_kfolds_train), axis=1)

# x_test_1 = np.concatenate(( svc_kfolds_test, gb_kfolds_test, ab_kfolds_test, lr_kfolds_test, rf_kfolds_test), axis=1)
warnings.filterwarnings("ignore")

# make the final prediction

import xgboost as xgb

from xgboost import XGBClassifier



gbm = xgb.XGBClassifier(

 learning_rate = 0.02,

 n_estimators= 2000, #2000

 max_depth= 4,#4

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9, #0.9               

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1)
clf = gbm



clf.fit(x_train_1, train_y)

predictions = clf.predict(x_test_1)
test = clf.predict(x_train_1)

acc = accuracy_score(train_y, test)

acc
result = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

result.to_csv("mysubmission3.csv", index=False)
print('OK')