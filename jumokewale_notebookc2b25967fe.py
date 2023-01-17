# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_file_path = '../input/dsn-competition2/Train.csv'

train_data = pd.read_csv(train_file_path)



test_file_path = '../input/dsn-competition2/Test.csv'

test_data = pd.read_csv(test_file_path)



test_id = test_data.Applicant_ID





from sklearn.preprocessing import LabelEncoder

def encode_dataset(dataset):

    label_encoder = LabelEncoder()

    #get the columns with object datatype

    for cols in dataset.columns:

        if dataset[cols].dtypes == 'object':

            dataset[cols] = label_encoder.fit_transform(dataset[cols])

        else:

            pass

    return dataset



encode_dataset(train_data)



encode_dataset(test_data)



train_data.head(20)


train_data.drop(['Applicant_ID'], axis=1, inplace=True)



test_data.drop(["Applicant_ID"], axis=1, inplace=True)





train_data.head()
train_data.describe()

missing_columns = [col for col in train_data.columns if train_data[col].isnull().any()]

print(missing_columns)

print('------------------------------')



for column in missing_columns:

    print('{}:   {}'.format(column, train_data[column].isnull().sum()))

    

    



    

train_data.fillna(-999, inplace=True)

test_data.fillna(-999, inplace=True)
train_data.head(10)
features = train_data.columns



y = train_data.pop('default_status')

X = train_data



y.head()


from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer

from sklearn.metrics import roc_auc_score



seed = 42



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=seed)



def model_auc(model):

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    test_auc =  roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print('Train Score: {}, Test_Score: {}'.format(train_auc, test_auc))

    return test_auc







standard_scaler = StandardScaler()

minmax = MinMaxScaler()





RAN = RandomForestClassifier(n_estimators=500, n_jobs=-1)

RAN2 = RandomForestClassifier(n_estimators=700, n_jobs=-1, random_state=seed, max_depth=2)

CAT = CatBoostClassifier(verbose=False, random_seed= 42,

 learning_rate= 0.05,

 leaf_estimation_iterations= 10,

 l2_leaf_reg= 3.162277660168379e-20,

 iterations= 500,

 depth=4)

CAT2 = CatBoostClassifier(verbose=False, random_state=seed)

LIG = LGBMClassifier(num_leaves= 80,

 #num_iterations= 500,

 max_bin= 100,

 learning_rate= 0.01,

 depth= 6, n_jobs=-1)

XGB = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01, n_job=-1)









models = [RAN, CAT, CAT2, LIG]



#for model in models:

 #   pipe = Pipeline([('Scaler', minmax),

  #                ('model', model)])

   # pipe.fit(X_train, y_train)

    #print('Model: {}, Score: {}'.format(model, model_auc(pipe)))
from sklearn.model_selection import StratifiedKFold



seeds = 42



skf = StratifiedKFold(n_splits=25, shuffle=True, random_state=seed)



classifiers = [('cat2', CAT2), ('lgm', LIG), ('ran', RAN)]



VC = VotingClassifier(estimators=classifiers, voting='soft')



preds = []

scores = []

i = 1

for train, test in skf.split(X, y):

    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]

    model = Pipeline([('scaler', standard_scaler),

                     ('model', VC)])

    model.fit(X_train, y_train)

    print('Number of splits trained {}'.format(i))

    score = model_auc(model)

    scores.append(score)

    preds.append(model.predict_proba(test_data)[:, 1])

    i += 1



#print(np.mean(scores))

#print(preds)









submission = pd.DataFrame({'Applicant_ID':test_id, 'default_status':model.predict_proba(test_data)[:, 1]})

submission.to_csv('./last2.csv', index=False)