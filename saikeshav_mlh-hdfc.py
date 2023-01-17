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
#Creating data directory and copying files into it.

!mkdir data

!cp /kaggle/input/* data
!ls data
def read_data(fileName):

    #Reading file

    dataFrame = pd.read_csv(fileName, low_memory = False)

    return dataFrame
#Reading Train File

dataFrame = read_data("data/Train.csv")
#Counting instances of each class

dataFrame.groupby('Col2')["Col1"].count()
#Preprocessing training data



from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler



def train_preprocess(dataFrame):

    

    #Dropping Target Variable and ID.

    data = dataFrame.drop(columns = ["Col1", "Col2"])

    #Converting object columns to numeric

    cols = data.select_dtypes(include=["object"]).columns

    for colName in cols:

        #Replacing '-' with 0

        data[colName] = data[colName].replace('-', 0)

        data[colName] = data[colName].astype(str).astype(float)

    

    #Filling NULLs and NAs with Mean

    data.fillna(data.mean(), inplace = True)

    #Storing target variable in labels

    labels = dataFrame[["Col2"]]

    

    #Under Sampling

    #rs = RandomUnderSampler(sampling_strategy = 0.25)

    #data, labels = rs.fit_resample(data, labels)

    #print(data.shape)

    #print(labels.groupby('Col2').count())

    #Over sampling

    #sm = SMOTE()

    #data, labels = sm.fit_resample(data, labels)

    

    return data, labels
#Preprocessing test data

def test_preprocess(dataFrame):

    

    #Dropping ID Column

    data = dataFrame.drop(columns = ["Col1"])

    #Converting object columns to numeric

    cols = data.select_dtypes(include=["object"]).columns

    for colName in cols:

        #Replacing '-' with 0

        data[colName] = data[colName].replace('-', 0)

        data[colName] = data[colName].astype(str).astype(float)

    

    #Filling NAs and NULLs with mean

    data.fillna(data.mean(), inplace = True)

    

    return data
#Preprocessing training data

data, labels = train_preprocess(dataFrame)
print(data.shape)

print(labels.shape)
#Standardizing data

from sklearn import preprocessing

def scaling(data):

    std_Scaler = preprocessing.StandardScaler().fit(data)

    data_scale = std_Scaler.transform(data)

    

    return data_scale
#Calling scaling function

data_scale = scaling(data)
#Dimensionality Reduction

from sklearn.decomposition import PCA



pca = PCA(n_components = 500)

#pca.fit(data_scale)

data_n_comp = pca.fit_transform(data_scale)
#Calculating sum of co-variance

np.sum(pca.explained_variance_ratio_)
data_n_comp.shape
#Lightgbm model

import lightgbm as lgb

from sklearn.metrics import accuracy_score



def lightgbm(data, labels):

    

    train_data = lgb.Dataset(data, label = labels)

    params = {}

    params['learning_rate'] = 0.003

    params['boosting_type'] = 'gbdt'

    params['objective'] = 'binary'

    params['metric'] = 'binary_logloss'

    params['sub_feature'] = 0.5

    params['num_leaves'] = 10

    params['min_data'] = 50

    params['max_depth'] = 10

    clf = lgb.train(params, train_data, 100)

    y_prob = clf.predict(data_scale)

    y_pred = [0 if i < 0.165 else 1 for i in y_prob]

    print(accuracy_score(y_pred, labels))

    

    return clf
#XGBClassifier Model

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

def xgbClassifier(data, labels):

    

    clf = XGBClassifier()

    clf.fit(data, labels)

    print(accuracy_score(clf.predict(data), labels))

    

    return clf
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

def decisionTree(data, labels):

    

    clf = DecisionTreeClassifier()

    clf.fit(data, labels)

    accuracy_score(clf.predict(data), labels)

    

    return clf
def predict(clf, model, scale = False):



    #Read data

    testDf = read_data("data/Test.csv")

    #Preprocess

    test_data = test_preprocess(testDf)

    #Standardize data

    if scale:

        test_data = scaling(test_data)

    #Predict

    preds = model.predict(test_data)

    #Create submission file

    submissionDf = pd.DataFrame()

    submissionDf["Col1"] = testDf["Col1"]

    submissionDf["Col2"] = list(preds)

    # Save the submission file

    submissionDf.to_csv('submission.csv',index=False)



    from IPython.display import FileLink

    FileLink('submission'+clf+'.csv')
#Model Building lightgbm

lgbm_clf = lightgbm(data_scale, labels)



#Model Building XGBClassifier

xgb_clf = xgbClassifier(data, labels)



#Model Building Decision Tree Classifier

dct_clf = decisionTree(data, labels)



#Call predict function

predict(clf = "XGBClassifier", model = xgb_clf, scale = False)

#Building model on top 10 features iteratively

from sklearn.feature_selection import SelectFromModel

ten_features = np.sort(dct_clf.feature_importances_)[::-1][:10]

for feature in ten_features:

    feature_select = SelectFromModel(dct_clf, threshold=feature, prefit = True)

    new_features = feature_select.transform(data)

    new_clf = DecisionTreeClassifier()

    new_clf.fit(new_features, labels)

    print(feature)

    print(accuracy_score(new_clf.predict(new_features), labels))
#Delete data directory

!rm -rf data