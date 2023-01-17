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
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/eeg-brainwave-dataset-feeling-emotions/emotions.csv')
df.head()
df.isnull().sum().any()
y = df['label']
from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()

y = le.fit_transform(y)
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.decomposition import PCA
df.drop('label', axis = 1, inplace=True)
X = df
#Using Correlation to remove features which are highly correlated

correlated_features = set()

correlation_matrix = X.corr()

correlation_matrix



for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.9:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)



#Total number of correlated features

print(len(correlated_features))

#Printing features that are correlated

print(correlated_features)



#Droping columns that are correlated

X.drop(labels=correlated_features, axis=1, inplace=True)
X.shape

# We are left with 632 columns
X_array = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size = 0.3)
#PCA

pca_result = PCA(n_components=25)

pca_result.fit_transform(X)

for index, var in enumerate(pca_result.explained_variance_ratio_):

    print("Explained Variance ratio by Principal Component ", (index+1), " : ", var)
#Voting Classifier Ensemble Technique

clf1 = LogisticRegression(multi_class='multinomial', random_state=1)

clf2 = RandomForestClassifier(n_estimators=50, random_state=1)

clf3 = GaussianNB()



eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

eclf1 = eclf1.fit(X_train, y_train)

vc_eclf1_y_pred = eclf1.predict(X_test)

print("Accuracy VC_eclf1:",accuracy_score(y_test,vc_eclf1_y_pred))





eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft')

eclf2 = eclf2.fit(X, y)

vc_eclf2_y_pred = eclf2.predict(X_test)

print("Accuracy VC_eclf2:",accuracy_score(y_test,vc_eclf2_y_pred))



eclf3 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[2,1,1],flatten_transform=True)

eclf3 = eclf3.fit(X, y)

vc_eclf3_y_pred = eclf3.predict(X_test)

print("Accuracy VC_eclf3:",accuracy_score(y_test,vc_eclf3_y_pred))



eclf4 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[1,2,1],flatten_transform=True)

eclf4 = eclf4.fit(X, y)

vc_eclf4_y_pred = eclf4.predict(X_test)

print("Accuracy VC_eclf4:",accuracy_score(y_test,vc_eclf4_y_pred))



eclf5 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[1,1,2],flatten_transform=True)

eclf5 = eclf5.fit(X, y)

vc_eclf5_y_pred = eclf5.predict(X_test)

print("Accuracy VC_eclf5:",accuracy_score(y_test,vc_eclf5_y_pred))



#Random forest is giving us the maximum accuracy from pool of Logistic Regression, Random Forest and Naive Bayes.
## Pipelines Creation

    ## 1. Data Preprocessing by using Standard Scaler

    ## 2. Reduce Dimension using PCA

    ## 3. Apply  Classifier
#Logistic Regression Pipeline with PCA

pipeline_lr=Pipeline([('scalar1',StandardScaler()),

                     ('pca1',PCA(n_components=25)),

                     ('lr_classifier',LogisticRegression(random_state=0))])
#Linear Support Vector Classifier Pipeline with PCA

pipeline_svc_pca=Pipeline([('scalar2',StandardScaler()),

                     ('pca2',PCA(n_components=25)),

                     ('svm_cl', LinearSVC())])
#Random Forest Pipeline with PCA

pipeline_randomforest=Pipeline([('scalar3',StandardScaler()),

                     ('pca3',PCA(n_components=25)),

                     ('rf_classifier',RandomForestClassifier())])
#Random Forest Pipeline without PCA

pipeline_randomforest_pca=Pipeline([('scalar3',StandardScaler()),

                     ('rf_classifier',RandomForestClassifier())])
#Linear Support Vector Classifier without PCA

svm_c = Pipeline(steps=[('scaler',StandardScaler()),

                             ('svm_cl', LinearSVC())])
#XGBoost Pipeline with PCA

pl_xgb_pca = Pipeline(steps=

                  [('pca4', PCA(n_components=25)) ,('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
#XGBoost Pipeline without PCA

pl_xgb = Pipeline(steps=

                  [('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
pipelines = [pipeline_lr, pipeline_svc_pca, pipeline_randomforest, pipeline_randomforest_pca,  svm_c, pl_xgb_pca, pl_xgb ]



best_accuracy=0.0

best_classifier=0

best_pipeline=""



pipe_dict = {0: 'Logistic Regression with PCA', 1: 'Support Vector with PCA', 2: 'RandomForest with PCA', 3: 'RandomForest without PCA', 4: 'Support Vector without PCA', 5: 'XGBoost with PCA', 6: 'XGBoost without PCA'}



# Fit the pipelines

for pipe in pipelines:

    pipe.fit(X_train, y_train)



for i,model in enumerate(pipelines):

    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))
for i,model in enumerate(pipelines):

    if model.score(X_test,y_test)>best_accuracy:

        best_accuracy=model.score(X_test,y_test)

        best_pipeline=model

        best_classifier=i

print('Classifier with best accuracy: {}'.format(pipe_dict[best_classifier]))
algo = []

accuracy = []

for i,model in enumerate(pipelines):

    algo.append(pipe_dict[i])

    accuracy.append((model.score(X_test,y_test))* 100)

accuracy_df = pd.DataFrame(list(zip(algo,accuracy)), index  = [0,1,2,3,4,5,6], 

                                              columns =['Algorithm', 'Accuracy']) 

plt.figure(figsize=(16,6))

sns.barplot(x="Algorithm", y="Accuracy", data=accuracy_df)

plt.title('Mean Accuracy for different Algorithms', fontsize=16)

plt.ylabel('Accuracy', fontsize=10)

plt.xlabel('Algorithm', fontsize=10)

plt.xticks(rotation='horizontal')