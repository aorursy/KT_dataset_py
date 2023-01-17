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
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.insert(loc=0,column='rowNum',value=np.arange(len(train_data)))
train_data=train_data.set_index('rowNum')
train_data.dropna(subset=['Embarked'],inplace=True)
%matplotlib inline
import matplotlib.pyplot as plt
train_data.hist(bins=50, figsize=(20,15))
plt.show()
corr_matrix = train_data.corr()
corr_matrix["Survived"].sort_values(ascending=False)
corr_matrix['Pclass'].sort_values(ascending=False)
import seaborn as sns
sns.catplot('Pclass', hue='Survived', data=train_data, kind='count')
sns.catplot('Pclass', hue='Age', data=train_data, kind='count')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
Y = train_data['Survived'].copy()
X_PassengerId=train_data['PassengerId'].copy()
X = train_data.drop(columns=['Survived'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#impute default values
def impute_def_values(df):
    "Extract the 'Cabin' and 'Family_size' features."
#     df['Cabin'] = df['Cabin'].map(lambda x: 'Known' if type(x) is str else 'Unknown')
    df['SibSp'].fillna(0,inplace= True)
    df['Parch'].fillna(0,inplace=True)
    df['Age'].fillna(df['Age'].median(),inplace= True)
    df['Embarked'].fillna('S',inplace=True)
    df['Fare'].fillna(df['Fare'].median(),inplace=True)
    return df
X=impute_def_values(X)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# num_attribs=['Pclass','Age','Fare']
# cat_attribs=['Sex','Cabin','Embarked']
drop_attribs=['Name','Ticket']
def extract_fetatures(X):
    "Extract the 'Cabin' and 'Family_size' features."
    X['Cabin'] = X['Cabin'].map(lambda x: 'Known' if type(x) is str else 'Unknown')
    
    X['Family']=X['SibSp'] + X['Parch']
    X['Family']=X['Family'].map(lambda x: 'No' if x==0
                                       else 'Few' if x in [1,2,3]
                                       else 'Many')
    return X.drop(columns=['SibSp','Parch'])
    
column_trans = make_column_transformer(
                    ('drop', drop_attribs),
                    (make_pipeline( # Extract the features and then encode them
                        FunctionTransformer(extract_fetatures),
                        OneHotEncoder()), ['Cabin','SibSp', 'Parch']),
                    (OneHotEncoder(), ['Pclass', 'Sex', 'Embarked']),
                    remainder=StandardScaler()) # Scale the continuous variables
preprocessor = make_pipeline(column_trans,
                             IterativeImputer(random_state=0)) # Impute missing values
extracted_X = preprocessor.fit_transform(X)
transfs = preprocessor[0].transformers_
extracted_featuers=transfs[1][1][1].get_feature_names(['Cabin','Family'])
encoded_features=transfs[2][1].get_feature_names(['Pclass', 'Sex', 'Embarked'])
remainder_features=['Age','Fare','X_PassengerId']
all_features=list(extracted_featuers)+list(encoded_features)+remainder_features
extracted_X_train=pd.DataFrame(extracted_X,X.index,all_features)
# extract_X.head()
# extracted_X_train=pd.concat([extracted_X_train,X_PassengerId],axis=1)
## Applying K-Folds cross-validator
kf=KFold(n_splits=4,shuffle=True)
# model = SVC(kernel='linear', C=100)
model = LinearSVC(loss="hinge", C=1)
for nbrOfFolds,(train_index, test_index) in enumerate(kf.split(extracted_X_train)):
    X_train, X_test = extracted_X_train.iloc[train_index], extracted_X_train.iloc[test_index]
    Y_train, Y_test  = Y.iloc[train_index], Y.iloc[test_index]
    model.fit(X_train,Y_train)
#     predTrain=model.predict((X_train))
#     tempTrain=tempTrain+accuracy_score(Y_train,predTrain)
    predTest=model.predict((X_test))
    n_correct=sum(predTest==Y_test)
    print(n_correct / len(predTest))
#     tempTest=tempTest+accuracy_score(Y_test,predTest)
    
# print(f'Number of Folds{nbrOfFolds+1}')
# train_accuracy.append(tempTrain*1.0/(nbrOfFolds+1))
# test_accuracy.append(tempTest*1.0/(nbrOfFolds+1))
# print("(Train, Test) accuracy=",tempTrain*1.0/(nbrOfFolds+1),tempTest*1.0/(nbrOfFolds+1))
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(model, extracted_X_train, Y, cv=4)
cross_val_pred_accuracy_score=accuracy_score(Y,y_train_pred)
print(f"Accuracy score is {cross_val_pred_accuracy_score}")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

conf_matrx=confusion_matrix(Y,y_train_pred)
plt.matshow(conf_matrx,cmap=plt.cm.gray)
plt.show()
rows_sum=conf_matrx.sum(axis=1,keepdims=True)
norm_conf_mx=conf_matrx/rows_sum

# print(rows_sum,norm_conf_mx)
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()
from sklearn.metrics import precision_score, recall_score
precision_score(Y,y_train_pred)
recall_score(Y,y_train_pred)
from sklearn.metrics import f1_score
f1_score(Y,y_train_pred)
y_train_scores = cross_val_predict(model, extracted_X_train, Y, cv=4,method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(Y, y_train_scores)
def plot_precision_recall_vs_thresold(precisions, recalls, thresholds):
    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds,recalls[:-1],"g-",label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc='center right')
    plt.ylim([0,1])
plot_precision_recall_vs_thresold(precisions, recalls, thresholds)
plt.show()
plt.plot(recalls,precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(Y, y_train_scores)
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
plot_roc_curve(fpr,tpr)
plt.show()
from sklearn.metrics import roc_auc_score
roc_auc_score(Y,y_train_scores)
from sklearn.ensemble import RandomForestClassifier
forest_clf=RandomForestClassifier(max_depth=8,random_state=13)
for nbrOfFolds,(train_index, test_index) in enumerate(kf.split(extracted_X_train)):
    X_train, X_test = extracted_X_train.iloc[train_index], extracted_X_train.iloc[test_index]
    Y_train, Y_test  = Y.iloc[train_index], Y.iloc[test_index]
    forest_clf.fit(X_train,Y_train)
#     predTrain=model.predict((X_train))
#     tempTrain=tempTrain+accuracy_score(Y_train,predTrain)
    predTest=model.predict((X_test))
    n_correct=sum(predTest==Y_test)
    print(n_correct / len(predTest))
y_train_forest_probas=cross_val_predict(forest_clf,extracted_X_train, Y, cv=4,method="predict_proba")
y_train_forest_pred = cross_val_predict(forest_clf, extracted_X_train, Y, cv=4)
cross_val_forest_pred_accuracy_score=accuracy_score(Y,y_train_forest_pred)
print(f"Accuracy score is {cross_val_forest_pred_accuracy_score}")
y_train_forest_scores=y_train_forest_probas[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(Y,y_train_forest_scores)
plt.plot(fpr,tpr,"b:",label='Linear SVC')
plot_roc_curve(fpr_forest, tpr_forest,"Random Forest")
plt.legend(loc="lower right")
plt.show()
roc_auc_score(Y, y_train_forest_scores)
test_data_X=pd.read_csv('/kaggle/input/titanic/test.csv')

test_data_X.insert(loc=0,column='rowNum',value=np.arange(len(test_data_X)))
test_data_X=test_data_X.set_index('rowNum')
X_test_PassengerId=test_data_X['PassengerId'].copy()
# test_data_X = test_data_X.drop(columns=['PassengerId'])
# test_data_X=test_data_X.set_index('PassengerId')
test_data_X=impute_def_values(test_data_X)
extracted_test_data_X = preprocessor.transform(test_data_X)
X_test=pd.DataFrame(extracted_test_data_X,test_data_X.index,all_features)
# X_test=pd.concat([X_test,X_test_PassengerId],axis=1)
# forest_clf.predict()
Y_pred = forest_clf.predict(X_test)
X_test_PassengerId
submission = pd.DataFrame({
        "Survived": Y_pred
    })
submission.insert(loc=0,column='rowNum',value=np.arange(len(submission)))
submission=submission.set_index('rowNum')
submission=pd.concat([X_test_PassengerId,submission],axis=1)
submission
submission.to_csv('submission.csv',index=False)
