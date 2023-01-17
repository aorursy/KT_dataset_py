#pip install catboost
#Importing Libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import RFECV

from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score,f1_score,log_loss,confusion_matrix,classification_report,precision_score,recall_score

#Reading Dataset

dataset = pd.read_csv('../input/adult-census-income/adult.csv')

dataset.head()
#Dropping education since we have its categorical coloumn education.num

dataset.drop(columns=['education'],inplace=True)
#data types of columns

dataset.dtypes
#List of columns present in Dataset

dataset.columns
#Shape of Dataset

dataset.shape
#Renaming few columns

dataset.rename(columns = {'education.num':'education_num', 'marital.status':'marital_status', 'capital.gain':'capital_gain',

                          'capital.loss':'capital_loss','hours.per.week':'hours_per_week','native.country':'native_country'}, inplace = True) 
#Checking the change

dataset.columns
#Replacing "?" with NAN

dataset['workclass'].replace('?', np.nan, inplace= True)

dataset['occupation'].replace('?', np.nan, inplace= True)

dataset['native_country'].replace('?', np.nan, inplace= True)
#A detailed description of the datset

dataset.describe(include='all')
#Number of null values in the dataset column wise

dataset.isnull().sum()
#Grouping Workclass

dataset.groupby(['workclass']).size().plot(kind="bar",fontsize=14)

plt.xlabel('Work Class Categories')

plt.ylabel('Count of People')

plt.title('Barplot of Workclass Variable')
#Grouping Occupation

dataset.groupby(['occupation']).size().plot(kind="bar",fontsize=14)

plt.xlabel('Occupation Categories')

plt.ylabel('Count of People')

plt.title('Barplot of Occupation Variable')
#Grouping Native Country

dataset.groupby(['native_country']).size().plot(kind="bar",fontsize=10)

plt.xlabel('Native Country Categories')

plt.ylabel('Count of People')

plt.title('Barplot of Native Country Variable')
#Droping null values in occupation column

dataset.dropna(subset=['occupation'],inplace=True)

dataset.isnull().sum()
#Imputing null values with Mode



dataset['native_country'].fillna(dataset['native_country'].mode()[0], inplace=True)
#Checking for null values

dataset.isnull().sum()
#Confirming the Categorical Features

categorical_feature_mask = dataset.dtypes==object

categorical_feature_mask

##Label encoding the all the categorical features





from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_list=['income','workclass','marital_status','occupation','relationship','race','sex','native_country']

dataset[cat_list]=dataset[cat_list].apply(lambda x:le.fit_transform(x))
#Number of categories in dataset

dataset.nunique()
#Finding Correlation between variables

corr = dataset.corr()

mask = np.zeros(corr.shape, dtype=bool)

mask[np.triu_indices(len(mask))] = True

plt.subplots(figsize=(10,7))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,annot=True,cmap='RdYlGn',mask = mask)
#Dropping "sex" variable since it is highly correlated with "relationship" variable 

dataset.drop(columns=['sex'],inplace=True)
#Slicing dataset into Independent(X) and Target(y) varibles

y = dataset.pop('income')

X = dataset

#Scaling the dependent variables

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
#Dividing dataset into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



print(X_train.shape)

print(y_train.shape)
#Performing Recursive Feauture Elimation with Cross Validation

#Using Random forest for RFE-CV and logloss as scoring

from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss

clf_rf=RandomForestClassifier(random_state=0)

rfecv=RFECV(estimator=clf_rf, step=1,cv=5,scoring='neg_log_loss')

rfecv=rfecv.fit(X_train,y_train)
#Optimal number of features

X_train = pd.DataFrame(X_train)

X_test = pd.DataFrame(X_test)

print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X_train.columns[rfecv.support_])
#Feauture Ranking

clf_rf = clf_rf.fit(X_train,y_train)

importances = clf_rf.feature_importances_



std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

#Selecting the Important Features

X_train = X_train.iloc[:,X_train.columns[rfecv.support_]]

X_test = X_test.iloc[:,X_test.columns[rfecv.support_]]
#Creating anew dataframe with column names and feature importance

dset = pd.DataFrame()

data1 = dataset



dset['attr'] = data1.columns





dset['importance'] = clf_rf.feature_importances_

#Sorting with importance column

dset = dset.sort_values(by='importance', ascending=True)



#Barplot indicating Feature Importance

plt.figure(figsize=(16, 14))

plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')

plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)

plt.xlabel('Importance', fontsize=14, labelpad=20)

plt.show()
classifier_lg = LogisticRegression(random_state=0)

classifier_dt = DecisionTreeClassifier(random_state=0)

classifier_nb = GaussianNB()

classifier_knn = KNeighborsClassifier()

classifier_rf = RandomForestClassifier(random_state=0)

classifier_xgb = XGBClassifier(random_state=0)

classifier_cgb = CatBoostClassifier(random_state=0)




# Instantiate the classfiers and make a list

classifiers = [classifier_lg,

               classifier_dt,

               classifier_nb,

               classifier_knn,

               classifier_rf,

               classifier_xgb,

               classifier_cgb]

# Define a result table as a DataFrame

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','Roc Auc','Accuracy','f1 Score','logloss','Confusion Matrix','Precision','Recall'])



# Train the models and record the results

for cls in classifiers:

    model = cls.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[::,1]

    y_pred = model.predict(X_test)

    print(cls, '\n','Confusion Matrix','\n',confusion_matrix(y_test,  y_pred))

    print('\n','Classification Report','\n',classification_report(y_test,  y_pred))

    print('='*170)

    fpr, tpr, _ = roc_curve(y_test,  y_proba)

    auc = roc_auc_score(y_test, y_proba)

    Accuracy = accuracy_score(y_test,y_pred)

    f1score = f1_score(y_test,y_pred)

    logloss = log_loss(y_test,y_proba)

    cm = confusion_matrix(y_test,  y_pred)

    precision = precision_score(y_test,  y_pred)

    recall = recall_score(y_test,  y_pred)

  

    

    result_table = result_table.append({'classifiers':cls.__class__.__name__,

                                        'fpr':fpr, 

                                        'tpr':tpr, 

                                        'Roc Auc':auc,

                                        'Accuracy':Accuracy,

                                        'f1 Score':f1score,

                                        'logloss':logloss,

                                        'Confusion Matrix': cm,

                                        'Precision':precision,

                                        'Recall':recall}, ignore_index=True)



# Set name of the classifiers as index labels

result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))



for i in result_table.index:

    plt.plot(result_table.loc[i]['fpr'], 

             result_table.loc[i]['tpr'], 

             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['Roc Auc']))

    

plt.plot([0,1], [0,1], color='orange', linestyle='--')



plt.xticks(np.arange(0.0, 1.1, step=0.1))

plt.xlabel("Flase Positive Rate", fontsize=15)



plt.yticks(np.arange(0.0, 1.1, step=0.1))

plt.ylabel("True Positive Rate", fontsize=15)



plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)

plt.legend(prop={'size':13}, loc='lower right')



plt.show()
result_table[['Roc Auc','Accuracy','f1 Score','logloss','Confusion Matrix','Precision','Recall']]