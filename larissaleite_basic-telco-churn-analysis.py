import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline 
import matplotlib as mlp
import matplotlib.pyplot as plt
import os

print(os.listdir("../input"))
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
data.dtypes
data.isnull().values.any()
def to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan

data['TotalCharges'] = data['TotalCharges'].apply(to_float)
data['SeniorCitizen'] = data['SeniorCitizen'].astype(object)

data.dtypes
data['Churn'].value_counts()
y_True = data['Churn'][data['Churn'] == 'Yes']
print(str( (y_True.shape[0] / float(data['Churn'].shape[0])) * 100 ) + "% of churn")
data.hist(bins=10, figsize=(12,7))
plt.show()
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
data.groupby('Churn').describe()
#comparing characteristics of those who churned and those who didn't
data.boxplot(column='TotalCharges', by='Churn')
data.boxplot(column='MonthlyCharges', by='Churn')
data.boxplot(column='tenure', by='Churn')
old_customers = data[data['tenure'] > 65]
old_customers.head()
old_customers['Churn'].value_counts()
churned_customers = data[data['Churn'] == 'Yes']
churned_customers.hist()
churned_customers.boxplot(column='TotalCharges', by='Churn')
churned_customers.boxplot(column='MonthlyCharges', by='Churn')
churned_customers.boxplot(column='tenure', by='Churn')
data.corr()
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper

def encode_categorical_cols(dataframe):
    categorical_columns = dataframe.dtypes.pipe(lambda x: x[x == 'object'])

    mapping = []
    mapping += ((col, LabelEncoder()) for col in categorical_columns.index if col != 'customerID')
    mapping += ((col, None) for col in dataframe.dtypes.index if col not in categorical_columns.index or col == 'customerID')
        
    mapper = DataFrameMapper(mapping, df_out=True)
    
    stages = []
    stages += [("pre_processing_mapper", mapper)]

    pipeline = Pipeline(stages)
    transformed_df = pipeline.fit_transform(dataframe)
    return transformed_df

encoded_data = encode_categorical_cols(data)
encoded_data.dtypes
encoded_data.head()
encoded_data.corr()
correlations = encoded_data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,20,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(encoded_data.columns)
ax.set_yticklabels(encoded_data.columns)
plt.xticks(rotation=90)
plt.show()
encoded_data.hist(bins=10, figsize=(12,7))
data['TechSupport'].value_counts()
encoded_data = encoded_data.loc[:,('MonthlyCharges', 'TotalCharges', 'tenure', 'Contract', 'Churn')]
encoded_data[['TotalCharges/tenure']] = encoded_data[['TotalCharges']].div(encoded_data.tenure, axis=0) 
encoded_data.head()
encoded_data.loc[encoded_data['MonthlyCharges']-encoded_data['TotalCharges/tenure'] > encoded_data['TotalCharges']*0.10]
encoded_data['ChargedMore'] = np.where(encoded_data['MonthlyCharges']-encoded_data['TotalCharges/tenure'] > encoded_data['TotalCharges']*0.10, 1, 0)
encoded_data.head()
# Binning numerical columns
encoded_data['CatTenure'] = pd.qcut(encoded_data.tenure, q=3, labels=False )
encoded_data['CatMonthlyCharges']= pd.qcut(encoded_data.MonthlyCharges, q=3, labels=False)
encoded_data.head()
final_df = encoded_data.loc[:,('Contract', 'Churn', 'CatTenure', 'CatMonthlyCharges', 'ChargedMore')]
final_df.head()
final_df.corr()
from sklearn.model_selection import *

target = final_df['Churn']
dataset = final_df.loc[:,('Contract', 'CatTenure', 'CatMonthlyCharges')]

train_data, test_data, train_target, expected = train_test_split(dataset, target, test_size=0.3, stratify=target)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

random_forest = RandomForestClassifier(n_estimators=10)
model = random_forest.fit(train_data, train_target)
predicted = model.predict(test_data)

print('Random Forest accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('Random Forest ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print('Random Forest F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(train_data, train_target)
predicted = logreg.predict(test_data)
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('Logistic regression ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print ('Logistic regression F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))
from sklearn.svm import SVC
svc = SVC()
svc.fit(train_data, train_target)
predicted = svc.predict(test_data)
print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('Support vector machine ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print ('Support vector machine F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(train_data, train_target)
predicted = knn.predict(test_data)
print('KNN accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('KNN ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print ('KNN F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))
random_forest = RandomForestClassifier()

n_estimators = [20, 50, 100]
max_features = ['auto', 'sqrt']
max_depth = [10, 20, None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
class_weight = [None, 'balanced']

parameters = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap, 'class_weight' : class_weight}

clf = GridSearchCV(random_forest, param_grid = parameters, cv = 3, n_jobs = -1)
model = clf.fit(train_data, train_target)
print(model.best_params_)
#{'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 20, 'min_samples_split': 2, 'max_features': 'auto', 'max_depth': 10, 'class_weight': None}
print(model.best_estimator_)
predicted = model.predict(test_data)

print('Random Forest accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('Random Forest ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print ('Random Forest F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))
C_param_range = [0.001,0.01,0.1,1,10,100]
penal = ['l1', 'l2']

logreg_params = {'C': C_param_range, 'penalty' : penal, 'class_weight' : class_weight}
logreg = GridSearchCV(LogisticRegression(), param_grid = logreg_params, cv = 10, n_jobs = -1)
logreg_model = logreg.fit(train_data, train_target)
print(logreg_model.best_params_)
print(logreg_model.best_estimator_)
predicted = logreg_model.predict(test_data)

print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('Logistic regression ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print ('Logistic regression F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))