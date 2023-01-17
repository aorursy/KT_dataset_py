import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline  

data = pd.read_csv('../input/KaggleV2-May-2016.csv')
data.head()
data.info()
data['No-show'].value_counts()
data['PatientId'].value_counts()
data['AppointmentDay'] =  pd.to_datetime(data['AppointmentDay'])
data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay']) # dt.normalize to ignore hours
data['DaysUntilAppointment'] = data['AppointmentDay'].sub(data['ScheduledDay'].dt.normalize(), axis=0) / np.timedelta64(1, 'D')
data['Period'] = pd.cut(data.ScheduledDay.dt.hour,[0,8,12,18,24],labels=['Night','Morning','Afternoon','Evening'])
data.dtypes
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper

def encode_categorical_cols(dataframe):
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    mapping = []
    mapping += ((col, LabelEncoder()) for col in categorical_columns)
    mapping += ((col, None) for col in numerical_columns if col not in ['AppointmentID'])
        
    mapper = DataFrameMapper(mapping, df_out=True)
    
    stages = []
    stages += [("pre_processing_mapper", mapper)]
    
    pipeline = Pipeline(stages)
    transformed_df = pipeline.fit_transform(dataframe)
    return transformed_df

encoded_data = encode_categorical_cols(data)
encoded_data.dtypes
encoded_data.corr()
correlations = encoded_data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,13,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(encoded_data.columns)
ax.set_yticklabels(encoded_data.columns)
plt.xticks(rotation=90)
plt.show()
from sklearn.model_selection import *

target = encoded_data['No-show']
dataset = encoded_data.drop(['No-show', 'PatientId'], axis=1)

train_data, test_data, train_target, expected = train_test_split(dataset, target, test_size=0.3)
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
features = dataset.columns
importances = random_forest.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
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
print("Min Age:", encoded_data.Age.min())
print("Max Age:", encoded_data.Age.max())
encoded_data = encoded_data[(encoded_data.Age >= 0) & (encoded_data.Age <= 100)] # removing outliers in Age
# Number of Appointments Missed by Patient
encoded_data['AppointmentsMissed'] = encoded_data.groupby('PatientId')['No-show'].apply(lambda x: x.cumsum())
encoded_data.head()
target = encoded_data['No-show']
dataset = encoded_data.drop(['No-show', 'PatientId'], axis=1)

train_data, test_data, train_target, expected = train_test_split(dataset, target, test_size=0.3)
logreg.fit(train_data, train_target)
predicted = logreg.predict(test_data)
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('Logistic regression ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print ('Logistic regression F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))
model = random_forest.fit(train_data, train_target)
predicted = model.predict(test_data)

print('Random Forest accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('Random Forest ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print('Random Forest F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))
n_estimators = [10, 20, 50]
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
print(model.best_estimator_)
predicted = model.predict(test_data)

print('Random Forest accuracy: {:.3f}'.format(accuracy_score(expected, predicted)))
print('Random Forest ROC accuracy: {:.3f}'.format(roc_auc_score(expected, predicted)))
print ('Random Forest F1 score: {:.3f}'.format(f1_score(expected, predicted, average='weighted')))
print('Confusion matrix')
print(confusion_matrix(expected, predicted))