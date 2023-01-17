# Quick load dataset and check
import pandas as pd
import matplotlib as plt
filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)
data_train.describe()
%matplotlib inline 

train_missing_count = (data_train == -1).sum()
plt.rcParams['figure.figsize'] = (20,10)
train_missing_count.plot.bar()
# Drop columns with many -1 values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
data_train.drop(vars_to_drop, inplace=True, axis=1)
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(missing_values=-1, strategy='mean')
mode_imputer = SimpleImputer(missing_values=-1, strategy='most_frequent')

data_train['ps_reg_03'] = mean_imputer.fit_transform(data_train[['ps_reg_03']]).ravel()
data_train['ps_car_12'] = mean_imputer.fit_transform(data_train[['ps_car_12']]).ravel()
data_train['ps_car_14'] = mean_imputer.fit_transform(data_train[['ps_car_14']]).ravel()
data_train['ps_car_11'] = mode_imputer.fit_transform(data_train[['ps_car_11']]).ravel()
## Select target and features
fea_col = data_train.columns[2:]
data_Y = data_train['target']
data_X = data_train[fea_col]
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.1, shuffle=True, random_state=42)
from sklearn.utils import resample

train = pd.concat([x_train, y_train], axis=1)
class_0 = train[train['target'] == 0]
class_1 = train[train['target'] == 1]

class_1_upsampled = resample(class_1, 
                          replace=True,
                          n_samples=len(class_0),
                          random_state=42)

upsampled = pd.concat([class_0, class_1_upsampled])

x_train = upsampled.drop('target', axis=1)
y_train = upsampled.target
from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy='minority', random_state=42)
x_train_over, y_train_over = sm.fit_sample(x_train, y_train)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()

param_grid = {'n_estimators': [50],
              'n_jobs': [-1],
              'class_weight': ['balanced'],
              'min_samples_leaf': [40, 50, 60], 
              'min_samples_split': [50, 80, 110, 140]
              }

clf = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', verbose=10)
clf.fit(x_train, y_train)
clf.best_params_
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier

rf_clf = RandomForestClassifier(n_estimators=50, 
                                n_jobs=-1,
                                class_weight='balanced',
                                min_samples_leaf=50, 
                                min_samples_split=50,
                                random_state=42,
                                verbose=5)

rf_clf.fit(x_train, y_train)
y_pred = rf_clf.predict(x_val)
sum(y_pred==y_val)/len(y_val)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_true=y_val, y_pred=y_pred)
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(rf_clf, x_val, y_val)
from sklearn.metrics import f1_score

f1_score(y_val, y_pred, average='macro')
data_test_X = data_test.drop(columns=['id', 'ps_car_03_cat', 'ps_car_05_cat'])
y_target = rf_clf.predict(data_test_X)
y_target0 = y_target[y_target == 0]
y_target1 = y_target[y_target == 1]
print(len(data_test_X), len(y_target0), len(y_target1))
sum(y_target==0)
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)
data_out




