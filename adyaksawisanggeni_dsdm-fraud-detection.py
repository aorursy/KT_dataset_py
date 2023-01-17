import pandas as pd
import numpy as np
import glob

path_name = "../input/tubes-dsdm-1/fraud_detection/"
train_inputs_filenames = glob.glob(path_name + "*Train.Inputs")
train_targets_filenames = glob.glob(path_name + "*Train.Targets")
test_inputs_filenames = glob.glob(path_name + "*Test.Inputs")
test_targets_filenames = glob.glob(path_name + "*Test.Targets")

train_inputs_filenames.sort()
train_targets_filenames.sort()
test_inputs_filenames.sort()
test_targets_filenames.sort()
# Check order of filenames in list
for (input, target) in zip(train_inputs_filenames, train_targets_filenames):
  print(input.split('/')[3] + " & " + target.split('/')[3])
print("\n")
for (input, target) in zip(test_inputs_filenames, test_targets_filenames):
  print(input.split('/')[3] + " & " + target.split('/')[3])
# Create Train Dataframe
df_train_input_list = []
for input in train_inputs_filenames:
  df = pd.read_csv(input, index_col=None, header=0)
  df_train_input_list.append(df)

df_train_full = pd.concat(df_train_input_list, axis=0, ignore_index=True)
# Create Train Targets
train_targets_list = []
for target in train_targets_filenames:
  df = pd.read_csv(target, index_col=None, header=None)
  train_targets_list.append(df[0])

train_targets = pd.concat(train_targets_list, ignore_index=True)
# Create Test Dataframe
df_test_input_list = []
for input in test_inputs_filenames:
  df = pd.read_csv(input, index_col=None, header=0)
  df_test_input_list.append(df)

df_test_full = pd.concat(df_test_input_list, axis=0, ignore_index=True)
# Create Test Targets
test_targets_list = []
for target in test_targets_filenames:
  df = pd.read_csv(target, index_col=None, header=None)
  test_targets_list.append(df[0])

test_targets = pd.concat(test_targets_list, ignore_index=True)
print("Train input shape: " + str(df_train_full.shape))
print("Train target shape: " + str(train_targets.shape))
print("Test input shape: " + str(df_test_full.shape))
print("Test target shape: " + str(test_targets.shape))
train_targets
df_train_full.describe()
# Remove duplicated data (if any)
df_train_full['target'] = train_targets.values

num_of_data = len(df_train_full)
df_train_full = df_train_full.drop_duplicates()
num_of_unique_data = len(df_train_full)

print('Initial data length: ', num_of_data)
print('Data length after the duplicates are dropped: ', num_of_unique_data)
print("Duplicated data: ", num_of_data - num_of_unique_data)
print('Initial data length: ', len(train_targets))
train_targets = train_targets[df_train_full.index]
print('Targets length after the duplicates are dropped: ', len(train_targets))
df_train_full.reset_index()
train_targets.reset_index()
# Check for null values
df_train_full.isnull().sum()
# Delete null values because there's only one row with it.
df_train_full = df_train_full.dropna(how='any', subset=['domain1'])
train_targets = train_targets[df_train_full.index]

df_train_full = df_train_full.reset_index(drop=True)
train_targets = train_targets.reset_index(drop=True)
# Show columns with numeric type and non-numeric type
df_train_full.info()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df_train_full['state1'] = le.fit_transform(df_train_full['state1'])
df_train_full['domain1'] = le.fit_transform(df_train_full['domain1'].astype(str))

df_test_full['state1'] = le.fit_transform(df_test_full['state1'])
df_test_full['domain1'] = le.fit_transform(df_test_full['domain1'].astype(str))

df_train_full.head()
# Show correlation matrix
df_train_full.corr().style.background_gradient(cmap='coolwarm')
existing_indices = df_train_full.index.values.tolist()
print(train_targets)
print(len(train_targets))
# Search indices where its targets = 1
fraud_indices = [i for i in existing_indices if (train_targets.iloc[i] == 1)]
print('Fraud transactions: ', len(fraud_indices))
# Get fraud transactions
df_train_fraud = df_train_full[df_train_full.index.isin(fraud_indices)]
df_train_fraud.head()
# Get real transactions
df_train_real = df_train_full.drop(fraud_indices, axis=0)
df_train_real.head()
# Describe real transactions
df_train_real.describe()
# Describe fraud transactions
df_train_fraud.describe()
from collections import Counter

amount_list = list(df_train_fraud.amount.values)
amount_counted_list = Counter(amount_list)
amount_counted_list.most_common(10)
hour_list = list(df_train_fraud.hour1.values)
hour_counted_list = Counter(hour_list)
hour_counted_list.most_common(10)
state_list = list(df_train_fraud.state1.values)
state_counted_list = Counter(state_list)
state_counted_list.most_common(10)
zip_list = list(df_train_fraud.zip1.values)
zip_counted_list = Counter(zip_list)
zip_counted_list.most_common(10)
zip_list = list(df_train_fraud.zip1.values)
zip_counted_list = Counter(zip_list)
zip_counted_list.most_common(10)
field1_list = list(df_train_fraud.field1.values)
field1_counted_list = Counter(field1_list)
field1_counted_list.most_common(10)
domain1_list = list(df_train_fraud.domain1.values)
domain1_counted_list = Counter(domain1_list)
domain1_counted_list.most_common(10)
field2_list = list(df_train_fraud.field2.values)
field2_counted_list = Counter(field2_list)
field2_counted_list.most_common(10)
flag1_list = list(df_train_fraud.flag1.values)
flag1_counted_list = Counter(flag1_list)
flag1_counted_list.most_common(10)
field3_list = list(df_train_fraud.field3.values)
field3_counted_list = Counter(field3_list)
field3_counted_list.most_common(10)
field4_list = list(df_train_fraud.field4.values)
field4_counted_list = Counter(field4_list)
field4_counted_list.most_common(10)
field5_list = list(df_train_fraud.field5.values)
field5_counted_list = Counter(field5_list)
field5_counted_list.most_common(30)
indicator1_list = list(df_train_fraud.indicator1.values)
indicator1_counted_list = Counter(indicator1_list)
indicator1_counted_list.most_common(10)
indicator2_list = list(df_train_fraud.indicator2.values)
indicator2_counted_list = Counter(indicator2_list)
indicator2_counted_list.most_common(10)
flag2_list = list(df_train_fraud.flag2.values)
flag2_counted_list = Counter(flag2_list)
flag2_counted_list.most_common(10)
flag3_list = list(df_train_fraud.flag3.values)
flag3_counted_list = Counter(flag3_list)
flag3_counted_list.most_common(10)
flag4_list = list(df_train_fraud.flag4.values)
flag4_counted_list = Counter(flag4_list)
flag4_counted_list.most_common(10)
flag5_list = list(df_train_fraud.flag5.values)
flag5_counted_list = Counter(flag5_list)
flag5_counted_list.most_common(20)
df_train_full.loc[fraud_indices, 'indicator2'] = 0
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

x_min_amount = df_train_full['amount'].min()
x_max_amount = df_train_full['amount'].max()

x_min_field3 = df_train_full['field3'].min()
x_max_field3 = df_train_full['field3'].max()

batas_atas_amount = 1
batas_bawah_amount = 0
def scaleAmount(x):
  return (((x-(x_min_amount))*(batas_atas_amount-(batas_bawah_amount)))/(x_max_amount-(x_min_amount)))+(batas_bawah_amount)

batas_atas_field3 = 1
batas_bawah_field3 = -1
def scaleField3(x):
  return (((x-(x_min_field3))*(batas_atas_field3-(batas_bawah_field3)))/(x_max_field3-(x_min_field3)))+(batas_bawah_field3)

def preprocess(df):
    le = preprocessing.LabelEncoder()
    df_new = df.copy()

    # drop column total and hour2
    df_new = df_new.drop(['total'], axis=1)
    #df_new = df_new.drop(['hour2'], axis=1)

    # label encode state1
    df_new['state1'] = le.fit_transform(df_new['state1'])

    # label encode domain1
    df_new['domain1'] = le.fit_transform(df_new['domain1'].astype(str))

    # scale amount's values as 0 to 1 values
    df_new['amount'] = df_new['amount'].apply(scaleAmount)

    # scale field3's values as -1 to 1 values
    df_new['field3'] = df_new['field3'].apply(scaleField3)

    return df_new
df_train_preprocessed = preprocess(df_train_full)
df_train_preprocessed['target'] = pd.Series(train_targets)
df_train_preprocessed = df_train_preprocessed.drop_duplicates()
X_train = df_train_preprocessed.drop(['target'], axis=1)
y_train = df_train_preprocessed['target']

df_test_processed = preprocess(df_test_full)
df_test_processed['target'] = pd.Series(test_targets)
df_test_processed = df_test_processed.drop_duplicates()
X_test = df_test_processed.drop(['target'], axis=1)
y_test = df_test_processed['target']
from xgboost import XGBClassifier
from numpy import mean
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from collections import Counter

model = XGBClassifier()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

counter = Counter(train_targets)

param_grid = dict(
    scale_pos_weight=[counter[0]/counter[1]],
    learning_rate=[0.01*i*i for i in range(1,21)],
    max_depth=[11+i for i in range(3)],
    tree_method=['gpu_hist'],
    predictor=['gpu_predictor']
)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='f1', verbose=1)

grid_result  = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
model = XGBClassifier(**grid_result.best_params_)
model.fit(X_train, y_train, verbose=True)

y_pred = model.predict(X_test)
from sklearn.metrics import f1_score, confusion_matrix
print("XGBoost F1 Score -> ", f1_score(y_pred, y_test)*100)
print(confusion_matrix(y_pred, y_test))
df_train_full
from sklearn.preprocessing import StandardScaler
features = df_train_full.columns
features
from sklearn.decomposition import PCA
#df_train_full = df_train_full.drop(['target'], axis=1)
pca = PCA(n_components=2)
pca.fit(df_train_full)
df_train_pca = pca.transform(df_train_full)
df_train_pca.shape
df_test_pca = ss.transform(df_test_full)
df_test_pca = pca.transform(df_test_pca)
X_train_pca = df_train_pca
y_train_pca = train_targets

X_test_pca = df_test_pca
y_test_pca = test_targets
counter = Counter(train_targets)
print(counter[0], counter[1])

model = XGBClassifier()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

param_grid = dict(
    scale_pos_weight=[counter[0]/counter[1]],
    learning_rate=[0.01*i*i for i in range(1,21)],
    max_depth=[4+i for i in range(3)],
    tree_method=['gpu_hist'],
    predictor=['gpu_predictor']
)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='f1', verbose=1)

grid_result  = grid.fit(X_train_pca, y_train_pca)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
model = XGBClassifier(**grid_result.best_params_)
model.fit(X_train_pca, y_train_pca, verbose=True)

y_pred_pca = model.predict(X_test_pca)
from sklearn.metrics import f1_score, confusion_matrix
print("XGBoost F1 Score PCA} -> ", f1_score(y_pred_pca, y_test_pca)*100)
print(confusion_matrix(y_pred_pca, y_test_pca))
temp = pd.DataFrame(df_train_pca)
df_train_pca = temp
df_train_pca['target'] = train_targets
df_train_pca
import matplotlib.pyplot as plt



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = df_train_pca['target'] == target
    ax.scatter(df_train_pca.loc[indicesToKeep, 0]
               , df_train_pca.loc[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()