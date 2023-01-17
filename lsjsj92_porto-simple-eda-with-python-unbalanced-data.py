import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape
train.describe()
train = pd.read_csv('../input/train.csv', na_values=['-1', '-1.0'])
test = pd.read_csv('../input/test.csv', na_values=['-1', '-1.0'])
train.head()
train.info()
train.target.unique()
train.target.value_counts().plot(kind='bar')
test['target'] = np.nan
df = pd.concat([train, test], axis=0)
def bar_plot(col, data, hue=None):
    f, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(x = col, hue=hue, data = data)
    plt.show()

def dist_plot(col, data):
    f, ax = plt.subplots(figsize=(12, 5))
    sns.distplot(data[col].dropna(), kde=False, bins=10)
    plt.show()
    
def bar_plot_ci(col, data):
    f, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(col, 'target', data=data)
    plt.show()
#binary variables
binary = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',
          'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin', 
          'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']
# categorical variables
category = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 
            'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 
            'ps_car_10_cat', 'ps_car_11_cat']
# integer varialbes
integer = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 
           'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 
           'ps_calc_14', 'ps_car_11']
# float variables
floats = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_car_12', 'ps_car_13',
          'ps_car_14', 'ps_car_15']
for col in binary:
    bar_plot(col, df)
for col in category:
    bar_plot(col, df)
for col in integer:
    bar_plot(col, df)
for col in floats:
    dist_plot(col, df)
for col in floats:
    dist_plot(col, df)
corr = df.corr()
f, ax = plt.subplots(figsize=(15, 7))
sns.heatmap(corr, cmap='summer')
for col in binary:
    bar_plot_ci(col, df)
for col in category:
    bar_plot_ci(col, df)
for col in integer:
    bar_plot_ci(col, df)
train.target.value_counts().plot(kind='bar')
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
labels = train.columns[2:]

X = train[labels]
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('accuracy : %.2f' %(accuracy * 100))
model = XGBClassifier()
model.fit(X_train[['ps_calc_01']], y_train)
y_pred = model.predict(X_test[['ps_calc_01']])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_true=y_test, y_pred = y_pred)
print(conf_mat)
labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)

count_class_0, count_class_1 = train.target.value_counts()
df_class_0 = train[train['target'] == 0]
df_class_1 = train[train['target'] == 1]
print(count_class_1)
print(count_class_0)
df_class_0_under = df_class_0.sample(count_class_1)
df_class_0_under.head()
df_class_0_under.shape
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
print(df_test_under.shape)
print(df_test_under.target.value_counts())
df_test_under.target.value_counts().plot(kind="bar")
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_class_1_over.shape
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.target.value_counts())

df_test_over.target.value_counts().plot(kind='bar', title='Count (target)');