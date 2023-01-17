import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

from time import time

import warnings
%matplotlib inline
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')
iris_df = pd.read_csv('../input/iris/Iris.csv')
# check the first five rows
iris_df.head()
columns = ['id','sepal_length', 'sepal_width','petal_length', 'petal_width', 'species']
iris_df.columns = columns
# frequency table for species
iris_df['species'].value_counts()
# replace label with numeric 
new_lab = {'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
iris_df['species'] = iris_df['species'].map(new_lab)
iris_df['species'].value_counts()
iris_df.info()
# Check missing values of features
print(len(iris_df[iris_df['sepal_length'].isna()]))
print(len(iris_df[iris_df['sepal_width'].isna()]))
print(len(iris_df[iris_df['petal_length'].isna()]))
print(len(iris_df[iris_df['petal_width'].isna()]))
iris_df.describe().T
# Calculate 1st, 2nd, 3rd quartiles, IQR, lower and upper bounds

# Sepal Length
sepal_length_q = pd.DataFrame(iris_df['sepal_length'].quantile([0.25, 0.50, 0.75]))
iqr_sl = sepal_length_q.iloc[2:3,:].values[0][0] - sepal_length_q.iloc[0:1,:].values[0][0]
lower_bound_sl = sepal_length_q.iloc[0:1,:].values[0][0] - (iqr_sl*1.5)
upper_bound_sl = sepal_length_q.iloc[2:3,:].values[0][0] + (iqr_sl*1.5)
print(sepal_length_q)

print('\nIQR:', str(iqr_sl))
print('Lower Bound:', str(lower_bound_sl))
print('Upper Bound:', str(upper_bound_sl),'\n')

# Sepal Width
sepal_width_q = pd.DataFrame(iris_df['sepal_width'].quantile([0.25, 0.50, 0.75]))
iqr_sw = sepal_width_q.iloc[2:3,:].values[0][0] - sepal_width_q.iloc[0:1,:].values[0][0]
lower_bound_sw = sepal_width_q.iloc[0:1,:].values[0][0] - (iqr_sw*1.5)
upper_bound_sw = sepal_width_q.iloc[2:3,:].values[0][0] + (iqr_sw*1.5)
print(sepal_width_q)

print('\nIQR:', str(iqr_sw))
print('Lower Bound:', str(lower_bound_sw))
print('Upper Bound:', str(upper_bound_sw),'\n')

# Petal Length
petal_length_q = pd.DataFrame(iris_df['petal_length'].quantile([0.25, 0.50, 0.75]))
iqr_pl = petal_length_q.iloc[2:3,:].values[0][0] - petal_length_q.iloc[0:1,:].values[0][0]
lower_bound_pl = petal_length_q.iloc[0:1,:].values[0][0] - (iqr_pl*1.5)
upper_bound_pl = petal_length_q.iloc[2:3,:].values[0][0] + (iqr_pl*1.5)
print(petal_length_q)

print('\nIQR:', str(iqr_pl))
print('Lower Bound:', str(lower_bound_pl))
print('Upper Bound:', str(upper_bound_pl),'\n')

# Petal Width
petal_width_q = pd.DataFrame(iris_df['petal_width'].quantile([0.25, 0.50, 0.75]))
iqr_pw = petal_width_q.iloc[2:3,:].values[0][0] - petal_width_q.iloc[0:1,:].values[0][0]
lower_bound_pw = petal_width_q.iloc[0:1,:].values[0][0] - (iqr_pw*1.5)
upper_bound_pw = petal_width_q.iloc[2:3,:].values[0][0] + (iqr_pw*1.5)
print(petal_width_q)

print('\nIQR:', str(iqr_pw))
print('Lower Bound:', str(lower_bound_pw))
print('Upper Bound:', str(upper_bound_pw))
# check distribution of sepal length, sepal width, petal length, petal width
fig_dims = (15, 12)
fig, ax = plt.subplots(2,2,figsize=fig_dims, sharex=True)

feat = ['sepal_length','sepal_width','petal_length','petal_width']
low_up_bounds = {lower_bound_sl: upper_bound_sl ,lower_bound_sw: upper_bound_sw,lower_bound_pl: upper_bound_pl,lower_bound_pw: upper_bound_pw}

for i in range(4):
    plt.subplot(221+i)
    plt.title(feat[i]+" distribution", fontsize=12)
    plt.axvline(x=np.mean(iris_df[feat[i]]), color='orange', linewidth=2)
    plt.axvline(x=np.median(iris_df[feat[i]]), color='red', linewidth=2)
    plt.axvline(x=list(low_up_bounds.keys())[i], color='grey', linestyle='--')
    plt.axvline(x=list(low_up_bounds.values())[i], color='grey', linestyle='--')
    sns.distplot(iris_df[feat[i]], kde=False, color='mediumseagreen')
    plt.ylabel('Frequency', fontsize=11)
    plt.xlabel(feat[i], fontsize=11)

plt.show()
sns.boxplot(iris_df['sepal_width'])
plt.show()
# Correlation check
iris_df.corr('pearson')
iris_df['sl_sw'] = iris_df['sepal_length']*iris_df['sepal_width']
iris_df['sl_pl'] = iris_df['sepal_length']*iris_df['petal_length']
iris_df['sl_pw'] = iris_df['sepal_length']*iris_df['petal_width']

iris_df['sw_pl'] = iris_df['sepal_width']*iris_df['petal_length']
iris_df['sw_pw'] = iris_df['sepal_width']*iris_df['petal_width']

iris_df['pl_pw'] = iris_df['petal_length']*iris_df['petal_width']
iris_df['species'].value_counts()
sns.relplot(x="petal_width", y="pl_pw",hue="species", data=iris_df)
ax1 = plt.gca()
ax1.set_title('Petal Width vs PL_PW')
ax1.set(xlabel='Petal Width')

sns.relplot(x="petal_length", y="pl_pw",hue="species", data=iris_df)
ax = plt.gca()
ax.set_title('Petal Length vs PL_PW')

sns.relplot(x="sepal_length", y="pl_pw",hue="species", data=iris_df)
ax = plt.gca()
ax.set_title('Sepal Length vs PL_PW')

plt.show()
iris_df.corr('pearson')
# take the features which have strong correlation (treshold 0.9)
iris_df_v2 = iris_df[['sl_pw','sw_pw','sw_pl','pl_pw','sl_pl','petal_length','petal_width','species']]
iris_df_v2.corr()
iris_df_v2.head()
features = iris_df_v2.drop('species', axis=1)
labels = iris_df_v2['species']
# Check the first five rows of our features
features.head()
# Check the first five rows of our labels
labels[:5]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42) #70% train set 30% test set
# Check
print(len(labels))
for dataset in [y_train, y_test]:
    print(round(len(dataset)/len(labels),2))
def printResult(results):
    print('Best Params: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std *2,3), params))
lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] # The C hyperparameter is a regularization parameter in logistic regression that controls how closly the model fits to the training data
}
cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(x_train, y_train.values.ravel())
printResult(cv)
lr_model = cv.best_estimator_
# Clean up
del cv, parameters
rf = RandomForestClassifier()

parameters = {
    'n_estimators': [5, 50, 250],
    'max_depth': [2, 4, 8, 16, 32, None]
}
cv = GridSearchCV(rf, parameters, cv= 5)
cv.fit(x_train, y_train.values.ravel())
printResult(cv)
rf_model = cv.best_estimator_
# Clean up
del cv, parameters
models = {'LR': lr_model, 'RF': rf_model}
def evaluateModel(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred, average='micro'), 3)
    recall = round(recall_score(labels, pred, average='micro'), 3)
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                  accuracy,
                                                                                  precision,
                                                                                  recall,
                                                                                  round(end - start)))
    
for name, model in models.items():
    evaluateModel(name, model, x_test, y_test)