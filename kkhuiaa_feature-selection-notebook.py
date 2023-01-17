# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style(style='darkgrid')



from sklearn.feature_selection import RFECV, VarianceThreshold

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.preprocessing import OrdinalEncoder

from ml_metrics import quadratic_weighted_kappa

random_state = 42



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/prudential-life-insurance-assessment/train.csv.zip')

train_df.set_index('Id', inplace=True)

display(train_df.head())

X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]

print(X_train.shape)

print(y_train.shape)
ordinal_y_mean_dict = {}



#transform the categorial columns into numeric

for col in X_train.select_dtypes(include='object').columns:

    ordinal_y_mean_dict[col]  = {index:i for i, index in enumerate(train_df.groupby(col)['Response'].mean().sort_values().index)}

    print(ordinal_y_mean_dict[col])

    X_train[col] = X_train[col].map(ordinal_y_mean_dict[col])

    

#check missing data

for col in X_train:

    if pd.isnull(X_train[col]).any():

        print('containing NA values:', col)



imputer = SimpleImputer(strategy='mean')

X_train2 = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

#check missing data

for col in X_train2:

    if pd.isnull(X_train2[col]).any():

        print('containing NA values:', col)
X_train2_unsup  = X_train2.copy() #deep copy
#Suppose you data now contains some missing data, you want to filter the data with very high missing rate.

X_train2_unsup['dump'] = np.nan

X_train2_unsup2 = X_train2_unsup.dropna(axis=1, thresh=0.8) #drop the columns with missing rate > 80%

print(X_train2_unsup.shape)

print(X_train2_unsup2.shape) #drop the demp column
selector = VarianceThreshold(0.7)

selector.fit(X_train2_unsup2)

X_train2_unsup3 = X_train2_unsup2[X_train2_unsup2.columns[selector.get_support(indices=True)]]

print('number of columns after dropping by variance threshold:', X_train2_unsup3.shape[1])
X_train2_sup = X_train2.copy() #deep copy
%%time

X_model, X_valid, y_model, y_valid = train_test_split(X_train2_sup, y_train, stratify=y_train, random_state=random_state, test_size=.8)



model_dict = {'LogisticRegression': LogisticRegression(penalty='l1', solver='saga', C=2, multi_class='multinomial', n_jobs=-1, random_state=random_state)

             , 'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=200, max_depth=3, min_samples_leaf=.06, n_jobs=-1, random_state=random_state)

              , 'RandomForestClassifier': RandomForestClassifier(n_estimators=20, max_depth=2, min_samples_leaf=.1, random_state=random_state, n_jobs=-1)

             }

estimator_dict = {}

importance_fatures_sorted_all = pd.DataFrame()

for model_name, model in model_dict.items():

    print('='*10, model_name, '='*10)

    model.fit(X_model, y_model)

    print('Accuracy in training:', accuracy_score(model.predict(X_model), y_model))

    print('Accuracy in valid:', accuracy_score(model.predict(X_valid), y_valid))

    importance_values = np.absolute(model.coef_) if model_name == 'LogisticRegression' else model.feature_importances_

    importance_fatures_sorted = pd.DataFrame(importance_values.reshape([-1, len(X_train2_sup.columns)]), columns=X_train2_sup.columns).mean(axis=0).sort_values(ascending=False).to_frame()

    importance_fatures_sorted.rename(columns={0: 'feature_importance'}, inplace=True)

    importance_fatures_sorted['ranking']= importance_fatures_sorted['feature_importance'].rank(ascending=False)

    importance_fatures_sorted['model'] = model_name

    print('Show top 10 important features:')

    display(importance_fatures_sorted.drop('model', axis=1).head(10))

    importance_fatures_sorted_all = importance_fatures_sorted_all.append(importance_fatures_sorted)

    estimator_dict[model_name] = model



plt.title('Feature importance ranked by number of features by model')

sns.lineplot(data=importance_fatures_sorted_all, x='ranking', y='feature_importance', hue='model')

plt.xlabel("Number of features selected")
selected_model = 'LogisticRegression'

number_of_features = 60

select_features_by_model = importance_fatures_sorted_all[importance_fatures_sorted_all['model'] == selected_model].index[:number_of_features].tolist()
%%time

#it takes much more time comparing 

rfecv = RFECV(estimator=model_dict['LogisticRegression'].set_params(max_iter=150, C=1), step=1, cv=StratifiedShuffleSplit(1, test_size=.2, random_state=random_state), scoring='accuracy', n_jobs=-1)

rfecv.fit(X_train2_sup[select_features_by_model], y_train)

plt.figure()

plt.title('Feature importance ranked by number of features by model')

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.plot(rfecv.n_features_, rfecv.grid_scores_[rfecv.n_features_-1], marker='o', label='Optimal number of feature')

plt.legend(loc='best')

plt.show()
rfecv_df = pd.DataFrame({'col': select_features_by_model})

rfecv_df['rank'] = np.nan

for index, support in enumerate(rfecv.get_support(indices=True)):

    rfecv_df.loc[support, 'rank'] = index

for index, rank in enumerate(rfecv.ranking_ -2):

    if rank >= 0:

        rfecv_df.loc[index, 'rank'] = rfecv.n_features_ + rank

rfecv_df