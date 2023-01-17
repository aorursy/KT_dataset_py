!pip install prince
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyo
import plotly.graph_objs as go
import prince
import warnings
warnings.filterwarnings("ignore")
# Set notebook mode to work in offline
pyo.init_notebook_mode()
from missingno import missingno

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
pio.templates.default = 'plotly_dark'
df = pd.read_csv('/kaggle/input/adult-income-dataset/adult.csv')
df.head()
types_count = df.dtypes.value_counts()
types_count.index = ['Categorical', 'Numeric']
types_count
df.income.value_counts()
df.workclass.value_counts()
df = df.replace("?", np.nan)
missingno.bar(df)
plt.show()
categorical = df.select_dtypes(include=['object'])
categorical.nunique()
df.groupby('education').agg({'educational-num': 'last'}).sort_values('educational-num').T
px.histogram(df.sort_values('educational-num'), 'education', color='income', barnorm='fraction', histnorm='percent', color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
def reduce_education(x):
    if x == 1:
        return 'Preschool'
    elif 1 < x < 9:
        return 'School'
    elif 8 < x < 13:
        return 'College_Assoc'
    elif 12 < x < 14:
        return 'Bachelors_Masters'
    else:
        return 'Prof_Doctorate'

df.education = df['educational-num'].apply(reduce_education)
new_education_enum = {'Preschool': 1, 'School': 2, 'College_Assoc': 3, 'Bachelors_Masters': 4, 'Prof_Doctorate': 5}       
df['educational-num'] = df.education.replace(new_education_enum).astype(int)
px.histogram(df.sort_values('educational-num'), 'education', color='income', barnorm='fraction', 
                                                     color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
regions_parser = {'United-States': 'United-States',
                 'Mexico': 'Latin America',
                 'Philippines': 'Asia',
                 'Germany': 'Europe',
                 'Puerto-Rico': 'Latin America',
                 'Canada': 'North America',
                 'El-Salvador': 'Latin America',
                 'India': 'Asia',
                 'Cuba': 'Latin America',
                 'England': 'Europe',
                 'China': 'Asia',
                 'South': 'Africa',
                 'Jamaica': 'Latin America',
                 'Italy': 'Europe',
                 'Dominican-Republic': 'Latin America',
                 'Japan': 'Asia',
                 'Guatemala': 'Latin America',
                 'Poland': 'Europe',
                 'Vietnam': 'Asia',
                 'Columbia': 'Latin America',
                 'Haiti': 'Latin America',
                 'Portugal': 'Europe',
                 'Taiwan': 'Asia',
                 'Iran': 'Middle-East',
                 'Greece': 'Europe',
                 'Nicaragua': 'Latin America',
                 'Peru': 'Latin America',
                 'Ecuador': 'Latin America',
                 'France': 'Europe',
                 'Ireland': 'Europe',
                 'Hong': 'Asia',
                 'Thailand': 'Asia',
                 'Cambodia': 'Asia',
                 'Trinadad&Tobago': 'Latin America',
                 'Laos': 'Asia',
                 'Outlying-US(Guam-USVI-etc)': 'United-States',
                 'Yugoslavia': 'Europe',
                 'Scotland': 'Europe',
                 'Honduras': 'Latin America',
                 'Hungary': 'Europe',
                 'Holand-Netherlands': 'Europe'}

df['native-country'] = df['native-country'].replace(regions_parser)
px.histogram(df[df['native-country']!='United-States'], 'native-country', barnorm='fraction', color='income', color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
px.histogram(df, 'relationship', color='income', barnorm='fraction',  color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
px.histogram(df, 'marital-status', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
# px.histogram(df, 'relationship', facet_col='marital-status', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
px.histogram(df, 'workclass', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
px.histogram(df, 'occupation', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
px.histogram(df, 'race', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
px.histogram(df, 'gender', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r).update_layout(bargap=0.02)
px.histogram(df, 'age', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=10).update_layout(bargap=0.02).update_layout(bargap=0.02)
def convert_age(x):
    if x <= 20:
        return '<= 20'
    elif 20 < x <= 30:
        return '20 < x <= 30'
    elif 30 < x <= 40:
        return '30 < x <= 40'
    elif 40 < x <= 60:
        return '30 < x <= 60'
    else:
        return '> 60'
df['age_cat'] = df.age.apply(convert_age)
fig_age = px.histogram(df.sort_values('age'), 'age_cat', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=10).update_layout(bargap=0.02)
fig_age
px.histogram(df, 'capital-gain', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=10).update_layout(bargap=0.02)
def convert_gain(x):
    if x <= 9e3:
        return '<= 9k'
    elif 9e3 < x <= 30e3:
        return '9k < x <= 30k'
    elif 30e3 < x <= 90e3:
        return '30k < x <= 90k'
    else:
        return '> 90k'
df['capital-gain_cat'] = df['capital-gain'].apply(convert_gain)
fig_gain = px.histogram(df, 'capital-gain_cat', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=10)
fig_gain
px.histogram(df, 'capital-loss', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=5).update_layout(bargap=0.02)
def convert_loss(x):
    if x <= 1e3:
        return '<= 1k'
    elif 1e3 < x <= 4e3:
        return '1k < x <= 4k'
    else:
        return '> 4k'
df['capital-loss_cat'] = df['capital-loss'].apply(convert_loss)
fig_loss = px.histogram(df, 'capital-loss_cat', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=10)
fig_loss
px.histogram(df, 'hours-per-week', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=10).update_layout(bargap=0.02)
def convert_hours(x):
    if x <= 40:
        return '<= 40'
    else:
        return '> 40'
df['hours-per-week_cat'] = df['hours-per-week'].apply(convert_hours)
fig_hours = px.histogram(df, 'hours-per-week_cat', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=10)
fig_hours
px.histogram(df, 'fnlwgt', color='income', barnorm='fraction', color_discrete_sequence=px.colors.qualitative.Safe_r, nbins=10).update_layout(bargap=0.02)
input_variables = [
    'age_cat',
    'capital-gain_cat', 
    'capital-loss_cat',
    'hours-per-week_cat',
    'workclass',
    'education',
    'marital-status',
    'race',
    'native-country',
    'occupation',
    'gender',
    
]

X = df[input_variables]
fill_mode = lambda col: col.fillna(col.mode()[0])
X = X.apply(fill_mode, axis=0)
from sklearn.naive_bayes import CategoricalNB, BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from category_encoders import OneHotEncoder
from sklearn.pipeline import Pipeline
import scikitplot as skplt

y = df.income.replace({'<=50K': 0, '>50K': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
ohce_nb = Pipeline(steps=[('ohe', OneHotEncoder()), ('naive-bayes', CategoricalNB())])

ohce_nb.fit(X_train, y_train)

y_probas = ohce_nb.predict_proba(X_test)
y_pred = ohce_nb.predict(X_test)
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
skplt.metrics.plot_roc_curve(y_test.values, y_probas, ax=ax[0, 0])
skplt.metrics.plot_precision_recall_curve(y_test, y_probas, ax=ax[0, 1])
skplt.metrics.plot_calibration_curve(y_test, y_probas.transpose().tolist(), ax=ax[1, 0])
skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax=ax[1, 1])
plt.show()
pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
from sklearn.model_selection import GridSearchCV
%%time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

ohce_nn = Pipeline(steps=[('ohe', OneHotEncoder()), ('knn', KNeighborsClassifier(n_jobs=-1))])

k_values = [300]

param_grid ={
    'knn__n_neighbors': k_values,
    'knn__metric': ['cosine']
}

search = GridSearchCV(ohce_nn, param_grid, scoring='neg_log_loss', cv=10, n_jobs=-1, return_train_score=True, verbose=1)

search.fit(X_train, y_train)

y_probas = search.predict_proba(X_test)
y_pred = search.predict(X_test)
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
skplt.metrics.plot_roc_curve(y_test.values, y_probas, ax=ax[0, 0])
skplt.metrics.plot_precision_recall_curve(y_test, y_probas, ax=ax[0, 1])
skplt.metrics.plot_calibration_curve(y_test, y_probas.transpose().tolist(), ax=ax[1, 0])
skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax=ax[1, 1])
plt.show()
pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
results = search.cv_results_

test_score = np.abs(results['mean_test_score'])
train_score = np.abs(results['mean_train_score'])
k_best = search.best_params_['knn__n_neighbors']
test_result_df = pd.DataFrame({'k_values': k_values, 'score': test_score, 'label': ['validation_score']*len(k_values)})
train_result_df = pd.DataFrame({'k_values': k_values, 'score': train_score, 'label': ['train_score']*len(k_values)})
test_result_df.append(train_result_df)
fig = px.line(test_result_df.append(train_result_df), x='k_values', y='score', color='label')
fig.update_layout(xaxis_type="log")
fig.data[0].update(mode='markers+lines')
fig.data[1].update(mode='markers+lines')
fig.show()
# from sklearn.model_selection import cross_val_score

# %%time
# cross_val_score_dict = dict()
# for k in [300]:
#     clf = ohce_nn = Pipeline(steps=[('ohe', OneHotEncoder()), ('knn', KNeighborsClassifier(n_neighbors=k, n_jobs=-1))])
#     cross_val_score_dict[k] = cross_val_score(clf, X_train, y_train, cv=30)

# cross_val_score_dict

# for k in [1000]:
#     clf = ohce_nn = Pipeline(steps=[('ohe', OneHotEncoder()), ('knn', KNeighborsClassifier(n_neighbors=k, n_jobs=-1))])
#     cross_val_score_dict[k] = cross_val_score(clf, X_train, y_train, cv=30)

# def wald_statistic(a, b):
#     A = np.mean(a - b)
#     N = len(a)    
    
#     return A / np.sqrt((1 / (N * (N - 1))) * sum((a - b - A) ** 2))

# print('k=300:k=200 ->', wald_statistic(cross_val_score_dict[300], cross_val_score_dict[200]))
# # print('k=300:k=400 ->', wald_statistic(cross_val_score_dict[300], cross_val_score_dict[400]))
# # print('k=300:k=500 ->', wald_statistic(cross_val_score_dict[300], cross_val_score_dict[500]))
# # print('k=300:k=1000 ->', wald_statistic(cross_val_score_dict[300], cross_val_score_dict[1000]))
mca = prince.MCA(n_components=3, random_state=42)
X_mca = mca.fit_transform(X)
X_mca = X_mca.merge(X, how='left', left_index=True, right_index=True).merge(df['income'].to_frame(), how='left', left_index=True, right_index=True)
px.scatter_3d(X_mca.sample(n=1000), 0, 1, 2, hover_data=X.columns, color='income', color_discrete_sequence=px.colors.qualitative.Safe_r)
def plot_decision_boundary(X, y, clf):

    clf.fit(X, y)

    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contour(xx, yy, Z, cmap=plt.cm.Dark2)
    ax.scatter(X[:, 0],X[:, 1], marker = 'o', c=y)
    
plot_decision_boundary(X_mca.iloc[:,:2].values, y, GaussianNB())
plot_decision_boundary(X_mca.iloc[:,:2].values, y, KNeighborsClassifier(n_neighbors=k_best, n_jobs=-1))
plot_decision_boundary(X_mca.iloc[:,:2].values, y, KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
plot_decision_boundary(X_mca.iloc[:,:2].values, y, KNeighborsClassifier(n_neighbors=3, n_jobs=-1))
# # PCA

# from sklearn.decomposition import PCA

# pca = PCA(n_components=3)

# X_decomp = pca.fit_transform(OneHotEncoder().fit_transform(X))

# X_decomp = pd.DataFrame(X_decomp)

# X_decomp['income'] = df.income

# X_decomp = X_decomp.merge(X, how='left', left_index=True, right_index=True)
# px.scatter_3d(X_decomp.sample(n=1000), 0, 1, 2, hover_data=X.columns, color='income', color_discrete_sequence=px.colors.qualitative.Safe_r)

# plot_decision_boundary(X_decomp.iloc[:,:2].values, y, GaussianNB())

# plot_decision_boundary(X_decomp.iloc[:,:2].values, y, KNeighborsClassifier(n_jobs=-1))

# # UMAP

# import umap

# X_decomp = umap.UMAP(n_components=3).fit_transform(OneHotEncoder().fit_transform(X))

# X_decomp = pd.DataFrame(X_decomp)

# X_decomp['income'] = df.income

# X_decomp = X_decomp.merge(X, how='left', left_index=True, right_index=True)
# px.scatter_3d(X_decomp.sample(n=1000), 0, 1, 2, hover_data=X.columns, color='income', color_discrete_sequence=px.colors.qualitative.Safe_r)

# plot_decision_boundary(X_decomp.iloc[:,:2].values, y, GaussianNB())

# plot_decision_boundary(X_decomp.iloc[:,:2].values, y, KNeighborsClassifier(n_jobs=-1))