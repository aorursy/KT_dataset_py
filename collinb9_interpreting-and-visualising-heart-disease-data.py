import numpy as np



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

from matplotlib import style

style.use('ggplot')



import seaborn as sns



import sklearn

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from sklearn.kernel_ridge import KernelRidge 

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



import pandas as pd



import eli5

from eli5.sklearn import PermutationImportance



from IPython.display import display, Image



from pdpbox import pdp, info_plots



import shap



#my modules

from data_prep import *

from model_selection import *

from visualisation import *



def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn



random_state = 250

#Importing the data set

df = pd.read_csv('../input/heart-disease-uci/heart.csv')

#Change target vector so that 1 = heart disease, 0 = no heart disease.

df['target'].replace([0,1], [1,0], inplace = True)

#Remove any duplicate rows

df.drop(df[df.duplicated(keep = 'first')].index[0], inplace = True)

#Change the column names to be more descriptive

df = rename_columns(df)

df.head()
df.info()
for feature in ['fasting blood sugar > 120 mg/dl', 'chest pain type', 'rest ecg results', 'exercise induced angina', 'slope of peak ST (ecg)', 'major vessels', 'thalassemia']:

    print(df.groupby(feature)['target'].count())
df['major vessels'].replace([4],[0], inplace = True)

df['thalassemia'].replace([0], [2], inplace = True)
df.describe()
df.hist(bins=50, figsize = (20,15), alpha = 0.8)



plt.show()
#Splitting fatures into numerical, binary and categorical features

num_features, bin_features, cat_features = separate_num_bin_cat_features(df)
#Reordering the columns

df = df[num_features + bin_features + cat_features + ['target']].copy()
X = df.drop(['target'], axis = 1).copy()

Y = df['target'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y,

                                                    test_size = 0.2, random_state = random_state, 

                                                    stratify = Y) 

im = sns.pairplot(df, hue = 'target', vars = num_features, diag_kind = 'kde',

            kind = 'scatter', markers = 'o', size = 3, plot_kws = {'alpha': 0.4, 's': 100})

plt.show()
X_train_target = X_train.copy()

X_train_target['target'] = y_train

corr_matrix = X_train_target.corr()

corr_matrix['target']
X_train_target = X_train.copy()

X_train_target['target'] = y_train

corr_matrix = X_train_target.corr()

corr_matrix['target']



plot_corr_matrix(corr_matrix.abs())

plt.show()
fig, ax = plt.subplots(figsize = (8,5))



sns.boxplot(x = 'thalassemia',y = 'age', data = X_train_target, ax = ax)

plt.show()
def rename_cat_values(df):

    df['chest pain type'].replace([0,1,2,3],['asymptomatic cp','atypical angina cp','non-anginal cp','typical angina cp'],inplace = True)

    df['rest ecg results'].replace([0,1,2],['left ventricular hypertrophy (ecg)', 'normal ecg', 'abnormal ecg'],inplace = True)

    df['slope of peak ST (ecg)'].replace([0,1,2],['downsloping','flat','upsloping'],inplace = True)

    df['thalassemia'].replace([1,2,3], ['fixed thal','no thal','reversable thal'], inplace = True)

 
#We rename the values of the categorical features to improve interpretability later on.

rename_cat_values(X_train)

rename_cat_values(X_test)

X_train.tail(3)
X_cat = X_train[cat_features].copy()

one_hot_encoder = OneHotEncoder(categories = 'auto', drop = None)

X_cat_encoded = one_hot_encoder.fit_transform(X_cat)

cat_features_encoded = one_hot_encoder.categories_
encoded_features = num_features + bin_features + [cat_features_encoded[i][j] for i in range(len(cat_features_encoded)) for

                                   j in range(len(cat_features_encoded[i]))]

cat_pipeline = Pipeline([('one_hot', OneHotEncoder(categories = cat_features_encoded, drop = None)) ])

num_pipeline = Pipeline([('standardise', StandardScaler()) ])
prep_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_features),

    ('bin', FunctionTransformer(), bin_features),

    ('cat', cat_pipeline, cat_features)

])
X_train_prepped = prep_pipeline.fit_transform(X_train)

X_test_prepped = prep_pipeline.transform(X_test)
#It will later become useful to have the training set where the numerical

#features are unscaled, but the categorical features are one-hot encoded

only_cat_pipeline = ColumnTransformer([

    ('num', FunctionTransformer(), num_features),

    ('bin', FunctionTransformer(), bin_features),

    ('cat', cat_pipeline, cat_features)

])

X_train = only_cat_pipeline.fit_transform(X_train)
#Perform a grid search to find the best parameters

random_forest = optimise_random_forest_params(X_train_prepped, y_train, scoring = 'accuracy',

                                                cv = 5, n_estimators = [100])

log_reg = optimise_log_reg_params(X_train_prepped, y_train, scoring = 'accuracy', cv = 5)
print('Random Forest')

scoring = {'Accuracy': 'accuracy'}

#Returns performance measures of a classifier on the training set and via cross validation.

profile_clf(X_train_prepped, y_train, random_forest, scoring = scoring, cv = 5)
print('Logistic Regression')

profile_clf(X_train_prepped, y_train, log_reg, scoring = scoring, cv = 5)
print('Random Forest:')

display(eli5.show_weights(random_forest, feature_names = encoded_features))

print('Logistic Regression:')

display(eli5.show_weights(log_reg, feature_names = encoded_features))
log_reg.fit(X_train_prepped, y_train)

random_forest.fit(X_train_prepped, y_train)



perm_log_reg = PermutationImportance(log_reg, random_state = random_state, cv = 'prefit')

perm_random_forest = PermutationImportance(random_forest, random_state = random_state, cv = 'prefit')



perm_log_reg.fit(X_test_prepped, y_test)

perm_random_forest.fit(X_test_prepped, y_test)



print('Random Forest')

display(eli5.show_weights(perm_random_forest, feature_names = encoded_features))

print('Logistic Regression')

display(eli5.show_weights(perm_log_reg, feature_names = encoded_features))



random_forest.fit(X_train_prepped, y_train)
explainer = shap.TreeExplainer(random_forest, X_train_prepped, model_output = 'probability')
data_point = X_test_prepped[0].reshape(1,-1)
shap.initjs()

shap_value = explainer.shap_values(data_point)

shap.force_plot(explainer.expected_value[1], shap_value[1],data_point, feature_names = encoded_features)
shap_values = explainer.shap_values(X_test_prepped)

shap.force_plot(explainer.expected_value[1], shap_values[1], X_test_prepped, feature_names = encoded_features)
shap.summary_plot(shap_values[1], X_test_prepped, feature_names = encoded_features, show = False)

plt.show()
#The pdpbox module only accepts pandas dataframes as input

X_train_prepped = pd.DataFrame(X_train_prepped, columns = encoded_features)

X_test_prepped = pd.DataFrame(X_test_prepped, columns = encoded_features)

X_train = pd.DataFrame(X_train, columns = encoded_features)

plot_1d_pdp(random_forest, X_train_prepped, y_train,X_train, feature = 'resting blood pressure',model_features = encoded_features,

            plot_pts_dist = True, plot_lines = True, figsize = (10,8))

plt.show()
important_features = ['no thal','asymptomatic cp', 'reversable thal', 'exercise induced angina', 'major vessels', 'age']
for feature in important_features:

    plot_1d_pdp(random_forest, X_train_prepped, y_train,X_train, model_features = encoded_features,

        feature = feature, plot_pts_dist = True, plot_lines = True, figsize = (10,8))

    plt.show()
for feature in important_features:

    fig= plt.figure(figsize = (8,5))

    ax = plt.subplot(111)

    shap.dependence_plot(feature, shap_values[1], X_test_prepped,

                        feature_names = encoded_features,

                         interaction_index = None, ax = ax, show = False

                        )

    mean = X_train[feature].mean()

    std = X_train[feature].std()

    #Unscale x values

    def unscale_ticks(x, pos):

        return ('%.1f' %(x*std + mean))

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(unscale_ticks))

    plt.show()
#Removing ouliers so the curve fits better

idx = np.argsort(X_test_prepped['age'])

X_test_temp = X_test_prepped.drop(idx[:2])

shap_values_temp_age  = np.delete(shap_values[1][:,0], idx[:2])



#Fitting polynomial regression model

kr = KernelRidge(kernel = 'sigmoid', gamma = .15, alpha = .005)

kr.fit(np.array(X_test_temp['age']).reshape(-1,1), shap_values_temp_age)



mean = X_train['age'].mean()

std = X_train['age'].std()

#Unscale x values

def unscale_ticks(x, pos):

    return ('%.0f' %(x*std + mean))



fig, ax = plt.subplots(figsize = (8,5))

shap.dependence_plot('age', shap_values[1], X_test_prepped,

                    feature_names = encoded_features,

                     interaction_index = None, ax = ax, show = False

                    )





plot_predicted_curve(kr, axes = [-2, 2, -0.05, 0.04], ax = ax, c = 'c', alpha = 0.7)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(unscale_ticks))

plt.show()
for feature in [feature for feature in important_features if feature != 'age']:

    plot_2d_pdp(random_forest, X_train_prepped, y_train,X_train, model_features = encoded_features,

                         features = ['age', feature], figsize = (10,8))    

    plt.show()
for feature in important_features:

    fig= plt.figure(figsize = (8,5))

    ax = plt.subplot(111)

    shap.dependence_plot(feature, shap_values[1], X_test_prepped,

                        feature_names = encoded_features,

                         interaction_index = 'sex', ax = ax, show = False, x_jitter = 0.2

                        )

    mean = X_train[feature].mean()

    std = X_train[feature].std()

    #Unscale x values

    def unscale_ticks(x, pos):

        return ('%.1f' %(x*std + mean))

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(unscale_ticks))

    plt.show()
feature = 'age'

shap_values = explainer.shap_values(X_train_prepped)

for feature in important_features:

    fig, ax = plt.subplots(figsize = (8,5))



    shap.dependence_plot(feature, shap_values[1], X_train_prepped,

                        feature_names = encoded_features,

                         interaction_index = 'sex', ax = ax, show = False, x_jitter = 0.2

                        )

    mean = X_train[feature].mean()

    std = X_train[feature].std()

    #Unscale x values

    def unscale_ticks(x, pos):

        return ('%.1f' %(x*std + mean))

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(unscale_ticks))

    plt.show()