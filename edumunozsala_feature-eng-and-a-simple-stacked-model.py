# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import gc

import seaborn as sns



from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict, cross_val_score



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier

from lightgbm import LGBMClassifier, plot_importance

from xgboost import XGBClassifier



from sklearn.decomposition import PCA



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

already_preprocessed=False

train_file='train.csv'

test_file='test.csv'



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        if filename==train_file:

            train_dir=dirname

        if filename==test_file:

            test_dir=dirname

            

# Any results you write to the current directory are saved as output.
import pip._internal as pip

pip.main(['install', '--upgrade', 'numpy==1.17.2'])



import numpy as np

print('> NumPy version: {}'.format(np.__version__))
# Reading the csv file with the whole dataset

data = pd.read_csv(train_dir+'/'+train_file, sep=',', header=0, index_col=0)

# Train data, erasing label column

X_train = data.drop('Cover_Type', axis=1)

y_train = data['Cover_Type']

# Read the test data

test = pd.read_csv(test_dir+'/'+test_file, sep=',', header=0, index_col=0)



n_train=len(X_train)
from sklearn.preprocessing import StandardScaler

all_data=X_train.append(test)



scaler = StandardScaler().fit(all_data)

test_scaled = scaler.transform(test)

X_train_scaled = scaler.transform(X_train)
from sklearn.mixture import GaussianMixture

# Num clusters o components?  

# Standarscaler for test data?



gmix = GaussianMixture(n_components=10)

gmix.fit(test_scaled)



x_train_gmix = gmix.predict(X_train_scaled)

test_gmix = gmix.predict(test_scaled)
X_train['Cluster_Test']=x_train_gmix

test['Cluster_Test']=test_gmix



#Convert to categorical

X_train = pd.get_dummies(X_train,columns = ['Cluster_Test'])

test = pd.get_dummies(test,columns = ['Cluster_Test'])
#Apply PCA to obtain new features based in all datasets (train and test)

def add_PCA_features(X):

    pca = PCA(n_components=0.99, random_state=0).fit(X)

    X_pca = pca.transform(X)

    

    return X_pca



components = add_PCA_features(all_data)



print('PCA components dimension: ',components.shape)



for i in range(components.shape[1]):

    col_name= 'pca'+str(i+1)

    X_train[col_name] = components[:n_train, i]

    test[col_name] = components[n_train:, i]



print(test.shape,X_train.shape)
# Plot the feature importance determined by the classifier clf

def feature_importances(clf, X, y):

    clf = clf.fit(X, y)

    

    importances = pd.DataFrame({'Features': X.columns, 

                                'Importances': clf.feature_importances_})

    

    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



    fig = plt.figure(figsize=(14, 4))

    sns.barplot(x='Features', y='Importances', data=importances)

    plt.xticks(rotation='vertical')

    plt.show()

    

# Calculate the accuracy using a cross validated approach 

def cv_accuracy(clf, X, y):

    scores = cross_val_score(clf, X, y, 

                             cv=5, 

                             scoring='accuracy',

                             verbose=0, 

                             n_jobs=-1)

    

    print('Cross-validation accuracy: {:.3f} {}'.format(np.mean(scores),  

                                                        np.around(scores, 3)))



# Shows the confusion matrix using criss validated predictions

def cv_confusion(clf, X, y):

    prediction = cross_val_predict(clf, X, y, 

                                   cv=5, 

                                   verbose=0, 

                                   n_jobs=-1)

    

    classes = sorted(y.unique().tolist())



    conf_mat = confusion_matrix(y, prediction)

    norm_conf_mat = np.around(conf_mat / conf_mat.sum(axis=1), 2)



    fig = plt.figure(figsize=(14, 8))



    fig.add_subplot(1, 2, 1)

    ax = sns.heatmap(norm_conf_mat, annot=True, cmap='OrRd', 

                     xticklabels=classes, yticklabels=classes)

    ax.set(xlabel='Predicted Class', ylabel='True Class', title='Normalized')





    fig.add_subplot(1, 2, 2)

    ax = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='OrRd', 

                     xticklabels=classes, yticklabels=classes)

    ax.set(xlabel='Predicted Class', ylabel='True Class', title ='Counts')



    #plt.tight_layout()

    plt.show()

    

# Plot correlations between numerical features

def plot_correlations(X, annot=False):

    numericals = [col for col in X.columns if 

                  'Soil_Type' not in col and 

                  'Wilderness_Area' not in col]



    numericals = sorted(numericals)



    # Place correlations in four bins

    corr = np.around(X[numericals].corr().abs(), 1)

    

    corr_bin = corr.copy()

    corr_bin = corr_bin.where(corr_bin > 0.30, 0.30)



    corr_bin = corr_bin.where((corr_bin <= 0.30) | 

                              (corr_bin > 0.50), 0.50)



    corr_bin = corr_bin.where((corr_bin <= 0.50) | 

                              (corr_bin > 0.70), 0.70)



    corr_bin = corr_bin.where(corr_bin <= 0.70, 1.0)

    

    if annot:

        annot = corr

        

    # Show binned correlation plot

    fig = plt.figure(figsize=(12, 12))

    sns.heatmap(corr_bin, annot=annot, linewidths=1, square=True, 

                cmap='BuPu', cbar_kws={'shrink':0.5})

    plt.title('Feature Correlations')

    plt.show()



# Drop features with mode frequency > 99% of data

# Those columns are irrelevant, they have almost just one value 

def drop_unimportant(X_):

    X = X_.copy()

    

    n_rows = X.shape[0]

    hi_freq_cols = []

    

    for col in X.columns:

        mode_frequency = 100.0 * X[col].value_counts().iat[0] / n_rows 

        

        if mode_frequency > 99.0:

            hi_freq_cols.append(col)

            

    X = X.drop(hi_freq_cols, axis='columns')

    

    return hi_freq_cols,X



def drop_correlated(X_):

    X = X_.copy()

   

    drop_cols=['Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology',

              'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Roadways']

    

    drop_cols = [col for col in drop_cols if col in X.columns]

    

    X = X.drop(drop_cols, axis='columns')

    

    return drop_cols,X
#This classifier is used to evaluate the performance of the features in diferent scenarios: importance, correlation and so.

clf = RandomForestClassifier(n_estimators=125,

                             min_samples_leaf=1,

                             max_depth=None,

                             verbose=0,

#                             class_weight ={1:0.4,2:0.4,3:0.05,4:0.05,5:0.05,6:0.05,7:0.01},

                             random_state=0)
#feature_importances(clf, X_train, y_train)

#cv_accuracy(clf, X_train, y_train)

#cv_confusion(clf, X_train, y_train)
# This new features has been extracted from the kernel previously citated:

# https://www.kaggle.com/kwabenantim/forest-cover-stacking-multiple-classifiers



def add_features(X_):

    X = X_.copy()

    

    X['Hydro_Elevation_sum'] = X[['Elevation',

                                  'Vertical_Distance_To_Hydrology']

                                 ].sum(axis='columns')

    

    X['Hydro_Elevation_diff'] = X[['Elevation',

                                   'Vertical_Distance_To_Hydrology']

                                  ].diff(axis='columns').iloc[:, [1]]



    X['Hydro_Euclidean'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 +

                                   X['Vertical_Distance_To_Hydrology']**2)



    X['Hydro_Manhattan'] = (X['Horizontal_Distance_To_Hydrology'] +

                            X['Vertical_Distance_To_Hydrology'].abs())

    

    

    X['Hydro_Distance_sum'] = X[['Horizontal_Distance_To_Hydrology',

                                 'Vertical_Distance_To_Hydrology']

                                ].sum(axis='columns')



    X['Hydro_Distance_diff'] = X[['Horizontal_Distance_To_Hydrology',

                                  'Vertical_Distance_To_Hydrology']

                                 ].diff(axis='columns').iloc[:, [1]]

    

    X['Hydro_Fire_sum'] = X[['Horizontal_Distance_To_Hydrology',

                             'Horizontal_Distance_To_Fire_Points']

                            ].sum(axis='columns')



    X['Hydro_Fire_diff'] = X[['Horizontal_Distance_To_Hydrology',

                              'Horizontal_Distance_To_Fire_Points']

                             ].diff(axis='columns').iloc[:, [1]].abs()



    X['Hydro_Fire_mean'] = X[['Horizontal_Distance_To_Hydrology',

                              'Horizontal_Distance_To_Fire_Points']

                             ].mean(axis='columns')



    X['Hydro_Fire_median'] = X[['Horizontal_Distance_To_Hydrology',

                                'Horizontal_Distance_To_Fire_Points']

                               ].median(axis='columns')

                               

    X['Hydro_Road_sum'] = X[['Horizontal_Distance_To_Hydrology',

                             'Horizontal_Distance_To_Roadways']

                            ].sum(axis='columns')



    X['Hydro_Road_diff'] = X[['Horizontal_Distance_To_Hydrology',

                              'Horizontal_Distance_To_Roadways']

                             ].diff(axis='columns').iloc[:, [1]].abs()



    X['Hydro_Road_mean'] = X[['Horizontal_Distance_To_Hydrology',

                              'Horizontal_Distance_To_Roadways']

                             ].mean(axis='columns')



    X['Hydro_Road_median'] = X[['Horizontal_Distance_To_Hydrology',

                                'Horizontal_Distance_To_Roadways']

                               ].median(axis='columns')

    

    X['Road_Fire_sum'] = X[['Horizontal_Distance_To_Roadways',

                            'Horizontal_Distance_To_Fire_Points']

                           ].sum(axis='columns')



    X['Road_Fire_diff'] = X[['Horizontal_Distance_To_Roadways',

                             'Horizontal_Distance_To_Fire_Points']

                            ].diff(axis='columns').iloc[:, [1]].abs()



    X['Road_Fire_mean'] = X[['Horizontal_Distance_To_Roadways',

                             'Horizontal_Distance_To_Fire_Points']

                            ].mean(axis='columns')



    X['Road_Fire_median'] = X[['Horizontal_Distance_To_Roadways',

                               'Horizontal_Distance_To_Fire_Points']

                              ].median(axis='columns')

    

    X['Hydro_Road_Fire_mean'] = X[['Horizontal_Distance_To_Hydrology',

                                   'Horizontal_Distance_To_Roadways',

                                   'Horizontal_Distance_To_Fire_Points']

                                  ].mean(axis='columns')



    X['Hydro_Road_Fire_median'] = X[['Horizontal_Distance_To_Hydrology',

                                     'Horizontal_Distance_To_Roadways',

                                     'Horizontal_Distance_To_Fire_Points']

                                    ].median(axis='columns')



    X['Hillshade_sum'] = X[['Hillshade_9am',

                            'Hillshade_Noon',

                            'Hillshade_3pm']

                           ].sum(axis='columns')



    X['Hillshade_mean'] = X[['Hillshade_9am',

                             'Hillshade_Noon',

                             'Hillshade_3pm']

                            ].mean(axis='columns')



    X['Hillshade_median'] = X[['Hillshade_9am',

                               'Hillshade_Noon',

                               'Hillshade_3pm']

                              ].median(axis='columns')



    X['Hillshade_min'] = X[['Hillshade_9am',

                            'Hillshade_Noon',

                            'Hillshade_3pm']

                           ].min(axis='columns')



    X['Hillshade_max'] = X[['Hillshade_9am',

                            'Hillshade_Noon',

                            'Hillshade_3pm']

                           ].max(axis='columns')

    

    X['Hillshade_std'] = X[['Hillshade_9am',

                            'Hillshade_Noon',

                            'Hillshade_3pm']

                           ].std(axis='columns')

    

    # Compute Soil_Type number from Soil_Type binary columns

    X['Stoneyness'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))

    

    # For all 40 Soil_Types, 1=rubbly, 2=stony, 3=very stony, 4=extremely stony, 0=?

    stoneyness = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1, 

                  1, 2, 1, 0, 0, 0, 0, 3, 0, 0, 

                  0, 4, 0, 4, 4, 3, 4, 4, 4, 4, 

                  4, 4, 4, 4, 1, 4, 4, 4, 4, 4]

    

    # Replace Soil_Type number with "stoneyness" value

    X['Stoneyness'] = X['Stoneyness'].replace(range(1, 41), stoneyness)



    return X
#Aspect

def transform_Aspect(X_):

    X = X_.copy()

    

    X['Aspect'] = X['Aspect'].astype(int) % 360

    

    from bisect import bisect

    

    cardinals = [i for i in range(45, 361, 90)]

    points = ['N', 'E', 'S', 'W']

    

    X['Cardinal'] = X.Aspect.apply(lambda x: points[bisect(cardinals, x) % 4])

    X.loc[:,'North']= X['Cardinal']=='N'

    X.loc[:,'East']= X['Cardinal']=='E'

    X.loc[:,'West']= X['Cardinal']=='W'

    X.loc[:,'South']= X['Cardinal']=='S'

    

    #X['Sin_Aspect'] = np.sin(np.radians(X['Aspect'])) # not important feature at all

    X['Cos_Aspect'] = np.cos(np.radians(X['Aspect']))

    

    return X



def transform_Slope(X_):

    X = X_.copy()

    

    X['Slope_hyd'] = np.arctan(X['Vertical_Distance_To_Hydrology']/(X['Horizontal_Distance_To_Hydrology']+0.001))

    X.Slope_hyd=X.Slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    

    X['Sin_Slope_hyd'] = np.sin(np.radians(X['Slope_hyd']))

    X['Cos_Slope_hyd'] = np.cos(np.radians(X['Slope_hyd']))



    #X['Sin_Slope'] = np.sin(np.radians(X['Slope'])) # not important feature at all

    X['Cos_Slope'] = np.cos(np.radians(X['Slope']))

    

    return X
# Add the new features to the dataset

X_train = add_features(X_train)

test = add_features(test)



print(test.shape,X_train.shape)



feature_importances(clf, X_train, y_train)

cv_accuracy(clf, X_train, y_train)
#Convert to categorical

X_train = pd.get_dummies(X_train,columns = ['Stoneyness'])

test = pd.get_dummies(test,columns = ['Stoneyness'])

print(test.shape,X_train.shape)
X_train.drop(['Soil_Type{}'.format(i) for i in range(1, 41)], axis='columns', errors='ignore', inplace=True)

test.drop(['Soil_Type{}'.format(i) for i in range(1, 41)], axis='columns', errors='ignore', inplace=True)

print(test.shape,X_train.shape)
# Add the new features to the dataset

#X_train = transform_Aspect(X_train)

#test = transform_Aspect(test)



# Dropping Cardinal columns

#X_train.drop('Cardinal',axis=1, inplace=True)

#test.drop('Cardinal',axis=1, inplace=True)



#print(test.shape,X_train.shape)

#cv_accuracy(clf, X_train, y_train)
# Add the new features to the dataset

#X_train = transform_Slope(X_train)

#test = transform_Slope(test)



#print(test.shape,X_train.shape)

#cv_accuracy(clf, X_train, y_train)
#Only execute in training or evaluating new features



#feature_importances(clf, X_train, y_train)

#cv_accuracy(clf, X_train, y_train)

#cv_confusion(clf, X_train, y_train)
# Drop umportant columns from train data

unimportant_cols,X_train = drop_unimportant(X_train)

# Drop umportant columns from test data

test = test.drop(unimportant_cols, axis='columns')

print(test.shape,X_train.shape)
#Only execute in training or evaluating new features

#cv_accuracy(clf, X_train, y_train)
plot_correlations(X_train, annot=False)

#Drop correlated columns from test data

dropped_cols, X_train = drop_correlated(X_train)

#Drop correlated columns from test data

test = test.drop(dropped_cols, axis='columns')

print(test.shape,X_train.shape)
#Lets evaluate our final dataset

#Only execute in training or evaluating new features



#cv_accuracy(clf, X_train, y_train)

#feature_importances(clf, X_train, y_train)

#cv_confusion(clf, X_train, y_train)
#Only executed when new combination of features is created



#df1 = X_train.assign(Cover_Type=y_train)

#df1.to_csv('train_fe.csv', sep=',', header=True, index=True, index_label='Id')

#test.to_csv('test_fe.csv', sep=',', header=True, index=True, index_label='Id')
# Tranform to numpy array of float type

X = X_train.values.astype('float64')

y = y_train.values.ravel()

test_ds= test.values.astype('float64')



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

print('X: ', X.shape)

print('y: ', y.shape)

print('test: ', test_ds.shape)
#Define a ratio for every class weighted

count = { 1: 0.37062,

 2: 0.49657,

 3: 0.05947,

 4: 0.00106,

 5: 0.01287, 

 6: 0.02698, 

 7: 0.03238} 



count_rf = { 1: 0.37062,

 2: 0.49657,

 3: 0.05947,

 4: 0.00106,

 5: 0.01287, 

 6: 0.02698} 



weight = [count[x]/(sum(count.values())) for x in range(1,7+1)]

class_weight_lgbm = {i: v for i, v in enumerate(weight)}
#Estimators: 400, 400, 300, 400, 250 (reduce to 100 100 100 100 for some tests)

models = {

    'Random Forest': RandomForestClassifier(criterion = 'gini',n_estimators=750, max_depth=None, min_samples_split=2, min_samples_leaf=1, 

                                    max_leaf_nodes=None,random_state = 0, class_weight = count_rf),

    'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion = 'gini', max_depth=None, min_samples_split=2,

                                                                         min_samples_leaf=1,max_leaf_nodes=None,max_features='auto',

                                                                         random_state = 0, class_weight = count_rf),

                                   n_estimators=750,learning_rate=0.2,random_state=0),

    

    'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion = 'gini', max_depth=None, min_samples_split=2,

                                                                         min_samples_leaf=1,max_leaf_nodes=None,max_features='auto',

                                                                         random_state = 0, class_weight = count_rf),

                                   n_estimators=500,max_features=0.75, max_samples=1.0, random_state=0,n_jobs=-1,verbose=0),

    #{'max_features': 0.75, 'max_samples': 1.0, 'n_estimators': 300}

    #'GBM': GradientBoostingClassifier(n_estimators=500, learning_rate= 0.2, max_depth=10, min_samples_leaf=1, 

    #                                  min_samples_split=2,random_state=0,verbose=1),

    'LGBM': LGBMClassifier(n_estimators=500, learning_rate= 0.1, objective= 'multiclass', num_class=7, random_state= 0, 

                           n_jobs=-1, class_weight = class_weight_lgbm),

    #'LGBM': LGBMClassifier(n_estimators=300, num_leaves=100, verbosity=0, random_state=0,n_jobs=-1),

    #'KNN': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),

    # 'XGB': XGBClassifier(n_estimator= 200, learning_rate= 0.1, max_depth= 50, objective= 'binary:logistic',random_state= 0,n_jobs=-1),

    'Extra Tree': ExtraTreesClassifier(criterion = 'gini', n_estimators=400, max_depth=None, min_samples_split=2,min_samples_leaf=1, 

                                      max_leaf_nodes=None,oob_score=False, warm_start=True, random_state = 0, 

                                      class_weight = count_rf)

}
from sklearn.model_selection import KFold, cross_val_score



# Create the splits for cross validation

cv = KFold(n_splits=5, shuffle=True, random_state=0)
def cross_validate_L1_Clfs(models,X,y,cv): 

    results= dict()

    for name, model in models.items():

        print('Evaluating Model: ',name)

        cv_results = cross_val_score(model, X, y,

                                    cv=cv, 

                                    scoring='accuracy')

        results[name] = cv_results

        print(name, 'Accuracy Mean {0:.4f}, Std {1:.4f}'.format(

                  cv_results.mean(), cv_results.std()))



    accuracies= dict()

    for name, accs in results.items():

        accuracies[name]=accs.mean()

    

    best_model=max(accuracies, key=accuracies.get)

    print('Best Model: ',best_model,' Accuracy: ',accuracies[best_model])

    

    return accuracies, best_model
#Cross Validate the L1 classifiers to extract the best model (Execute only on evaluation time)

# accs, best_model = cross_validate_L1_Clfs(models,X,y,cv)

#meta_model=models[best_model]

# Estimators: 150

meta_model=RandomForestClassifier(criterion = 'entropy',n_estimators=250, max_depth=None, min_samples_split=2, min_samples_leaf=1, 

                                    max_leaf_nodes=None,random_state = 0, class_weight = count)

#meta_model=LogisticRegression(max_iter=1000, n_jobs=-1, solver= 'lbfgs',multi_class = 'multinomial')
from mlxtend.classifier import StackingCVClassifier



clfs = [m for m in models.values()]



stacked_model = StackingCVClassifier(classifiers=clfs,

                             meta_classifier=meta_model,

                             cv=cv,

                             use_probas=True,

                             drop_last_proba=False,

                             use_features_in_secondary=True,

                             verbose=1,

                             #store_train_meta_features=True,

                             random_state=0,

                             n_jobs=-1)

#It is time for evaluating the stacked model (Executed only in training or evluation time)

#scores = cross_val_score(stacked_model, X, y, cv=cv, scoring='accuracy', verbose=0)



#print('Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))
# Fit and predict the stacked model on both train and test data

stacked_model.fit(X, y)

predictions = stacked_model.predict(X)

predictions_test = stacked_model.predict(test_ds)

print('Stacked Model Accuracy: ',round(accuracy_score(y, predictions),4))
submission = pd.DataFrame({ 'Id': test.index.values,

                            'Cover_Type': predictions_test })

submission.to_csv("submission_data.csv", index=False)