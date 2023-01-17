import pandas as pd

import numpy as np

import sklearn

import imblearn

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
matplotlib.rcParams[ 'figure.dpi' ] = 350.0

matplotlib.rcParams[ 'axes.linewidth' ] = 1.0

matplotlib.rcParams[ 'axes.grid' ] = True

matplotlib.rcParams[ 'legend.borderpad' ] = 0.5

matplotlib.rcParams[ 'legend.framealpha' ] = 1.0

matplotlib.rcParams[ 'legend.frameon' ] = True

matplotlib.rcParams[ 'legend.fancybox' ] = False

matplotlib.rcParams[ 'legend.borderaxespad' ] = 0.5

matplotlib.rcParams[ 'grid.linewidth' ] = 1.0

matplotlib.rcParams[ 'grid.alpha' ] = 0.5

matplotlib.rcParams[ 'grid.linestyle' ] = '--'

matplotlib.rcParams[ 'lines.linewidth' ] = 2.0



import warnings

warnings.filterwarnings( 'ignore' )
RANDOM_STATE_SEED = 0
df = pd.read_csv( '../input/pima-indians-diabetes-database/diabetes.csv' )

cols = [ "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Diabetes" ]

df.columns = cols
df.info()
df.head()
df.hist( bins=40, figsize=( 7.0, 5.0 ) )

plt.tight_layout( True )

plt.show()
df.drop( "Diabetes", axis=1 ).isin( [ 0 ] ).sum()
columns_with_missing_values = [ "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI" ]



for col in columns_with_missing_values:

    df[ col ] = df[ col ].replace( to_replace=0, value=np.NaN )
df.hist( bins=40, figsize=( 7.0, 5.0 ) )

plt.tight_layout( True )

plt.show()
num_diabetes = df[ "Diabetes" ].sum()

num_no_diabetes = df.shape[ 0 ] - num_diabetes

perc_diabetes = num_diabetes / df.shape[ 0 ] * 100

perc_no_diabetes = num_no_diabetes / df.shape[ 0 ] * 100



print( "There are %d (%.2f%%) people who have diabetes and the remaining %d (%.2f%%) who have not been diagnosed with the desease." % ( num_diabetes, perc_diabetes, num_no_diabetes, perc_no_diabetes ) )



def plot_diabetes_value_counts( normalize ):

    plt.grid( False )

    df.Diabetes.value_counts( normalize=normalize ).plot( kind="bar", grid=False, color=[ sns.color_palette()[ 0 ], sns.colors.xkcd_rgb.get( 'dusty orange' ) ] )

    plt.xticks( [ 0, 1 ], [ 'No', 'Yes' ], rotation=0 )

    plt.xlabel( "Diabetes" )

    

    if ( normalize == False ):

        plt.ylabel( "Count" )

    else:

        plt.ylabel( "Percentage" )    

        

    return

    

plt.subplot( 1, 2, 1 )

plot_diabetes_value_counts( False )

plt.subplot( 1, 2, 2 )

plot_diabetes_value_counts( True )

plt.tight_layout( True )

plt.show()
df.describe().round( 2 )
plt.figure( figsize=( 5.5, 5.0 ) )

plt.grid( False )

plt.xticks( range( df.shape[ 1 ] ), df.columns[ 0: ], rotation=0 )

plt.yticks( range( df.shape[ 1 ] ), df.columns[ 0: ], rotation=0 )

sns.heatmap( df.corr(), cbar=True, annot=True, square=False, fmt='.2f', cmap=plt.cm.Blues, robust=False, vmin=0 )

plt.show()
sns.pairplot( df.dropna(), vars=[ 'Glucose', 'Insulin', 'BMI', 'SkinThickness' ], size=1.5, diag_kind='kde', hue='Diabetes' )

plt.tight_layout( False )

plt.show()
plt.figure( figsize=( 7.0, 5.0 ) )



for i in range( 8 ):

    plt.subplot( 2, 4, i + 1 )

    plt.grid( False )

    sns.boxplot( x='Diabetes', y=df.columns[ i ], data=df )

    plt.xticks( [ 0, 1 ], [ 'No', 'Yes' ], rotation=0 )



plt.tight_layout( True )

plt.show()
pd.crosstab( pd.cut( df.Glucose, bins=25 ), df.Diabetes ).plot( kind='bar', figsize=( 6.5, 3.2 ) )

plt.ylabel( "Frequency" )

plt.show()
pd.crosstab( pd.cut( df.Insulin, bins=25 ), df.Diabetes ).plot( kind='bar', figsize=( 6.5, 3.4 ), yticks=[ 0, 10, 20, 30, 40, 50, 60, 70 ] )

plt.ylabel( "Frequency" )

plt.show()
from sklearn.model_selection import train_test_split



df_X = df.drop( [ "Diabetes" ], axis=1 )

df_y = df.Diabetes



X_train, X_test, y_train, y_test = train_test_split( df_X, df_y, test_size=0.20, random_state=RANDOM_STATE_SEED, shuffle=True, stratify=df_y )



train_size = np.shape( X_train )[ 0 ]

train_num_diabetes = np.sum( y_train )

train_num_no_diabetes = train_size - train_num_diabetes

train_perc_diabetes = train_num_diabetes / train_size * 100

train_perc_no_diabetes = train_num_no_diabetes / train_size * 100



test_size = np.shape( X_test )[ 0 ]

test_num_diabetes = np.sum( y_test )

test_num_no_diabetes = test_size - test_num_diabetes

test_perc_diabetes = test_num_diabetes / test_size * 100

test_perc_no_diabetes = test_num_no_diabetes / test_size * 100



print( "The training set is composed by %d samples: %d (%.2f%%) with diabetes and %d (%.2f%%) without diabetes." % ( train_size, train_num_diabetes, train_perc_diabetes, train_num_no_diabetes, train_perc_no_diabetes ) )

print( "The test set is composed by %d samples: %d (%.2f%%) with diabetes and %d (%.2f%%) without diabetes." % ( test_size, test_num_diabetes, test_perc_diabetes, test_num_no_diabetes, test_perc_no_diabetes ) )
from sklearn.impute import SimpleImputer



imputer = SimpleImputer( missing_values=np.nan, strategy='median' )



X_train_imputed = imputer.fit_transform( X_train )

X_test_imputed = imputer.transform( X_test )
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



X_train_normalized = sc.fit_transform( X_train_imputed )

X_test_normalized = sc.transform( X_test_imputed )



df_X_train_normalized = pd.DataFrame( X_train_normalized, columns=cols[ 0:8 ], index=y_train.index )

df_y_train_normalized = pd.DataFrame( y_train, columns = [ cols[ 8 ] ] )

df_train_normalized = df_X_train_normalized.join( df_y_train_normalized )
from sklearn.decomposition import PCA



pca = PCA( whiten=True )

pca.fit( X_train_normalized )



pca_evr = pca.explained_variance_ratio_

pca_evr_cum = np.cumsum( pca_evr )



x = np.arange( 1, len( pca_evr ) + 1 )

y = np.linspace( 0.1, 1, 10 )



plt.bar( x, pca_evr, alpha=1, align='center', label='Individual' )

plt.step( x, pca_evr_cum, where='mid', label='Cumulative', color=sns.colors.xkcd_rgb.get( 'dusty orange' ) )

plt.ylabel( 'Explained Variance Ratio' )

plt.xlabel( 'Principal Components' )

plt.legend()

plt.xticks( x )

plt.yticks( y )

plt.show()
pca = PCA( n_components=6 )



X_train_pca = pca.fit_transform( X_train_normalized )

X_test_pca = pca.transform( X_test_normalized )
print( "The training set transormed by PCA is composed by %d rows and %d columns." % ( X_train_pca.shape[ 0 ], X_train_pca.shape[ 1 ] ) )

print( "The test set transormed by PCA is composed by %d rows and %d columns." % ( X_test_pca.shape[ 0 ], X_test_pca.shape[ 1 ] ) )
from imblearn.over_sampling import SMOTE



smote = SMOTE( random_state=RANDOM_STATE_SEED )

X_train_smote, y_train_smote = smote.fit_resample( X_train_normalized, y_train )



df_X_train_smote = pd.DataFrame( X_train_smote, columns=cols[ 0:8 ] )

df_y_train_smote = pd.DataFrame( y_train_smote, columns = [ cols[ 8 ] ] )

df_train_smote = df_X_train_smote.join( df_y_train_smote )
num_diabetes_smote = df_train_smote[ "Diabetes" ].sum()

num_no_diabetes_smote = df_train_smote.shape[ 0 ] - num_diabetes_smote

perc_diabetes_smote = num_diabetes_smote / df_train_smote.shape[ 0 ] * 100

perc_no_diabetes_smote = num_no_diabetes_smote / df_train_smote.shape[ 0 ] * 100



print( "There are %d (%.2f%%) people with diabetes and %d (%.2f%%) people without diabetes." % ( num_diabetes_smote, perc_diabetes_smote, num_no_diabetes_smote, perc_no_diabetes_smote ) )



def plot_diabetes_value_counts( normalize ):

    plt.grid( False )

    df_train_smote[ 'Diabetes' ].value_counts( normalize=normalize ).plot( kind="bar", grid=False, color=[ sns.color_palette()[ 0 ], sns.colors.xkcd_rgb.get( 'dusty orange' ) ] )

    plt.xticks( [ 0, 1 ], [ 'No', 'Yes' ], rotation=0 )

    plt.xlabel( "Diabetes" )

    

    if ( normalize == False ):

        plt.ylabel( "Count" )

    else:

        plt.ylabel( "Percentage" )    

        

    return

    

plt.subplot( 1, 2, 1 )

plot_diabetes_value_counts( False )

plt.subplot( 1, 2, 2 )

plot_diabetes_value_counts( True )

plt.tight_layout( True )

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve



from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
def get_estimator_names( estimator_name ):

    estimator_name_smote = estimator_name + " (SMOTE)"

    estimator_name_pca = estimator_name + " (PCA)"

    

    return [ estimator_name, estimator_name_smote, estimator_name_pca ]
def print_grid_search_cross_validation_model_details( gscv_model, estimator_name, scoring ):

    print()

    print( estimator_name )

    print( "Best parameters: ", gscv_model.best_params_ )

    

    return
def grid_search_cv_fit( estimator, param_grid, X_train, scoring='f1' ):

    gscv = GridSearchCV( estimator=estimator, param_grid=param_grid, cv=5, n_jobs=-1, scoring=scoring )

    gscv.fit( X=X_train, y=y_train )

    

    return gscv
def grid_search_cv_fit_smote( estimator, param_grid, X_train, scoring='accuracy' ):

    pipeline_param_grid = {}



    try:

        for key in param_grid.keys():

            pipeline_param_grid[ "estimator__" + key ] = param_grid[ key ]

            

    except:

        pipeline_param_grid = []

        

        for d in param_grid:

            grid = {}

            

            for key in d.keys():

                grid[ "estimator__" + key ] = d[ key ]

                

            pipeline_param_grid.append( grid )

    

    

    smote = SMOTE( random_state=RANDOM_STATE_SEED, n_jobs=-1 )

    pipeline = Pipeline( [ ( 'smote', smote ), ( 'estimator', estimator ) ] )

    

    return grid_search_cv_fit( pipeline, pipeline_param_grid, X_train, scoring )
def grid_search_cross_validation( estimator, param_grid, estimator_names ):

    gscv = grid_search_cv_fit( estimator, param_grid, X_train_normalized )

    gscv_smote = grid_search_cv_fit_smote( estimator, param_grid, X_train_normalized )

    gscv_pca = grid_search_cv_fit( estimator, param_grid, X_train_pca )

    

    print_grid_search_cross_validation_model_details( gscv, estimator_names[ 0 ], 'f1' )

    print_grid_search_cross_validation_model_details( gscv_smote, estimator_names[ 1 ], 'accuracy' )

    print_grid_search_cross_validation_model_details( gscv_pca, estimator_names[ 2 ], 'f1' )

    

    return [ gscv, gscv_smote, gscv_pca ]
def print_confusion_matrix( confusion_matrix, estimator_name ):

    plt.grid( False )

    plt.title( estimator_name )

    sns.heatmap( confusion_matrix, cbar=False, annot=True, square=False, fmt='.0f', cmap=plt.cm.Blues, robust=True, linewidths=0, linecolor='black', vmin=0 )

    plt.xlabel( "Predicted labels" )

    plt.ylabel( "True labels" )

    

    return
def print_compared_cofusion_matrices( test_predictions, estimator_names ):

    confusion_matrix_ = confusion_matrix( y_test, test_predictions[ 0 ] )

    confusion_matrix_smote = confusion_matrix( y_test, test_predictions[ 1 ] )

    confusion_matrix_pca = confusion_matrix( y_test, test_predictions[ 2 ] )

    

    plt.figure( figsize=( 7.0, 2.8 ) )

    

    axs = plt.subplot( 1, 3, 1 )

    print_confusion_matrix( confusion_matrix_, estimator_names[ 0 ] )

    axs.set_xlabel( "Predicted labels" )

    axs.set_ylabel( "True labels" )

    plt.subplot( 1, 3, 2 )

    print_confusion_matrix( confusion_matrix_smote, estimator_names[ 1 ] )

    plt.subplot( 1, 3, 3 )

    print_confusion_matrix( confusion_matrix_pca, estimator_names[ 2 ] )

    

    plt.tight_layout( True )

    plt.show()

    

    return
def test_predictions( estimators ):

    predictions = estimators[ 0 ].predict( X_test_normalized )

    predictions_smote = estimators[ 1 ].predict( X_test_normalized )

    predictions_pca = estimators[ 2 ].predict( X_test_pca )

    

    return [ predictions, predictions_smote, predictions_pca ]
def evaluate_test_predictions( test_predictions, estimator_name ):

    test_f1 = f1_score( y_test, test_predictions )

    test_accuracy = accuracy_score( y_test, test_predictions )

    test_precision = precision_score( y_test, test_predictions )

    test_recall = recall_score( y_test, test_predictions )

    

    results = { 

        'F1' : [ test_f1 ],

        'Accuracy' : [ test_accuracy ], 

        'Precision' : [ test_precision ], 

        'Recall' : [ test_recall ]

    }

    

    df_results = pd.DataFrame( results, index=[ estimator_name ] )

    

    return df_results
def merge_and_sort_results( results ):

    df_results = pd.DataFrame( columns=results[ 0 ].columns )



    for result in results:

        df_results = df_results.append( result )

        

    df_results = df_results.sort_values( "F1", ascending=False )

    

    return df_results.round( 3 )
def evaluate_test_results( test_predictions, estimator_names ):

    df_results = evaluate_test_predictions( test_predictions[ 0 ], estimator_names[ 0 ] )

    df_smote_results = evaluate_test_predictions( test_predictions[ 1 ], estimator_names[ 1 ] )

    df_pca_results = evaluate_test_predictions( test_predictions[ 2 ], estimator_names[ 2 ] )



    df_overall_results = merge_and_sort_results( [ df_results, df_pca_results, df_smote_results ] )

    

    return df_overall_results
def evaluate_best_estimators_results( best_estimators ):

    df_results = []

    

    for estimator in best_estimators:

        if estimator[ 1 ].endswith( "(PCA)" ):

            test_predictions = estimator[ 0 ].predict( X_test_pca )

        else:

            test_predictions = estimator[ 0 ].predict( X_test_normalized )

            

        df_result = evaluate_test_predictions( test_predictions, estimator[ 1 ] )

        df_results.append( df_result )

        

    df_results = merge_and_sort_results( df_results )

    

    return df_results
def plot_learning_curve( estimator, X_train_, estimator_name, legend_location='best', scoring='f1', scoring_name='F1' ):

    train_sizes = np.linspace( 0.2, 1.0, 6 )

    train_size, train_scores, test_scores = learning_curve( estimator, X_train_, y_train, train_sizes=train_sizes, cv=5, n_jobs=-1, shuffle=True, random_state=RANDOM_STATE_SEED, scoring=scoring )

    

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)

    

    plt.title( "Learning curve: " + estimator_name )

    plt.plot( train_size, train_mean, label='Train', marker='o', markerfacecolor='white', markeredgewidth=2.0 )

    plt.fill_between(train_size,train_mean + train_std,train_mean - train_std, alpha=0.2 )

    plt.plot( train_size, test_mean, label='Validation', marker='o', markerfacecolor='white', markeredgewidth=2.0 )

    plt.fill_between(train_size,test_mean + test_std,test_mean - test_std, alpha=0.2 )

    

    plt.xlabel( 'Number of training samples' )

    plt.ylabel( scoring_name )

    

    plt.legend( loc=legend_location )

    

    plt.plot()
best_estimators = []
from sklearn.neighbors import KNeighborsClassifier



knn_param_grid = {

    'n_neighbors' : [ 5, 9, 15, 21 ],

    'weights' : [ 'uniform', 'distance' ]

}



knn_estimator_names = get_estimator_names( "kNN" )

knn_best_estimators = grid_search_cross_validation( KNeighborsClassifier(), knn_param_grid, knn_estimator_names )
knn_test_predictions = test_predictions( knn_best_estimators )
print_compared_cofusion_matrices( knn_test_predictions, knn_estimator_names )
df_knn_overall_results = evaluate_test_results( knn_test_predictions, knn_estimator_names )

df_knn_overall_results
knn_best_estimator = knn_best_estimators[ 1 ].best_estimator_

knn_best_estimator_name = knn_estimator_names[ 1 ]

best_estimators.append( [ knn_best_estimator, knn_best_estimator_name ] )
plot_learning_curve( knn_best_estimator, X_train_normalized, knn_best_estimator_name, 'lower right' )
plt.figure( figsize=( 7.0, 5.0 ) )



for i in range( 8 ):

    plt.subplot( 3, 3, i + 1 )

    sns.distplot( df_train_normalized[ df_train_normalized.Diabetes == 0 ][ cols[ i ] ], bins=15 )

    sns.distplot( df_train_normalized[ df_train_normalized.Diabetes == 1 ][ cols[ i ] ], bins=15 )



plt.tight_layout( True )

plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



plot_learning_curve( LinearDiscriminantAnalysis( priors=[ ( train_perc_diabetes / 100 ), ( train_perc_no_diabetes / 100 ) ] ), X_train_normalized, "LDA" )
from sklearn.svm import SVC



svc_param_grid = [ 

    {

     'kernel': [ 'linear' ],

     'C': [ 0.001, 0.01, 0.1, 1, 10, 100 ],

     'class_weight': [ None, 'balanced' ] },

    {

     'kernel': [ 'rbf' ],

     'C': [ 0.001, 0.01, 0.1, 1, 10, 100 ],

     'gamma': [ 0.001, 0.01, 0.1, 1, 10, 100 ],

     'class_weight': [ None, 'balanced' ] } ]







svc_estimator_names = get_estimator_names( "SVM" )

svc_best_estimators = grid_search_cross_validation( SVC( random_state=RANDOM_STATE_SEED ), svc_param_grid, svc_estimator_names )
svc_test_predictions = test_predictions( svc_best_estimators )
print_compared_cofusion_matrices( svc_test_predictions, svc_estimator_names )
df_svc_overall_results = evaluate_test_results( svc_test_predictions, svc_estimator_names )

df_svc_overall_results
svc_best_estimator = svc_best_estimators[ 0 ].best_estimator_

svc_best_estimator_name = svc_estimator_names[ 0 ]

best_estimators.append( [ svc_best_estimator, svc_best_estimator_name ] )
plot_learning_curve( svc_best_estimator, X_train_normalized, svc_best_estimator_name, 'lower right' )
from sklearn.tree import DecisionTreeClassifier



decision_tree_param_grid = {

    'max_depth': [ 10, 15, 20, None ],

    'criterion' : [ 'gini', 'entropy' ],

    'class_weight': [ None, 'balanced' ]

}



decision_tree_estimator_names = get_estimator_names( "Decision Tree" )

decision_tree_best_estimators = grid_search_cross_validation( DecisionTreeClassifier( random_state=RANDOM_STATE_SEED ), decision_tree_param_grid, decision_tree_estimator_names )
decision_tree_test_predictions = test_predictions( decision_tree_best_estimators )
print_compared_cofusion_matrices( decision_tree_test_predictions, decision_tree_estimator_names )
df_decision_tree_overall_results = evaluate_test_results( decision_tree_test_predictions, decision_tree_estimator_names )

df_decision_tree_overall_results
decision_tree_best_estimator_name = decision_tree_estimator_names[ 0 ]

decision_tree_best_estimator = decision_tree_best_estimators[ 0 ].best_estimator_

best_estimators.append( [ decision_tree_best_estimator, decision_tree_best_estimator_name ] )
plot_learning_curve( decision_tree_best_estimator, X_train_normalized, decision_tree_best_estimator_name, 'center left' )
from sklearn.ensemble import RandomForestClassifier



random_forest_param_grid = {

    'max_depth': [ 10, 15, 20, None ],

    'criterion': [ 'gini', 'entropy' ],

    'n_estimators': [ 25, 50, 100 ],

    'class_weight': [ None, 'balanced' ]

}



random_forest_estimator_names = get_estimator_names( "Random Forest" )

random_forest_best_estimators = grid_search_cross_validation( RandomForestClassifier( random_state=RANDOM_STATE_SEED ), random_forest_param_grid, random_forest_estimator_names )
random_forest_test_predictions = test_predictions( random_forest_best_estimators )
print_compared_cofusion_matrices( random_forest_test_predictions, random_forest_estimator_names )
df_random_forest_overall_results = evaluate_test_results( random_forest_test_predictions, random_forest_estimator_names )

df_random_forest_overall_results
random_forest_best_estimator_name = random_forest_estimator_names[ 1 ]

random_forest_best_estimator = random_forest_best_estimators[ 1 ].best_estimator_

best_estimators.append( [ random_forest_best_estimator, random_forest_best_estimator_name ] )
plot_learning_curve( random_forest_best_estimator, X_train_normalized, random_forest_best_estimator_name, 'center left' )
df_results = evaluate_best_estimators_results( best_estimators )

df_results
df_best_estimator = pd.DataFrame( svc_best_estimator.get_params(), index=[ 'SVM' ] )

df_best_estimator