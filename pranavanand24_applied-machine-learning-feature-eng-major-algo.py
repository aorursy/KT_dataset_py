import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn



ipl_auction_df = pd.read_csv("../input/ipl-2013/IPL 2013.csv")

ipl_auction_df.head(5)
ipl_auction_df.info()
X_features = ['AGE','COUNTRY', 'PLAYING ROLE', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B','ODI-WKTS', 'ODI-SR-BL',

             'CAPTAINCY EXP', 'RUNS-S', 'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C','WKTS', 'AVE-BL', 'ECON', 'SR-BL']
#Initialize the list with the catagorical feature names

categorical_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'CAPTAINCY EXP']



#get_dummies() is invoked to return the dummy features

ipl_auction_encoded_df =pd.get_dummies(ipl_auction_df[X_features],

                                      columns = categorical_features,

                                      drop_first = True)
#To display all features along with new dummy feautures we use the following

ipl_auction_encoded_df.columns
X = ipl_auction_encoded_df

Y = ipl_auction_df['SOLD PRICE']
from sklearn.preprocessing import StandardScaler
#Initialize the standardscaler

X_scaler = StandardScaler()

#Standardize all the feature columns

X_scaled = X_scaler.fit_transform(X)



# Standardizing Y explicitly by subtracting mean and dividing by standard deviation

Y = (Y - Y.mean()) / Y.std()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

X_scaled,

Y,

test_size = 0.2,

random_state = 42)
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train, y_train)
linreg.coef_
#the dataframe has two columns to store feature name and the corresponding coefficient values

columns_coef_df = pd.DataFrame({ 'columns': ipl_auction_encoded_df.columns,

                               'coef': linreg.coef_ })



#sorting the feature by cofficient values in descending order

sorted_coef_vals = columns_coef_df.sort_values('coef',

                                              ascending=False)
#Creating a bar plot

plt.figure(figsize = (14, 12))

sn.barplot(x="coef", y="columns", data=sorted_coef_vals);

plt.xlabel("Coefficients from Linear Regression")

plt.ylabel("Features")
from sklearn import metrics



#Take a model as parameter

#Print the RMSE on train and test set



def get_train_test_rmse(model):

    #predicting on training dataset

    y_train_pred = model.predict(X_train)

    #comapre the actual y with predicted y in the training datasets

    rmse_train = round(np.sqrt(metrics.mean_squared_error(y_train,

                                                         y_train_pred)),3)

    

    #predicting on test dataset

    y_test_pred = model.predict(X_test)

    #comapare the actual y with predicted y in the test dataset

    rmse_test = round(np.sqrt(metrics.mean_squared_error(y_test,

                                                        y_test_pred)), 3)

    print("train: ", rmse_train, "test: ", rmse_test)
get_train_test_rmse(linreg)
#Importing Ridge Regerssion

from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 1, max_iter = 500)

ridge.fit(X_train, y_train)
get_train_test_rmse(ridge)
# Importing Lasso Regression

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.01, max_iter = 500)

lasso.fit(X_train, y_train)
get_train_test_rmse( lasso)
#storing the feature names and coefficient values in the Dataframe

lasso_coef_df = pd.DataFrame ({ 'columns': ipl_auction_encoded_df.columns,

                              'coef': lasso.coef_})
#filtering out coefficients with zeros

lasso_coef_df[lasso_coef_df.coef == 0]
bank_df =pd.read_csv("../input/bank-dataset/bank.csv")

bank_df.head(5)
bank_df.info()
bank_df.subscribed.value_counts()
#sklearn.utils has resample method to help with upsampling

from sklearn.utils import resample



#seperate the case of yes-subscriber and no-subscriber

bank_subscribed_no = bank_df[bank_df.subscribed == 'no']

bank_subscribed_yes = bank_df[bank_df.subscribed == 'yes']



#upsampling the yes-subscribed cases

df_minority_upsampled = resample(bank_subscribed_yes,

                                 replace=True,

                                 n_samples=2000)
#combine majority class with unsampled minority class

new_bank_df = pd.concat([ bank_subscribed_no, df_minority_upsampled])
from sklearn.utils import shuffle

new_bank_df = shuffle(new_bank_df)
#assigning list of columns names in the DataFrame

X_features = list (new_bank_df.columns)

#remove the response variable from the list

X_features.remove('subscribed')

X_features
#get_dummies() will convert all the columns with datatypes as object

encoded_bank_df = pd.get_dummies( new_bank_df [X_features ],

                                drop_first = True)

X = encoded_bank_df
#encoding the subscribed columns and assigning to Y

Y = new_bank_df.subscribed.map( lambda x: int(x == 'yes'))
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X,

                                                   Y,

                                                   test_size = 0.3,

                                                   random_state = 42)
from sklearn.linear_model import LogisticRegression



#Initializing the model

logit = LogisticRegression()

## Fitting the model with X and Y values of the dataset

logit.fit( train_X, train_y)
pred_y = logit.predict(test_X)
from sklearn import metrics

#defining the matrix to draw the confusion  matrix from actual and predicted class lebels

def draw_cm( actual, predicted ):

    cm = metrics.confusion_matrix( actual, predicted, [1,0])

    #the labels are configured to better interpretation from the plot 

    sn.heatmap(cm, annot=True, fmt='.2f',

              xticklabels = ["Subscribed", "Not Subscribed"],

              yticklabels = ["Subscribed", "Not Subscribed"])

    

    plt.ylabel('True lebels')

    plt.xlabel('Preicted label')

    plt.show()

cm = draw_cm(test_y, pred_y)
print(metrics.classification_report(test_y, pred_y))
#predicting the probability values for test cases

predict_proba_df = pd.DataFrame(logit.predict_proba(test_X))

predict_proba_df.head()
test_results_df = pd.DataFrame({'actual': test_y})

test_results_df = test_results_df.reset_index()

#assigning the probability values for class label 1

test_results_df['chd_1'] = predict_proba_df.iloc[:,1:2]
test_results_df.head(5)
#passing the actual class labels and predicted probability values to compute ROC AUC score

auc_score = metrics.roc_auc_score(test_results_df.actual,

                                 test_results_df.chd_1)

round(float(auc_score ), 2)
def draw_roc_curve( model, test_X, test_y ):

## Creating and initializing a results DataFrame with actual labels

    test_results_df = pd.DataFrame( { 'actual': test_y } )

    test_results_df = test_results_df.reset_index()

    # predict the probabilities on the test set

    predict_proba_df = pd.DataFrame( model.predict_proba( test_X ) )

    ## selecting the probabilities that the test example belongs to class 1

    test_results_df['chd_1'] = predict_proba_df.iloc[:,1:2]

    ## Invoke roc_curve() to return the fpr, tpr and threshold values.

    ## threshold values contain values from 0.0 to 1.0

    fpr, tpr, thresholds = metrics.roc_curve( test_results_df.actual,

    test_results_df.chd_1,

    drop_intermediate = False )

    ## Getting the roc auc score by invoking metrics.roc_auc_score method

    auc_score = metrics.roc_auc_score( test_results_df.actual, test_results_df.chd_1 )

    

    ## Setting the size of the plot

    plt.figure(figsize=(8, 6))

    ## plotting the actual fpr and tpr values

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    ## plotting th diagnoal line from (0,1)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    ## Setting labels and titles

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()

    return auc_score, fpr, tpr, thresholds
## Invoking draw_roc_curve with the logistic regresson model

_, _, _, _ = draw_roc_curve( logit, test_X, test_y )
# importing the knn classifier algorithms

from sklearn.neighbors import KNeighborsClassifier



#Initialize the classifier

knn_clf = KNeighborsClassifier()

#Fitting the model with the training set

knn_clf.fit(train_X, train_y)
#invoking draw_roc_curve with the knn model

_, _, _, _ = draw_roc_curve(knn_clf, test_X, test_y)
## Predicting on test set

pred_y = knn_clf.predict(test_X)

## Drawing the confusion matrix for KNN model

draw_cm( test_y, pred_y )
print( metrics.classification_report( test_y, pred_y ) )
## Importing GridSearchCV

from sklearn.model_selection import GridSearchCV

## Creating a dictionary with hyperparameters and possible values for searching

tuned_parameters = [{'n_neighbors': range(5,10),

'metric': ['canberra', 'euclidean', 'minkowski']}]

## Configuring grid search

clf = GridSearchCV(KNeighborsClassifier(),

tuned_parameters,

cv=10,

scoring='roc_auc')

## fit the search with training set

clf.fit(train_X, train_y )
clf.best_score_
clf.best_params_
# Importing Random Forest Classifier from the sklearn.ensemble

from sklearn.ensemble import RandomForestClassifier

# Initializing the Random Forest Classifier with max_dept and n_estimators

radm_clf = RandomForestClassifier( max_depth=10, n_estimators=10)

radm_clf.fit( train_X, train_y )
_, _, _, _ = draw_roc_curve( radm_clf, test_X, test_y );
# Configuring parameters and values for searched

tuned_parameters = [{'max_depth': [10, 15],

'n_estimators': [10,20],

'max_features': ['sqrt', 'auto']}]

# Initializing the RF classifier

radm_clf = RandomForestClassifier()

# Configuring search with the tunable parameters

clf = GridSearchCV(radm_clf,

tuned_parameters,

cv=5,

scoring='roc_auc')

# Fitting the training set

clf.fit(train_X, train_y )
clf.best_score_
clf.best_params_
# Initializing the Random Forest Mode with the optimal values

radm_clf = RandomForestClassifier( max_depth=15, n_estimators=20, max_features =

'auto')

# Fitting the model with the training set

radm_clf.fit( train_X, train_y )
_, _, _, _ = draw_roc_curve( clf, test_X, test_y )
pred_y = radm_clf.predict( test_X )

draw_cm( test_y, pred_y )
print( metrics.classification_report( test_y, pred_y ) )
import numpy as np

# Create a dataframe to store the featues and their corresponding importances

feature_rank = pd.DataFrame( { 'feature': train_X.columns,

'importance': radm_clf.feature_importances_ } )

# Sorting the features based on their importances with most important feature at top.

feature_rank = feature_rank.sort_values('importance', ascending = False)

plt.figure(figsize=(8, 6))

# plot the values

sn.barplot( y = 'feature', x = 'importance', data = feature_rank );
# Importing Adaboost classifier

from sklearn.ensemble import AdaBoostClassifier

## Initializing logistic regression to use as base classifier

logreg_clf = LogisticRegression()

## Initilizing adaboost classifier with 50 classifers

ada_clf = AdaBoostClassifier(logreg_clf, n_estimators=50)

## Fitting adaboost model to training set

ada_clf.fit(train_X, train_y )
_, _, _, _ = draw_roc_curve( ada_clf, test_X, test_y )
## Importing Gradient Boosting classifier

from sklearn.ensemble import GradientBoostingClassifier

## Initializing Gradient Boosting with 500 estimators and max depth as 10.

gboost_clf = GradientBoostingClassifier( n_estimators=500, max_depth=10)

## Fitting gradient boosting model to training set

gboost_clf.fit(train_X, train_y )
_, _, _, _ = draw_roc_curve( gboost_clf, test_X, test_y )
from sklearn.model_selection import cross_val_score

gboost_clf = GradientBoostingClassifier( n_estimators=500, max_depth=10)

cv_scores = cross_val_score( gboost_clf, train_X, train_y, cv = 10, scoring = 'roc_auc' )
print( cv_scores )

print( "Mean Accuracy: ", np.mean(cv_scores), " with standard deviation of: ",

np.std(cv_scores))
gboost_clf.fit(train_X, train_y )

pred_y = gboost_clf.predict( test_X )

draw_cm( test_y, pred_y )
print( metrics.classification_report( test_y, pred_y ) )
import numpy as np

# Create a dataframe to store the featues and their corresponding importances

feature_rank = pd.DataFrame( { 'feature': train_X.columns,

'importance': gboost_clf.feature_importances_ } )

## Sorting the features based on their importances with most important feature at top.

feature_rank = feature_rank.sort_values('importance', ascending = False)

plt.figure(figsize=(8, 6))

# plot the values

sn.barplot( y = 'feature', x = 'importance', data = feature_rank );