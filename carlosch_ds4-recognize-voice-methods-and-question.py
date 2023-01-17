# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pyplot # plot in python

import seaborn as sns # data visualization in python 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/voice.csv")

data.head()
#type of data

data.info()
#count the null value 

data.isnull().sum()
#share of data

print( "dimension of dataframe is: ",data.shape)

#how many men and women are in data?

print("Number of male: {}".format(data[data.label == 'male'].shape[0]))

print("Number of female: {}".format(data[data.label == 'female'].shape[0]))
#Library what i will use

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestClassifier
#separate in features and label data

features = data.iloc[  : , :-1]

labels = data.iloc[ : , -1 ]

#preprocessing of data: i think that raw data can affect to performance of algorithms

gender_encoder = LabelEncoder()

y = gender_encoder.fit_transform(labels)

scaler = StandardScaler()

scaler.fit(features)

x = scaler.transform(features)
#create a selection

selection = SelectKBest( f_classif,  k = 5 )

model = selection.fit( x , y )

#preparate the scores

scores_features = pd.DataFrame()

scores_features["Value"] = selection.scores_

scores_features["Attribute"] = data.drop("label", axis = 1 ).columns

scores_features = scores_features.sort_values( ["Attribute"] , ascending = False )

#plots the value

plot1 = sns.barplot( x = scores_features['Value'] , y = scores_features['Attribute'] , data = scores_features   )

plot1.set_title('Feature Importance')
#create a dataset using features selection

features_selection = data[ ["IQR", "Q25" , "sd" , "sp.ent" , "meanfun" , "sfm" , "label" ] ]

features_fs = features_selection.iloc[  : , :-1]

labels_fs = features_selection.iloc[ : , -1 ]

#normalize the data

gender_encoder = LabelEncoder()

y_fs = gender_encoder.fit_transform( labels_fs )

scaler = StandardScaler()

scaler.fit( features_fs )

x_fs = scaler.transform( features_fs )

#create trainset and testset

x_fs_train , x_fs_test , y_fs_train  , y_fs_test = train_test_split( x_fs  , y_fs , test_size = 0.2 , random_state = 1 )

# the future plot

data_plot = pd.DataFrame( columns =("Method","Accuracy") )
#create a SVM and see the precision

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=1)

svm_linear = SVC( kernel = 'linear' )

svm_linear.fit( x_train, y_train )

prediction = svm_linear.predict( x_test )

print("Accuracy Score ( default linear ): ",metrics.accuracy_score( y_test, prediction ) )

#with cross-validation 

score_cv = cross_val_score( svm_linear , x_test , y_test , scoring = "accuracy",cv = 10 )

print("-->Average accuracy score ( default linear ): ",score_cv.mean() )

data_plot.loc[ len(data_plot) ] = [ "SVM( default + linear) ", score_cv.mean( ) ]

# with tuned hiperparameter with gridsearch

tuned_parameter = { 'C': (np.arange(0.1 , 1 , 0.1 ) )  , 'kernel':[ 'linear' ] }

tuned_svm_linear = GridSearchCV( SVC() , tuned_parameter , cv = 10 , scoring = 'accuracy'  )

tuned_svm_linear.fit( x_train , y_train )

print("-->Accuracy score ( GridSearchCV defaultlinear ):",tuned_svm_linear.best_score_ )

print("Best parameter ( GridSearchCV default linear ):",tuned_svm_linear.best_params_)

data_plot.loc[ len(data_plot) ] = [ "SVM ( GridSearchCv + linear + default )" , tuned_svm_linear.best_score_ ]

#with C = 0.1

customer_svm_linear = SVC( kernel = 'linear', C= 0.1 )

customer_svm_linear.fit( x_train, y_train )

prediction = customer_svm_linear.predict( x_test )

print("Accuracy Score ( customer linear ) : ",metrics.accuracy_score( y_test, prediction ) )

#with cross-validation

score_cv = cross_val_score( customer_svm_linear , x_test , y_test , scoring = "accuracy",cv = 10 )

print("-->Average accuracy score ( customer linear ): ",score_cv.mean() )

data_plot.loc[ len( data_plot ) ] = [ "SVM ( Customer + linear )" , score_cv.mean() ]

#with rbf kernel default + cross validation

svm_rbf = SVC( kernel = "rbf")

svm_rbf.fit(x_train , y_train )

score_cv = cross_val_score( svm_rbf , x_test , y_test , scoring = "accuracy" , cv= 10 )

print("-->Average accuracy score( default RBF ): ",score_cv.mean() )

data_plot.loc[ len( data_plot ) ] = [ "SVM ( default + rbf )" , score_cv.mean() ]

#with rbf kernel + gridsearch

tuned_parameter = { 'C' : np.arange( 0.1 , 1 , 0.1) , 'gamma' : [ 0.01 , 0.02, 0.03 , 0.04 , 0.05] , 'kernel':['rbf'] }

tuned_svm_rbf = GridSearchCV( SVC() , tuned_parameter , cv = 10 , scoring = "accuracy" )

tuned_svm_rbf.fit( x_train , y_train )

print("-->Acurracy score ( GridSearchCV RBF ): ",tuned_svm_rbf.best_score_ )

print("Best parameter ( GridSearchCV rbf ): ",tuned_svm_rbf.best_params_)

data_plot.loc[ len( data_plot ) ] = [ "SVM ( GridSearchCV + rbf )" , tuned_svm_rbf.best_score_ ]
from sklearn.tree import DecisionTreeClassifier

#CART using default parameter and use CV

dt = DecisionTreeClassifier()

dt.fit( x_train , y_train )

score_dt = cross_val_score(dt , x_test , y_test , cv = 10 , scoring = "accuracy")

print("--> Average Accuracy score ( default ): ",score_dt.mean() )

data_plot.loc[ len( data_plot ) ] = [ "CART ( default ) ", score_dt.mean() ]

#using features selection

dt_fs = DecisionTreeClassifier()

dt_fs.fit( x_fs_train , y_fs_train )

score_dt = cross_val_score(dt_fs , x_fs_test , y_fs_test , cv = 10 , scoring = "accuracy")

print("--> Average Accuracy score ( feature selection default ): ",score_dt.mean() )

data_plot.loc[ len( data_plot ) ] = [ "CART ( features selection ) ", score_dt.mean() ]

#using GridSearchCV

tune_parameter = { 'criterion':["gini","entropy"] , 'max_depth': [ 3, 4 ,5 ,6 , 7 , 8, 9 ,10 ] }

tuned_dt = GridSearchCV( DecisionTreeClassifier() , tune_parameter , cv = 10 , scoring = "accuracy" )

tuned_dt.fit( x_train , y_train )

print("-->Acurracy score ( GridSearchCV default ): ",tuned_dt.best_score_ )

print("Best parameter ( GridSearchCV default ): ",tuned_dt.best_params_)

data_plot.loc[ len( data_plot ) ] = [ "CART ( GridSearchCV + default ) ", tuned_dt.best_score_ ]

#using GridSearchCV + feature selection

tune_parameter = { 'criterion':["gini","entropy"] , 'max_depth': [ 3, 4 ,5 ,6 , 7 , 8, 9 ,10 ] }

tuned_dt_fs = GridSearchCV( DecisionTreeClassifier() , tune_parameter , cv = 10 , scoring = "accuracy" )

tuned_dt_fs.fit( x_fs_train , y_fs_train )

print("-->Acurracy score ( GridSearchCV feature selection ): ",tuned_dt_fs.best_score_ )

print("Best parameter ( GridSearchCV feature selection ): ",tuned_dt_fs.best_params_) 

data_plot.loc[ len( data_plot ) ] = [ "CART ( GridSearchCV + feature selection ) ", tuned_dt_fs.best_score_ ]

#using entropy default

dt_entropy = DecisionTreeClassifier( criterion = 'entropy')

dt_entropy.fit( x_train , y_train )

score_dt_entropy = cross_val_score( dt_entropy ,x_test , y_test , cv = 10 , scoring = "accuracy" )

print("--> Average accuracy score ( entropy default ): ",score_dt_entropy.mean() )

data_plot.loc[ len( data_plot ) ] = [ "CART ( entropy + default ) ", score_dt_entropy.mean()]

#using entropy + features selection

dt_entropy = DecisionTreeClassifier( criterion = 'entropy')

dt_entropy.fit( x_fs_train , y_fs_train )

score_dt_entropy = cross_val_score( dt_entropy ,x_fs_test , y_fs_test , cv = 10 , scoring = "accuracy" )

print("--> Average accuracy score ( entropy feature selection ): ",score_dt_entropy.mean() )

data_plot.loc[ len( data_plot ) ] = [ "CART ( entropy + features selection ) ", score_dt_entropy.mean()]
from sklearn.linear_model import LogisticRegression

#Logistic Regression default

lr_default = LogisticRegression()

lr_default.fit( x_train , y_train )

prediction = lr_default.predict( x_test )

print("Accuracy ( default ): ",metrics.accuracy_score( y_test, prediction) )

scores = cross_val_score( lr_default , x_test , y_test , cv = 10 , scoring = "accuracy")

print("--> Average Accuracy ( default ): ",scores.mean() )

data_plot.loc[ len( data_plot ) ] = [ "Logistic Regression ( default ) ", scores.mean() ]

#Logistic Regression + GridSearchCV

tuned_parameter = { 'penalty': ['l1','l2'], 'solver':['liblinear'] , 'C': np.arange( 0.1 , 5 , 0.1 ) }

tuned_lr = GridSearchCV( LogisticRegression() , tuned_parameter, cv = 10 , scoring = "accuracy")

tuned_lr.fit( x_train , y_train )

print("--> Accurancy ( GridSearch default ): ",tuned_lr.best_score_ )

print("Best Parameter: ", tuned_lr.best_params_ )

data_plot.loc[ len( data_plot ) ] = [ "Logistic Regression ( GridSearchCV + default ) ", tuned_lr.best_score_ ]

#Logistic Regression + Features selection

lr_fs = LogisticRegression()

lr_fs.fit( x_fs_train , y_fs_train )

prediction = lr_fs.predict( x_fs_test )

print("Accuracy ( features selection ): ",metrics.accuracy_score( y_fs_test , prediction ) )

scores = cross_val_score( lr_fs , x_fs_test , y_fs_test , cv = 10 , scoring = "accuracy")

print("--> Average Accuracy ( features selection default ): ",scores.mean() )

data_plot.loc[ len( data_plot ) ] = [ "Logistic Regression ( features selection ) ", scores.mean()  ]

#Logistic Regression + Features selection + GridSearchCV

tuned_parameter = { 'penalty': ['l1','l2'], 'solver':['liblinear'] , 'C': np.arange( 0.1 , 5 , 0.1 ) }

tuned_lr_fs = GridSearchCV( LogisticRegression() , tuned_parameter, cv = 10 , scoring = "accuracy")

tuned_lr_fs.fit( x_fs_train , y_fs_train )

print("--> Accurancy ( GridSearch features selection ): ",tuned_lr_fs.best_score_ )

print("Best Parameter: ", tuned_lr_fs.best_params_ )

data_plot.loc[ len( data_plot ) ] = [ "Logistic Regression ( GridSearchCV + features selection ) ", tuned_lr_fs.best_score_  ]
#default

rf_default = RandomForestClassifier()

rf_default.fit( x_train , y_train )

prediction = rf_default.predict( x_test )

print("Accuracy ( default ): ",metrics.accuracy_score( prediction, y_test ) )

#with CV

score = cross_val_score( rf_default , x_test , y_test, cv= 20 , scoring = "accuracy" )

print("--> Average Accurancy ( default ): ",score.mean() )

data_plot.loc[ len( data_plot ) ] = [ "RandomForest ( default )" , score.mean() ]

#with GridSearchCV

tuned_parameter = {'criterion': ["gini","entropy"] , 'max_features': ["auto","log2"] , "max_depth": [ 3 , 4 ,5 , 6, 7 , 8 ,9 ,10,11,12 ] }

tuned_rf = GridSearchCV( RandomForestClassifier() , tuned_parameter , cv = 15 , scoring = "accuracy" )

tuned_rf.fit( x_train , y_train )

print("-->Accuracy ( GridSearch Default ): ",tuned_rf.best_score_ )

print("Best Parameter: ",tuned_rf.best_params_)

data_plot.loc[ len( data_plot ) ] = [ "RandomForest ( GridSearchCV + default )" , tuned_rf.best_score_   ]

#with features selection

rf_default = RandomForestClassifier()

rf_default.fit( x_fs_train , y_fs_train )

prediction = rf_default.predict( x_fs_test )

print("Accuracy ( features selection ): ",metrics.accuracy_score( prediction, y_fs_test ) )

score = cross_val_score( rf_default , x_fs_test , y_fs_test, cv= 20 , scoring = "accuracy" )

print("--> Average Accurancy ( features selection ): ",score.mean() )

data_plot.loc[ len( data_plot ) ] = [ "RandomForest ( features selection )" , score.mean() ]

#features selection + Gridsearch

tuned_parameter = {'criterion': ["gini","entropy"] , 'max_features': ["auto","log2"] , "max_depth": [ 3 , 4 ,5 , 6, 7 , 8 ,9 ,10,11,12 ] }

tuned_rf = GridSearchCV( RandomForestClassifier() , tuned_parameter , cv = 15 , scoring = "accuracy" )

tuned_rf.fit( x_fs_train , y_fs_train )

print("-->Accuracy ( GridSearch Default + features selection ): ",tuned_rf.best_score_ )

print("Best Parameter: ",tuned_rf.best_params_)

data_plot.loc[ len( data_plot ) ] = [ "RandomForest ( GridSearchCV + features selection )" , tuned_rf.best_score_   ]

# Customed

rf_tuned = RandomForestClassifier( criterion="gini", max_features = "log2", max_depth = 9 )

rf_tuned.fit( x_train , y_train )

score = cross_val_score( rf_tuned , x_test , y_test , cv = 20 , scoring = "accuracy" )

print("-->Average Accuracy ( customed default ): ",score.mean() )

rf_tuned.fit(x_fs_train , y_fs_train )

score = cross_val_score( rf_tuned, x_fs_test , y_fs_test , cv = 20 , scoring = "accuracy" )

print("-->Average accuracy (customed + features selection ): ", score.mean() )

data_plot.loc[ len( data_plot ) ] = [ "RandomForest ( Customer )" , score.mean()   ]
from sklearn.ensemble import  AdaBoostClassifier

#defaul

ab = AdaBoostClassifier()

ab.fit( x_train , y_train )

score = cross_val_score(ab , x_test , y_test , cv = 10 , scoring = "accuracy" )

print("--> Average accuracy ( default ): ",score.mean() )

data_plot.loc[ len( data_plot ) ] = [" adaboost ( default ) " , score.mean() ]

#GridSearchCV + default

tuned_parameter = {"n_estimators":[40 , 50 ,60 ,70 ,80 ] , 'algorithm':["SAMME","SAMME.R"]}

ab_tuned = GridSearchCV( AdaBoostClassifier() , tuned_parameter , cv = 10 , scoring = "accuracy" )

ab_tuned.fit( x_train , y_train )

print("--> Accuracy ( GridSearchCV + default ): ", ab_tuned.best_score_ )

print("best parameter: ",ab_tuned.best_params_)

data_plot.loc[ len( data_plot ) ] = [" adaboost ( GridSearchCV + default ) " , ab_tuned.best_score_ ]

# with feature selection

ab = AdaBoostClassifier()

ab.fit( x_fs_train , y_fs_train )

score = cross_val_score(ab , x_fs_test , y_fs_test , cv = 10 , scoring = "accuracy" )

print("--> Average accuracy ( feature selection ): ",score.mean() )

data_plot.loc[ len( data_plot ) ] = [" adaboost ( feature selection ) " , score.mean() ]

#GridSearchCV + feature selection

tuned_parameter = {"n_estimators":[40 , 50 ,60 ,70 ,80 ] , 'algorithm':["SAMME","SAMME.R"]}

ab_tuned = GridSearchCV( AdaBoostClassifier() , tuned_parameter , cv = 10 , scoring = "accuracy" )

ab_tuned.fit( x_fs_train , y_fs_train )

print("--> Accuracy ( GridSearchCV + selection ): ", ab_tuned.best_score_ )

print("best parameter: ",ab_tuned.best_params_)

data_plot.loc[ len( data_plot ) ] = [" adaboost ( GridSearchCV +  features selection ) " , ab_tuned.best_score_ ]
data_plot.sort_values( by = "Accuracy", ascending = False ).reset_index(drop = True )
# will separate data

data_female = data.loc[  data[ "label" ] == "female", : ]

data_male = data.loc[ data[ "label" ] == "male", :]

#plot

plotf1 = sns.distplot( data_female["meanfreq"] , color = "blue") 

plotm1 = sns.distplot( data_male["meanfreq"] , color = "red")
#see the relationship between some selected features

plot2 = sns.pairplot( data[ ["meanfreq","dfrange","meanfun","meandom","sfm","label"] ]  , hue = "label" , size = 3 )