# Load packages and data.



# Packages for data manipulation and data visualization.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Packages for data modelling. 

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.grid_search import GridSearchCV



%matplotlib inline



df = pd.read_csv("../input/diabetes.csv")
# Inspect the data set.



df.shape
df.info()
# We can see that the data frame is made up of numerical data only. 



df.describe()
# Some takeaways from the data above:

# Around 35% of the patients have diabetes.

# There average age is around 33.

# The bulk of patients are aged between 24 and 41.

# There is a wide diversity in insulin levels.

# The patients have had just under 4 pregnancies on average.



df.head()
# Check for nulls.



df.apply(lambda x: sum(x.isnull()))
# There are no null values in the data set. We tend to see clean data sets like this on UCI.



# Plot a heatmap using seaborn. 



corr = df.corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax=1, square=True)
# The heatmap shows that all of the variables have some degree of correlation

# with the target variable 'outcome'. The variable with the strongest correlation

# is glucose. We can also see that there is correlation between the variables.

# Some of these make intuitive sense e.g pregnancies and age.
# Show precise correlations with the target variable.



cor_dict = corr['Outcome'].to_dict()

del cor_dict['Outcome']

print("List the numerical features decendingly by their correlation with Outcome:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*ele))
# Plot a pairplot using seaborn.



sns.pairplot(df)
sns.distplot(df.BMI, bins=40) 
# BMI guide



#under 18.5 - underweight

#18.5 to 25 - healthy

#25 to 30 - overweight

#over 30 - obese



# Most people in the data set are overweight or obese. Worryingly there are even people with

# very low BMIs (this could be an error with caused during data entry).
# Let's make our data ready for modelling. 



# Separate the target and features.

target = df.Outcome

features = df.drop('Outcome', axis=1)



# The column names will be used later to help us make sense of the models.

cols = features.columns.values
# Standardize features by removing the mean and scaling to unit variance.

standard_scaler = StandardScaler()

features = standard_scaler.fit_transform(features)
# Split the data up in train and test sets using Sklearn's train_test_split module.



X_train, X_test, y_train, y_test = train_test_split(features, target)
# Create a function for running logisitic regression on the data. 

# Use grid search to find optimal hyperparameters.

# The reason we have to use a function is due to the way Python uses parallelization on Windows.

# http://tinyurl.com/h3g3m8m

# For more info on Sklearn's logisitc regression function please visit the link below.

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html



def log_reg_model(X_train, X_test, y_train, y_test):

    if __name__ == '__main__':



        param_grid = {'penalty' : ['l1', 'l2'],

                      'C' : [0.001, 0.01, 0.1, 1, 10]}



        classifier = GridSearchCV(estimator=LogisticRegression(),

                                  param_grid=param_grid,

                                  n_jobs=-1,

                                  cv=3)



        classifier.fit(X_train, y_train)



        best_params = classifier.best_params_



        print('Best parameters: ', best_params)



        validation_accuarcy = classifier.score(X_test, y_test)

        

        print('Validation accuracy: ', validation_accuarcy)



        coefficients = classifier.best_estimator_.coef_

        print('Coefficients: ', list(zip(cols, coefficients[0])))
#Find model with best paramters.



log_reg_model(X_train, X_test, y_train, y_test)
# Results from 5 runs of the model.



'''

1st run: Best parameters:  {'penalty': 'l1', 'C': 1}

         Validation accuracy:  0.776041666667

    

2nd run: Best parameters:  {'penalty': 'l1', 'C': 1}

         Validation accuracy:  0.734375

    

3rd run: Best parameters:  {'penalty': 'l1', 'C': 0.1}

         Validation accuracy:  0.78125

    

4th run: Best parameters:  {'penalty': 'l2', 'C': 1}

         Validation accuracy:  0.791666666667

    

5th run: Best parameters:  {'penalty': 'l2', 'C': 0.1}

         Validation accuracy:  0.760416666667

'''
avg_val_accuarcy = ((0.776 + 0.734 + 0.781 + 0.791 + 0.76) / 5) * 100

print('average validation accuracy: ',

      round(avg_val_accuarcy,2), '%')
# Let's see if using Linear SVC will provide a better model for our data.

# For more info on Linear SVC in Sklearn please see the link below.

# http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html



def svc_model(X_train, X_test, y_train, y_test):

    if __name__ == '__main__':



        param_grid = {'C':[0.001, 0.01, 0.1, 1.0],

                      'class_weight':[None, 'balanced']}



        classifier = GridSearchCV(estimator=LinearSVC(),

                                  param_grid=param_grid,

                                  n_jobs=-1,

                                  cv=3)



        classifier.fit(X_train, y_train)



        best_params = classifier.best_params_



        print('Best parameters: ', best_params)



        validation_accuarcy = classifier.score(X_test, y_test)

        

        print('Validation accuracy: ', validation_accuarcy)



        coefficients = classifier.best_estimator_.coef_

        print('Coefficients: ', list(zip(cols, coefficients[0])))
# Results from 5 runs of the model.



'''

1st run: Best parameters:  {'C': 0.1, 'class_weight': None}

         Validation accuracy:  0.796875

        

2nd run: Best parameters:  {'class_weight': None, 'C': 0.01}

         Validation accuracy:  0.713541666667



3rd run: Best parameters:  {'class_weight': None, 'C': 0.1}

         Validation accuracy:  0.802083333333



4th run: Best parameters:  {'class_weight': None, 'C': 0.1}

         Validation accuracy:  0.760416666667



5th run: Best parameters:  {'class_weight': None, 'C': 0.001}

         Validation accuracy:  0.734375

'''
avg_val_accuarcy = ((0.797 + 0.714 + 0.802 + 0.76 + 0.734) / 5) * 100

print('average validation accuracy: ',

      round(avg_val_accuarcy,2), '%')
# We see a similar performance to logisitc regression. Next up it's a gradient boosting ensemble

# algorithm: Gradient Boosting Classifier.



# Note numerous different values were used in the param_grid to hone in on the best paramater

# combinations. The param grid below is what I ended up with after running the model several 

# times.



def gradient_boosting_model(X_train, X_test, y_train, y_test):

    if __name__ == '__main__':



        param_grid = {'learning_rate': [0.015, 0.013, 0.011],

                      'max_depth': [20, 25, None],

                      'min_samples_leaf': [9, 10, 11],

                      'max_features': [0.25, 0.27, 0.3],

                      'n_estimators': [225, 250, 235]} 



        classifier = GridSearchCV(estimator=GradientBoostingClassifier(),

                                  param_grid=param_grid,

                                  n_jobs=-1,

                                  cv=5)



        classifier.fit(X_train, y_train)



        best_params = classifier.best_params_



        print('Best parameters: ', best_params)



        validation_accuarcy = classifier.score(X_test, y_test)

        

        print('Validation accuracy: ', validation_accuarcy)



        feature_importances = classifier.best_estimator_.feature_importances_

        

        print('Feature importances: ', feature_importances)
# Results from 5 runs of the model.



'''

1st run: Validation accuracy:  0.776

         Best Parameters: {'max_depth': 20, 'min_samples_leaf': 9,

         'n_estimators': 250, 'max_features': 0.25, 'learning_rate': 0.011}

        

2nd run: Validation accuracy:  0.776

         Best Paramaters: {'learning_rate': 0.011, 'min_samples_leaf': 9,

                 'n_estimators': 225, 'max_features': 0.25, 'max_depth': None}



3rd run: Validation accuracy:  0.781

         Best Parameters: {'max_depth': 20, 'min_samples_leaf': 11, 'n_estimators': 250,

                 'max_features': 0.27, 'learning_rate': 0.013}



4th run: Validation accuracy:  0.781

         Best Parameters: {'max_depth': 20, 'n_estimators': 225, 'max_features': 0.3,

                'min_samples_leaf': 9, 'learning_rate': 0.011}



5th run: Validation accuracy:  0.786

         Best Parameters: {'learning_rate': 0.015, 'max_depth': 20,

              'n_estimators': 250,'min_samples_leaf': 11, 'max_features': 0.3}

'''    
avg_val_accuarcy = ((0.776 + 0.7776 + 0.781 + 0.781 + 0.786) / 5) * 100

print('average validation accuracy: ',

      round(avg_val_accuarcy,2), '%')
# The more complex model marginally improved the accuracy on the validation set.