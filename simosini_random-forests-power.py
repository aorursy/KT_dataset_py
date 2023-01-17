# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) 

# will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# I'm using python2 so I need true division between integers

from __future__ import division



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns

# option to print pandas objects with fewer digits

pd.set_option('display.float_format', lambda x: '%.3f' % x)



%matplotlib inline
#import and check data

data = pd.read_csv('../input/HR_comma_sep.csv')

data.head()

                   


print(data._get_numeric_data().shape[1], data.select_dtypes(include=['object']).shape[1])
#We only have 2 categorical features 'sales' and 'salary' so let's check them out

print(data.salary.unique(), data.sales.unique())
#group by salary we check how many left per group and then compare with total number per group

salary_left = data.groupby(['salary']).agg({'left': np.sum}) 

salary_count = data.salary.value_counts()

# concat the two columns

salary_tot = pd.concat([salary_count, salary_left], axis=1)

# add ratio per group of people left over total number

salary_tot['ratio'] = salary_tot.apply(lambda x: x['left'] / x['salary'], axis=1)

salary_tot
#the same with sales

sales_left = data.groupby(['sales']).agg({'left': np.sum})

sales_count = data.sales.value_counts()

# concat the two columns

sales_tot = pd.concat([sales_count, sales_left], axis=1)

# add ratio per group of people left over total number

sales_tot['ratio'] = sales_tot.apply(lambda x: x['left'] / x['sales'], axis=1)

sales_tot
#check if there is any nan values. A similar result could be achieved using data.info 

data.isnull().values.any()
dummies_data = pd.get_dummies(data)

dummies_data.shape
# compute correlation matrix to check if there are any linear relation between features

corr_mat = dummies_data.corr()
# We are actually interested in correlation between the target feature and the others

# select only best values and use seaborn to plot heatmap

target_corr = corr_mat['left'].abs() #even large negative correlations are meaningful

#we take the best 5 plus the obvious left column. This choice is due to the fact

#that all other features have very low correlation wrt target feature 'left' (< 0.02)

target_corr = target_corr.nlargest(6) 

cols = target_corr.index.tolist() # columns to use to plot correlation matrix

new_corr_mat = dummies_data[cols].corr()

plt.figure(figsize=[12,12]) #we only plot correlation between what we think are the best features

sns.heatmap(new_corr_mat, annot=True, fmt=".2f", cmap='viridis')

plt.show()
#filter data keeping only features with highest correlation

filtered_data = dummies_data[cols]

filtered_data.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
#a general function to evaluate our models



def evaluate(model, df, target_col, iterations=10, test_size=0.3):

    tot_score = 0.

    tot_rmse = 0.

    for i in range(iterations):

        train, test = train_test_split(df, test_size=test_size)

    

        X_train = train.drop([target_col], axis=1)

        y_train = train[target_col]

        X_test = test.drop([target_col], axis=1)

        y_test = test[target_col]    

    

        # Train the model using the training sets

        model.fit(X_train, y_train)



        # Coefficients: this is the m term in the formula f(x) = mx + q

        #print('Coefficients: \n', regr.coef_, regr.intercept_)

        pred = model.predict(X_test)

        #emp_pred_values = pd.Series(pred, index=X_test.index)

        #print(pd.concat([emp_pred_values.head(),y_test.head()], axis=1))

        # Mean squared error

        rmse = mean_squared_error(y_test, pred)**0.5

        print("Root mean squared error: %.2f"% rmse)

        # Explained variance score: 1 is perfect prediction

        score = model.score(X_test, y_test)

        print('Variance score: %.2f' % score)

        tot_rmse += rmse

        tot_score += score

    

    print('\nAverage score: %.2f' % float(tot_score/iterations))

    print('Average rmse: %.2f' % float(tot_rmse/iterations))
################## Helper functions #######################



from sklearn.metrics import confusion_matrix



# another general function yielding data needed for the confusion matrix

def predict(model, X_train, y_train, X_test) : #X_train, y_train to fit the model, X_test to evaluate it

    #fit and make predictions

    model.fit(X_train, y_train)

    return model.predict(X_test)



# a function to compute and plot the confusion matrix

def plot_conf_matrix(actual_values, predicted_values) :

    #compute confusion matrix

    conf_mat = confusion_matrix(predicted_values, actual_values)

    #name blocks

    idx = ['remained','left']

    #convert matrix to dataframe to plot

    df_cm = pd.DataFrame(conf_mat, index = idx, columns = idx)

    #plot confusion matrix

    plt.figure(figsize = (10,7))

    plt.xlabel('Actual values')

    plt.ylabel('Predicted values')

    sns.heatmap(df_cm, annot=True, fmt='d')

    

#prepare data for prediction

def split_data(df, target_column, test_size) :

    train, test = train_test_split(df, test_size=test_size)

    X_train = train.drop([target_column], axis=1)

    y_train = train[target_column]

    X_test = test.drop([target_column], axis=1)

    y_test = test[target_column]

    return X_train, y_train, X_test, y_test
logReg_model = LogisticRegression()

evaluate(logReg_model, df=filtered_data, iterations=10, target_col='left', test_size=0.3)
#we now plot a confusion matrix to check our model accuracy



#prepare train and test sets

X_train, y_train, X_test, y_test = split_data(filtered_data, 'left', 0.3)



#fit and make predictions

pred = predict(logReg_model, X_train, y_train, X_test)



#compute and plot confusion matrix

plot_conf_matrix(y_test, pred); #semicolon to avoid memory location printing of the graph
from sklearn import tree

tree_model = tree.DecisionTreeClassifier()

evaluate(tree_model, df=filtered_data, iterations=10, target_col='left', test_size=0.3)
#we now plot a confusion matrix to check our model accuracy



#prepare train and test sets

X_train, y_train, X_test, y_test = split_data(filtered_data, 'left', 0.3)



#fit and make predictions

pred = predict(tree_model, X_train, y_train, X_test)



#compute and plot confusion matrix

plot_conf_matrix(y_test, pred); #semicolon to avoid memory location printing of the graph
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()

evaluate(rf_model, df=filtered_data, iterations=10, target_col='left', test_size=0.3)
#we now plot a confusion matrix to check our model accuracy



#prepare train and test sets

X_train, y_train, X_test, y_test = split_data(filtered_data, 'left', 0.3)



#fit and make predictions

pred = predict(rf_model, X_train, y_train, X_test)



#compute and plot confusion matrix

plot_conf_matrix(y_test, pred); #semicolon to avoid memory location printing of the graph
numeric_features = data.drop(['sales','salary'], axis=1)

evaluate(df=numeric_features,iterations=5,model=rf_model,target_col='left',test_size=0.3)
#we now plot a confusion matrix to check our model accuracy



#prepare train and test sets

X_train, y_train, X_test, y_test = split_data(numeric_features, 'left', 0.3)



#fit and make predictions

pred = predict(rf_model, X_train, y_train, X_test)



#compute and plot confusion matrix

plot_conf_matrix(y_test, pred); #semicolon to avoid memory location printing of the graph
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

encoded = data

encoded.sales = le.fit_transform(encoded.sales)

encoded.salary = le.fit_transform(encoded.salary)
encoded.head()
evaluate(df=encoded,iterations=5,model=rf_model,target_col='left',test_size=0.3)
#we now plot a confusion matrix to check our model accuracy



#prepare train and test sets

X_train, y_train, X_test, y_test = split_data(encoded, 'left', 0.3)



#fit and make predictions

pred = predict(rf_model, X_train, y_train, X_test)



#compute and plot confusion matrix

plot_conf_matrix(y_test, pred); #semicolon to avoid memory location printing of the graph
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.metrics import classification_report



# we select the number of features to keep, in this case I choose at random 8

best_feats = SelectKBest(k=8)



# pipeline steps

steps = [('feature_selection', best_feats),

        ('random_forest', rf_model)]



# create pipeline

pipeline = Pipeline(steps)



#prepare train and test sets

X_train, y_train, X_test, y_test = split_data(encoded, 'left', 0.3)



#fit and make predictions

pred = predict(pipeline, X_train, y_train, X_test)



# test accuracy with metrics.classification_report this gives more info wrt to the confusion matrix

print(classification_report(y_test, pred))


# create parameters to evaluate with the GridSearchCV we use a dict here, 

# but it's not the only possible choice

# gridSearch will try any possible combination and choose the best for us, 

# this could be a bit slow so be patient.



# In order to change parameters we have to use the syntax name__parameter where the name

# is the one chosen in the pipeline steps and the parameters are the one of the model



parameters = dict(feature_selection__k=[6,8,9], #how many features to consider each iteration

                  random_forest__n_estimators=[15,20,25], #change number of estimators

                  random_forest__min_samples_split=[2,4] #vary number of sample splits

                 )



# prepare Grid Search

grid_search = GridSearchCV(pipeline, param_grid=parameters)



# prepare train and test sets

X_train, y_train, X_test, y_test = split_data(encoded, 'left', 0.3)





# make prediction on X_test

pred = predict(grid_search, X_train, y_train, X_test)



#print best parameters

print(grid_search.best_params_)



#print report 

report = classification_report(y_test, pred)

print(report)



plot_conf_matrix(y_test, pred)