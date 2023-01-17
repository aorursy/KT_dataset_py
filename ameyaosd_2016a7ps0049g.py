import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



pd.set_option('display.max_columns', 100)
# Change name of .csv file if necessary

df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
df.head()
df.info()
# replace "?" with NaN

# df.replace('?',np.nan, inplace = True)
df.isnull().head(5)
# Count number of NaNs in each column

missing_count = df.isnull().sum(axis = 0) #axis = 1 to count in each row

missing_count[missing_count > 0]
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
numerical_features = []

# Convert these numerical features to float

df[numerical_features] = df[numerical_features].astype(np.float)
df.dtypes
df.fillna(df.mean(), inplace = True)
# df['---'].value_counts()
# df['---'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent 

# df['num-of-doors'].fillna(df['num-of-doors'].value_counts().idxmax(), inplace = True)
df.head()
df.isnull().any().any()
df.describe()
df.describe(include='object')
df['type'].unique()
df_group = df[['type','feature1','rating']]
grouped_test1 = df_group.groupby(['type'],as_index=False).mean()

grouped_test1
for i, col in enumerate(df.columns):

  if col not in ['type','rating']:

    plt.figure(i)

    sns.regplot(x=col, y = 'rating', data=df)

    # print(col)
sns.boxplot(x = 'rating',y='type', data = df)
df.corr()
# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
numerical_features = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']

categorical_features = ['type']
X = df[numerical_features + categorical_features].copy()

y = df['rating'].copy()
X.head()
y.head()
for i, col in enumerate(numerical_features):

    plt.figure(i)

    plt.hist(X[col], bins=50)

    plt.xlabel(col)

    plt.show()
from statsmodels.graphics.gofplots import qqplot

for i, col in enumerate(numerical_features):

    plt.figure(i)

    qqplot(X[col], line = 's')

    plt.xlabel(col)

    plt.show()
temp_code = {'old':0,'new':1}

X['type'] = X['type'].map(temp_code)

X.head()
from sklearn.model_selection import train_test_split

import random



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state=random.randint(1,500))  #Checkout what does random_state do

#TODO

from sklearn.preprocessing import RobustScaler



transformer1 = RobustScaler().fit(X_train[numerical_features])

transformer1.transform(X_train[numerical_features])



transformer2 = RobustScaler().fit(X_val[numerical_features])

transformer2.transform(X_val[numerical_features])



X_train[numerical_features].head()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X_train_lda = lda.fit_transform(X_train, y_train)

print(X_train_lda)



X_val_lda = lda.transform(X_val)

print(X_val_lda)

# X_train_lda = X_train

# X_val_lda = X_val
# min_max_scaler_features = ['feature1','feature6','feature8']

# standard_scaler_features = ['feature4','feature7']

# robust_scaler_features = ['feature2','feature3','feature5','feature9','feature10','feature11']
X.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# Initialize and train

clf1 = DecisionTreeClassifier().fit(X_train_lda,y_train)

clf2 = RandomForestClassifier().fit(X_train_lda,y_train)
from sklearn.metrics import accuracy_score  #Find out what is accuracy_score

from sklearn.metrics import r2_score, mean_squared_error, make_scorer



y_pred_1 = clf1.predict(X_val_lda)

y_pred_2 = clf2.predict(X_val_lda)



acc1 = accuracy_score(y_pred_1,y_val)*100

acc2 = accuracy_score(y_pred_2,y_val)*100



print("Accuracy score of clf1: {}".format(acc1))

print("Accuracy score of clf2: {}".format(acc2))



mse1 = mean_squared_error(y_val, y_pred_1)

mse2 = mean_squared_error(y_val, y_pred_2)



print("Mse score of clf1: {}".format(mse1))

print("Mse score of clf2: {}".format(mse2))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



#TODO

clf = RandomForestClassifier()        #Initialize the classifier object



parameters = {'n_estimators':[10,30,50,80,100],

              'criterion': ['gini','entropy']

              }    #Dictionary of parameters



scorer = make_scorer(mean_squared_error)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train_lda,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf.fit(X_train_lda, y_train)).predict(X_val_lda)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_val_lda)        #Same, but use the best estimator



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression



reg = LinearRegression().fit(X_train_lda, y_train)

y_pred1 = reg.predict(X_val_lda)

y_pred1 = np.rint(y_pred1)



regr = RandomForestRegressor(n_estimators=100).fit(X_train_lda, y_train)

y_pred2 = regr.predict(X_val_lda)

y_pred2 = np.rint(y_pred2)



mse1 = mean_squared_error(y_val, y_pred1)

mse2 = mean_squared_error(y_val, y_pred2)



print("Mse score of clf1: {}".format(mse1))

print("Mse score of clf2: {}".format(mse2))
clf = RandomForestRegressor()        #Initialize the classifier object



parameters = {'n_estimators':[10,30,50,80,100,200,250,300],

              'criterion': ['mse']

              }    #Dictionary of parameters



scorer = make_scorer(mean_squared_error)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer, verbose = 2)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train_lda,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(best_clf)



unoptimized_predictions = (clf.fit(X_train_lda, y_train)).predict(X_val_lda)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_val_lda)        #Same, but use the best estimator



unoptimized_predictions = np.rint(unoptimized_predictions)

optimized_predictions = np.rint(optimized_predictions)



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))



mse1 = mean_squared_error(y_val,unoptimized_predictions)

mse2 = mean_squared_error(y_val,optimized_predictions)



print("mse unop ",mse1)

print("mse op",mse2)
test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
test.head()
test_X = test[numerical_features + categorical_features].copy()
test_X.isnull().any().any()
missing_count = test_X.isnull().sum(axis = 0) #axis = 1 to count in each row

missing_count[missing_count > 0]
test_X_dtype_nunique = pd.concat([test_X.dtypes, test_X.nunique()],axis=1)

test_X_dtype_nunique.columns = ["dtype","unique"]

test_X_dtype_nunique
test_X.fillna(test_X.mean(), inplace = True)
test.head()
temp_code = {'old':0,'new':1}

test_X['type'] = test_X['type'].map(temp_code)

test_X.head()
# Make sure that the scaler is same as the one used before

transformer = RobustScaler().fit(test_X[numerical_features])

transformer.transform(test_X[numerical_features])
test_X = lda.transform(test_X)
# Enter the classifier

predicted_val = best_clf.predict(test_X)

predicted_val = np.rint(predicted_val)
my_submission = pd.DataFrame({'Id': test.id, 'rating': predicted_val})
my_submission.to_csv('submission.csv', index=False)