import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines=False)

df.head()
df.describe() # Generate the summary table of the data
df.dtypes # Check the data types of all columns
df.isnull().sum() # Check if there's any missing value
from sklearn.preprocessing import OrdinalEncoder



encoding = {'language_code':{'en-US': 'eng', 'en-GB': 'eng', 'en-CA': 'eng'}} # Unify the langauge codes

df.replace(encoding, inplace=True)



enc = OrdinalEncoder()

enc.fit(df[['language_code']])

df[['language_code']] = enc.fit_transform(df[['language_code']]) # Apply ordinal encoding on language_code to convert it into numerical column
df['publication_date'] = pd.to_datetime(df['publication_date'], format='%m/%d/%Y', errors='coerce') # Convert data type of publication_date from object into date type

df[df['publication_date'].isnull()]
df.loc[df.bookID == 31373, 'publication_date'] = '1999-10-01 00:00:00'

df.loc[df.bookID == 45531, 'publication_date'] = '1975-10-01 00:00:00'
df['year'] = pd.DatetimeIndex(df['publication_date']).year # Extract year of publication in a separate column



df.rename(columns = {'  num_pages': 'num_pages'}, inplace=True) # Rename the column to remove leading whitespaces
df['num_occ'] = df.groupby('title')['title'].transform('count') # Add a new feature which has the number of occurences of each book
df['rate_occ'] = df['average_rating'] * df['num_occ']

df['rate_weight'] = df['average_rating'] * df['text_reviews_count']

df['rate_weight_2'] = df['average_rating'] * df['ratings_count']

df['rate_per_pages'] = df['average_rating'] * df['num_pages']
fig = plt.gcf()

fig.set_size_inches(26, 10)

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True)
sns.relplot(x="num_occ", y="average_rating", data=df, height=7, aspect = 2)
fig = plt.gcf()

fig.set_size_inches(26, 10)

sns.lineplot(x="year", y="average_rating", data=df)
sns.relplot(x="language_code", y="average_rating", data=df, height=9, aspect = 2)
sns.relplot(x="text_reviews_count", y="average_rating", data=df, height=9, aspect = 2)
sns.relplot(x="num_pages", y="average_rating", data=df, height=9, aspect = 2)
fig = plt.gcf()

fig.set_size_inches(26, 10)

sns.lineplot(x="year", y="text_reviews_count", data=df)
label = df['average_rating'].values

df.drop(['bookID', 'title', 'authors', 'isbn', 'isbn13', 'publication_date', 'publisher', 'average_rating'], axis=1, inplace=True)
# Split the Data into 70% - 30%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.3)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor



model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))



parameters = {

    'learning_rate': [0.001, 0.01, 0.02, 0.1, 0.2, 1.0],

    'n_estimators': [10, 50, 100, 200]

}



grad_Ada = GridSearchCV(model, parameters, refit=True)

grad_Ada.fit(X_train, y_train)



print('Best Score: ', grad_Ada.best_score_*100, '\nBest Parameters: ', grad_Ada.best_params_)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV



model  = LinearRegression()



parameters = {

    'fit_intercept': [True, False],

    'normalize': [True, False],

    

}



grad_Linear = GridSearchCV(model, parameters, refit=True)

grad_Linear.fit(X_train, y_train)



print('Best Score: ', grad_Linear.best_score_*100, '\nBest Parameters: ', grad_Linear.best_params_)
from sklearn.linear_model import Ridge



model = Ridge()



parameters = {

    'fit_intercept': [True, False],

    'normalize': [True, False],

    'max_iter': [1000, 100, 10000],

    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}



grad_ridge = GridSearchCV(model, parameters, refit=True)

grad_ridge.fit(X_train, y_train)



print('Best Score: ', grad_ridge.best_score_*100, '\nBest Parameters: ', grad_ridge.best_params_)
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor()



parameters = {

    'n_estimators': [50, 100, 150, 200],

    'max_depth': [3, 5, 7, 10, 12, 15],

    'min_samples_split': [5, 10, 15],

    'min_samples_leaf': [5, 10, 15]

}



grad_rf = GridSearchCV(model, parameters, refit=True, cv=10)

grad_rf.fit(X_train, y_train)



print('Best Score: ', grad_rf.best_score_*100, '\nBest Parameters: ', grad_rf.best_params_)
l = []

l.append(('AdaBoost', grad_Ada.best_score_*100))

l.append(('Linear Regression', grad_Linear.best_score_*100))

l.append(('Ridge Regression', grad_ridge.best_score_*100))

l.append(('Random Forest', grad_rf.best_score_*100))

scores = pd.DataFrame(l, columns =['Model', 'Train Score'])
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# AdaBoost Model

pred_adaboost = grad_Ada.predict(X_test)



# Check Model Score

print("Residual sum of squares: ",  np.mean((pred_adaboost - y_test) ** 2))

print('RMSE: '+str(np.sqrt(mean_squared_error(y_test, pred_adaboost))))

print('Model Score on Test Data: ', grad_Ada.score(X_test, y_test))
from eli5.sklearn import PermutationImportance

import eli5

perm = PermutationImportance(grad_Ada.best_estimator_, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plt.figure(figsize=(19,10))

sns.regplot(pred_adaboost, y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})
# Linear Regression Model

pred_lr = grad_Linear.predict(X_test)



# Check Model Score

print("Residual sum of squares: ",  np.mean((pred_lr - y_test) ** 2))

print('RMSE: '+str(np.sqrt(mean_squared_error(y_test, pred_lr))))

print('Model Score on Test Data: ', grad_Linear.score(X_test, y_test))
perm = PermutationImportance(grad_Linear.best_estimator_, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plt.figure(figsize=(19,10))

sns.regplot(pred_lr, y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})
# Ridge Regression Model

pred_ridge = grad_ridge.predict(X_test)



# Check Model Score

print("Residual sum of squares: ",  np.mean((pred_ridge - y_test) ** 2))

print('RMSE: '+str(np.sqrt(mean_squared_error(y_test, pred_ridge))))

print('Model Score on Test Data: ', grad_ridge.score(X_test, y_test))
perm = PermutationImportance(grad_ridge.best_estimator_, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plt.figure(figsize=(19,10))

sns.regplot(pred_ridge,y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})
# Random Forest Model

pred_rf = grad_rf.predict(X_test)



# Check Model Score

print("Residual sum of squares: ",  np.mean((pred_rf - y_test) ** 2))

print('RMSE: '+str(np.sqrt(mean_squared_error(y_test, pred_rf))))

print('Model Score on Test Data: ', grad_rf.score(X_test, y_test))
perm = PermutationImportance(grad_ridge.best_estimator_, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plt.figure(figsize=(19,10))

sns.regplot(pred_rf,y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})
l2 = []

l2.append(('AdaBoost', grad_Ada.score(X_test, y_test)*100))

l2.append(('Linear Regression', grad_Linear.score(X_test, y_test)*100))

l2.append(('Ridge Regression', grad_ridge.score(X_test, y_test)*100))

l2.append(('Random Forest', grad_rf.score(X_test, y_test)*100))



test_scores = pd.DataFrame(l2, columns =['Model', 'Test Score'])
scores['Test Score'] = test_scores['Test Score']

scores
scores.plot.bar()