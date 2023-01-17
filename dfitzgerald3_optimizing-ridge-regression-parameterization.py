import pandas as pd

import numpy as np



from sklearn.linear_model import Ridge

from sklearn.preprocessing import Normalizer

from sklearn.cross_validation import cross_val_score



from scipy.stats import skew



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('ggplot')
train = '../input/train.csv'

test = '../input/test.csv'



df_train = pd.read_csv(train)

df_train_dummies = df_train.select_dtypes(include = ['object'])

df_train = df_train.select_dtypes(exclude = ['object'])



df_test = pd.read_csv(test)

df_test_dummies = df_test.select_dtypes(include = ['object'])

df_test = df_test.select_dtypes(exclude = ['object'])



df_train_dummies = pd.get_dummies(df_train_dummies)

df_test_dummies = pd.get_dummies(df_test_dummies)
def is_outlier(points, thresh = 3.5):

    if len(points.shape) == 1:

        points = points[:,None]

    median = np.median(points, axis=0)

    diff = np.sum((points - median)**2, axis=-1)

    diff = np.sqrt(diff)

    med_abs_deviation = np.median(diff)



    modified_z_score = 0.6745 * diff / med_abs_deviation



    return modified_z_score > thresh
# Define Target Data

target = df_train[df_train.columns.values[-1]]

target_log = np.log(target)



# Identify indexes of outliers

outliers = np.where(is_outlier(target))

outliers_log = np.where(is_outlier(target_log))



# Create arrays for plotting purposes

x = target.iloc[outliers].astype(float).values

y = np.zeros(len(outliers[0]))



x_log = target_log.iloc[outliers_log].astype(float).values

y_log = np.zeros(len(outliers_log[0]))



# Plot Original Data

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(target, bins=50)

plt.scatter(x, y, c='b', s=50)

plt.title('Original Data')

plt.xlabel('Sale Price')



# Plot Log Data

plt.subplot(1,2,2)

sns.distplot(target_log, bins=50)

plt.scatter(x_log, y_log, c='b', s=50)

plt.title('Natural Log of Data')

plt.xlabel('Natural Log of Sale Price')

plt.tight_layout()
df_train = df_train.drop(df_train.index[outliers_log])

df_train_dummies = df_train_dummies.drop(df_train_dummies.index[outliers_log])

target_log = target_log.drop(target_log.index[outliers_log])
def clean_data(data):

    for col in data.columns.values:

        if np.sum(data[col].isnull()) > 50:

            #print("Removing Column: {}".format(col))

            data = data.drop(col, axis = 1)

            

        elif np.sum(data[col].isnull()) > 0:

            #print("Replacing with Median: {}".format(col))

            median = data[col].median()

            data[col] = data[col].fillna(median)



            if skew(data[col]) > 0.75:

                #print("Skewness Detected: {}".format(col))

                data[col] = np.log(data[col])

                data[col] = data[col].apply(lambda x: 0 if x == -np.inf else x)



            data[col] = Normalizer().fit_transform(data[col].reshape(1,-1))[0]

            

#        else:

#            if skew(data[col]) > 0.75:

#                #print("Skewness Detected: {}".format(col))

#                data[col] = np.log(data[col])

#                data[col] = data[col].apply(lambda x: 0 if x == -np.inf else x)

#

#            data[col] = Normalizer().fit_transform(data[col].reshape(1,-1))[0]

            

    return(data)
df_train = clean_data(df_train[df_train.columns[:-1]])

df_train = df_train.join(df_train_dummies)

df_train['SalePrice'] = target_log



df_test = clean_data(df_test)

df_test = df_test.join(df_test_dummies)



np.where(df_train.isnull())

#np.where(df_test.isnull())
labels_train = df_train[df_train.columns[1:-1]]

target_train = df_train[df_train.columns[-1]]



labels_test = df_test[df_test.columns[1:]]
from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, False)



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

solvers = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']



cv_score = []

for i in solvers:

    for ii in alphas:

        clf = Ridge(alpha = ii, solver = i)

        cv_score.append(np.sqrt(-cross_val_score(estimator=clf, 

                                            X=labels_train, 

                                            y=target_train, 

                                            cv=15, 

                                            scoring = "neg_mean_squared_error")).mean())
cv_score = np.reshape(cv_score, (5,11))

score = pd.DataFrame(cv_score).T

score.columns = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']

score['alpha'] = alphas

score.set_index(['alpha'])



plt.figure(figsize=(10,5))

plt.bar(range(len(score[score.columns[:-1]].min().values)), score[score.columns[:-1]].min().values)

plt.title('Cross Validation Score')

plt.ylabel('RMSE')

plt.xlabel('Iteration')





score
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(labels_train, target_train)



clf = Ridge(alpha = 15)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



plt.figure(figsize=(10, 5))

plt.scatter(y_test, y_pred)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])

plt.title('Predicted vs. Actual')

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')

plt.tight_layout()





cv_score = np.sqrt(-cross_val_score(clf, 

                                    labels_train, 

                                    target_train, 

                                    cv=15, 

                                    scoring = "neg_mean_squared_error"))



print("RMSE Score: {}".format(cv_score.mean()))
# Fit model with training data

clf.fit(labels_train, target_train)



# Output feature importance coefficients, map them to their feature name, and sort values

coef = pd.Series(clf.coef_ , index = labels_train.columns).sort_values(ascending=False)



plt.figure(figsize=(10, 5))

coef.head(25).plot(kind='bar')

plt.title('Feature Significance')

plt.tight_layout()