import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None) #This displays all the columns instead of the default 20 
# train.csv contains the predictor variables and target variable, Attrition 

df = pd.read_csv('https://raw.githubusercontent.com/thepankj/IIT-G-Summer-Analytics-2020/master/train.csv').drop(['Attrition'], axis = 1)



df_test = pd.read_csv('https://raw.githubusercontent.com/thepankj/IIT-G-Summer-Analytics-2020/master/test.csv') #test.csv contains all the features except Attrition



target_var = pd.read_csv('https://raw.githubusercontent.com/thepankj/IIT-G-Summer-Analytics-2020/master/train.csv').Attrition



df
#combining the train and test datasets for preprocessing

df_combined = pd.concat([df, df_test], axis=0, sort = False, ignore_index = True)

df_combined 
(df_combined.dtypes).sort_values()
no_unique_values = df_combined.nunique().sort_values()

no_unique_values
df_combined.drop(['Behaviour', 'Id'], axis = 1, inplace = True)

df_combined.head()
#This checks for missing values 

sns.heatmap(df_combined.isnull(), yticklabels=False);
nom_col_names = list(df_combined.select_dtypes('object').columns)

df_hot_enc = pd.get_dummies(df_combined, columns=nom_col_names, drop_first=True) #one hot encoding the nominal variables

df_hot_enc.head()
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler(feature_range=(0, 1))

df_min_maxed = pd.DataFrame(scalar.fit_transform(df_hot_enc), columns=df_hot_enc.columns)

df_min_maxed.head()
numr_col_list = list(df_combined.select_dtypes('int64').columns)

numr_col_list.remove('PerformanceRating')

for _ in numr_col_list:

    fig, ax = plt.subplots(1, 3)

    sns.distplot(df_min_maxed[_], ax = ax[0], color='orange')

    sns.distplot(np.sqrt(df_min_maxed[_]), ax = ax[1], color='blue')

    sns.distplot(np.log(df_min_maxed[_]+1), ax = ax[2], color='green')
#applying the necessary transformations to the needed columns

sqrt_cols = ['DistanceFromHome', 'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',

             'YearsSinceLastPromotion', 'YearsWithCurrManager']

log_cols = ['MonthlyIncome', 'NumCompaniesWorked']



sqrt_transformed = df_min_maxed[sqrt_cols].apply(np.sqrt)

log_transformed = (df_min_maxed[log_cols]+1).apply(np.log)
#Replacing the columns with the transformed columns

temp = df_min_maxed.drop(sqrt_cols+log_cols, axis = 1)

df_transformed = pd.concat([temp, sqrt_transformed, log_transformed], axis = 1)

df_transformed
#selecting 30 best features for the model

from sklearn.feature_selection import SelectKBest, chi2



Xk = df_transformed.iloc[0:1628, :]

yk = target_var



bestFeatures = SelectKBest(score_func=chi2, k = 30)

fit = bestFeatures.fit(Xk, yk)



scores = pd.DataFrame(fit.scores_, index=Xk.columns, columns=['Scores'])



best30 = scores.sort_values(['Scores'], ascending = False)[0:30]

print(best30)

best30_features = best30.index



df_preprocessed = df_transformed[best30_features]
#Finally the Pre-processed DataFrame is

X = df_preprocessed.iloc[:1628, :]

y = target_var

X_test = df_preprocessed.iloc[1628:, :]

X_test.index = range(470)
#Splitting the dataset

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=4)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

logModel = LogisticRegression()

param_dist = {'C':list(range(1, 5)),

              'random_state':list(range(0,5)),

              'max_iter':[50, 100, 150, 200, 250]}

log_search = RandomizedSearchCV(logModel, param_distributions=param_dist, n_iter=50)

log_search.fit(X_train, y_train)
pred = log_search.predict(X_val)

accuracy_score(y_val, pred)
from sklearn.svm import SVC

svcModel = SVC(gamma="auto")

svcModel.fit(X_train, y_train)

pred = svcModel.predict(X_val)

accuracy_score(y_val, pred)