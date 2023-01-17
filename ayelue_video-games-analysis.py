import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import chi2_contingency

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split, GridSearchCV 

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge, Lasso, MultiTaskLasso

from sklearn.metrics import r2_score

%matplotlib inline
df = pd.read_csv('../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv')
df.isna().sum()
df.rename(columns={'Name': 'name',

                    'Platform': 'platform',

                    'Year_of_Release': 'year',

                    'Genre': 'genre',

                    'Publisher': 'publisher',

                    'NA_Sales': 'na_sales',

                    'EU_Sales': 'eu_sales',

                    'JP_Sales': 'jp_sales',

                    'Other_Sales': 'other_sales',

                    'Global_Sales': 'global_sales',

                    'Critic_Score': 'critic_score',

                    'Critic_Count': 'critic_count',

                    'User_Score': 'user_score',

                    'User_Count': 'user_count',

                    'Developer': 'developer',

                    'Rating': 'rating'},inplace=True)
df.columns
cat = ['platform','genre','publisher','developer','rating']

num = ['year','na_sales',

       'eu_sales', 'jp_sales', 'other_sales', 'global_sales', 'critic_score',

       'critic_count', 'user_score', 'user_count']
corr_matrix = df[num].corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr_matrix,annot=True)
df['global_sales'].describe()
# sns.lmplot(x='critic_count',y='global_sales', data=df)

values = df[~df['global_sales'].isna()]['global_sales']

values = shuffle(values)

x = list(range(len(values)))

plt.figure(figsize=(20,20))

plt.plot(x, values, 'D')

plt.show()
df = df[df['global_sales']<50]
values = df[~df['na_sales'].isna()]['na_sales']

values = shuffle(values)

x = list(range(len(values)))

plt.figure(figsize=(20,20))

plt.plot(x, values, 'D')

plt.show()
df = df[df['na_sales']<16]
values = df[~df['eu_sales'].isna()]['eu_sales']

values = shuffle(values)

x = list(range(len(values)))

plt.figure(figsize=(20,20))

plt.plot(x, values, 'D')

plt.show()
df = df[df['eu_sales']<10]
values = df[~df['jp_sales'].isna()]['jp_sales']

values = shuffle(values)

x = list(range(len(values)))

plt.figure(figsize=(20,20))

plt.plot(x, values, 'D')

plt.show()
df = df[df['jp_sales']<6]
values = df[~df['other_sales'].isna()]['other_sales']

values = shuffle(values)

x = list(range(len(values)))

plt.figure(figsize=(20,20))

plt.plot(x, values, 'D')

plt.show()
df = df[df['other_sales']<4]
sns.pairplot(df[['na_sales','eu_sales','jp_sales','other_sales']])
name_sales = ['na_sales','eu_sales','jp_sales','other_sales','global_sales']
result = {sale_name : {c_name: chi2_contingency(pd.crosstab(df[sale_name], df[c_name]))[1] for c_name in cat} for sale_name in name_sales}
result
df['na_sales']
# plt.figure(figsize=(10,10))



# sns.distplot(df['na_sales'])

# # sns.distplot(df['eu_sales'])

# # sns.distplot(df['jp_sales'])
df[df['name'].isna()]
df = df[~df['name'].isna()]
df['name'].isna().sum()
df[df['year'].isna()]
df.loc[:,'year'] = df.loc[:,'year'].fillna(df['year'].median())
df['year'].isna().sum()
df['publisher'].isna().sum()
df.loc[:,'publisher'] = df.loc[:,'publisher'].fillna(df['publisher'].mode()[0])
df['publisher'].isna().sum()
df['critic_score'].isna().sum()
df.loc[:,'critic_score'] = df.loc[:,'critic_score'].fillna(df['critic_score'].median())
df['critic_score'].isna().sum()
df['critic_count'].isna().sum()
df.loc[:,'critic_count'] = df.loc[:,'critic_count'].fillna(df['critic_count'].median())
df['critic_count'].isna().sum()
df['user_score'].isna().sum()
df['user_score'].describe()
df.loc[:,'user_score'] = df.loc[:,'user_score'].apply(lambda x: None if x=='tbd' else x)
df['user_score'] = df['user_score'].astype('float')
df.loc[:,'user_score'] = df.loc[:,'user_score'].fillna(df['user_score'].median())
df['user_score'].isna().sum()
df['user_count'].isna().sum()
df['user_count'].describe()
df.loc[:,'user_count'] = df.loc[:,'user_count'].fillna(df['user_count'].median())
df['user_count'].isna().sum()
df['developer'].isna().sum()
df.loc[:,'developer'] = df.loc[:,'developer'].fillna('Unknown')
df['developer'].isna().sum()
df['rating'].isna().sum()
df['rating'].describe()
df['rating'].value_counts()
df.loc[:,'rating'] = df.loc[:,'rating'].fillna(df['rating'].mode()[0])
df['rating'].isna().sum()
df_model = df.drop(['name','publisher','developer','other_sales','global_sales'],axis=1)
df_model = pd.get_dummies(df_model)
df_model
corr_matrix = df[['na_sales',

       'eu_sales', 'jp_sales', 'other_sales']].corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr_matrix,annot=True)
def prepare_data(y_name):    

    labels = df_model[y_name].to_numpy()

    if y_name!='jp_name':   

        features = df_model.drop([y_name],axis=1)

    else:

        features = df_model.drop([y_name,'na_sales'],axis=1)

    features_names = list(features.columns)

    features = features.to_numpy()

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

    return train_features, test_features, train_labels, test_labels
# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'bootstrap': [True],

#     'max_depth': [80, 90, 100, 110],

#     'max_features': [2, 3],

#     'min_samples_leaf': [3, 4, 5],

#     'min_samples_split': [8, 10, 12],

#     'n_estimators': [100, 200, 300, 1000]

#     }

# # Create a based model

# rf = RandomForestRegressor()

# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3,  verbose = 2)
predict_values_names = ['na_sales','eu_sales','jp_sales']
param_grid_rfr = {'bootstrap': [True],

 'max_depth': [90],

 'max_features': [3],

 'min_samples_leaf': [3],

 'min_samples_split': [8],

 'n_estimators': [100]}





param_grid_lin = {'alpha': [0.01+i*0.05 for i in range(20)]}

models = {

            'Lasso': GridSearchCV(estimator = Lasso(), param_grid = param_grid_lin, cv = 3),

            'Ridge': GridSearchCV(estimator = Ridge(), param_grid = param_grid_lin, cv = 3),

            'RFR': GridSearchCV(estimator = RandomForestRegressor(), param_grid = param_grid_rfr, cv = 3)

         }
result = {}

for predict_name in predict_values_names:

    result[predict_name] = {}

    train_features, test_features, train_labels, test_labels = prepare_data(predict_name)

    for key in models.keys():

        models[key].fit(train_features, train_labels)

        result[predict_name][key] = {}

        result[predict_name][key]['r^2 train data'] = r2_score(train_labels,models[key].best_estimator_.predict(train_features))

        predictions = models[key].best_estimator_.predict(test_features)

        result[predict_name][key]['r^2 test data'] = r2_score(test_labels,predictions)

        se_predict = predictions.std()/len(predictions)**0.5

        result[predict_name][key]['confidence_interval'] =  [predictions.mean()-1.96*se_predict,predictions.mean()+1.96*se_predict]

        result[predict_name][key]['test_mean'] = test_labels.mean()
# grid_search.fit(train_features, train_labels)

# grid_search.best_params_

# # best_grid = grid_search.best_estimator_

# # grid_accuracy = evaluate(best_grid, test_features, test_labels)
result_list = []

for key in result.keys():

    

    result_list.append((key,pd.DataFrame(result[key])))
print(result_list[0][0])

result_list[0][1]

print(result_list[1][0])

result_list[1][1]
print(result_list[2][0])

result_list[2][1]