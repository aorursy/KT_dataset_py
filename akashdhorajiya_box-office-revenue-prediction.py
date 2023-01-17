import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt



box_train = pd.read_csv("../input/train.csv", index_col='id')
print(box_train.info())
print(box_train.describe())
example = 2

for i in range(len(box_train.iloc[example])):

    print(box_train.columns[i],' ==> ', box_train.iloc[example,i])
box_train.loc[:,['original_title','title']].head(10)
box_train.loc[:,['homepage']].head()
print(box_train.status.unique())

rumored = box_train[box_train['status'] == 'Rumored']

rumored
box_train.nunique()
box_test = pd.read_csv("../input/test.csv", index_col='id')



box = pd.concat([box_train.drop('revenue', axis=1), box_test])

box.info()
not_useful_columns = ['homepage', 'imdb_id', 'original_title', 'overview', 'poster_path',

                     'tagline', 'title']

box_train.drop(not_useful_columns, axis=1, inplace=True)

box_train.info()
num_box = box_train[['budget', 'popularity', 'runtime', 'revenue']]
import seaborn



corr_mx = num_box.corr()

print(corr_mx)

seaborn.heatmap(corr_mx)

plt.show()
num_box.hist(bins=100, figsize=(16,12))

plt.show()
import ast



# Function to transfer unique dictionary value in DataFrame



def get_as_dataframe(df, column):

    as_dataframe = pd.DataFrame()

    for i in range(1,len(df)+1):

        if str(df.loc[i, column]) != 'nan':

            temp_list = ast.literal_eval(df.loc[i, column])

            temp = pd.DataFrame(temp_list)

            as_dataframe = pd.concat([as_dataframe, temp], ignore_index=True)

    return as_dataframe.drop_duplicates()
def get_as_dataframe_for_top_5_cast(df, column):

    as_dataframe = pd.DataFrame()

    for i in range(1,len(df)+1):

        if str(df.loc[i, column]) != 'nan':

            temp_list = ast.literal_eval(df.loc[i, column])

            if temp_list:

                temp = pd.DataFrame(temp_list)

                temp = temp.iloc[:5,:]

                as_dataframe = pd.concat([as_dataframe, temp], ignore_index=True)

    return as_dataframe.drop_duplicates()
def get_as_dataframe_for_director(df, column):

    as_dataframe = pd.DataFrame()

    for i in range(1,len(df)+1):

        if str(df.loc[i, column]) != 'nan':

            temp_list = ast.literal_eval(df.loc[i, column])

            if temp_list:

                temp = pd.DataFrame(temp_list)

                temp = temp[temp['job'] == 'Director']

                as_dataframe = pd.concat([as_dataframe, temp], ignore_index=True)

    return as_dataframe.drop_duplicates()
dataframes = {}
def prepare_as_dataframe(df, column, index_col):

    temp = get_as_dataframe(df, column)

    print(column,' - length : ',str(len(temp)))

    temp = temp.set_index(index_col).sort_index()

    temp['count'], temp['popularity'], temp['budget'] = 0, 0.0, 0.0

    return temp
def prepare_belongs_to_collection_as_dataframe(df):

    temp = get_as_dataframe(df, 'belongs_to_collection')

    temp = temp[['id', 'name']]

    print('belongs_to_collection - length : ', str(len(temp)))

    temp = temp.set_index('id').sort_index()

    temp['count'], temp['popularity'], temp['budget'] = 0, 0.0, 0.0

    return temp
def prepare_top_5_cast_as_dataframe(df):

    temp = get_as_dataframe_for_top_5_cast(df, 'cast')

    temp = temp[['id', 'name']].drop_duplicates()

    temp = temp.set_index('id').sort_index()

    print('cast - length : ', str(len(temp)))

    temp['count'], temp['popularity'], temp['budget'] = 0, 0.0, 0.0

    return temp
def prepare_director_as_dataframe(df):

    temp = get_as_dataframe_for_director(df, 'crew')

    temp = temp[['id', 'name']].drop_duplicates()

    temp = temp.set_index('id').sort_index()

    print('director - length : ', str(len(temp)))

    temp['count'], temp['popularity'], temp['budget'] = 0, 0.0, 0.0

    return temp
dataframes['belongs_to_collection'] = prepare_belongs_to_collection_as_dataframe(box)

dataframes['genres'] = prepare_as_dataframe(box, 'genres', 'id')

dataframes['production_companies'] = prepare_as_dataframe(box, 'production_companies', 'id')

dataframes['production_countries'] = prepare_as_dataframe(box, 'production_countries', 'iso_3166_1')

dataframes['spoken_languages'] = prepare_as_dataframe(box, 'spoken_languages', 'iso_639_1')

dataframes['Keywords'] = prepare_as_dataframe(box, 'Keywords', 'id')

dataframes['cast'] = prepare_top_5_cast_as_dataframe(box)

dataframes['crew'] = prepare_director_as_dataframe(box)
dataframes['belongs_to_collection'].head()
dataframes['genres'].head()
dataframes['production_companies'].head()
dataframes['production_countries'].head()
dataframes['spoken_languages'].head()
dataframes['Keywords'].head()
dataframes['cast'].head()
dataframes['crew'].head()
def add_popularity_and_budget(df, raw_id, popularity, budget):

    cnt = df.loc[raw_id, 'count']

    pop = df.loc[raw_id, 'popularity']

    bdgt = df.loc[raw_id, 'budget']

    pop = (pop*cnt + popularity)

    bdgt = (bdgt*cnt + budget)

    cnt = cnt+1

    pop = pop/cnt

    bdgt = bdgt/cnt

    df.loc[raw_id, 'count'] = cnt

    df.loc[raw_id, 'popularity'] = pop   

    df.loc[raw_id, 'budget'] = bdgt
def update_popularity_and_budget(df, column, index_name):

    for i in range(1,len(df)+1):

        if str(df.loc[i, column]) != 'nan':

            temp_list = ast.literal_eval(df.loc[i, column])

            if temp_list:

                temp = pd.DataFrame(temp_list)

                temp = temp.set_index(index_name)

                for j in temp.index:

                    add_popularity_and_budget(dataframes[column], j, df.loc[i,'popularity'], df.loc[i, 'budget'])   
def update_popularity_and_budget_for_top_5_cast(df, column, index_name):

    for i in range(1,len(df)+1):

        if str(df.loc[i, column]) != 'nan':

            temp_list = ast.literal_eval(df.loc[i, column])

            if temp_list:

                temp = pd.DataFrame(temp_list)

                temp = temp.iloc[:5,:]

                temp = temp.set_index(index_name)

                for j in temp.index:

                    add_popularity_and_budget(dataframes[column], j, df.loc[i,'popularity'], df.loc[i, 'budget'])
def update_popularity_and_budget_for_directors(df, column, index_name):

    for i in range(1,len(df)+1):

        if str(df.loc[i, column]) != 'nan':

            temp_list = ast.literal_eval(df.loc[i, column])

            if temp_list:

                temp = pd.DataFrame(temp_list)

                temp = temp[temp['job'] == 'Director']

                temp = temp.set_index(index_name)

                for j in temp.index:

                    add_popularity_and_budget(dataframes[column], j, df.loc[i,'popularity'], df.loc[i, 'budget'])
update_popularity_and_budget(box, 'belongs_to_collection', 'id')

update_popularity_and_budget(box, 'genres', 'id')

update_popularity_and_budget(box, 'production_companies', 'id')

update_popularity_and_budget(box, 'production_countries', 'iso_3166_1')

update_popularity_and_budget(box, 'spoken_languages', 'iso_639_1')

update_popularity_and_budget(box, 'Keywords', 'id')

update_popularity_and_budget_for_top_5_cast(box, 'cast', 'id')

update_popularity_and_budget_for_directors(box, 'crew', 'id')
# parameters

# column: column name

# entity: 'popularity' or 'budget'

# title: title for plot

# n: top n result, default is 20.

def plot_top_items(column, entity, title, n=20):

    temp = dataframes[column].sort_values(entity, ascending=False).iloc[:n, :]

    plt.rcParams["figure.figsize"] = [16, 14]

    fig, ax = plt.subplots()

    y_pos = np.arange(len(temp))

    ax.barh(y_pos, temp[entity])

    ax.set_yticks(y_pos)

    ax.set_yticklabels(temp['name'])

    ax.invert_yaxis()

    ax.set_xlabel(entity)

    ax.set_title(title)

    plt.show()
plot_top_items(column='belongs_to_collection', entity='popularity', title='Top 20 popular collections')
plot_top_items(column='belongs_to_collection', entity='budget', title='Top 20 most expensive collections')
plot_top_items(column='genres', entity='popularity', title='Genres popularity')
plot_top_items(column='genres', entity='budget', title='Genres budget')
def add_popularity_budget_features(df, column, index_name):

    df['has_{}'.format(column)] = 0

    df['{}_popularity'.format(column)] = 0

    df['{}_budget'.format(column)] = 0

    for i in range(df.index[0], len(df)+df.index[0]):

#         pop = 0

#         bud = 0

        if str(df.loc[i, column]) != 'nan':

            temp_list = ast.literal_eval(df.loc[i, column])

            if temp_list:

#                 df.loc[i, 'has_{}'.format(column)] = 1

                temp = pd.DataFrame(temp_list)

                temp = temp.set_index(index_name)

                if len(temp) > 0:

                    df.loc[i, '{}_popularity'.format(column)] = dataframes[column].loc[temp.index, 'popularity'].mean()

                    df.loc[i, '{}_budget'.format(column)] = dataframes[column].loc[temp.index, 'budget'].mean()

                    df.loc[i, 'has_{}'.format(column)] = 1

#                 for j in temp.index:

#                     pop = pop + dataframes[column].loc[j, 'popularity']

#                     bud = bud + dataframes[column].loc[j, 'budget']

#                 df.loc[i, '{}_popularity'.format(column)] = pop/len(temp)

#                 df.loc[i, '{}_budget'.format(column)] = bud/len(temp)
def add_popularity_budget_for_director(df, column, index_name):

    df['has_director'] = 0

    df['director_popularity'] = 0

    df['director_budget'] = 0

    for i in range(df.index[0], len(df)+df.index[0]):

#         pop = 0

#         bud = 0

        if str(df.loc[i, column]) != 'nan':

            temp_list = ast.literal_eval(df.loc[i, column])

            if temp_list:

                temp = pd.DataFrame(temp_list)

                temp = temp[temp['job'] == 'Director']

                temp = temp.set_index(index_name)

#                 has_director = 0

                if len(temp) > 0:

                    df.loc[i, 'director_popularity'] = dataframes[column].loc[temp.index, 'popularity'].mean()

                    df.loc[i, 'director_budget'] = dataframes[column].loc[temp.index, 'budget'].mean()

                    df.loc[i, 'has_director'] = 1

#                 for j in temp.index:

#                     pop = pop + dataframes[column].loc[j, 'popularity']

#                     bud = bud + dataframes[column].loc[j, 'budget']

#                     has_director = 1

#                 df.loc[i, 'director_popularity'] = pop/len(temp)

#                 df.loc[i, 'director_budget'] = bud/len(temp)

#                 df.loc[i, 'has_director'] = has_director
def add_popularity_budget_for_5_cast(df, column, index_name):

    for n in range(1,6):

        df['has_{}_{}'.format(column, n)] = 0

        df['{}_{}_popularity'.format(column, n)] = 0

        df['{}_{}_budget'.format(column, n)] = 0

    for i in range(df.index[0], len(df)+df.index[0]):

        try:

            if str(df.loc[i, column]) != 'nan':

                temp_list = ast.literal_eval(df.loc[i, column])

                if temp_list:

                    temp = pd.DataFrame(temp_list)

                    if len(temp) > 5:

                        temp = temp.iloc[:5,:]

                        temp = temp.set_index(index_name)

                        cnt = 1

                        for j in temp.index:

                            df.loc[i, 'has_{}_{}'.format(column, cnt)] = 1

                            df.loc[i, '{}_{}_popularity'.format(column, cnt)] = dataframes[column].loc[j, 'popularity']

                            df.loc[i, '{}_{}_budget'.format(column, cnt)] = dataframes[column].loc[j, 'budget']

                            cnt = cnt+1

        except ValueError:

            print('Oops ! Not valid value. Row # ', i)
from datetime import datetime



def add_date_features(df, column):

    df['release_day'] = 1

    df['release_month'] = 1

    df['release_year'] = 2000

    for i in range(df.index[0], len(df)+df.index[0]):

        if str(df.loc[i, column]) != 'nan':

            release_date = datetime.strptime(str(df.loc[i, column]), '%m/%d/%y')

            df.loc[i, 'release_day'] = release_date.day

            df.loc[i, 'release_month'] = release_date.month

            df.loc[i, 'release_year'] = release_date.year
from sklearn.preprocessing import OneHotEncoder



def add_catagoriacal_features(df, column):

    df[column].fillna(column+'_unknown', inplace=True)

    cat_encoder = OneHotEncoder(sparse=False, categories='auto', dtype=np.int)

    features = cat_encoder.fit_transform(df[[column]])

    temp_df = pd.DataFrame(features, 

                           index=range(df.index[0], len(df)+df.index[0]), 

                           columns=[column+'_'+str(i) for i in cat_encoder.categories_[0]])

    df = pd.concat([df, temp_df], axis=1)

    return df
# Preparing data for machine learning Model



box_train_feature = box_train.copy()



# Popularity and budget related features

add_popularity_budget_features(box_train_feature, 'belongs_to_collection', 'id')

add_popularity_budget_features(box_train_feature, 'genres', 'id')

add_popularity_budget_features(box_train_feature, 'production_companies', 'id')

add_popularity_budget_features(box_train_feature, 'production_countries', 'iso_3166_1')

add_popularity_budget_features(box_train_feature, 'spoken_languages', 'iso_639_1')

add_popularity_budget_features(box_train_feature, 'Keywords', 'id')

add_popularity_budget_for_director(box_train_feature, 'crew', 'id')

add_popularity_budget_for_5_cast(box_train_feature, 'cast', 'id')



# Release date related features

add_date_features(box_train_feature, 'release_date')



# Categorical features

# box_train_feature = add_catagoriacal_features(box_train_feature, 'status')

# box_train_feature = add_catagoriacal_features(box_train_feature, 'original_language')

box_train_feature = add_catagoriacal_features(box_train_feature, 'release_day')

box_train_feature = add_catagoriacal_features(box_train_feature, 'release_month')



# Filling not available numerical feature

box_train_feature.runtime.fillna(box_train_feature.runtime.median(), inplace=True)



# Removing un-necessary columns

un_necessary_columns = ['belongs_to_collection', 'genres', 'original_language', 

                        'production_companies', 'production_countries', 'release_date', 

                        'spoken_languages', 'status', 'Keywords', 'crew', 'cast', 

                        'release_day', 'release_month']

box_train_feature.drop(un_necessary_columns, axis=1, inplace=True)
numerical_ftr = ['budget', 

                 'belongs_to_collection_budget', 'genres_budget', 'production_companies_budget', 

                 'production_countries_budget', 'spoken_languages_budget', 'Keywords_budget', 

                 'cast_1_budget', 'cast_2_budget', 'cast_3_budget', 'cast_4_budget', 'cast_5_budget', 

                 'director_budget',

                 'popularity', 

                 'belongs_to_collection_popularity', 'genres_popularity', 'production_companies_popularity', 

                 'production_countries_popularity', 'spoken_languages_popularity', 'Keywords_popularity', 

                 'cast_1_popularity', 'cast_2_popularity', 'cast_3_popularity', 'cast_4_popularity', 

                 'cast_5_popularity', 'director_popularity', 

                 'runtime', 

                 'revenue']



box_numerical_ftr = box_train_feature[numerical_ftr]

corr_mx_ftr = box_numerical_ftr.corr()

pd.DataFrame(corr_mx_ftr, index=numerical_ftr, columns=numerical_ftr)
seaborn.heatmap(corr_mx_ftr)

plt.show()
# Separating features and target



X = box_train_feature.drop('revenue', axis=1)

# X = box_train_feature[['budget', 'popularity']]

y = box_train_feature['revenue']



# from sklearn.decomposition import PCA

# pca = PCA(n_components=0.99)

# X = pca.fit_transform(X)

# print(pca.explained_variance_ratio_)



# Spliting dataset to test and val

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2)
# from sklearn.preprocessing import StandardScaler



# std_scaler = StandardScaler()

# X_scaled = std_scaler.fit_transform(X)
from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



def sqrt_mean_squared_log_error(actual, pred):

    return np.sqrt(mean_squared_log_error(actual, pred))



param_grid = [{'n_estimators':[100, 500, 1000], 'max_features':['auto', 'log2'], 

              'min_samples_split':[5, 12, 20], 'max_depth':[5, 20, 50], 'n_jobs':[-1]}]



rf_reg = RandomForestRegressor()



grid = GridSearchCV(rf_reg, param_grid, scoring='neg_mean_squared_log_error', n_jobs=-1)

grid.fit(X, y)



# rf_reg.fit(X_train, y_train)
rf_reg = grid.best_estimator_
grid.best_params_
y_train_pred = rf_reg.predict(X)

y_train_pred[y_train_pred < 0] = 0



error = np.sqrt(mean_squared_log_error(y, y_train_pred))

print(error)

plt.scatter(y, y_train_pred)

plt.show()
# y_val_pred = rf_reg.predict(X_val)

# y_val_pred[y_val_pred < 0] = 0



# error = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

# print(error)

# plt.scatter(y_val, y_val_pred)

# plt.show()
box_test_feature = box_test.copy()



# Popularity and budget related features

add_popularity_budget_features(box_test_feature, 'belongs_to_collection', 'id')

add_popularity_budget_features(box_test_feature, 'genres', 'id')

add_popularity_budget_features(box_test_feature, 'production_companies', 'id')

add_popularity_budget_features(box_test_feature, 'production_countries', 'iso_3166_1')

add_popularity_budget_features(box_test_feature, 'spoken_languages', 'iso_639_1')

add_popularity_budget_features(box_test_feature, 'Keywords', 'id')

add_popularity_budget_for_director(box_test_feature, 'crew', 'id')

add_popularity_budget_for_5_cast(box_test_feature, 'cast', 'id')



# Release date related features

add_date_features(box_test_feature, 'release_date')



# Categorical features

# box_test_feature = add_catagoriacal_features(box_test_feature, 'status')

# box_test_feature = add_catagoriacal_features(box_test_feature, 'original_language')

box_test_feature = add_catagoriacal_features(box_test_feature, 'release_day')

box_test_feature = add_catagoriacal_features(box_test_feature, 'release_month')



# Filling not available numerical feature

box_test_feature.runtime.fillna(box_test_feature.runtime.median(), inplace=True)



# Removing un-necessary columns

un_necessary_columns = ['belongs_to_collection', 'genres', 'original_language', 

                        'production_companies', 'production_countries', 'release_date', 

                        'spoken_languages', 'status', 'Keywords', 'crew', 'cast', 

                        'release_day', 'release_month']

box_test_feature.drop(un_necessary_columns+not_useful_columns, axis=1, inplace=True)
X_test = box_test_feature.copy()



y_test_pred = rf_reg.predict(X_test)

y_test_pred[y_test_pred < 0] = 0

# y_test_pred.shape

# np.array(range(3001, 3001+len(y_test_pred))).reshape()

prediction = pd.DataFrame({'id':np.array(range(3001, 3001+len(y_test_pred))), 'revenue':y_test_pred})

prediction.to_csv('submission.csv', index=False)