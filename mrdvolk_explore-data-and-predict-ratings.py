import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
apps_df = pd.read_csv('../input/googleplaystore.csv')
reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
apps_df.head(10)
reviews.head()
apps_df['Rating'].describe()
extraordinary_app = apps_df[apps_df['Rating'] > 5.]
extraordinary_app.head()
extraordinary_app_reviews = reviews[reviews['App'] == extraordinary_app.iloc[0]['App']]
extraordinary_app_reviews.head()
apps_df = apps_df.drop(apps_df[apps_df['Rating'] > 5].index, axis=0)
apps_df['Rating'].isnull().sum()
apps_df = apps_df.drop(apps_df[apps_df['Rating'].isnull()].index, axis=0)
apps_df['Rating'].describe()
apps_df['Rating'].hist(bins=50)
plt.figure(figsize=(18,10))
apps_df['Category'].value_counts().plot(kind='bar')
apps_df['Content Rating'].value_counts().plot(kind='bar')
apps_df['Type'].value_counts().plot(kind='bar')
apps_df['Price'] = apps_df['Price'].apply(lambda x: float(x.replace('$', ''))).astype('float64')
plt.figure(figsize=(18,10))
apps_df[apps_df['Price'] > 0.]['Price'].hist(bins=100)
apps_df['Genres'].value_counts()
apps_df['Genres'].apply(lambda x: len(x.split(';'))).value_counts()
apps_df['Main Genre'] = apps_df['Genres'].apply(lambda x: x.split(';')[0])
apps_df['Sub Genre'] = apps_df['Genres'].apply(lambda x: x.split(';')[1] if len(x.split(';')) > 1 else 'no sub genre')
plt.figure(figsize=(18,10))
apps_df['Main Genre'].value_counts().plot(kind='bar')
apps_df[apps_df['Sub Genre'] != 'no sub genre']['Sub Genre'].value_counts().plot(kind='bar')
plt.figure(figsize=(18,10))
apps_df['Installs'].value_counts().plot(kind='bar')
apps_df['Reviews'] = apps_df['Reviews'].astype('int64')
plt.figure(figsize=(18,10))
apps_df['Reviews'].hist(bins=100)
spans = [1000, 10000, 100000, 1000000, 10000000, 100000000]

plt.figure(figsize=(18, 4 * len(spans)))
prev=0
for i, span in enumerate(spans):
    plt.subplot(len(spans), 1, i+1)
    subset = apps_df[(apps_df['Reviews'] > prev) & (apps_df['Reviews'] < span)]
    subset['Reviews'].hist(bins=100)
    plt.title("{:,}".format(prev) + ' - ' + "{:,}".format(span))
    prev=span
installs_categories = apps_df['Installs'].value_counts().index
installs_categories_list = [(x, int(x.replace(',', '').replace('+', ''))) for x in installs_categories]
sorted_installs = sorted(installs_categories_list, key= lambda x: x[1])

plt.figure(figsize=(18, 5 * len(sorted_installs)))
for i, installs in enumerate(sorted_installs):
    plt.subplot(len(sorted_installs), 1, i+1)
    subset = apps_df[apps_df['Installs'] == installs[0]]
    subset['Reviews'].hist(bins=100)
    plt.title("Installs: "+installs[0])
apps_df['Installs (int)'] = apps_df['Installs'].apply(lambda x: int(x.replace(',', '').replace('+', '')))
sns.lmplot("Installs (int)", "Reviews", data=apps_df, aspect=2)
ax = plt.gca()
_ = ax.set_title('Overall correlation between installs and reviews')
sns.lmplot("Installs (int)", "Reviews", data=apps_df, aspect=2, hue='Type')
apps_df.corr()['Reviews']
apps_df[apps_df['Type'] == 'Free'].corr()['Reviews']
apps_df[apps_df['Type'] == 'Paid'].corr()['Reviews']
sns.lmplot("Installs (int)", "Reviews", data=apps_df[apps_df['Type'] == 'Paid'], aspect=2)
sns.lmplot("Installs (int)", "Reviews", data=apps_df[(apps_df['Type'] == 'Paid') & (apps_df['Installs (int)'] < 1e7)], aspect=2)
apps_df[(apps_df['Type'] == 'Paid') & (apps_df['Installs (int)'] < 1e7)].corr()['Reviews']
apps_under10mil_installs_df = apps_df[apps_df['Installs (int)'] < 1e7]
columns_to_encode = ['Category', 'Main Genre', 'Content Rating']

for col in columns_to_encode:
    def get_prefix(col):
        if col == 'Main Genre':
            return 'Genre'
        else:
            return col
    
    col_labels_ctg = apps_under10mil_installs_df[col].astype('category')
    col_dummies = pd.get_dummies(col_labels_ctg, prefix=get_prefix(col))
    
    apps_under10mil_installs_df = pd.concat([apps_under10mil_installs_df, col_dummies], axis=1)
    del apps_under10mil_installs_df[col]
subgenres = set(list(apps_df[apps_df['Sub Genre'] != 'no sub genre']['Sub Genre']))

for subgenre in subgenres:
    col_name = 'Genre_' + subgenre
    apps_under10mil_installs_df[col_name] = apps_under10mil_installs_df['Sub Genre'].apply(
        lambda x: 1 if x == subgenre else 0)
    
del apps_under10mil_installs_df['Sub Genre']
def get_strongest_correlations(col, num):
    corrs = apps_under10mil_installs_df.corr()[col]
    max_num = len(list(corrs))
    if num > max_num:
        num = max_num
        print ('Features limit exceeded. Max number of features: ', max_num)
        
    corrs = corrs.drop(col)
    idx = list(corrs.abs().sort_values(ascending=False).iloc[:num].index)
    return corrs[idx], idx
installs_corrs, _ = get_strongest_correlations('Installs (int)', 20)
plt.figure(figsize=(18,10))
installs_corrs.plot(kind='bar')
reviews_corrs, _ = get_strongest_correlations('Reviews', 20)
plt.figure(figsize=(18,10))
reviews_corrs.plot(kind='bar')
rating_corrs, _ = get_strongest_correlations('Rating', 20)
plt.figure(figsize=(18,10))
rating_corrs.plot(kind='bar')
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
train = apps_under10mil_installs_df.sample(frac=0.8)
test_and_validation = apps_under10mil_installs_df.loc[~apps_under10mil_installs_df.index.isin(train.index)]
validation = test_and_validation.sample(frac=0.5)
test = test_and_validation.loc[~test_and_validation.index.isin(validation.index)]

print(train.shape, validation.shape, test.shape)
def get_features(num_features):
    col_to_predict = 'Rating'
    rating_corrs, idx = get_strongest_correlations(col_to_predict, num_features)
    if col_to_predict in idx:
        idx.remove(col_to_predict)
    return idx

def compare_predictions(predicted, test_df, target_col):
    check_df = pd.DataFrame(data=predicted, index=test_df.index, columns=["Predicted "+target_col])
    check_df = pd.concat([check_df, test_df[[target_col]]], axis=1)
    check_df["Error, %"] = np.abs(check_df["Predicted "+target_col]*100/check_df[target_col] - 100)
    check_df['Error, val'] = check_df["Predicted "+target_col] - check_df[target_col]
    return (check_df.sort_index(), check_df["Error, %"].mean())

def evaluate_predictions(model, train_df, test_df, features, target_col):
    train_pred = model.predict(train_df[features])
    train_rmse = mean_squared_error(train_pred, train_df[target_col]) ** 0.5

    test_pred = model.predict(test_df[features])
    test_rmse = mean_squared_error(test_pred, test_df[target_col]) ** 0.5

    print("RMSEs:")
    print(train_rmse, test_rmse)
    
    return test_pred
def rfr_model_evaluation(num_features=30, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_leaf_nodes=None, use_test=False):
    rfr = RandomForestRegressor(random_state=42, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_leaf_nodes=max_leaf_nodes)
    features = get_features(num_features)
    rfr.fit(train[features], train['Rating'])
    if use_test:
        rfr_test_predictions = evaluate_predictions(rfr, train, test, features, 'Rating')
        check_df, avg_error = compare_predictions(rfr_test_predictions, test, 'Rating')
        print("Average test error:", avg_error)
    else:
        rfr_validation_predictions = evaluate_predictions(rfr, train, validation, features, 'Rating')
        check_df, avg_error = compare_predictions(rfr_validation_predictions, validation, 'Rating')
        print("Average validation error:", avg_error)
    return check_df, avg_error
check, error = rfr_model_evaluation()
num_features_list = [1] + [x for x in range(5, 91, 5)] + [96]
max_depth_list = [None] + [x for x in range(3, 11)] + [x for x in range(15, 36, 5)]
min_samples_split_list = [x for x in range(2, 11)] + [x for x in range(15, 101, 5)]
min_samples_leaf_list = [x for x in range(1, 15)] + [0.001] + list(np.linspace(0.005,0.1,20).round(3))
min_weight_fraction_leaf_list = [0., 0.001] + list(np.linspace(0.005, 0.3, 30).round(3))
max_leaf_nodes_list = [None] + [x for x in range(5, 101, 5)]
n_estimators = [x for x in range(10, 220, 20)]

hyperparams = {
    'num_features': num_features_list,
    'max_depth': max_depth_list,
    'min_samples_split': min_samples_split_list,
    'min_samples_leaf': min_samples_leaf_list,
    'min_weight_fraction_leaf': min_weight_fraction_leaf_list,
    'max_leaf_nodes': max_leaf_nodes_list,
    'n_estimators': n_estimators
}

validation_results = []
for hp_name, hp_list in hyperparams.items():
    errors = []
    for hp_val in hp_list:
        if hp_name == 'num_features':
            _, error = rfr_model_evaluation(num_features=hp_val)
        elif hp_name == 'max_depth':
            _, error = rfr_model_evaluation(max_depth=hp_val)
        elif hp_name == 'min_samples_split':
            _, error = rfr_model_evaluation(min_samples_split=hp_val)
        elif hp_name == 'min_samples_leaf':
            _, error = rfr_model_evaluation(min_samples_leaf=hp_val)
        elif hp_name == 'min_weight_fraction_leaf':
            _, error = rfr_model_evaluation(min_weight_fraction_leaf=hp_val)
        elif hp_name == 'max_leaf_nodes':
            _, error = rfr_model_evaluation(max_leaf_nodes=hp_val)
        elif hp_name == 'n_estimators':
            _, error = rfr_model_evaluation(n_estimators=hp_val)
            
        errors.append(error)
    validation_results.append((hp_name, errors))
fig = plt.figure(figsize=(18, 30))

for i, result in enumerate(validation_results):
    ax = fig.add_subplot(len(validation_results), 1, i+1)
    hp_name = result[0]
    hp_errors = result[1]
    
    ax.set_title(hp_name)
    ax.plot(range(0, len(hp_errors)), hp_errors)
    plt.sca(ax)
    x_labels = hyperparams[hp_name]
    plt.xticks(range(0, len(hp_errors)), x_labels)
    
fig.tight_layout()
plt.show()
check, error = rfr_model_evaluation(num_features=96, n_estimators=110, max_depth=10, min_samples_split=45,
                                    min_samples_leaf=14, min_weight_fraction_leaf=0.005, max_leaf_nodes=75)
check, error = rfr_model_evaluation(num_features=96, n_estimators=110, max_depth=10, min_samples_split=45,
    min_samples_leaf=14, min_weight_fraction_leaf=0.005, max_leaf_nodes=75, use_test=True)
