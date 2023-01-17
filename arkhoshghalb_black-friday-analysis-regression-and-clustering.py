import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
data: pd.DataFrame = pd.read_csv('../input/BlackFriday.csv')
describe = data.describe()
describe.loc['#unique'] = data.nunique()
display(describe)
purchase_desc = data['Purchase'].describe()
purchase_desc.drop(['count', 'std'], inplace=True)
purchase_desc.loc['sum'] = data['Purchase'].sum()
purchase_desc.loc['mean_by_user'] = data['Purchase'].sum() / data['User_ID'].nunique()
display(pd.DataFrame(purchase_desc).T)
null_percent = (data.isnull().sum() / len(data))*100
display(pd.DataFrame(null_percent[null_percent > 0].apply(lambda x: "{:.2f}%".format(x)),columns=['Null %']))
cat_describe = data[['Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Marital_Status', 'Product_Category_1']].astype('object').describe()
cat_describe.loc['percent'] = 100*cat_describe.loc['freq'] / cat_describe.loc['count']
display(cat_describe)
plt.figure(figsize=(13, 6))
gender_gb = data[['Gender', 'Purchase']].groupby('Gender').agg(['count', 'sum'])
params = {
#     'colors': [(255/255, 102/255, 102/255, 1), (102/255, 179/255, 1, 1)],
    'labels': gender_gb.index.map({'M': 'Male', 'F': 'Female'}),
    'autopct': '%1.1f%%',
    'startangle': -30, 
    'textprops': {'fontsize': 15},
    'explode': (0.05, 0),
    'shadow': True
}
plt.subplot(121)
plt.pie(gender_gb['Purchase']['count'], **params)
plt.title('Number of transactions', size=17)
plt.subplot(122)
plt.pie(gender_gb['Purchase']['sum'], **params)
plt.title('Sum of purchases', size=17)
plt.show()
gender_gb = data[['Gender', 'Purchase']].groupby('Gender', as_index=False).agg('mean')
sns.barplot(x='Gender', y='Purchase', data=gender_gb)
plt.ylabel('')
plt.xlabel('')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.title('Mean purchase amount by gender', size=14)
plt.show()
plt.figure(figsize=(16, 8))
plt.subplot(121)
sns.countplot(y='Age', data=data, order=sorted(data.Age.unique()))
plt.title('Number of transactions by age group', size=14)
plt.xlabel('')
plt.ylabel('Age Group', size=13)
plt.subplot(122)
age_gb = data[['Age', 'Purchase']].groupby('Age', as_index=False).agg('mean')
sns.barplot(y='Age', x='Purchase', data=age_gb, order=sorted(data.Age.unique()))
plt.title('Mean purchase amount by age group', size=14)
plt.xlabel('')
plt.ylabel('')
plt.show()
age_product_gb = data[['Age', 'Product_ID', 'Purchase']].groupby(['Age', 'Product_ID']).agg('count').rename(columns={'Purchase': 'count'})
age_product_gb.sort_values('count', inplace=True, ascending=False)
ages = sorted(data.Age.unique())
result = pd.DataFrame({
    x: list(age_product_gb.loc[x].index)[:5] for x in ages
}, index=['#{}'.format(x) for x in range(1,6)])
display(result)
men = data[data.Gender == 'M']['Occupation'].value_counts(sort=False)
women = data[data.Gender == 'F']['Occupation'].value_counts(sort=False)
pd.DataFrame({'M': men, 'F': women}, index=range(0,21)).plot.bar(stacked=True)
plt.gcf().set_size_inches(10, 4)
plt.title("Count of different occupations in dataset (Separated by gender)", size=14)
plt.legend(loc="upper right")
plt.xlabel('Occupation label', size=13)
plt.ylabel('Count', size=13)
plt.show()
import random
color_mapping = {}
def random_color(val):    
    if val in color_mapping.keys():
        color = color_mapping[val]
    else:
        r = lambda: random.randint(0,255)
        color = 'rgba({}, {}, {}, 0.4)'.format(r(), r(), r())
        color_mapping[val] = color
    return 'background-color: %s' % color

occ_product_gb = data[['Occupation', 'Product_ID', 'Purchase']].groupby(['Occupation', 'Product_ID']).agg('count').rename(columns={'Purchase': 'count'})
occ_product_gb.sort_values('count', inplace=True, ascending=False)
result = pd.DataFrame({
    x: list(occ_product_gb.loc[x].index)[:5] for x in range(21)
}, index=['#{}'.format(x) for x in range(1,6)])
display(result.style.applymap(random_color))
stay_years = [data[data.Stay_In_Current_City_Years == x]['City_Category'].value_counts(sort=False).iloc[::-1] for x in sorted(data.Stay_In_Current_City_Years.unique())]

f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 2]}, sharey=True)

years = sorted(data.Stay_In_Current_City_Years.unique())
pd.DataFrame(stay_years, index=years).T.plot.bar(stacked=True, width=0.3, ax=ax1, rot=0, fontsize=11)
ax1.set_xlabel('City Category', size=13)
ax1.set_ylabel('# Transactions', size=14)
ax1.set_title('# Transactions by city (separated by stability)', size=14)

sns.countplot(x='Stay_In_Current_City_Years', data=data, ax=ax2, order=years)
ax2.set_title('# Transactions by stability', size=14)
ax2.set_ylabel('')
ax2.set_xlabel('Years in current city', size=13)

plt.gcf().set_size_inches(15, 6)
plt.show()
out_vals = data.Marital_Status.value_counts()
in_vals = np.array([data[data.Marital_Status==x]['Gender'].value_counts() for x in [0,1]]).flatten()

fig, ax = plt.subplots(figsize=(7, 7))

size = 0.3
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(2)*4)
inner_colors = cmap(np.array([1, 2, 5, 6]))

ax.pie(out_vals, radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'), labels=['Single', 'Married'],
       textprops={'fontsize': 15}, startangle=50)

ax.pie(in_vals, radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'), labels=['M', 'F', 'M', 'F'],
       labeldistance=0.75, textprops={'fontsize': 12, 'weight': 'bold'}, startangle=50)

ax.set(aspect="equal")
plt.title('Marital Status / Gender', fontsize=16)
plt.show()
col_names = ['Product_ID', 'Product_Category_1', 'User_ID']
renames = ['Product', 'Category', 'User']
results = []
for col_name, new_name in zip(col_names, renames):
    group = data[[col_name, 'Purchase']].groupby(col_name, as_index=False).agg('count')
    result = group.sort_values('Purchase', ascending=False)[:10]
    result.index = ['#{}'.format(x) for x in range(1,11)]
    results.append(result.rename(columns={col_name: new_name}))    
    
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline; padding-right: 3em !important;"'),raw=True)
display_side_by_side(*results)
train = data.drop(['Product_Category_2', 'Product_Category_3'], axis=1)\
            .rename(columns={
                'Product_ID': 'Product',
                'User_ID': 'User',
                'Product_Category_1': 'Category',
                'City_Category': 'City',
                'Stay_In_Current_City_Years': 'City_Stay'
})
y = train.pop('Purchase')

train.loc[:, 'Gender'] = np.where(train['Gender'] == 'M', 1, 0)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV

# Label encoding Product ID and User ID
for col in ['Product', 'User']:    
    train.loc[:, col] = LabelEncoder().fit_transform(train[col])
        
# One hot encoding other features
categoricals = ['Occupation', 'Age', 'City', 'Gender', 'Category', 'City_Stay']
encoder = OneHotEncoder().fit(train[categoricals])
train = pd.concat([train, pd.DataFrame(encoder.transform(train[categoricals]).toarray(), index=train.index, columns=encoder.get_feature_names(categoricals))], axis=1)
train.drop(categoricals, axis=1, inplace=True)

# Splitting train and test sets
X_train, X_test, y_train, y_test = train_test_split(train, y)

# Standardizing
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

params = {
    'n_estimators': [10, 30, 100, 300],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
preds = grid_search.predict(X_test)
print("Best params found: {}".format(grid_search.best_params_))
print("RMSE score: {}".format(mean_squared_error(y_test, preds) ** 0.5))
sizes, train_scores, test_scores = learning_curve(RandomForestRegressor(**grid_search.best_params_), X_train, y_train, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
train_scores = np.mean((-1*train_scores)**0.5, axis=1)
test_scores = np.mean((-1*test_scores)**0.5, axis=1)
sns.lineplot(sizes, train_scores, label="Train")
sns.lineplot(sizes, test_scores, label="Test")
plt.xlabel("Size of training set", size=13)
plt.ylabel("Round Mean Squared Error", size=13)
plt.show()
model = grid_search.best_estimator_
impo = pd.Series(model.feature_importances_[:10], index=train.columns[:10]).sort_values()
impo_plot = sns.barplot(x=impo.index, y=impo.values)
for item in impo_plot.get_xticklabels():
    item.set_rotation(50)
plt.gcf().set_size_inches(8, 4)
plt.title("Top 10 most important features", size=14)
plt.show()
train = data.drop(['Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase'], axis=1).groupby('User_ID')
train = train.agg(lambda x: x.value_counts().index[-1])
feaures_list = list(train.columns.values)
encoder = OneHotEncoder().fit(train[feaures_list])
train = pd.concat([train, pd.DataFrame(encoder.transform(train[feaures_list]).toarray(), index=train.index, columns=encoder.get_feature_names(feaures_list))], axis=1)
train.drop(feaures_list, axis=1, inplace=True)
columns = ['Product_ID', 'Product_Category_1']
for column in columns:
    top_100 = data[column].value_counts().index[:100]    
    user_purchase = pd.pivot_table(
        data[['User_ID', column, 'Purchase']],
        values='Purchase',
        index='User_ID',
        columns=column,
        aggfunc=np.sum
    ).fillna(0)[top_100]  
    train = train.join(user_purchase)
train = train.join(data[['User_ID', 'Purchase']].groupby('User_ID').agg('sum'))
train_scaled = StandardScaler().fit_transform(train)
from sklearn.cluster import KMeans

k_values = np.arange(1, 11)
models = []
dists = []
for k in k_values:
    model = KMeans(k).fit(train_scaled)
    models.append(model)
    dists.append(model.inertia_)

plt.figure(figsize=(9, 6))
plt.plot(k_values, dists, 'o-')
plt.ylabel('Sum of squared distances', size=13)
plt.xlabel('K', size=13)
plt.xticks(k_values)
plt.show()
from sklearn.metrics import silhouette_score

model = models[2]
print("Silhouette score: {:.2f}".format(silhouette_score(train_scaled, model.predict(train_scaled))))