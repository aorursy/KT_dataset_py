import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline
zomato_data = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')
df = zomato_data.copy()
df.head()
df.info()
add_group = df.groupby('address')
group_1 = add_group.filter(lambda x: len(x) > 1)[add_group.filter(lambda x: len(x) > 1)['name'] == 'Onesta']
group_1['address'][7]
group_1[add_group.filter(lambda x: len(x) > 1)[add_group.filter(lambda x: len(x) > 1)['name'] == 'Onesta'].address == '2469, 3rd Floor, 24th Cross, Opposite BDA Complex, 2nd Stage, Banashankari, Bangalore']
df = df.drop(['url', 'listed_in(type)', 'listed_in(city)'], axis=1).reset_index().drop(['index'], axis=1).drop_duplicates()
df = df.drop(['phone'],axis=1)
plt.figure(figsize=(12, 8))
sns.set_style('white')
rest = df.groupby(['address', 'name'])
rest_chains = rest.name.nunique().index.to_frame()['name'].value_counts()[:15] # first 15

ax = sns.barplot(x=rest_chains, y=rest_chains.index, palette='Greys_d')
sns.despine()
plt.title('Top 15 Restaurant chains in Bangalore')
plt.xlabel('Number of Outlets')
plt.ylabel('Name')
for p in ax.patches:
    width = p.get_width()
    ax.text(width+0.007, p.get_y()+p.get_height()/2. + 0.2, format(width), ha="left", color="black")
plt.show()
#Preprocessing restaurant types
rest_type_df = df.groupby(['address', 'rest_type']).rest_type.nunique().index.to_frame()
temp_df = pd.DataFrame()
temp_df = rest_type_df.rest_type.str.strip().str.split(',', expand=True)
rest_type_temp_df = pd.DataFrame()
rest_type_temp_df = pd.concat([temp_df.iloc[:,0].str.strip(), temp_df.iloc[:, 1].str.strip()]).value_counts()
rest_type_temp_df
plt.figure(figsize=(12,8))
sns.set_style('white')
restaurant_types = rest_type_temp_df[:15]
ax = sns.barplot(x=restaurant_types, y=restaurant_types.index, palette="Greys_d")
sns.despine()
plt.title('Top 15 restaurant types in Bangalore')
plt.xlabel('Number of restaurants')
plt.ylabel('Restaurant types')
total_rest = len(rest_type_df)     #restaurants after grouping by address and rest_type
for p in ax.patches:
    percentage='{:.1f}%'.format(100*p.get_width()/total_rest)    # find % of rest types
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x,y), ha="left", color='black')
plt.show()
#Preprocessing of Cuisines
cuisines_df = df.groupby(['address', 'cuisines']).cuisines.nunique().index.to_frame()
cuisines_temp_df = pd.DataFrame()
cuisines_temp_df = cuisines_df.cuisines.str.strip().str.split(',', expand=True)
cuisines = pd.DataFrame()
cuisines = pd.concat([cuisines_temp_df.iloc[:,0].str.strip(), cuisines_temp_df.iloc[:,1].str.strip(), cuisines_temp_df.iloc[:,2].str.strip(),
                     cuisines_temp_df.iloc[:,3].str.strip(),cuisines_temp_df.iloc[:,4].str.strip(),cuisines_temp_df.iloc[:,5].str.strip(),
                     cuisines_temp_df.iloc[:,6].str.strip(),cuisines_temp_df.iloc[:,7].str.strip()]).value_counts()
plt.figure(figsize=(12,8))
sns.set_style('white')
cuisine = cuisines[:15]
ax = sns.barplot(x=cuisine, y=cuisine.index, palette="Greys_d")
sns.despine()
plt.title('Top 15 cuisines served in Bangalore')
plt.xlabel('Number of restaurants')
plt.ylabel('Name of cuisines')
total_rest = len(cuisines_df)     #restaurants after grouping by address and rest_type
for p in ax.patches:
    percentage='{:.1f}%'.format(100*p.get_width()/total_rest)    # find % of rest types
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height()/2
    ax.annotate(percentage, (x,y), ha="left", color='black')
plt.show()
cuisines[cuisines.index == "Healthy Food"]
location_df = df.groupby(['address', 'location']).location.nunique().index.to_frame()
print(location_df['location'].value_counts()[location_df['location'].value_counts().index.str.contains('Koramangala')])
print("Number of restaurants in Koramangala:",
    sum(location_df['location'].value_counts()[location_df['location'].value_counts().index.str.contains('Koramangala')]))
plt.figure(figsize=(12, 8))
sns.set_style('white')
locations= location_df['location'].value_counts()[:15]
ax = sns.barplot(x= locations, y = locations.index, palette='Greys_d')
sns.despine()
plt.title('Top 15 locations for foodies in Bangalore')
plt.xlabel('Number of restaurants')
plt.ylabel('Name of Location')
for p in ax.patches:
    width = p.get_width()
    ax.text(width+0.007, p.get_y() + p.get_height() / 2. + 0.2, format(width), 
            ha="left", color='black')
plt.show()
#Preprocessing dish liked
dish_liked_df = df.groupby(['address', 'dish_liked']).dish_liked.nunique().index.to_frame()
dish_liked_temp_df = pd.DataFrame()
dish_liked_temp_df = dish_liked_df.dish_liked.str.strip().str.split(',', expand=True)
dish_liked = pd.DataFrame()
dish_liked = pd.concat([dish_liked_temp_df.iloc[:,0].str.strip(), dish_liked_temp_df.iloc[:,1].str.strip()]).value_counts()
dish_liked
plt.figure(figsize=(12, 8))
sns.set_style('white')
dishes = dish_liked[:15]
ax = sns.barplot(x= dishes, y = dishes.index, palette='Greys_d')
sns.despine()
plt.title('Top 15 commonly served dishes in Bangalore')
plt.xlabel('Number of restaurants')
plt.ylabel('Name of dishes')
total = len(dish_liked_df)

for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y),
        ha="left", color='black')
plt.show()
#Replacing restaurants with their ratings given as New to NAN and dropping them finally
rating_df = df.copy()
rating_df['rate'] = rating_df['rate'].replace('NEW',np.NaN)
rating_df['rate'] = rating_df['rate'].replace('-',np.NaN)
rating_df.dropna(how = 'any', inplace = True)
rating_df['rate'] = rating_df.loc[:,'rate'].replace('[ ]','',regex = True)
rating_df['rate'] = rating_df['rate'].astype(str)
rating_df['rate'] = rating_df['rate'].apply(lambda r: r.replace('/5',''))
rating_df['rate'] = rating_df['rate'].apply(lambda r: float(r))
plt.figure(figsize=(20, 8))
sns.set_style('white')
ax = sns.countplot(x='rate', hue='book_table', data=rating_df, palette='Greys_d')
sns.despine()
plt.title('Rating of Restaurants vs Table Booking')
plt.xlabel('Rating')
plt.ylabel('Number of restaurants')
plt.show()
plt.figure(figsize=(20, 8))
sns.set_style('white')
ax = sns.countplot(x='rate', hue='online_order', data=rating_df, palette='Greys_d')
sns.despine()
plt.title('Rating of Restaurants vs Online Delivery')
plt.xlabel('Rating')
plt.ylabel('Number of restaurants')
plt.show()
rating_df.loc[rating_df['rate'] >= 4, 'rating_category'] = 'Above 4'
rating_df.loc[(rating_df['rate'] >= 3) & (rating_df['rate'] < 4), 'rating_category'] = 'Above 3'
rating_df.loc[(rating_df['rate'] >= 2) & (rating_df['rate'] < 3), 'rating_category'] = 'Above 2'
rating_df.loc[rating_df['rate'] < 2, 'rating_category'] = 'Above 1'
def dish_served_per_rest_type(rest_type):
    dishLiked_restType_df = df.groupby(['address', 'dish_liked', 'rest_type']).dish_liked.nunique().index.to_frame()
    dishLiked_restType_temp_df = pd.DataFrame()
    dishLiked_restType_temp_df = dishLiked_restType_df[dishLiked_restType_df['rest_type'] == rest_type].dish_liked.str.strip().str.split(',', expand=True)
    dish_liked_temp_df = pd.DataFrame()
    dish_liked_temp_df = pd.concat([dishLiked_restType_temp_df.iloc[:,0].str.strip(), dishLiked_restType_temp_df.iloc[:,1].str.strip()]).value_counts()
    temp_df = pd.DataFrame({'dishes':dish_liked_temp_df[:10], 'group':dish_liked_temp_df[:10].index})
    norm = matplotlib.colors.Normalize(vmin=min(dish_liked_temp_df[:10]), vmax=max(dish_liked_temp_df[:10]))
    colors = [matplotlib.cm.Blues(norm(value)) for value in dish_liked_temp_df[:10]]
    squarify.plot(sizes=temp_df['dishes'], label=temp_df['group'], alpha=0.8, color=colors)
    plt.title("Top 10 dishes served in "+rest_type, fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.show()
dish_served_per_rest_type('Quick Bites')
dish_served_per_rest_type('Casual Dining')
dish_served_per_rest_type('Cafe')
dish_served_per_rest_type('Delivery')
dish_served_per_rest_type('Dessert Parlor')
dish_served_per_rest_type('Takeaway')
dish_served_per_rest_type('Bar')
costfortwo_df = df.groupby(['address', 'approx_cost(for two people)'])
cost_df = costfortwo_df['approx_cost(for two people)'].nunique().index.to_frame()
cost_df['approx_cost(for two people)'] = cost_df['approx_cost(for two people)'].str.replace(',', '').astype(float)
plt.figure(figsize=(8,8))
cost_dist = cost_df['approx_cost(for two people)'].dropna()
sns.distplot(cost_dist, bins=20, kde_kws={'color':'k', 'lw':3, 'label':'KDE'})
plt.show()
rating_df_temp = rating_df.groupby(['address', 'rate'])
plt.figure(figsize=(8,8))
rating = rating_df_temp.rate.nunique().index.to_frame()['rate']
sns.distplot(rating,bins=20,kde_kws={"color": "k", "lw": 3, "label": "KDE"})
plt.show()
rating.describe()
cost_rating_temp_df = df.groupby(['address', 'rate', 'approx_cost(for two people)', 'book_table', 'online_order'])
cost_rating_df = cost_rating_temp_df[['rate', 'approx_cost(for two people)', 'book_table', 'online_order']].nunique().index.to_frame()
cost_rating_df['approx_cost(for two people)'] = cost_rating_df['approx_cost(for two people)'].str.replace(',','').astype(float)
cost_rating_df['rate'] = cost_rating_df['rate'].apply(lambda x: float(x.split('/')[0]) if len(x) > 3 else np.nan).dropna()
fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
sns.scatterplot(x='rate', y='approx_cost(for two people)', hue='book_table', data=cost_rating_df, ax=axis[0])
sns.scatterplot(x='rate', y='approx_cost(for two people)', hue='online_order', data=cost_rating_df, ax=axis[1])
plt.show()
regression_df = zomato_data.copy()
regression_df=regression_df.drop(['url','dish_liked','phone'],axis=1)
regression_df.dropna(how='any', inplace=True)
#Preprocessing rate
regression_df = regression_df.loc[regression_df['rate'] !='NEW']
regression_df = regression_df.loc[regression_df['rate'] !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
regression_df['rate'] = regression_df['rate'].apply(remove_slash).str.strip().astype('float')
#Preprocessing approx_cost(for two people)
regression_df['approx_cost(for two people)'] = regression_df['approx_cost(for two people)'].dropna()
regression_df['approx_cost(for two people)'] = regression_df['approx_cost(for two people)'].astype(str)
regression_df['approx_cost(for two people)'] = regression_df['approx_cost(for two people)'].apply(lambda x: x.replace(',','.'))
regression_df['approx_cost(for two people)'] = regression_df['approx_cost(for two people)'].astype(float)
regression_df['name'] = regression_df['name'].apply(lambda x: x.title())
regression_df['online_order'].replace(('Yes', 'No'), (True, False), inplace=True)
regression_df['book_table'].replace(('Yes', 'No'), (True, False), inplace=True)
def encoding(regression_df):
    for column in regression_df.columns[~regression_df.columns.isin(['rate', 'approx_cost(for two people)', 'votes'])]:
        regression_df[column] = regression_df[column].factorize()[0]
    return regression_df

regression_df_en = encoding(regression_df)
plt.figure(figsize=(15,10))
sns.heatmap(regression_df_en.corr(), annot=True, cmap='Greys')
plt.title('Correlation Heatmap', size=20)
plt.show()
#Independent variables and dependent variable
x = regression_df_en.iloc[:, [2,3,5,6,7,8,9,11]]
y = regression_df_en['rate']
#Splitting dataset into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
#Random Forest Regression
model = RandomForestRegressor(n_estimators=500)
model.fit(x_train, y_train)
print('Accuracy: {}'.format(model.score(x_test, y_test)))
