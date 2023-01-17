# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)
data = pd.read_csv('../input/houseRent/housing_train.csv')
data.shape
data.head()
data.tail()
data.info()
data.columns
data.describe()
data.isna().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(data.isnull(), ax=ax, cmap="YlGnBu", center=0).set(
            title = 'Missing Data', 
            xlabel = 'Columns', 
            ylabel = 'Data Points');
# make a list of the variables that contain missing values
vars_with_na = [var for var in data.columns if data[var].isnull().sum() > 0]
data[vars_with_na].isnull().mean()
def analyse_na_value(df, var):
    df = df.copy()
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    df[var] = np.where(df[var].isna(), 1, 0)
    grs = df.groupby(var)['price'].median().reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(x=grs[var], y=grs['price'])
    plt.title(var)
    plt.show()


# let's run the function on each variable with missing data
for var in vars_with_na:
    analyse_na_value(data, var)
# make a list of the categorical variables that contain missing values

vars_with_na = [
    var for var in data.columns
    if data[var].isnull().sum() > 0 and data[var].dtypes == 'O'
]
print(vars_with_na)
data[vars_with_na].isna().mean()
data[vars_with_na].head()
data.description[0]
data[['region','state']].head(15)
data.groupby('region')['state'].value_counts()
data.groupby('type')['laundry_options'].value_counts()
data.groupby('type')['parking_options'].value_counts()
# make a list with the numerical variables that contain missing values
vars_with_na = [
    var for var in data.columns
    if data[var].isnull().sum() > 0 and data[var].dtypes != 'O'
]
print(vars_with_na)
# print percentage of missing values per variable
data[vars_with_na].isnull().mean()
data.groupby('region')['lat'].value_counts()
data.groupby('region')['long'].value_counts()
bool_vars = [var for var in data if data[var].nunique() == 2]

data[bool_vars].head()
# make list of numerical variables
num_vars = [var for var in data.columns if data[var].dtypes != 'O' and var not in bool_vars]

print('Number of numerical variables: ', len(num_vars))

# visualise the numerical variables
data[num_vars].head()
print('Number of House Id labels: ', len(data.id.unique()))
print('Number of Houses in the Dataset: ', len(data))
plt.scatter(x=data['long'], y=data['lat'],alpha=0.01)
plt.xlim(right=-50)
plt.ylim(bottom=20,top=60)
plt.show()
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame


geometry = [Point(xy) for xy in zip(data['long'], data['lat'])]
gdf = GeoDataFrame(data, geometry=geometry)   

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);
#  let's male a list of discrete variables
discrete_vars = [var for var in num_vars if len(
    data[var].unique()) < 20 and var not in ['id', 'price']]


print('Number of discrete variables: ', len(discrete_vars))
# let's visualise the discrete variables

data[discrete_vars].head()
def analyse_discrete(df, var):
    df = df.copy()
    grs = df.groupby(var)['price'].median().reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(x=grs[var], y=grs['price'])
    plt.title(var.upper())
    plt.show()
    
    
for var in discrete_vars:
    analyse_discrete(data, var)
# make list of continuous variables
cont_vars = [
    var for var in num_vars if var not in discrete_vars+['id']]

print('Number of continuous variables: ', len(cont_vars))
# let's visualise the continuous variables

data[cont_vars].head()
# Let's go ahead and analyse the distributions of these variables
def analyse_continuous(df, var):
    df = df.copy()  
    df = df.dropna(axis=0)
    plt.figure(figsize=(10,6))
    sns.set_style("darkgrid")
    sns.distplot(df[var], hist=True)
    plt.legend(['Skewness={:.2f} Kurtosis={:.2f}'.format(
            data[var].skew(), 
            data[var].kurt())
        ],
        loc='best')
    plt.title(var)
    plt.show()

for var in cont_vars:
    analyse_continuous(data, var)
# Let's go ahead and analyse the distributions of these variables
# after applying a logarithmic transformation
def analyse_transformed_continuous(df, var):
    df = df.copy()
    df = df.dropna(axis=0)

    # log does not take 0 or negative values, so let's be
    # careful and skip those variables
    if var == 'lat' or var == 'long':
        pass
    else:
        # log transform the variable
        df[var] = np.log1p(df[var])
    plt.figure(figsize=(10,6))
    sns.set_style("darkgrid")
    sns.distplot(df[var], hist=True)
    plt.legend(['Skewness={:.2f} Kurtosis={:.2f}'.format(
            data[var].skew(), 
            data[var].kurt())
        ],
        loc='best')
    plt.title(var)
    plt.show()


for var in cont_vars:
    analyse_transformed_continuous(data, var)
# let's make boxplots to visualise outliers in the continuous variables


def find_outliers(df, var):
    df = df.copy()

    # log does not take negative values, so let's be
    # careful and skip those variables
    if var == 'lat' or var == 'long':
        pass
    else:
        # log transform the variable
        df[var] = np.log1p(df[var])
    ax = sns.boxplot(x=data[var], palette="muted", orient="vertical")
    plt.title(var)
    plt.ylabel(var)
    plt.show()


for var in cont_vars:
    find_outliers(data, var)
def out_iqr(df , column):
    global lower,upper
    q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
    # calculate the IQR
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    # calculate the lower and upper bound value
    lower, upper = q25 - cut_off, q75 + cut_off
    print('The IQR is',iqr)
    print('The lower bound value is', lower)
    print('The upper bound value is', upper)
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > upper]
    df2 = df[df[column] < lower]
    return print('Total number of outliers are', df1.shape[0]+ df2.shape[0])
def out_std(df, column):
    global lower,upper
    # calculate the mean and standard deviation of the data frame
    data_mean, data_std = df[column].mean(), df[column].std()
    # calculate the cutoff value
    cut_off = data_std * 3
    # calculate the lower and upper bound value
    lower, upper = data_mean - cut_off, data_mean + cut_off
    print('The lower bound value is', lower)
    print('The upper bound value is', upper)
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > upper]
    df2 = df[df[column] < lower]
    return print('Total number of outliers are', df1.shape[0]+ df2.shape[0])

fig, ax = plt.subplots()
ax.scatter(x = data['sqfeet'], y = data['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel('sqfeet', fontsize=13)
plt.show()
out_iqr(data, 'price')
out_std(data,'price')

fig, ax = plt.subplots()
ax.scatter(x = data['sqfeet'], y = data['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel('sqfeet', fontsize=13)
plt.show()
out_iqr(data, 'sqfeet')
out_std(data,'sqfeet')

fig, ax = plt.subplots()
ax.scatter(x = data['sqfeet'], y = data['beds'])
plt.ylabel('beds', fontsize=13)
plt.xlabel('sqfeet', fontsize=13)
plt.show()
out_iqr(data, 'beds')
out_std(data,'beds')
out_iqr(data, 'baths')
out_std(data,'baths')
out_iqr(data.dropna(axis=0), 'lat')
out_iqr(data.dropna(axis=0), 'long')
# capture categorical variables in a list
cat_vars = [var for var in data.columns if data[var].dtypes == 'O']

print('Number of categorical variables: ', len(cat_vars))
# let's visualise the values of the categorical variables
data[cat_vars].head()
data[cat_vars].nunique().sort_values(ascending=False)
data[cat_vars].nunique() / len(data)
# recapture categorical variables in a list
cat_vars = [var for var in cat_vars if var not in ['url', 'image_url', 'description', 'region_url']]
data[cat_vars].nunique()
def analyse_rare_labels(df, var, rare_perc):
    df = df.copy()

    # determine the % of observations per category
    tmp = df.groupby(var)['price'].count() / len(df)

    # return categories that are rare
    return tmp[tmp < rare_perc]

# print categories that are present in less than
# 1 % of the observations


for var in cat_vars:
    print(analyse_rare_labels(data, var, 0.01))
    print()
def find_frequent_labels(df, var, rare_perc):
    # function finds the labels that are shared by more than
    # a certain % of the houses in the dataset
    df = df.copy()
    tmp = df.groupby(var)['price'].count() / len(df)
    return tmp[tmp > rare_perc].index.values

frequent_ls = {}
for var in cat_vars:
    frequent_ls[var] = find_frequent_labels(data, var, 0.01)
    
frequent_ls
grdsp = data.groupby(["type"])[["price"]].mean().reset_index()

fig = px.pie(grdsp,
             values="price",
             names="type",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
data['state'].value_counts().sort_values(ascending=False)
df = data[((data['long']>-125) & (data['long']<-45)) & ((data['lat']>30) & (data['lat']<45))]
df = df[df.price<2400]
df.plot(kind="scatter", x="lat", y="long", alpha=0.4, 
        s=df["state"].value_counts()[1]/100, label="no_of_houses", 
        c="price", cmap=plt.get_cmap("jet"), colorbar=True,
       figsize=(12,12))
plt.title('House Rent Across State')
plt.legend()
data['region'].value_counts().sort_values(ascending=False)
df = data[((data['long']>-125) & (data['long']<-45)) & ((data['lat']>30) & (data['lat']<45))]
df = df[df.price<2400]
df.plot(kind="scatter", x="lat", y="long", alpha=0.4, 
        s=df["region"].value_counts()[1]/100, label="no_of_houses", 
        c="price", cmap=plt.get_cmap("jet"), colorbar=True,
        figsize=(12,12))
plt.title('House Rent Across Region')
plt.legend()
corr_matrix = data.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

fig, ax = plt.subplots(figsize=(12,12)) 

sns.heatmap(corr_matrix, 
            annot=True, 
            mask=mask,
            ax=ax, 
            cmap='BrBG').set(
    title = 'Feature Correlation', xlabel = 'Columns', ylabel = 'Columns')

ax.set_yticklabels(corr_matrix.columns, rotation = 0)
ax.set_xticklabels(corr_matrix.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})