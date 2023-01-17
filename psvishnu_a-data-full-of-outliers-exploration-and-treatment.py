!pip install -U scikit-learn
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import warnings

import sklearn

from sklearn.model_selection import train_test_split

warnings.filterwarnings ( "ignore" )



# pd.set_option(""display.height"", 10)

# pd.set_option(""display.max_columns"", 500)

#pd.set_option(""display.width"", 1000)

pd.set_option("display.max_rows", 100)



# Version check

print('Current Version'.center(20,'-'))

print(f"Pandas:{pd.__version__}")

print(f"Numpy:{np.__version__}")

print(f"Seaborn:{sns.__version__}")

print(f"Scikit learn: {sklearn.__version__}")
data = pd.read_csv ( "../input/online-news-popularity/OnlineNewsPopularity.csv" )

print("Dataset Shape: {}".format(data.shape))

print(data.columns.tolist())
for i in data.columns:

    data = data.rename ( columns = { i : i.lstrip ( ) } )



data.columns[:10]
print(data['timedelta'].nunique())



data['timedelta'].describe()
df = data.drop ( [ "url" , "timedelta" ] , axis = 1 )



df.describe ( )
df.isnull ( ).sum ( ).sum ( )
num = df.select_dtypes ( include = "number" )
counter = 1

plt.figure(figsize=(15,18))

for col in num.columns:

    if np.abs(df [col].skew ( )) > 1 and df[col].nunique() < 10:

        plt.subplot(5,3,counter)

        counter += 1

        df [col].value_counts().plot.bar()

        plt.xticks(rotation = 45)

        plt.title(f'{col}\n(skewness {round(df [ col ].skew ( ),2)})')



plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

plt.show ( )
%%time

# > 10 sec

counter = 1

truly_conti = []

plt.figure(figsize=(18,40))

for i in num.columns:

    if np.abs(df [ i ].skew ( )) > 1 and df[i].nunique() > 10:

        plt.subplot(20,3,counter)

        counter += 1

        truly_conti.append(i)

        sns.distplot ( df [ i ] )

        plt.title(f'{i} (skewness {round(df [ i ].skew ( ),1)})')

        plt.xticks(rotation = 45)

plt.tight_layout()

plt.show ( )
plt.figure(figsize=(18,3))

sns.boxenplot(data=df.loc[:,truly_conti[0]],orient='h')

plt.title(truly_conti[0])

plt.show()
# plt.rcParams['figure.figsize'] = 18,10

plt.figure(figsize=(18,6))

sns.boxenplot(data=df.loc[:,truly_conti[1:4]],orient='h')

plt.show()
# plt.rcParams['figure.figsize'] = 18,10

plt.figure(figsize=(18,6))

sns.boxenplot(data=df.loc[:,truly_conti[4:10]],orient='h')

plt.show()
plt.figure(figsize=(18,3))

sns.boxenplot(data=df.loc[:,truly_conti[10]],orient='h')

plt.title(truly_conti[10])

plt.show()
plt.figure(figsize=(18,3))

sns.boxenplot(data=df.loc[:,truly_conti[11]],orient='h')

plt.title(truly_conti[11])

plt.show()
plt.figure(figsize=(18,10))

sns.boxenplot(data=df.loc[:,truly_conti[12:19]],orient='h')

plt.show()
plt.figure(figsize=(18,10))

sns.boxenplot(data=df.loc[:,truly_conti[19:30]],orient='h')

plt.show()
plt.figure(figsize=(18,3))

sns.boxenplot(data=df.loc[:,truly_conti[30]],orient='h')

plt.title(truly_conti[30])

plt.show()
df['shares'].describe(percentiles=[.1,.25,.375,.50,.625,.75,.90,.95,1])
# Will use the following convention

# 0: Less popular

# 1: Popular

df['Popularity'] = pd.cut(data.shares,(0,1400,843300), labels=[0,1])

df['Popularity'] = df['Popularity'].astype(int)

df['Popularity'].value_counts(normalize=True)
words = [

    'n_tokens_title','n_tokens_content','average_token_length','n_non_stop_words','n_unique_tokens','n_non_stop_unique_tokens'

]
plt.figure(figsize=(10,8))

sns.heatmap(df[words].corr(),cmap='coolwarm',annot=True)

plt.show()
section = df[words + ['Popularity']]

sns.heatmap(pd.DataFrame(section.corr()['Popularity']),cmap='coolwarm',annot=True)

plt.show()
sns.pairplot(data=df,diag_kind='kde',vars=words,hue='Popularity')

plt.show()
links = [

    'num_hrefs','num_self_hrefs','num_imgs','num_videos'

]
plt.figure(figsize=(10,8))

sns.heatmap(df[links].corr(),cmap='Accent',annot=True)

plt.show()
section = df[links + ['Popularity']]

sns.heatmap(pd.DataFrame(section.corr()['Popularity']),cmap='coolwarm',annot=True)

plt.show()
plt.figure(figsize=(10,8))

sns.pairplot(data=df,diag_kind='kde',vars=section)

plt.show()
time = [

    'weekday_is_monday','weekday_is_tuesday','weekday_is_wednesday','weekday_is_thursday','weekday_is_friday','weekday_is_saturday','weekday_is_sunday','is_weekend'

]
merge = df [ [ "weekday_is_monday" , "weekday_is_tuesday" , "weekday_is_wednesday" ,

               "weekday_is_thursday" , "weekday_is_friday" , "weekday_is_saturday" , 

                "weekday_is_sunday" ] ]

arr = [ ]

for i in range ( merge.shape [ 0 ] ):

    for j in range ( merge.shape [ 1 ] ):

        if j == 0 and merge.iloc [ i , j ] == 1:

            arr.append ( "Monday" )

        elif j == 1 and merge.iloc [ i , j ] == 1:

            arr.append ( "Tuesday" )

        elif j == 2 and merge.iloc [ i , j ] == 1:

            arr.append ( "Wednesday" )

        elif j == 3 and merge.iloc [ i , j ] == 1:

            arr.append ( "Thursday" )

        elif j == 4 and merge.iloc [ i , j ] == 1:

            arr.append ( "Friday" )

        elif j == 5 and merge.iloc [ i , j ] == 1:

            arr.append ( "Saturday" )

        elif j == 6 and merge.iloc [ i , j ] == 1:

            arr.append ( "Sunday" )



df [ "Day" ] = arr
%%time

plt.figure(1)

g = sns.PairGrid(data=df, y_vars="shares",

                 x_vars=time[:-1],

                 height=5, aspect=.5)

g.map(sns.pointplot, scale=1.2, errwidth=4).add_legend()

sns.despine(fig=g.fig, left=True)

plt.title('Total shares each day')



plt.figure(2)

g = sns.catplot(x="Day", y="shares",

                capsize=.2, palette="YlGnBu_d", height=6, aspect=3,

                kind="point", data=df)

plt.tight_layout()

plt.show()
%%time

g = sns.PairGrid(df, y_vars="shares",

                 x_vars=time[:-1],

                 hue="Popularity",

                 height=5, aspect=.5)

plt.title('Distribution of shares per day')

g.map(sns.pointplot, scale=1.3, errwidth=4)

sns.despine(fig=g.fig, left=True)



g = sns.catplot(x="Day", y="shares", hue="Popularity",

                capsize=.2, palette="YlGnBu_d", height=6, aspect=2.5,

                kind="point", data=df)

g.despine(left=True)

plt.tight_layout()

plt.show()
df.drop(columns=['Day'],inplace=True)
plt.figure(figsize=(10,8))

sns.heatmap(df[time].corr(),cmap='coolwarm',annot=True)

plt.show()
section = df[time + ['Popularity']]

sns.heatmap(pd.DataFrame(section.corr()['Popularity']),cmap='coolwarm',annot=True)

plt.show()
# import matplotlib.gridspec as gridspec



# plt.close('all')

# fig = plt.figure()



# gs1 = gridspec.GridSpec(4, 3)

# ax1 = fig.add_subplot(gs1[0])

# ax2 = fig.add_subplot(gs1[1])
keywords = ['num_keywords','kw_min_min','kw_max_min','kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg']

target = 'Popularity'



# fig,ax = plt.subplots(4,3,figsize=(10,8))

for i,col in enumerate(keywords,1):

    mean_per_cat = pd.pivot_table(df,values=col,index=target,aggfunc="mean")

    mean_per_cat.plot.barh(alpha=0.4)

    plt.title(col)

plt.show ( )
plt.figure(figsize=(10,8))

sns.heatmap(df[keywords].corr(),cmap='Accent',annot=True)

plt.show()
section = df[keywords + ['Popularity']]

sns.heatmap(pd.DataFrame(section.corr()['Popularity']),cmap='coolwarm',annot=True)

plt.show()
channels = [

    'data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world'

]

channel_names = ['Lifestyle','Entertainment','BUS','SocMed','Tech','World']
dict_channels = { k:v for k,v in zip(channels,range(1,7))}

dict_channels
channel_type = df.loc[:,channels].apply(lambda x: np.argmax(x),axis=1)

channel_type
df['channels'] = channel_type.apply(lambda x: dict_channels[x])
g = sns.catplot(x="channels", y="shares",

                capsize=.2, palette="YlGnBu_d", height=6, aspect=3,

                kind="point", data=df)

plt.xticks(range(6),channel_names)

plt.title("Total shares per category")
g = sns.catplot(x="channels", y="shares",hue="Popularity",

                capsize=.2, palette="YlGnBu_d", height=6, aspect=3,

                kind="point", data=df)



plt.xticks(range(6),channel_names)

plt.title('Channels based on popularity')

plt.show()
df.drop(columns=['channels'],inplace=True)
plt.figure(figsize=(10,8))

sns.heatmap(df[channels].corr(),cmap='Accent',annot=True)

plt.show()
section = df[channels + ['Popularity']]

sns.heatmap(pd.DataFrame(section.corr()['Popularity']),cmap='coolwarm',annot=True)

plt.show()
from scipy.stats import zscore



target = 'Popularity'

X = df.drop(columns=[target,'shares'])

y = df[target]



print(f"Original shape of dataframe: {X.shape}")
outliers = (X.apply(zscore)<3).all(axis=1)

X[outliers].shape
f"Choosing this approach we will loose {39644 - 24627} datapoints "
# This will be the data without any outliers

X[outliers].to_csv('data.csv',index=False)
# Since we have large amount of data planning to split it in 70-30 fashion

# train will be 70

# Of the 30, 10 validation and 20 test set

# By default, the data will be split in stratified fashion

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=42)

print(X_train.shape)

print(X_test.shape)
from fancyimpute import KNN    
outliers = np.abs(X_train.apply(zscore)) > 3

outliers.sum().sum()
from tqdm import tqdm
for col_name in tqdm(X_train.columns):

    mean = X_train[col_name].mean()

    std = X_train[col_name].std()

    X_train[col_name] = X_train[col_name].apply(lambda x: np.nan if abs((x-mean)/std)>3 else x)

#     X_train[col_name].loc[(X_train[col_name] < lower_limit) | (X_train[col_name] > upper_limit)] = np.NaN
print(X_train.isna().sum().sum())
# KNN based imputation 

X_filled_knn = KNN(k=3).fit_transform(X_train)
knn_train = pd.DataFrame(X_filled_knn,columns=X_train.columns)

knn_train['Popularity'] = y_train.values

knn_train.to_csv('train.csv',index=False)

print(knn_train.shape)
knn_train.head()
test = pd.concat((X_test,y_test),axis=1)

test.to_csv('test.csv',index=False)