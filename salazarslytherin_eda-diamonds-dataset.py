import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import seaborn as sns

sns.set_palette("husl")

from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings('ignore')

plt.style.use('bmh')
df = pd.read_csv('../input/diamonds/diamonds.csv')

df.head()

df['volume'] = df['x']*df['y']*df['z']
df.info()
print(df['price'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['price'], color='g', bins=100, hist_kws={'alpha': 0.4});
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])

df_num.head()
df_num.hist(figsize=(10, 10), bins=50, xlabelsize=8, ylabelsize=8);
plt.figure(figsize=[12,12])



# First subplot showing the diamond carat weight distribution

plt.subplot(221)

plt.hist(df['carat'],bins=20,color='b')

plt.xlabel('Carat Weight')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Carat Weight')



# Second subplot showing the diamond depth distribution

plt.subplot(222)

plt.hist(df['depth'],bins=20,color='r')

plt.xlabel('Diamond Depth (%)')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Depth')



# Third subplot showing the diamond price distribution

plt.subplot(223)

plt.hist(df['price'],bins=20,color='g')

plt.xlabel('Price in USD')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Price')



# Fourth subplot showing the diamond volume distribution

plt.subplot(224)

plt.hist(df['volume'],bins=20,color='m')

plt.xlabel('Volume in mm cubed')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Volume')
df_num_corr = df_num.corr()['price'][:-1] # -1 because the latest row is SalePrice

golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)

print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['price'])
corr = df_num.drop('price', axis=1).corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
quantitative_features_list = ['carat','cut','color','clarity','depth','table','x','y','z']

df_quantitative_values = df[quantitative_features_list]

df_quantitative_values.head()
features_to_analyse = [x for x in quantitative_features_list if x in golden_features_list]

features_to_analyse.append('price')

features_to_analyse
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.regplot(x = 'carat', y = 'price', data=df, ax = saxis[0,0])

sns.regplot(x = 'volume', y = 'price', data=df, ax = saxis[0,1])

# Order the plots from worst to best

sns.barplot(x = 'cut', y = 'price', order=['Fair','Good','Very Good','Premium','Ideal'], data=df, ax = saxis[1,0])

sns.barplot(x = 'color', y = 'price', order=['J','I','H','G','F','E','D'], data=df, ax = saxis[1,1])

sns.barplot(x = 'clarity', y = 'price', order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], data=df, ax = saxis[1,2])
df['cut'] = df['cut'].apply(lambda x: 1 if x=='Fair' else(2 if x=='Good' 

                                           else(3 if x=='Very Good' 

                                           else(4 if x=='Premium' else 5))))



df['color'] = df['color'].apply(lambda x: 1 if x=='J' else(2 if x=='I'

                                          else(3 if x=='H'

                                          else(4 if x=='G'

                                          else(5 if x=='F'

                                          else(6 if x=='E' else 7))))))



df['clarity'] = df['clarity'].apply(lambda x: 1 if x=='I1' else(2 if x=='SI2'

                                          else(3 if x=='SI1'

                                          else(4 if x=='VS2'

                                          else(5 if x=='VS1'

                                          else(6 if x=='WS2'

                                          else 7 if x=='WS1' else 8))))))



scaler = MinMaxScaler()

df[['cut','color','clarity']] = scaler.fit_transform(df[['cut','color','clarity']])

df['diamond score'] = df['cut'] + df['color'] + df['clarity']

sns.regplot(x = 'diamond score', y = 'price', data=df)
fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))



for i, ax in enumerate(fig.axes):

    if i < len(features_to_analyse) - 1:

        sns.regplot(x=features_to_analyse[i],y='price', data=df[features_to_analyse], ax=ax)
plt.figure(figsize=(12, 12))

correlation = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

sns.heatmap(correlation, vmax=1, annot=True,square=True)
# quantitative_features_list[:-1] as the last column is SalePrice and we want to keep it

categorical_features = [a for a in quantitative_features_list[:-1] + df.columns.tolist() if (a not in quantitative_features_list[:-1]) or (a not in df.columns.tolist())]

df_categ = df[categorical_features]

df_categ.head()
df_not_num = df_categ.select_dtypes(include = ['O'])

print('There is {} non numerical features including:\n{}'.format(len(df_not_num.columns), df_not_num.columns.tolist()))
fig, axes = plt.subplots(round(len(df_not_num.columns) / 3), 3, figsize=(12, 30))



for i, ax in enumerate(fig.axes):

    if i < len(df_not_num.columns):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

        sns.countplot(x=df_not_num.columns[i], alpha=0.7, data=df_not_num, ax=ax)



fig.tight_layout()