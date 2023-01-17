import pandas as pd

import numpy as np

import copy

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

import pylab

import math

%matplotlib inline

import os
print(os.listdir("../input"))
wholesale_customer_df = pd.read_csv('../input/WholesaleCustomersData.csv')
wholesale_customer_df.head()
wholesale_customer_df.info()
msno.matrix(wholesale_customer_df, figsize = (30,4))
wholesale_customer_drop_df = copy.deepcopy(wholesale_customer_df)

wholesale_customer_drop_df
del wholesale_customer_drop_df['Buyer/Spender']
wholesale_customer_drop_df
wholesale_customer_drop_df['Region'].value_counts()
wholesale_customer_drop_df['Channel'].value_counts()
def categorical_multi(i,j):

    pd.crosstab(wholesale_customer_drop_df[i],wholesale_customer_drop_df[j]).plot(kind='bar')

    plt.show()

    print(pd.crosstab(wholesale_customer_drop_df[i],wholesale_customer_drop_df[j]))



categorical_multi(i='Channel',j='Region')    
print('Descriptive Statastics of our Data:')

wholesale_customer_drop_df.describe().T
print('Descriptive Statastics of our Data including Channel & Retail:')

wholesale_customer_drop_df.describe(include='all').T
def plot_distribution(df, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(df.shape[1]) / cols)

    for i, column in enumerate(df.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        if df.dtypes[column] == np.object:

            g = sns.countplot(y=column, data=df)

            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]

            g.set(yticklabels=substrings)

            plt.xticks(rotation=25)

        else:

            g = sns.distplot(df[column])

            plt.xticks(rotation=25)

    

plot_distribution(wholesale_customer_drop_df, cols=3, width=20, height=20, hspace=0.45, wspace=0.5)
# Let’s remove the categorical columns:

products = wholesale_customer_drop_df[wholesale_customer_drop_df.columns[+2:wholesale_customer_drop_df.columns.size]]



#Let’s plot the distribution of each feature

def plot_distribution(df2, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(df2.shape[1]) / cols)

    for i, column in enumerate(df2.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        g = sns.boxplot(df2[column])

        plt.xticks(rotation=25)

    

plot_distribution(products, cols=3, width=20, height=10, hspace=0.45, wspace=0.5)
sns.set(style="ticks")

g = sns.pairplot(products,corner=True,kind='reg')

g.fig.set_size_inches(15,15)
# Compute the correlation matrix

corr = products.corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .6},annot=True)



plt.title("Pearson correlation", fontsize =20)
print('Descriptive Statastics of our Data:')

wholesale_customer_drop_df.describe().T
print('Descriptive Statastics of our Data including Channel & Retail:')

wholesale_customer_drop_df.describe(include='all').T
#created Summation of all the products into a new column - Spending

# there are many ways to create a new column, I have selected the below approach

wholesale_customer_spending_df = copy.deepcopy(wholesale_customer_drop_df)

wholesale_customer_spending_df['Spending'] =wholesale_customer_drop_df['Fresh']+wholesale_customer_drop_df['Milk']+wholesale_customer_drop_df['Grocery']+wholesale_customer_drop_df['Frozen']+wholesale_customer_drop_df['Detergents_Paper']+wholesale_customer_drop_df['Delicatessen']

wholesale_customer_spending_df
regiondf = wholesale_customer_spending_df.groupby('Region')['Spending'].sum()

print(regiondf)

print()

channeldf = wholesale_customer_spending_df.groupby('Channel')['Spending'].sum()

print(channeldf)
region_channel_df = wholesale_customer_spending_df.groupby(['Region','Channel'])['Spending'].sum()

print(region_channel_df)

#different way

#region_channel_df_1 = wholesale_customer_spending_df.groupby(['Region','Channel']).agg({'Spending' : 'sum'})

#print(region_channel_df_1)
wholesale_customer_drop_df_1 = copy.deepcopy(wholesale_customer_df)

del wholesale_customer_drop_df_1['Buyer/Spender']
data1 = wholesale_customer_drop_df_1.drop(columns=['Region'])

mean1 = data1.groupby('Channel').mean()

mean1.round(2)
data2 = wholesale_customer_drop_df_1.drop(columns=['Channel'])

mean2 = data2.groupby('Region').mean()

mean2.round(2)
sns.set(style="ticks", color_codes=True)

sns.catplot(x="Channel", y="Fresh", hue ="Region", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Fresh')
sns.catplot(x="Channel", y="Fresh", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Fresh')
sns.catplot(x="Region", y="Fresh", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Fresh')
sns.set(style="ticks", color_codes=True)

sns.catplot(x="Channel", y="Milk", hue ="Region", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Milk')
sns.catplot(x="Channel", y="Milk", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Milk')
sns.catplot(x="Region", y="Milk", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Milk')
sns.set(style="ticks", color_codes=True)

sns.catplot(x="Channel", y="Grocery", hue ="Region", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Grocery')
sns.catplot(x="Channel", y="Grocery", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Grocery')
sns.catplot(x="Region", y="Grocery", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Grocery')
sns.set(style="ticks", color_codes=True)

sns.catplot(x="Channel", y="Frozen", hue ="Region", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Frozen')
sns.catplot(x="Channel", y="Frozen", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Frozen')
sns.catplot(x="Region", y="Frozen", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Frozen')
sns.set(style="ticks", color_codes=True)

sns.catplot(x="Channel", y="Detergents_Paper", hue ="Region", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Detergents_Paper')
sns.catplot(x="Channel", y="Detergents_Paper", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Detergents_Paper')
sns.catplot(x="Region", y="Detergents_Paper", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Detergents_Paper')
sns.set(style="ticks", color_codes=True)

sns.catplot(x="Channel", y="Delicatessen", hue ="Region", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Delicatessen')
sns.catplot(x="Channel", y="Delicatessen", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Delicatessen')
sns.catplot(x="Region", y="Delicatessen", kind="bar", ci=None, data=wholesale_customer_drop_df)

plt.title('Item - Delicatessen')
standard_deviation_items = products.std() #use standard deviation to check the measure of variabilty

standard_deviation_items.round(2)
cv_fresh = np.std(products['Fresh']) / np.mean(products['Fresh'])

cv_fresh
cv_milk = np.std(products['Milk']) / np.mean(products['Milk'])

cv_milk
cv_grocery = np.std(products['Grocery']) / np.mean(products['Grocery'])

cv_grocery
cv_frozen = np.std(products['Frozen']) / np.mean(products['Frozen'])

cv_frozen
cv_detergents_paper = np.std(products['Detergents_Paper']) / np.mean(products['Detergents_Paper'])

cv_detergents_paper
cv_delicatessen = np.std(products['Delicatessen']) / np.mean(products['Delicatessen'])

cv_delicatessen
from scipy.stats import variation

print(variation(products, axis = 0))
variance_items = products.var()

variance_items
products.describe().T
pylab.style.use('seaborn-pastel')

products.plot.area(stacked=False,figsize=(11,5))

pylab.grid(); pylab.show()
plt.figure(figsize=(15,8))

sns.boxplot(data=products, orient="h", palette="Set2")
def plot_distribution(items, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(items.shape[1]) / cols)

    for i, column in enumerate(items.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        g = sns.boxplot(items[column])

        plt.xticks(rotation=25)

    

plot_distribution(products, cols=3, width=20, height=10, hspace=0.45, wspace=0.5)
# visual analysis via histogram

products.hist(figsize=(6,6));
def out_std(s, nstd=3.0, return_thresholds=False):

    data_mean, data_std = s.mean(), s.std()

    cut_off = data_std * nstd

    lower, upper = data_mean - cut_off, data_mean + cut_off

    if return_thresholds:

        return lower, upper

    else:

        return [True if x < lower or x > upper else False for x in s]



def out_iqr(s, k=1.5, return_thresholds=False):

    # calculate interquartile range

    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)

    iqr = q75 - q25

    # calculate the outlier cutoff

    cut_off = iqr * k

    lower, upper = q25 - cut_off, q75 + cut_off

    if return_thresholds:

        return lower, upper

    else: # identify outliers

        return [True if x < lower or x > upper else False for x in s]
# outlier_mask is a boolean list identifies the indices of the outliers

outlier_mask = out_std(products['Fresh'], nstd=3.0)

# first 10 elements

outlier_mask[:10]
products['Fresh'][outlier_mask]
plt.figure(figsize=(8,6))

sns.distplot(products['Fresh'], kde=False);

plt.vlines(products['Fresh'][outlier_mask], ymin=0, ymax=110, linestyles='dashed');
# outlier_mask is a boolean list identifies the indices of the outliers

outlier_mask_Milk = out_std(products['Milk'], nstd=3.0)

# first 10 elements

outlier_mask_Milk[:10]
products['Milk'][outlier_mask_Milk]
plt.figure(figsize=(8,6))

sns.distplot(products['Milk'], kde=False);

plt.vlines(products['Milk'][outlier_mask_Milk], ymin=0, ymax=110, linestyles='dashed');
# outlier_mask is a boolean list identifies the indices of the outliers

outlier_mask_Frozen = out_std(products['Frozen'], nstd=3.0)

# first 10 elements

outlier_mask_Frozen[:10]
products['Frozen'][outlier_mask_Frozen]
plt.figure(figsize=(8,6))

sns.distplot(products['Frozen'], kde=False);

plt.vlines(products['Frozen'][outlier_mask_Frozen], ymin=0, ymax=110, linestyles='dashed');
# outlier_mask is a boolean list identifies the indices of the outliers

outlier_mask_Grocery= out_std(products['Grocery'], nstd=3.0)

# first 10 elements

outlier_mask_Grocery[:10]
products['Grocery'][outlier_mask_Grocery]
plt.figure(figsize=(8,6))

sns.distplot(products['Grocery'], kde=False);

plt.vlines(products['Grocery'][outlier_mask_Grocery], ymin=0, ymax=110, linestyles='dashed');
# outlier_mask is a boolean list identifies the indices of the outliers

outlier_mask_Detergents_Paper= out_std(products['Detergents_Paper'], nstd=3.0)

# first 10 elements

outlier_mask_Detergents_Paper[:10]
products['Detergents_Paper'][outlier_mask_Detergents_Paper]
plt.figure(figsize=(8,6))

sns.distplot(products['Detergents_Paper'], kde=False);

plt.vlines(products['Detergents_Paper'][outlier_mask_Detergents_Paper], ymin=0, ymax=110, linestyles='dashed');
# outlier_mask is a boolean list identifies the indices of the outliers

outlier_mask_Delicatessen = out_std(products['Delicatessen'], nstd=3.0)

# first 10 elements

outlier_mask_Delicatessen[:10]
products['Delicatessen'][outlier_mask_Delicatessen]
plt.figure(figsize=(8,6))

sns.distplot(products['Delicatessen'], kde=False);

plt.vlines(products['Delicatessen'][outlier_mask_Delicatessen], ymin=0, ymax=110, linestyles='dashed');
# For comparison, make one array each using standard deviations of 2.0, 3.0 and 4.0.

std2 = products.apply(out_std, nstd=2.0)

std3 = products.apply(out_std, nstd=3.0)

std4 = products.apply(out_std, nstd=4.0)



# For comparison, make one array each at varying values of k.

iqr1 = products.apply(out_iqr, k=1.5)

iqr2 = products.apply(out_iqr, k=2.0)

iqr3 = products.apply(out_iqr, k=3.0)
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(ncols=3, nrows=2, figsize=(22, 12));

ax1.set_title('Outliers using 2 standard deviations');

ax2.set_title('Outliers using 3 standard deviations');

ax3.set_title('Outliers using 4 standard deviations');

ax4.set_title('Outliers using a 1.5 IQR cutoff');

ax5.set_title('Outliers using a 2.5 IQR cutoff');

ax6.set_title('Outliers using a 3.0 IQR cutoff');



sns.heatmap(std2, cmap='YlGn', ax=ax1);

sns.heatmap(std3, cmap='YlGn', ax=ax2);

sns.heatmap(std4, cmap='YlGn', ax=ax3);

sns.heatmap(iqr1, cmap='YlGn', ax=ax4);

sns.heatmap(iqr2, cmap='YlGn', ax=ax5);

sns.heatmap(iqr3, cmap='YlGn', ax=ax6);



plt.savefig('outliers.png')

plt.show()
def plot_cutoff(dataframe, col, nstd=2.0, color='red'):

    lower, upper = out_std(dataframe[col], nstd=nstd, return_thresholds=True)

    plt.axvspan(min(dataframe[col][dataframe[col] < lower], default=dataframe[col].min()), lower, alpha=0.2, color=color);

    plt.axvspan(upper, max(dataframe[col][dataframe[col] > upper], default=dataframe[col].max()), alpha=0.2, color=color);
column = 'Fresh'

sns.distplot(products[column], kde=False)

plot_cutoff(products, column, nstd=2.0, color='red');

plot_cutoff(products, column, nstd=3.0, color='green');

plot_cutoff(products, column, nstd=4.0, color='yellow');
column = 'Milk'

sns.distplot(products[column], kde=False)

plot_cutoff(products, column, nstd=2.0, color='red');

plot_cutoff(products, column, nstd=3.0, color='green');

plot_cutoff(products, column, nstd=4.0, color='yellow');
column = 'Grocery'

sns.distplot(products[column], kde=False)

plot_cutoff(products, column, nstd=2.0, color='red');

plot_cutoff(products, column, nstd=3.0, color='green');

plot_cutoff(products, column, nstd=4.0, color='yellow');
column = 'Frozen'

sns.distplot(products[column], kde=False)

plot_cutoff(products, column, nstd=2.0, color='red');

plot_cutoff(products, column, nstd=3.0, color='green');

plot_cutoff(products, column, nstd=4.0, color='yellow');
column = 'Detergents_Paper'

sns.distplot(products[column], kde=False)

plot_cutoff(products, column, nstd=2.0, color='red');

plot_cutoff(products, column, nstd=3.0, color='green');

plot_cutoff(products, column, nstd=4.0, color='yellow');
column = 'Delicatessen'

sns.distplot(products[column], kde=False)

plot_cutoff(products, column, nstd=2.0, color='red');

plot_cutoff(products, column, nstd=3.0, color='green');

plot_cutoff(products, column, nstd=4.0, color='yellow');
cols_prd = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
from sklearn.ensemble import IsolationForest

fig, axs = plt.subplots(2, 3, figsize=(22, 12), facecolor='w', edgecolor='k')

axs = axs.ravel()



for i, column in enumerate(cols_prd):

    isolation_forest = IsolationForest(contamination='auto')

    isolation_forest.fit(products[column].values.reshape(-1,1))



    xx = np.linspace(products[column].min(), products[column].max(), len(products)).reshape(-1,1)

    anomaly_score = isolation_forest.decision_function(xx)

    outlier_iso_forest = isolation_forest.predict(xx)

    

    axs[i].plot(xx, anomaly_score, label='anomaly score')

    axs[i].fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 

                     where=outlier_iso_forest==-1, color='r', 

                     alpha=.4, label='outlier region')

    axs[i].legend()

    axs[i].set_title(column)
wholesale_customer_drop_df.groupby(['Channel', 'Region']).agg(['mean', 'std']).round(1)
def hist_plot(column):

    fig = plt.figure()

    ax = fig.add_subplot(111) # stands for subplot(1,1,1)

    ax.hist(products[column], bins=25)

    plt.title('Histgram plot of ' + column)

    plt.show()



columns = ['Milk', 'Grocery', 'Detergents_Paper']

for c in columns:

    hist_plot(c)
#Display the distribution accross all features

features = products.columns.values





fig = plt.figure(figsize=(15,10))

for i in range(len(features)):

    ax = fig.add_subplot(2,3,i+1)

    ax.set_title(features[i])

    ax.hist(products[features[i]], bins = 100)

plt.show()
from scipy.stats import iqr

print('IQR of Fresh item            ' + str(iqr(wholesale_customer_drop_df['Fresh'])))

print('IQR of Milk item             ' + str(iqr(wholesale_customer_drop_df['Milk'])))

print('IQR of Grocery item          ' + str(iqr(wholesale_customer_drop_df['Grocery'])))

print('IQR of Frozen item           ' + str(iqr(wholesale_customer_drop_df['Frozen'])))

print('IQR of Detergents_Paper item ' + str(iqr(wholesale_customer_drop_df['Detergents_Paper'])))

print('IQR of Delicatessen item     ' + str(iqr(wholesale_customer_drop_df['Delicatessen'])))

print(wholesale_customer_drop_df.skew())
plt.scatter(x = wholesale_customer_df['Milk'], y = wholesale_customer_df['Grocery'])
from scipy.stats import boxcox, probplot, norm, shapiro



shapiro_test = {}

plt.figure(figsize=(15, 10))

for i in range(0,6):

    ax = plt.subplot(2,3,i+1)

    probplot(x = products[products.columns[i]], dist=norm, plot=ax)

    plt.title(products.columns[i])

    shapiro_test[products.columns[i]] = shapiro(products[products.columns[i]])

    

plt.show()



pd.DataFrame(shapiro_test, index=['Test Statistic', 'p-value']).transpose()


products_log = np.log(products)



shapiro_test = {}



plt.figure(figsize=(15, 10))

for i in range(6):

    ax = plt.subplot(2,3,i+1)

    probplot(x = products_log[products_log.columns[i]], dist=norm, plot=ax)

    plt.title(products_log.columns[i])

    shapiro_test[products.columns[i]] = shapiro(products[products.columns[i]])

    

plt.show()



pd.DataFrame(shapiro_test, index=['Test Statistic', 'p-value']).transpose()
from scipy.stats import boxcox



shapiro_test = {}

lambdas = {}



plt.figure(figsize=(15, 10))

plt.title('BoxCox Transformation')

for i in range(6):

    ax = plt.subplot(2,3,i+1)

    x, lbd = boxcox(products[products.columns[i]])

    probplot(x = x, dist=norm, plot=ax)

    plt.title(products.columns[i])

    shapiro_test[products.columns[i]] = shapiro(x)

    lambdas[products.columns[i]] = lbd

    

plt.show()



pd.DataFrame(shapiro_test, index=['Test Statistic', 'p-value']).transpose()
products.corr()
print('Correlation Heat map of the data')

plt.figure(figsize=(10,6))

sns.heatmap(products.corr(),annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='Spectral')

plt.show()
def scatterplot(i,j):

    sns.regplot(data=products_log,x=i,y=j)

    plt.show()
scatterplot(i='Milk',j='Grocery')
scatterplot(i='Milk',j='Detergents_Paper')
scatterplot(i='Detergents_Paper',j='Grocery')
pd.plotting.scatter_matrix(products, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
def plot_corr(df,size=10):

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''



    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    cax = ax.matshow(df, interpolation='nearest')

    ax.matshow(corr)

    fig.colorbar(cax)

    plt.xticks(range(len(corr.columns)), corr.columns);

    plt.yticks(range(len(corr.columns)), corr.columns);









plot_corr(products)