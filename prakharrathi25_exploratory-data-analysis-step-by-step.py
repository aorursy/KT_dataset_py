import pandas as pd 

import numpy as np
cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

data = pd.read_csv("../input/imports-85.data.txt", names = cols)

data.shape
data.head()
data = data.replace('?', np.NaN)

data.head(10)
data.isnull().sum()
avg_norm_loss = data['normalized-losses'].astype('float').mean()

print("The average is {}".format(avg_norm_loss))

data["normalized-losses"].replace(np.NaN, avg_norm_loss, inplace = True)
#Bore Values

avg_bore = data['bore'].astype('float').mean()

data['bore'].replace(np.NaN, avg_bore, inplace = True)



#Stroke Values

avg_stroke = data['stroke'].astype('float').mean(axis = 0)

data['stroke'].replace(np.nan, avg_stroke, inplace = True)



#horsepower Values 

avg_horsepower = data['horsepower'].astype('float').mean(axis=0)

data['horsepower'].replace(np.nan, avg_horsepower, inplace=True)



#Peak RPM 

avg_peakrpm= data['peak-rpm'].astype('float').mean(axis=0)

data['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
data['num-of-doors'].value_counts()

data['num-of-doors'].value_counts().idxmax()

data["num-of-doors"].replace(np.nan, "four", inplace=True)

data.head()
data.dropna(subset = ['price'], axis = 0, inplace = True)

data.shape
data.dtypes
data[["bore", "stroke"]] = data[["bore", "stroke"]].astype("float")

data[["normalized-losses"]] = data[["normalized-losses"]].astype("int")

data[["price"]] = data[["price"]].astype("float")

data[["peak-rpm"]] = data[["peak-rpm"]].astype("float")

data.head()
import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
# list the data types for each column

print(data.dtypes)
data.corr()
data[['bore','stroke' ,'compression-ratio','horsepower']].corr()
sns.regplot(x = 'engine-size', y = 'price', data = data)

plt.ylim(0, )
data['engine-size'].corr(data['price'])
sns.regplot(x = 'highway-mpg', y = 'price', data = data)

plt.title("Highway Miles per Gallon vs Price")

plt.ylabel("Price")
data['highway-mpg'].corr(data['price'])
sns.regplot(x = 'peak-rpm', y = 'price', data = data)
data['peak-rpm'].corr(data['price'])
sns.boxplot(x = 'body-style', y = 'price', data = data)
data.describe()
data.describe(include = ['object'])
data['drive-wheels'].value_counts()
drive_counts = data['drive-wheels'].value_counts()

drive_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)

drive_counts
drive_counts.index.name = 'drive-wheels'

drive_counts
data['drive-wheels'].unique()
group_one = data[['drive-wheels','body-style','price']]

# grouping results 

group_one = group_one.groupby(['drive-wheels'], as_index = False).mean()

group_one
# grouping results

gptest = data[['drive-wheels','body-style','price']]

grouped_test1 = gptest.groupby(['drive-wheels','body-style'], as_index = False).mean()

grouped_test1
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')

grouped_pivot
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0

grouped_pivot
#use the grouped results

plt.pcolor(grouped_pivot, cmap='RdBu')

plt.colorbar()

plt.show()
fig, ax = plt.subplots()

im = ax.pcolor(grouped_pivot, cmap='RdBu')



# label names

row_labels = grouped_pivot.columns.levels[1]

col_labels = grouped_pivot.index



# move ticks and labels to the center

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor = False)

ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor = False)



# insert labels

ax.set_xticklabels(row_labels, minor = False)

ax.set_yticklabels(col_labels, minor = False)



# rotate label if too long

plt.xticks(rotation = 90)



fig.colorbar(im)

plt.show()
data.corr()
pearson_coef, p_value = stats.pearsonr(data['wheel-base'], data['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
pearson_coef, p_value = stats.pearsonr(data['horsepower'].astype('float'), data['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
pearson_coef, p_value = stats.pearsonr(data['length'], data['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
plt.pcolor(data.corr(), cmap = 'RdBu')

plt.colorbar()

plt.show()
sns.heatmap(data.corr(), annot = True)

sns.set(rc={'figure.figsize':(15, 15)})
grouped_test2 = gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

grouped_test2.head(2)
gptest.head(10)
grouped_test2.get_group('4wd')['price']
# ANOVA

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  

 

print( "ANOVA results: F=", f_val, ", P =", p_val)   
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  

 

print( "ANOVA results: F=", f_val, ", P =", p_val )
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  

   

print( "ANOVA results: F=", f_val, ", P =", p_val)   
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  

 

print("ANOVA results: F =", f_val, ", P =", p_val)   