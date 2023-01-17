import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
automobile = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')
automobile.head(2)
automobile.dtypes
automobile.describe()
automobile.info()
automobile.isnull().any().sum()
automobile.columns
obj_col = automobile.select_dtypes('object').columns
obj_col
col_list = []
for col in automobile.columns:
    i = automobile[col][automobile[col] == '?'].count()
    if i > 0:
        col_list.append(col)
        print(col, i)
col_list
automobile[col_list].dtypes
automobile[col_list].head(4)
null_list = ['normalized-losses','bore','stroke', 'horsepower','peak-rpm','price']
for col in null_list:
    automobile[col] = pd.to_numeric(automobile[col], errors = 'coerce')
automobile.isnull().sum().sort_values(ascending = False)
for col in null_list:
    automobile[col] = automobile[col].fillna(automobile[col].mean())
automobile.isnull().any().sum()
automobile[automobile['num-of-doors'] == '?']
automobile.drop(index = [27,63], inplace = True)
automobile[automobile['num-of-doors'] == '?']
automobile.make.value_counts().head(10).plot(kind = 'bar', figsize = (8,2))
plt.title('Number of Vehicles by make')
plt.ylabel('Number of vehicles')
plt.xlabel('Make')
automobile.symboling.hist(bins = 6, color = 'g')
plt.title('Insurance risk rating of vehicles')
plt.ylabel('Number of vehicles')
plt.xlabel('Risk rating')
automobile['normalized-losses'].hist(bins = 5, color = 'orange')
plt.title('Normalized losses of vehicles')
plt.ylabel('Number of vehicles')
plt.xlabel('Normalized losses')
automobile.head(3)
automobile['fuel-type'].value_counts().plot(kind = 'bar', color = 'purple')
automobile['aspiration'].value_counts().plot.pie(figsize = (5,5), autopct = '%.2f')
automobile.horsepower[np.abs(automobile.horsepower-automobile.horsepower.mean())<=(3*automobile.horsepower.std())].hist(bins=5,color='red')
plt.title('Horse power histogram')
plt.ylabel('Number of vehicles')
plt.xlabel('Horse power')
automobile['curb-weight'].hist(bins=5, color = 'brown')
plt.title('curb weight histogram')
plt.ylabel('Number of vehicles')
plt.xlabel('Curb weight')
automobile['drive-wheels'].value_counts().plot(kind ='bar', color = 'grey')
plt.title('Drive Wheels diagram')
plt.ylabel('Number of vehicles')
plt.xlabel('Drive wheels')
automobile['num-of-doors'].value_counts().plot(kind = 'bar', color = 'purple')
plt.title('Number of doors frequency diagram')
plt.ylabel('Number of vehicles')
plt.xlabel('Number of doors')
sns.set_context('notebook', font_scale =1.0, rc = {'line.linewidth': 2.5})
plt.figure(figsize = (13, 7))
a = sns.heatmap(automobile.corr(), annot = True, fmt = '.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation = 90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation = 30)
plt.rcParams['figure.figsize'] = (23,10)
ax = sns.boxplot(x='make', y='price', data = automobile)
g = sns.lmplot('price', 'engine-size', automobile)
g = sns.lmplot('normalized-losses','symboling', automobile)
plt.scatter(automobile['engine-size'], automobile['peak-rpm'])
plt.xlabel('Engine size')
plt.ylabel('peak RPM')
g = sns.lmplot('city-mpg', 'curb-weight', automobile)
g = sns.lmplot('highway-mpg',"curb-weight", automobile,  fit_reg=False)
fig = plt.figure(figsize = (6,4))
automobile.groupby('drive-wheels')['city-mpg'].mean().plot(kind='bar', color = 'peru')
plt.title("Drive wheels City MPG")
plt.ylabel('City MPG')
plt.xlabel('Drive wheels')
fig = plt.figure(figsize = (6,4))
automobile.groupby('drive-wheels')['highway-mpg'].mean().plot(kind='bar', color = 'peru')
plt.title("Drive wheels Highway MPG")
plt.ylabel('Highway MPG')
plt.xlabel('Drive wheels')
plt.rcParams['figure.figsize']=(10,5)
ax = sns.boxplot(x="drive-wheels", y="price", data=automobile)
pd.pivot_table(automobile,index=['body-style','num-of-doors'], values='normalized-losses').plot(kind='bar',color='purple')
plt.title("Normalized losses based on body style and no. of doors")
plt.ylabel('Normalized losses')
plt.xlabel('Body style and No. of doors')
