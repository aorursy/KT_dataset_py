import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl

import numpy as np

import seaborn as sns



%matplotlib inline
# winequality-white.csv is a ';' separated csv file.



white_wine = pd.read_csv('../input/white-wine-quality/winequality-white.csv', sep=';')

red_wine = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
red_wine.head()
# store wine type as an attribute

red_wine['wine_type'] = 'red'   

white_wine['wine_type'] = 'white'



red_wine.head()
# bucket wine quality scores into qualitative quality labels



red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low' 

                                                          if value <= 5 else 'medium' 

                                                              if value <= 7 else 'high')

red_wine.head()
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'], 

                                           categories=['low', 'medium', 'high'])



red_wine.head()
white_wine.columns




white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low' 

                                                              if value <= 5 else 'medium' 

                                                                  if value <= 7 else 'high')

white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'], 

                                             categories=['low', 'medium', 'high'])



# merge red and white wine datasets

wines = pd.concat([red_wine, white_wine])   # concatenating horizontally since by default axis=0



# re-shuffle records just to randomize data points

wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)
wines.head()
subset_attributes = ['residual sugar', 'total sulfur dioxide', 'sulphates', 'alcohol', 'volatile acidity', 'quality']

rs = round(red_wine[subset_attributes].describe(),2)

ws = round(white_wine[subset_attributes].describe(),2)



pd.concat([rs, ws], axis=1, keys=['Red Wine Statistics', 'White Wine Statistics'])
subset_attributes = ['alcohol', 'volatile acidity', 'pH', 'quality']

ls = round(wines[wines['quality_label'] == 'low'][subset_attributes].describe(),2)

ms = round(wines[wines['quality_label'] == 'medium'][subset_attributes].describe(),2)

hs = round(wines[wines['quality_label'] == 'high'][subset_attributes].describe(),2)

pd.concat([ls, ms, hs], axis=1, keys=['Low Quality Wine', 'Medium Quality Wine', 'High Quality Wine'])
wines.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,

              xlabelsize=8, ylabelsize=8, grid=False)



# command to give space b/t diff. plots 

plt.tight_layout(rect=(0, 0, 1.2, 1.2))   
fig = plt.figure(figsize = (6,4))

title = fig.suptitle("Sulphates Content in Wine", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax = fig.add_subplot(1,1, 1)

ax.set_xlabel("Sulphates")

ax.set_ylabel("Frequency") 

ax.text(1.2, 800, r'$\mu$='+str(round(wines['sulphates'].mean(),2)), 

         fontsize=12)

freq, bins, patches = ax.hist(wines['sulphates'], color='steelblue', bins=15,

                                    edgecolor='black', linewidth=1)
fig = plt.figure(figsize = (6, 4))

title = fig.suptitle("Sulphates Content in Wine", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax1 = fig.add_subplot(1,1, 1)

ax1.set_xlabel("Sulphates")

ax1.set_ylabel("Density") 

sns.kdeplot(wines['sulphates'], ax=ax1, shade=True, color='steelblue')
w_q=wines['quality'].value_counts()   # most of the wines are having quality 6

w_q
list(w_q.index)
list(w_q.values)
fig = plt.figure(figsize = (6, 4))

title = fig.suptitle("Wine Quality Frequency", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax = fig.add_subplot(1,1, 1)

ax.set_xlabel("Quality")

ax.set_ylabel("Frequency") 

w_q = wines['quality'].value_counts()

w_q = (list(w_q.index), list(w_q.values))

ax.tick_params(axis='both', which='major', labelsize=8.5)

bar = ax.bar(w_q[0], w_q[1], color='steelblue', 

        edgecolor='black', linewidth=1)
f, ax = plt.subplots(figsize=(12, 8))

corr = wines.corr()

hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',

            linewidths=.05)

f.subplots_adjust(top=0.93)

t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)
wines.head()
cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity', 'quality_label']

pp = sns.pairplot(wines[cols], size=1.8, hue = 'quality_label' , aspect=1.8,

                  plot_kws=dict(edgecolor="k", linewidth=0.5),

                  diag_kind="kde", diag_kws=dict(shade=True))



fig = pp.fig  

fig.subplots_adjust(top=0.93, wspace=0.5)

t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity']

subset_df = wines[cols]



from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

scaled_df = ss.fit_transform(subset_df)

scaled_df

scaled_df = pd.DataFrame(scaled_df, columns=cols)

final_df = pd.concat([scaled_df, wines['wine_type']], axis=1)

final_df.head()
from pandas.plotting import parallel_coordinates



pc = parallel_coordinates(final_df, 'wine_type', color=('#FFE888', '#FF9999'))
plt.scatter(wines['sulphates'], wines['alcohol'],

            alpha=0.4, edgecolors='w')



plt.xlabel('Sulphates')

plt.ylabel('Alcohol')

plt.title('Wine Sulphates - Alcohol Content',y=1.05)
jp = sns.jointplot(x='sulphates', y='alcohol', data=wines,

              kind='reg', space=0, height=5, ratio=4)
fig = plt.figure(figsize = (10, 4))

title = fig.suptitle("Wine Type - Quality", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax1 = fig.add_subplot(1,2, 1)

ax1.set_title("Red Wine")

ax1.set_xlabel("Quality")

ax1.set_ylabel("Frequency") 



rw_q = red_wine['quality'].value_counts()

rw_q = (list(rw_q.index), list(rw_q.values))



ax1.set_ylim([0, 2500])

ax1.tick_params(axis='both', which='major', labelsize=8.5)

bar1 = ax1.bar(rw_q[0], rw_q[1], color='red', 

        edgecolor='black', linewidth=1)





ax2 = fig.add_subplot(1,2, 2)

ax2.set_title("White Wine")

ax2.set_xlabel("Quality")

ax2.set_ylabel("Frequency") 



ww_q = white_wine['quality'].value_counts()

ww_q = (list(ww_q.index), list(ww_q.values))



ax2.set_ylim([0, 2500])

ax2.tick_params(axis='both', which='major', labelsize=8.5)

bar2 = ax2.bar(ww_q[0], ww_q[1], color='white', 

        edgecolor='black', linewidth=1)
# multiple bar

cp = sns.countplot(x="quality", hue="wine_type", data=wines, 

                   palette={"red": "#FF9999", "white": "#FFE888"})
fig = plt.figure(figsize = (10,4))

title = fig.suptitle("Sulphates Content in Wine", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax1 = fig.add_subplot(1,2, 1)

ax1.set_title("Red Wine")

ax1.set_xlabel("Sulphates")

ax1.set_ylabel("Frequency") 

ax1.set_ylim([0, 1200])

ax1.text(1.2, 800, r'$\mu$='+str(round(red_wine['sulphates'].mean(),2)), 

         fontsize=12)

r_freq, r_bins, r_patches = ax1.hist(red_wine['sulphates'], color='red', bins=15,

                                     edgecolor='black', linewidth=1)



ax2 = fig.add_subplot(1,2, 2)

ax2.set_title("White Wine")

ax2.set_xlabel("Sulphates")

ax2.set_ylabel("Frequency")

ax2.set_ylim([0, 1200])

ax2.text(0.8, 800, r'$\mu$='+str(round(white_wine['sulphates'].mean(),2)), 

         fontsize=12)

w_freq, w_bins, w_patches = ax2.hist(white_wine['sulphates'], color='white', bins=15,

                                     edgecolor='black', linewidth=1)
fig = plt.figure(figsize = (10, 4))

title = fig.suptitle("Sulphates Content in Wine", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax1 = fig.add_subplot(1,2, 1)

ax1.set_title("Red Wine")

ax1.set_xlabel("Sulphates")

ax1.set_ylabel("Density") 

sns.kdeplot(red_wine['sulphates'], ax=ax1, shade=True, color='r')



ax2 = fig.add_subplot(1,2, 2)

ax2.set_title("White Wine")

ax2.set_xlabel("Sulphates")

ax2.set_ylabel("Density") 

sns.kdeplot(white_wine['sulphates'], ax=ax2, shade=True, color='y')
fig = plt.figure(figsize = (6, 4))

title = fig.suptitle("Sulphates Content in Wine", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1,1, 1)

ax.set_xlabel("Sulphates")

ax.set_ylabel("Frequency") 



g = sns.FacetGrid(wines, hue='wine_type', palette={"red": "r", "white": "y"})

g.map(sns.distplot, 'sulphates', kde=False, bins=15, ax=ax)

ax.legend(title='Wine Type')

plt.close(2)
f, (ax) = plt.subplots(1, 1, figsize=(12, 4))

f.suptitle('Wine Quality - Alcohol Content', fontsize=14)



sns.boxplot(x="quality", y="alcohol", data=wines,  ax=ax)

ax.set_xlabel("Wine Quality",size = 12,alpha=0.8)

ax.set_ylabel("Wine Alcohol %",size = 12,alpha=0.8)
f, (ax) = plt.subplots(1, 1, figsize=(12, 4))

f.suptitle('Wine Quality - Sulphates Content', fontsize=14)



sns.violinplot(x="quality", y="sulphates", data=wines,  ax=ax)

ax.set_xlabel("Wine Quality",size = 12,alpha=0.8)

ax.set_ylabel("Wine Sulphates",size = 12,alpha=0.8)
cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity', 'wine_type']

pp = sns.pairplot(wines[cols], hue='wine_type', height=1.8, aspect=1.8, 

                  palette={"red": "#FF9999", "white": "#FFE888"},

                  plot_kws=dict(edgecolor="black", linewidth=0.5))

fig = pp.fig 

fig.subplots_adjust(top=0.93, wspace=0.3)

t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')



xs = wines['residual sugar']

ys = wines['fixed acidity']

zs = wines['alcohol']

ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')



ax.set_xlabel('Residual Sugar')

ax.set_ylabel('Fixed Acidity')

ax.set_zlabel('Alcohol')
# using size for the 3rd dimension

sc = sns.scatterplot(wines['fixed acidity'], wines['alcohol'], 

                size=wines['residual sugar'])



# using color for the 3rd dimension

sc = sns.scatterplot(wines['fixed acidity'], wines['alcohol'], 

                hue=wines['residual sugar'], alpha=0.9)
# example depicting representing 3-D continous data

# using color and facets

quantile_list = [0, .25, .5, .75, 1.]

quantile_labels = ['0', '25', '50', '75']

wines['res_sugar_labels'] = pd.qcut(wines['residual sugar'], 

                                    q=quantile_list, labels=quantile_labels)

wines['alcohol_levels'] = pd.qcut(wines['alcohol'], 

                                    q=quantile_list, labels=quantile_labels)

wines.head()
g = sns.FacetGrid(wines, col="res_sugar_labels", 

                  hue='alcohol_levels')



g.map(plt.scatter, "fixed acidity", "alcohol", alpha=.7)

g.add_legend();
# Visualizing 3-D categorical data using bar plots

# leveraging the concepts of hue and facets



fc = sns.catplot(x="quality", hue="wine_type", col="quality_label", 

                    data=wines, kind="count",

                    palette={"red": "#FF9999", "white": "#FFE888"})
jp = sns.pairplot(wines, x_vars=["sulphates"], y_vars=["alcohol"], size=4.5,

                  hue="wine_type", palette={"red": "#FF9999", "white": "#FFE888"},

                  plot_kws=dict(edgecolor="k", linewidth=0.5))
lp = sns.lmplot(x='sulphates', y='alcohol', hue='wine_type', 

                palette={"red": "#FF9999", "white": "#FFE888"},

                data=wines, fit_reg=True, legend=True,

                scatter_kws=dict(edgecolor="k", linewidth=0.5))
ax = sns.kdeplot(white_wine['sulphates'], white_wine['alcohol'],

                  cmap="YlOrBr", shade=True, shade_lowest=False)

ax = sns.kdeplot(red_wine['sulphates'], red_wine['alcohol'],

                  cmap="Reds", shade=True, shade_lowest=False)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

f.suptitle('Wine Type - Quality - Acidity', fontsize=14)



sns.violinplot(x="quality", y="volatile acidity",

               data=wines, inner="quart", linewidth=1.3,ax=ax1)

ax1.set_xlabel("Wine Quality",size = 12,alpha=0.8)

ax1.set_ylabel("Wine Volatile Acidity",size = 12,alpha=0.8)



sns.violinplot(x="quality", y="volatile acidity", hue="wine_type", 

               data=wines, split=True, inner="quart", linewidth=1.3,

               palette={"red": "#FF9999", "white": "white"}, ax=ax2)

ax2.set_xlabel("Wine Quality",size = 12,alpha=0.8)

ax2.set_ylabel("Wine Volatile Acidity",size = 12,alpha=0.8)

l = plt.legend(loc='upper right', title='Wine Type')
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

f.suptitle('Wine Type - Quality - Alcohol Content', fontsize=14)



sns.boxplot(x="quality", y="alcohol", hue="wine_type",

               data=wines, palette={"red": "#FF9999", "white": "white"}, ax=ax1)

ax1.set_xlabel("Wine Quality",size = 12,alpha=0.8)

ax1.set_ylabel("Wine Alcohol %",size = 12,alpha=0.8)



sns.boxplot(x="quality_label", y="alcohol", hue="wine_type",

               data=wines, palette={"red": "#FF9999", "white": "white"}, ax=ax2)

ax2.set_xlabel("Wine Quality Class",size = 12,alpha=0.8)

ax2.set_ylabel("Wine Alcohol %",size = 12,alpha=0.8)

l = plt.legend(loc='best', title='Wine Type')
fig = plt.figure(figsize = (10,8))

t = fig.suptitle('Wine Residual Sugar - Alcohol Content - Acidity - Type', fontsize=14)

ax = fig.add_subplot(111, projection='3d')



xs = list(wines['residual sugar'])

ys = list(wines['alcohol'])

zs = list(wines['fixed acidity'])



data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]

colors = ['red' if wt == 'red' else 'yellow' for wt in list(wines['wine_type'])]

data_points[0]



for data, color in zip(data_points, colors):

    x, y, z = data

    ax.scatter(x, y, z,

               c=color,

               alpha=0.4,

               s=30,

               edgecolors='black'          # to remove edgecolors comment this parameter

              )



ax.set_xlabel('Residual Sugar')

ax.set_ylabel('Alcohol')

ax.set_zlabel('Fixed Acidity')
size = wines['residual sugar']*25

fill_colors = ['#FF9999' if wt=='red' else '#FFE888' for wt in list(wines['wine_type'])]

edge_colors = ['red' if wt=='red' else 'orange' for wt in list(wines['wine_type'])]



plt.scatter(wines['fixed acidity'], wines['alcohol'], s=size, 

            alpha=0.4, color=fill_colors, edgecolors=edge_colors)



plt.xlabel('Fixed Acidity')

plt.ylabel('Alcohol')

plt.title('Wine Alcohol Content - Fixed Acidity - Residual Sugar - Type',y=1.05)
g = sns.FacetGrid(wines, col="wine_type", hue='quality_label', 

                  col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],

                  aspect=1.2, size=3.5, palette=sns.light_palette('navy', 4)[1:])

g.map(plt.scatter, "volatile acidity", "alcohol", alpha=0.9, 

      edgecolor='white', linewidth=0.5, s=100)

fig = g.fig 

fig.subplots_adjust(top=0.8, wspace=0.3)

fig.suptitle('Wine Type - Alcohol - Quality - Acidity', fontsize=14)

l = g.add_legend(title='Wine Quality Class')
g = sns.FacetGrid(wines, col="wine_type", hue='quality_label', 

                  col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],

                  aspect=1.2, size=3.5, palette=sns.light_palette('green', 4)[1:])

g.map(plt.scatter, "volatile acidity", "total sulfur dioxide", alpha=0.9, 

      edgecolor='white', linewidth=0.5, s=100)

fig = g.fig 

fig.subplots_adjust(top=0.8, wspace=0.3)

fig.suptitle('Wine Type - Sulfur Dioxide - Acidity - Quality', fontsize=14)

l = g.add_legend(title='Wine Quality Class')
fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection='3d')

t = fig.suptitle('Wine Residual Sugar - Alcohol Content - Acidity - Total Sulfur Dioxide - Type', fontsize=14)



xs = list(wines['residual sugar'])

ys = list(wines['alcohol'])

zs = list(wines['fixed acidity'])

data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]



ss = list(wines['total sulfur dioxide'])

colors = ['red' if wt == 'red' else 'yellow' for wt in list(wines['wine_type'])]



for data, color, size in zip(data_points, colors, ss):

    x, y, z = data

    ax.scatter(x, y, z, alpha=0.4, c=color, s=size, 

               edgecolors='black'

              )



ax.set_xlabel('Residual Sugar')

ax.set_ylabel('Alcohol')

ax.set_zlabel('Fixed Acidity')
g = sns.FacetGrid(wines, col="wine_type", hue='quality_label', 

                  col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],

                  aspect=1.2, size=3.5, palette=sns.light_palette('black', 4)[1:])



g.map(plt.scatter, "residual sugar", "alcohol", alpha=0.8, 

      edgecolor='white', linewidth=0.5, 

      #s=np.ravel(wines['total sulfur dioxide']*2)

     )



fig = g.fig 

fig.subplots_adjust(top=0.8, wspace=0.3)

fig.suptitle('Wine Type - Sulfur Dioxide - Residual Sugar - Alcohol - Quality', fontsize=14)

l = g.add_legend(title='Wine Quality Class')
fig = plt.figure(figsize=(12, 8))

t = fig.suptitle('Wine Residual Sugar - Alcohol Content - Acidity - Total Sulfur Dioxide - Type - Quality', fontsize=14)

ax = fig.add_subplot(111, projection='3d')



xs = list(wines['residual sugar'])

ys = list(wines['alcohol'])

zs = list(wines['fixed acidity'])

data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]



ss = list(wines['total sulfur dioxide'])

colors = ['red' if wt == 'red' else 'yellow' for wt in list(wines['wine_type'])]

markers = [',' if q == 'high' else 'x' if q == 'medium' else 'o' for q in list(wines['quality_label'])]



for data, color, size, mark in zip(data_points, colors, ss, markers):

    x, y, z = data

    ax.scatter(x, y, z, alpha=0.4, c=color, 

               edgecolors='black',

               s=size, marker=mark)



ax.set_xlabel('Residual Sugar')

ax.set_ylabel('Alcohol')

ax.set_zlabel('Fixed Acidity')
g = sns.FacetGrid(wines, row='wine_type', col="quality", hue='quality_label', size=4)

g.map(plt.scatter,  "residual sugar", "alcohol", alpha=0.5, 

      edgecolor='k', linewidth=0.5, 

      #s=wines['total sulfur dioxide']*2

     )

fig = g.fig 

fig.set_size_inches(18, 8)

fig.subplots_adjust(top=0.85, wspace=0.3)

fig.suptitle('Wine Type - Sulfur Dioxide - Residual Sugar - Alcohol - Quality Class - Quality Rating', fontsize=14)

l = g.add_legend(title='Wine Quality Class')