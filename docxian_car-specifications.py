import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



from statsmodels.graphics.mosaicplot import mosaic
df = pd.read_csv('../input/car-specifications/cars specifications.csv')

df.head()
df.shape
df.describe(include='all')
# cleanse "?" in horsepower column

df.horsepower[374] = 0

df.horsepower = pd.to_numeric(df.horsepower)
df.describe(include='all')
# numeric colums

col_numeric = list(df.columns) 

col_numeric.remove('car name')

col_numeric.remove('origin')

col_numeric.remove('cylinders')

col_numeric
for c in col_numeric:

    print(c)

    print(df[c].describe())

    print('\n')
for c in col_numeric:

    plt.figure()

    df[c].plot(kind='hist')

    plt.grid()

    plt.title(c)
df.cylinders.value_counts()
df.cylinders.value_counts().plot(kind='bar')

plt.title('cylinders')

plt.grid()

plt.show()
df.origin.value_counts()
df.origin.value_counts().plot(kind='bar')

plt.title('origin')

plt.grid()

plt.show()
# pairwise scatter plot

sns.pairplot(df[col_numeric])

plt.show()
# correlations

cor_numeric = (df[col_numeric]).corr()

cor_numeric
plt.rcParams['figure.figsize']=(6,5)

sns.heatmap(cor_numeric, cmap=plt.cm.plasma)

plt.show()
# group by origin    

df_origin = df.groupby('origin', as_index=False).agg(

    mean_mpg = pd.NamedAgg(column='mpg', aggfunc=np.mean),

    mean_disp = pd.NamedAgg(column='displacement', aggfunc=np.mean),

    mean_hp = pd.NamedAgg(column='horsepower', aggfunc=np.mean),

    mean_weight = pd.NamedAgg(column='weight', aggfunc=np.mean),

    mean_acc = pd.NamedAgg(column='acceleration', aggfunc=np.mean),

    mean_year = pd.NamedAgg(column='model year', aggfunc=np.mean))



df_origin
# violinplots

for c in col_numeric:

    plt.figure(figsize=(6,4))

    sns.violinplot(x='origin', y=c, data=df)

    plt.title(c)

    plt.grid()

    plt.show()
def first_piece(i_string):

    return i_string.split()[0]



df['manufacturer'] = list(map(first_piece, df['car name']))
# clean up manufacturer levels

df.manufacturer.loc[df.manufacturer=='maxda'] = 'mazda'

df.manufacturer.loc[df.manufacturer=='vw'] = 'volkswagen'

df.manufacturer.loc[df.manufacturer=='vokswagen'] = 'toyota'

df.manufacturer.loc[df.manufacturer=='chevy'] = 'chevrolet'

df.manufacturer.loc[df.manufacturer=='chevroelt'] = 'chevrolet'

df.manufacturer.loc[df.manufacturer=='toyouta'] = 'toyota'

df.manufacturer.loc[df.manufacturer=='mercedes'] = 'mercedes-benz'
df.manufacturer.value_counts()
plt.figure(figsize=(12,5))

df.manufacturer.value_counts().plot(kind='bar')

plt.grid()

plt.show()
# select relevant features

features = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year']

df4pca = df.loc[:,features]
# standardize first

df4pca_std = StandardScaler().fit_transform(df4pca)
# define PCA

pc_model = PCA(n_components=2)

# apply PCA

pc = pc_model.fit_transform(df4pca_std)

# convert to data frame

df_pc = pd.DataFrame(data = pc, columns = ['pc_1', 'pc_2'])

# add origin column

df_pc['origin'] = df.origin

# and look at result

df_pc.head()
# add PCA data to original data frame, so we have all data in one place

df['pc_1'] = df_pc.pc_1

df['pc_2'] = df_pc.pc_2

df.head()
# plot PCA results by origin

sns.lmplot( x='pc_1', y='pc_2', data=df_pc, fit_reg=True, hue='origin', legend=False)

plt.legend(['1 - US','2 - Europe','3 - Japan'])

plt.title('Principal Component Analysis')

plt.grid()

plt.show()
# Let's make an interactive version of the plot where we can also see the car name

fig = px.scatter(df, x='pc_1', y='pc_2', color='origin', size='horsepower',

                 hover_data=['car name'])

fig.show()
# The bubble size seems to increase in x-direction,

# so the first component looks strongly correlated to feature 'horsepower'.

# Let's check:

plt.figure(figsize=(6,6))

plt.scatter(df.pc_1, df.horsepower)

plt.title('Horsepower vs. PC 1')

plt.xlabel('Principal Component 1')

plt.ylabel('Horsepower')

plt.grid()

plt.show()
# let's look at the same plot for mpg instead of horsepower

plt.figure(figsize=(6,6))

plt.scatter(df.pc_1, df.mpg)

plt.title('MPG vs. PC 1')

plt.xlabel('Principal Component 1')

plt.ylabel('MPG')

plt.grid()

plt.show()
# define 3D PCA

pc_model = PCA(n_components=3)

# apply PCA

pc = pc_model.fit_transform(df4pca_std)

# convert to data frame

df_pc = pd.DataFrame(data = pc, columns = ['pc_1', 'pc_2','pc_3'])

# add origin column

df_pc['origin'] = df.origin

# and look at result

df_pc.head()
# add PCA data to original data frame, so we have all data in one place

df['pc3d_1'] = df_pc.pc_1

df['pc3d_2'] = df_pc.pc_2

df['pc3d_3'] = df_pc.pc_3

df.head()
fig = px.scatter_3d(df, x='pc3d_1', y='pc3d_2', z='pc3d_3',

                    color='origin',

                    hover_data=['car name'],

                    opacity=0.5)

fig.update_layout(title='PCA 3D')

fig.show()
def target_plot(my_feature):

    my_feature_bin = my_feature + '_bin'

    df[my_feature_bin] = pd.qcut(df[my_feature], q=5) # bin by quantiles

    plt.rcParams["figure.figsize"]=(6,4)

    df[my_feature_bin].value_counts().plot(kind='bar')

    plt.grid()

    plt.show()



    # plot (multiclass) target vs feature

    plt.rcParams["figure.figsize"]=(12,6)

    mosaic(df, [my_feature_bin, 'origin'], title='Origin vs binned feature' + ' ' + my_feature)

    plt.show()
target_plot('mpg')
target_plot('horsepower')
target_plot('displacement')
target_plot('weight')
target_plot('acceleration')
target_plot('model year')
df.columns
# plot also for cylinders_

mosaic(df, ['cylinders', 'origin'], title='Origin vs cylinders')

plt.show()
df[df.cylinders==3]
df[df.cylinders==5]
df[df.cylinders==8]