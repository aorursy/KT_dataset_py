import pandas as pd

import numpy as np
df = pd.read_csv("../input/magic.csv")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df.head()
df.columns
df.rentalrates_value_3.unique()
df = df.drop(['buyrates_value_1', 'buyratesqqa_image','buyratesqqa_image/_alt', 

         'buyrates_value_3', 'rentalrates_value_1', 'rentalrates_value_3', 

         'rentalrates_value_5'], axis=1)



df.head()
df.info()
df['buy_rate_min'] = df.buyrates_value_2.str.split(' - ').str[0]

df['buy_rate_max'] = df.buyrates_value_2.str.split(' - ').str[1]

df['buy_rate_max'] = df.buy_rate_max.str.split('/').str[0]
df.head()
df.rentalrates_value_2.unique()
df['rent_1_min'] = df.rentalrates_value_2.str.split(' - ').str[0]

df['rent_1_max'] = df.rentalrates_value_2.str.split(' - ').str[1]

df.head()
df['rent_2_min'] = df.rentalrates_value_4.str.split(' - ').str[0]

df['rent_2_max'] = df.rentalrates_value_4.str.split(' - ').str[1]

df.head()
df['rent_3_min'] = df.rentalrates_value_6.str.split(' - ').str[0]

df['rent_3_max'] = df.rentalrates_value_6.str.split(' - ').str[1]

df.head()
df = df.drop(['buyrates_value_2','rentalrates_value_2', 'rentalrates_value_4', 'rentalrates_value_6'], axis = 1)

df.head()
df.describe()
test = df.buy_rate_max.str.replace(',', '')

test.astype('float64')[:5]
df['buy_rate_max'] = df.buy_rate_max.str.replace(',', '').astype('float64')

df.head()
df.buy_rate_min.unique()
def remove_comma(x):

    try:

        if ',' in x:

            return x.replace(',', '')

    except:

        return np.NAN
df['buy_rate_min'] = df['buy_rate_min'].apply(remove_comma).astype(float)
df.buy_rate_max.hist()
df['rent_1_min'] = df['rent_1_min'].apply(remove_comma).astype(float)

df['rent_1_max']= df['rent_1_max'].apply(remove_comma).astype(float)

df['rent_2_min'] = df['rent_2_min'].apply(remove_comma).astype(float)

df['rent_2_max'] = df['rent_2_max'].apply(remove_comma).astype(float)

df['rent_3_min'] = df['rent_3_min'].apply(remove_comma).astype(float)

df['rent_3_max'] = df['rent_3_max'].apply(remove_comma).astype(float)
%matplotlib inline
import seaborn as sns

import matplotlib.pyplot as plt
df.columns
def remove_percent(x):

    try:

        if '%' in x:

            return x.replace('%', '')

    except:

        return np.NAN
df['buyratesqqa_value'] = df['buyratesqqa_value'].apply(remove_percent).astype(float)
df.head()
df.buyratesqqa_value.min()
df.buyratesqqa_value.max()
df.describe()
corr_mat = df[['buy_rate_max', 'rent_1_max', 'rent_2_max', 'rent_3_max']].corr()
corr_mat
sns.heatmap(corr_mat, vmax=1.0, square=True)
df['buy_rate_avg'] = (df['buy_rate_max'] - df['buy_rate_min'])/2

df.head()
corr_mat = df[['buy_rate_avg', 'rent_1_max', 'rent_2_max', 'rent_3_max']].corr()

sns.heatmap(corr_mat, vmax=1.0, square=True)
df['rent_1_avg'] = (df['rent_1_max'] - df['rent_1_min'])/2

df['rent_2_avg'] = (df['rent_2_max'] - df['rent_2_min'])/2

df['rent_3_avg'] = (df['rent_3_max'] - df['rent_3_min'])/2

df.head()
corr_mat = df[['buy_rate_avg', 'rent_1_avg', 'rent_2_avg', 'rent_3_avg']].corr()

sns.heatmap(corr_mat, vmax=1.0, square=True)
corr_mat
df[['buy_rate_avg', 'rent_1_avg', 'rent_2_avg', 'rent_3_avg']].hist()
df.mean()
from sklearn import preprocessing
df.columns
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(df[['buyratesqqa_value', 

        'buy_rate_min',

       u'buy_rate_max', u'rent_1_min', u'rent_1_max', u'rent_2_min',

       u'rent_2_max', u'rent_3_min', u'rent_3_max']])
df_imp = df.fillna(0)
df_imp.head()
df_scaled = df_imp - df_imp.mean()/df_imp.std()
df_scaled.head()
df_scaled.hist(bins=10)
corr_mat = df_scaled.corr()

sns.heatmap(corr_mat, vmax=1.0, square=True)
df_imp.head()
df_top = df_imp.sort_values('buy_rate_avg',ascending=False).head(20)
df_top
from sklearn.cluster import KMeans
X = df_imp.as_matrix()
locality = X[:,0]
locality
X1 = np.delete(X, 0, 1)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X1)
kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1],

            marker='x', s=169, linewidths=3,

            color='w', zorder=10)
from sklearn.decomposition import PCA
reduced_data = PCA(n_components=2).fit_transform(X1)

kmeans = KMeans(init='k-means++', n_clusters=10, n_init=5)

kmeans.fit(reduced_data)
# Step size of the mesh. Decrease to increase the quality of the VQ.

h = 100     # point in the mesh [x_min, x_max]x[y_min, y_max].



# Plot the decision boundary. For that, we will assign a color to each

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1

y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
y_min
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



# Obtain labels for each point in mesh. Use last trained model.

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(1)

plt.clf()

plt.imshow(Z, interpolation='nearest',

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap=plt.cm.Paired,

           aspect='auto', origin='lower')
# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(1)

plt.clf()

plt.imshow(Z, interpolation='nearest',

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap=plt.cm.Paired,

           aspect='auto', origin='lower')



plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)

# Plot the centroids as a white X

centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1],

            marker='x', s=169, linewidths=3,

            color='w', zorder=10)

plt.title('K-means clustering on the property dataset (PCA-reduced data)\n'

          'Centroids are marked with white cross')

plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xticks(())

plt.yticks(())

plt.show()
y_data = np.arange(0, X.shape[0])
vis_data = reduced_data



#vis_data = bh_sne(X1.astype('float64'))



# plot the result

vis_x = vis_data[:, 0]

vis_y = vis_data[:, 1]
from bokeh.io import output_notebook, show

output_notebook()
from bokeh.charts import Scatter
df_plot = pd.DataFrame(vis_data, index=locality, columns=['x', 'y'])

df_plot.head()
df_plot.index.name = 'locality'

df_plot = df_plot.reset_index()

df_plot.head()
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
source = ColumnDataSource(data=df_plot)

labels = LabelSet(x='x', y='y', 

                  text='locality', level='glyph', x_offset=5, y_offset=5, source=source, 

                  render_mode='canvas')
plt.scatter(x=vis_x, y=vis_y)
plt.rcParams['figure.figsize'] = (15.0, 15.0)

plt.scatter(vis_x, vis_y)

for i, txt in enumerate(locality):

    plt.annotate(txt, (vis_x[i],vis_y[i]))
import pip



for package in sorted(pip.get_installed_distributions(), key=lambda package: package.project_name):

    print("{} ({})".format(package.project_name, package.version))