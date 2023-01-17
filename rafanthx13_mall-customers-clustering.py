import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configs
pd.options.display.float_format = '{:,.3f}'.format
sns.set(style="whitegrid")
plt.style.use('seaborn')
seed = 42
np.random.seed(seed)
file_path = '/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'
df = pd.read_csv(file_path)

print("DataSet = {} rows and {} columns".format(df.shape[0], df.shape[1]))

quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

print("\nQualitative Variables: (Numerics)", "\n=>", quantitative,
      "\n\nQuantitative Variable: (Strings)\n=>", qualitative )

df.head()
# No Missing Data
df.isnull().sum().max()
# Restructure DataFrame
dict_rename = {'Gender': 'gender', 'Age': 'age', 'Annual Income (k$)': 'income', 'Spending Score (1-100)': 'spending_score'}
df = df.rename(dict_rename, axis=1).drop(['CustomerID'], axis =1)
df.describe()
def eda_categ_feat_desc_plot(series_categorical, title = ""):
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)
    if(title != ""):
        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.8)

    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
    for index, row in val_concat.iterrows():
        s.text(row.name, row['quantity'], row['quantity'], color='black', ha="center")

    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
                             title="Percentage Plot")

    ax[1].set_ylabel('')
    ax[0].set_title('Quantity Plot')

    plt.show()
def eda_numerical_feat(series, title="", with_label=True, number_format=""):
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 5), sharex=False)
    if(title != ""):
        f.suptitle(title, fontsize=18)
    sns.distplot(series, ax=ax1)
    sns.boxplot(series, ax=ax2)
    if(with_label):
        describe = series.describe()
        labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 
              'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],
              'Q3': describe.loc['75%']}
        if(number_format != ""):
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',
                         size=10, color='white', bbox=dict(facecolor='#445A64'))
        else:
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',
                     size=10, color='white', bbox=dict(facecolor='#445A64'))
    plt.show()
df['age_cat'] = np.nan
lst = [df]

for col in lst:
    col.loc[(col['age'] >= 0) & (col['age'] <= 30), 'age_cat'] = 'Young Adult'
    col.loc[(col['age'] >  30) & (col['age'] <= 50), 'age_cat'] = 'Adult'
    col.loc[(col['age'] >  50) & (col['age'] <= 60), 'age_cat'] = 'Senior'
    col.loc[ col['age'] >  60, 'age_cat'] = 'Elder'
    
df.head()
eda_categ_feat_desc_plot(df['gender'])
eda_numerical_feat(df['age'])
eda_numerical_feat(df['income'])
eda_numerical_feat(df['spending_score'])
fig, ((ax1, ax2), (ax3,ax4), (ax5, ax6)) = plt.subplots(
    figsize = (16,14), ncols=2, nrows=3, sharex=False, sharey=False)

# age
sns.violinplot(x="gender", y="age", data=df, ax=ax1)
sns.swarmplot(x="gender", y="age", data=df, ax=ax1, palette='rocket')
sns.distplot(df[ df['gender'] == 'Male']['age'], ax=ax2, label="Male")
sns.distplot(df[ df['gender'] == 'Female']['age'], ax=ax2, label="Female")

# income
# sns.boxplot(x="gender", y="income", data=df, ax=ax3)
sns.violinplot(x="gender", y="income", data=df, ax=ax3)
sns.swarmplot(x="gender", y="income", data=df, ax=ax3, palette='rocket')
sns.distplot(df[ df['gender'] == 'Male']['income'], ax=ax4, label="Male")
sns.distplot(df[ df['gender'] == 'Female']['income'], ax=ax4, label="Female")

# spending_score
sns.violinplot(x="gender", y="spending_score", data=df, ax=ax5)
sns.swarmplot(x="gender", y="spending_score", data=df, ax=ax5, palette='rocket')
sns.distplot(df[ df['gender'] == 'Male']['spending_score'], ax=ax6, label="Male")
sns.distplot(df[ df['gender'] == 'Female']['spending_score'], ax=ax6, label="Female")

# Config Titles
fig.suptitle('Features by gender', fontsize=20)
font_size = 16
ax1.set_title('age by gender')
ax2.set_title('age by gender')
ax3.set_title('income by gender')
ax4.set_title('income by gender')
ax5.set_title('spending_score by gender')
ax6.set_title('spending_score by gender')

plt.legend();
plt.show()
fig, (ax1, ax2) = plt.subplots(figsize = (16,4), ncols=2, sharex=False, sharey=False)

sns.scatterplot(x="age", y="income", data=df, ax=ax1)
sns.scatterplot(x="age", y="spending_score", data=df, ax=ax2)
ax1.set_title("Income by age")
ax2.set_title('spending_score by age')
plt.show()
fig, ax1 = plt.subplots(figsize = (8,4))

sns.scatterplot(x="spending_score", y="income", data=df, ax=ax1)
ax1.set_title('score by income')
plt.show()
fig, (ax1, ax2) = plt.subplots(figsize = (16,4), ncols=2, sharex=False, sharey=False)

sns.scatterplot(x="spending_score", y="income", hue='gender', data=df, ax=ax1, alpha = 0.8)
ax1.set_title('score by income and gender')

sns.scatterplot(x="spending_score", y="income", hue='age_cat', data=df, ax=ax2, alpha = 0.8)
ax2.set_title('score by income and gender')

plt.show()
# Convert Gender to LabelEnconder: 0/1
label_encoder = LabelEncoder()
df_pre_processing = df.drop('age_cat',axis=1)
df_pre_processing['gender'] = label_encoder.fit_transform(df['gender'])

X0 = df_pre_processing.values
X0[0]
map_labels = {0: 'g1', 1: 'g2', 2: 'g3', 3: 'g4', 4: 'g5', 5: 'g6', 6: 'g7',
               7: 'g8', 8: 'g9', 9: 'g10', 10: 'g11', 11: 'g12', 12: 'g13'}
df.head()
fig, (ax1, ax2, ax3) = plt.subplots(figsize = (8,13), nrows=3, sharex=False, sharey=False)

X1 = df[['age','spending_score']].values
X2 = df[['income','spending_score']].values
X3 = df[['income','age']].values

model = KMeans()

viz1 = KElbowVisualizer(model, k=(3,12), ax=ax1)
viz1.fit(X1)
viz1.finalize()
ax1.set_title('Best K to Kmeans: age x spending_score')

viz2 = KElbowVisualizer(model, k=(3,12), ax=ax2)
viz2.fit(X2)
viz2.finalize()
ax2.set_title('Best K to Kmeans: income x spennding_score')

viz3 = KElbowVisualizer(model, k=(3,12), ax=ax3)
viz3.fit(X3)
viz3.finalize()
ax3.set_title('Best K to Kmeans: income x age')

plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(figsize = (18,4), ncols=3)

# K-MEANS-01

X, f1, f2, clusters = X1, 'age', 'spending_score', 4

kmeans = KMeans(n_clusters = clusters ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
kmeans.fit(X)
labels, centroids = [map_labels[label_num] for label_num in kmeans.labels_], kmeans.cluster_centers_

sns.scatterplot(x =f1 ,y = f2 , data=df, hue=labels, ax=ax1)
sns.scatterplot(x = centroids[: , 0] , y =  centroids[: , 1] ,
                palette = 'Set2' , alpha = 0.5, s=400, ax=ax1)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # legend outside
ax1.set_title('Kmeans: {} groups: {} x {}'.format(str(clusters), f1, f2), fontsize=16)

# K-MEANS-02

X, f1, f2, clusters = X2, 'income', 'spending_score', 5
kmeans = KMeans(n_clusters = clusters ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
kmeans.fit(X)
labels, centroids = [map_labels[label_num] for label_num in kmeans.labels_], kmeans.cluster_centers_

sns.scatterplot(x =f1 ,y = f2 , data=df, hue=labels, ax=ax2)
sns.scatterplot(x = centroids[: , 0] , y =  centroids[: , 1] ,
                palette = 'Set3' , alpha = 0.5, s=400, ax=ax2)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # legend outside
ax2.set_title('Kmeans: {} groups: {} x {}'.format(str(clusters), f1, f2), fontsize=16)

# K-MEANS-03

X, f1, f2, clusters = X3, 'income', 'age', 6
kmeans = KMeans(n_clusters = clusters ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
kmeans.fit(X)
labels, centroids = [map_labels[label_num] for label_num in kmeans.labels_], kmeans.cluster_centers_

sns.scatterplot(x =f1 ,y = f2 , data=df, hue=labels, ax=ax3)
sns.scatterplot(x = centroids[: , 0] , y =  centroids[: , 1] ,
                palette = 'Set3' , alpha = 0.5, s=400, ax=ax3)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # legend outside
ax3.set_title('Kmeans: {} groups: {} x {}'.format(str(clusters), f1, f2), fontsize=16)

plt.show()
!pip install pycaret
from pycaret.clustering import *
df.head()
pycaret_cluster_setup = setup(df, normalize = True,
                              categorical_features = ['gender'],
                              ignore_features = ['age_cat'],
                              session_id = 42)
kmeans = create_model('kmeans')
kmean_results = assign_model(kmeans)
kmean_results.head()
plot_model(kmeans)
plot_model(kmeans, plot = 'elbow')
plot_model(kmeans, plot = 'silhouette')
plot_model(kmeans, plot = 'distribution') #to see size of clusters
fig, (ax1, ax2, ax3) = plt.subplots(figsize = (18,4), ncols=3)
fig.suptitle('Evaluate pycaret cluster by feature pairs')

sns.scatterplot(data=kmean_results, x="age", y="spending_score", hue="Cluster", ax=ax1)
ax1.set_title('Compare pycaret clusters by age and score')

sns.scatterplot(data=kmean_results, x="income", y="spending_score", hue="Cluster", ax=ax2)
ax2.set_title('Compare pycaret clusters by income and score')

sns.scatterplot(data=kmean_results, x="income", y="age", hue="Cluster", ax=ax3)
ax3.set_title('Compare pycaret clusters by income and age')

plt.show()