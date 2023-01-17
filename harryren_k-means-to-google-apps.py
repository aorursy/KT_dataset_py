# Thanks @Lavanya Gupta whose kernel inspired me!

# Please check https://www.kaggle.com/lava18/all-that-you-need-to-know-about-the-android-market



# import required packages

from matplotlib import pyplot as plt

import plotly

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import pandas as pd

import numpy as np

import seaborn as sns

import tensorflow as tf

from scipy import stats

plt.style.use("ggplot")

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/googleplaystore.csv') # read the data
df.shape
# data cleanning



df["Size"] = df["Size"].apply(lambda x: str(x).replace('Varies with device', 'NaN') 

                              if "Varies with device" in str(x) else x)

df["Size"] = df["Size"].apply(lambda x: str(x).replace('M','') 

                              if 'M' in str(x) else x)

df["Size"] = df["Size"].apply(lambda x: str(x).replace(',','') 

                              if ',' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: str(x).replace('+', '') 

                              if '+' in str(x) else x)

df['Size'] = df["Size"].apply(lambda x: float(str(x).replace('k',''))/1024 

                              if 'k' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in x else x)

df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in x else x)

df['Installs'] = df['Installs'].apply(lambda x: x.replace('Free', 'NaN') if 'Free' in x else x)

df['Installs'] = df['Installs'].apply(lambda x: x.replace('Paid', 'NaN') if 'Paid' in x else x)
df["Price"] = df["Price"].apply(lambda x: str(x).replace('$','') if '$' in x else x)
df = df.dropna()
df['Size'] = df['Size'].apply(lambda x:float(x))

df['Installs'] = df['Installs'].apply(lambda x:int(x))

df['Reviews'] = df['Reviews'].apply(lambda x:int(x))

df['Price'] = df['Price'].apply(lambda x:float(x))

df = df.dropna()
data = df[['Rating', 'Size', 'Installs', 'Reviews', 'Price']]
def mean_norm(x):

    x = np.array(x)

    x = (x - np.mean(x))/np.std(x)

    return x
data['Rating'] = mean_norm(data['Rating'])

data['Size'] = mean_norm(data['Size'])

data['Installs'] = mean_norm(data['Installs'])

data['Reviews'] = mean_norm(data['Reviews'])

data['Price'] = mean_norm(data['Price'])

result = data
data = np.array(data)

print(data[0:5])
def input_fn():

    return tf.data.Dataset.from_tensors(tf.convert_to_tensor(data, dtype=tf.float32)).repeat(1)
x = input_fn()

print(x)
num_clusters = 3

kmeans = tf.contrib.factorization.KMeansClustering(

    num_clusters=num_clusters, use_mini_batch=False)
num_iterations = 20

previous_centers = None

for _ in range(num_iterations):

    kmeans.train(input_fn)

    cluster_centers = kmeans.cluster_centers()

    if previous_centers is not None:

        dis = previous_centers - cluster_centers

        print('dleta' + str(dis))

    previous_centers = cluster_centers

    print("socre:" + str(kmeans.score(input_fn)))

print("the centers are: " + str(previous_centers))

center = previous_centers
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
result['Group'] = cluster_indices
result['Group'] = result['Group'].apply(lambda x: str(x))

result['Group'] = result['Group'].apply(lambda x: x.replace('0', 'Group1') if '0' == x else x)

result['Group'] = result['Group'].apply(lambda x: x.replace('1', 'Group2') if '1' == x else x)

result['Group'] = result['Group'].apply(lambda x: x.replace('2', 'Group3') if '2' == x else x)
ra = df['Rating']

s = df['Size']

i = df['Installs']

re = df['Reviews']

c = result['Group']

p = df['Price']



sns.pairplot(pd.DataFrame(list(zip(ra, s, np.log(i), np.log(re), c, p)), 

                        columns=['Rating','Size', 'Installs', 'Reviews', 'Group', 'Price']),hue='Group', palette="Set2")
num_of_app_in_group = result['Group'].value_counts().sort_values(ascending=True)

data1 = [go.Pie(

        labels = num_of_app_in_group.index,

        values = num_of_app_in_group.values

)]

plotly.offline.iplot(data1, filename='apps_per_group')