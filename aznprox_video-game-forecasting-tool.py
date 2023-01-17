import os

import numpy as np

from matplotlib import pyplot as plt

import matplotlib.style as style

import seaborn as sns

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from random import randint



#import and parse the CSV file

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        df = pd.read_csv(os.path.join(dirname, filename))
df['Year'].fillna(df['Year'].mode()[0], inplace=True)

df['Publisher'].replace(np.nan, df['Publisher'].mode()[0], inplace=True)

df['Platform'].replace('2600', 'Atari', inplace=True)

df = df[df.Year < 2017]
style.use('seaborn-poster')

genre_global_sales = df.groupby(['Genre'])['Global_Sales'].sum().sort_values(ascending=False)

sns.barplot(x=genre_global_sales.index, y=genre_global_sales.values, ec='Black', palette='rainbow')

plt.xticks(rotation=20, fontsize=12)

plt.xlabel('Genre', fontsize=18)

plt.ylabel('Global Sales (in Millions)', fontsize=18)

plt.title('Global Sales of Genres from 1980-2016', fontweight='bold', fontsize=22)

plt.tight_layout()

plt.show()
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import StandardScaler

from random import randint
categorical_labels = ['Platform', 'Genre', 'Publisher']

numerical_lables = ['Global_Sales']

enc = LabelEncoder()

encoded_df = pd.DataFrame(columns=['Platform', 'Genre', 'Publisher', 'Global_Sales'])

encoders = {'Platform': LabelEncoder(),

            'Genre': LabelEncoder(),

            'Publisher': LabelEncoder(),

            'Global_Sales': LabelEncoder()}



for label in categorical_labels:

    temp_column = df[label]

    encoded_temp_col = encoders[label].fit_transform(temp_column)



    encoded_df[label] = encoded_temp_col





for label in numerical_lables:

    encoded_df[label] = df[label].values
x = encoded_df.iloc[:, 0:3]

y = encoded_df.iloc[:,3:]



scalar = StandardScaler()



x = scalar.fit_transform(x)



linear_reg = LinearRegression()



linear_reg.fit(x, y)



y_pred = linear_reg.predict(x)

new_pred = linear_reg.predict([[randint(0, 30),randint(0, 11),randint(0, 577)]])





import ipywidgets as widgets

from IPython.display import display

button = widgets.Button(description="Random Query Predict")

display(button)



def fun(a):

    linear_reg.fit(x, y)

    new_column_dictionary = {'Platform': [randint(0, 30)],

                             'Genre': [randint(0, 11)],

                             'Publisher': [randint(0, 577)]}

    randomDataFrame = pd.DataFrame.from_dict(new_column_dictionary)

    for columnEntry in list(encoded_df):

        if 'Global_Sales' not in columnEntry:

            print( str(columnEntry) + ": " + str(encoders[columnEntry].inverse_transform(randomDataFrame[columnEntry])[0]))

    new_prediction = linear_reg.predict(randomDataFrame)

    print(str("new prediction is $") + str(round(new_prediction[0][0] * 1000000, 2)))



def on_button_clicked(b):

    fun('a')



button.on_click(on_button_clicked)

new_x = encoded_df.iloc[:, [0,1,2,3]].values







Error =[]

for i in range(1, 11):

    temp_kmeans = KMeans(n_clusters = i).fit(new_x)

    temp_kmeans.fit(new_x)

    Error.append(temp_kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1, 11), Error)

plt.title('Elbow Chart')

plt.xlabel('No of clusters')

plt.ylabel('Error')

plt.show()

kmeans3 = KMeans(n_clusters=3)

y_kmeans3 = kmeans3.fit_predict(new_x)



kmeans3.cluster_centers_



plt.scatter(new_x[:,0], new_x[:,3], c=y_kmeans3, cmap='rainbow')

plt.title('Platform vs Global Sales')

plt.xlabel('Platform')

plt.ylabel('Global Sales')

plt.show()
plt.scatter(new_x[:,1], new_x[:,3], c=y_kmeans3, cmap='prism')

plt.title('Genre vs Global Sales')

plt.xlabel('Genre')

plt.ylabel('Global Sales')

plt.show()
plt.scatter(new_x[:,2], new_x[:,3], c=y_kmeans3, cmap='nipy_spectral')

plt.title('Publisher vs Global Sales')

plt.xlabel('Publisher')

plt.ylabel('Global Sales')

plt.show() 