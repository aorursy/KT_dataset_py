import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.express as px



%config InlineBackend.figure_format='svg'

%matplotlib inline 
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.tree import plot_tree
df = pd.read_csv('../input/tv-power-consumption-data/Q3-data.csv')

df.head()
df[['TV', 'Agg']].describe()
df['House'].value_counts()
df.isnull().sum()
px.scatter(df, 

           x='Time', 

           y='TV',                   

           color=df['House'].astype(str),

           facet_row='House', 

           render_mode='svg', 

           labels={'TV': 'TV consumption', 'color': 'House'}, 

           title='TV consumption across time')
is_on = ( ((df['TV'] > 15) & (df['House'] == 1)) |

    ((df['TV'] > 50) & (df['House'] == 2)) |

    ((df['TV'] > 40) & (df['House'] == 3)) ).apply(int)

df = df.assign(ON=is_on)

df.head()
px.scatter(df, 

           x='Time', 

           y='TV', 

           color=df['ON'].astype(str), 

           color_discrete_sequence=['Black', 'Green'], 

           facet_row='House', 

           render_mode='svg', 

           labels={'TV': 'TV consumption'}, 

           title='TV consumption across time')
px.line(df, 

        x='Time', 

        y='Agg', 

        color='House', 

        facet_row='House', 

        render_mode='svg', 

        labels={'Agg': 'Consumption'}, 

        title='Aggregate consumption across time')
df_no_outliers = df[(df['Agg'] < 900)]

px.line(df_no_outliers, 

        x='Time', 

        y=['Agg','TV'], 

        facet_row='House', 

        render_mode='svg', 

        labels={'value': 'Consumption'}, 

        title='Aggregate and TV consumption across time')
X = df.drop(['ON', 'Time'], axis = 1)

y = df['ON']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 1)
forest = DecisionTreeClassifier(random_state=1, criterion='entropy')

forest.fit(X_train, y_train)
print(forest.score(X_test, y_test))



y_pred = forest.predict(X_test)

confusion_matrix(y_pred, y_test)
result = forest.predict(X)

px.scatter(df, 

           x='Time', 

           y='TV', 

           color=result.astype(str), 

           color_discrete_sequence=['Black', 'Green'],

           facet_row='House', 

           render_mode='svg', 

           labels={'TV': 'TV consumption', 'color': 'TV State'}, 

           title='TV consumption across time')
plt.figure(figsize=(20,7))

plot_tree(forest, 

          feature_names=X.columns, 

          class_names=['ON', 'OFF'], 

          proportion=True, 

          filled = True)

plt.show()