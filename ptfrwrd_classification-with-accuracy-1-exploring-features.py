import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/star-dataset/6 class csv.csv')

data.head(10)
print('shape: ', data.shape)
data.describe()
sns.set(style="darkgrid")

stars_types = pd.DataFrame(data['Star type'].value_counts().sort_values(ascending=False))

plt.figure(figsize=(15,5))

ax = sns.barplot(x = stars_types.index, y = 'Star type' , data = stars_types, palette='pastel')
stars_color = pd.DataFrame(data['Star color'].value_counts().sort_values(ascending=False))

plt.figure(figsize=(15,5))

ax = sns.barplot(x = stars_color.index, y = 'Star color' , data = stars_color, palette='pastel')

ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
stars_spectral = pd.DataFrame(data['Spectral Class'].value_counts().sort_values(ascending=False))

plt.figure(figsize=(15,5))

ax = sns.barplot(x = stars_spectral.index, y = 'Spectral Class' , data = stars_spectral, palette='pastel')

ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
stars_data = {

    'temperature': data['Temperature (K)'],

    'luminosity': data['Luminosity(L/Lo)'],

    'radius': data['Radius(R/Ro)'],

    'absolute_magnitude': data['Absolute magnitude(Mv)'],

    'star_type': data['Star type'],

    'star_color': data['Star color'],

    'spectral_class': data['Spectral Class']

}

stars_data = pd.DataFrame.from_dict(stars_data)

stars_data['star_type'] = stars_data['star_type'].astype('category').cat.codes

stars_data['star_color'] = stars_data['star_color'].astype('category').cat.codes

stars_data['spectral_class'] = stars_data['spectral_class'].astype('category').cat.codes



corr = stars_data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 10))

cmap = sns.diverging_palette(200, 21, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

corr
f, axes = plt.subplots(2, 2, figsize=(10, 10))

sns.despine(left=True)



sns.distplot(stars_data['temperature'], color='b', ax=axes[0, 0])

sns.distplot(stars_data['luminosity'], color='m', ax=axes[0, 1])

sns.distplot(stars_data['radius'], color='r', ax=axes[1, 0])

sns.distplot(stars_data['absolute_magnitude'], color='g', ax=axes[1, 1])

plt.setp(axes, yticks=[])

plt.tight_layout()
ax = sns.catplot(x = 'Star color', y = 'Temperature (K)', kind = "box", data = data, palette='pastel')

ax = ax.fig.set_size_inches(30, 5)
ax = sns.catplot(x = 'Star color', y = 'Luminosity(L/Lo)', kind = "box", data = data, palette='pastel')

ax = ax.fig.set_size_inches(30, 5)
ax = sns.catplot(x = 'Star color', y = 'Radius(R/Ro)', kind = "box", data = data, palette='pastel')

ax = ax.fig.set_size_inches(30, 5)
ax = sns.catplot(x = 'Star color', y = 'Absolute magnitude(Mv)', kind = "box", data = data, palette='pastel')

ax = ax.fig.set_size_inches(30, 5)
import plotly.express as px



fig = px.scatter(data, x="Temperature (K)", y="Luminosity(L/Lo)", size="Radius(R/Ro)", color="Star color",

           hover_name="Star type", log_x=True, size_max=60)

fig.show()

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(stars_data.drop('star_type',axis=1), stars_data['star_type'], test_size=0.40, random_state=42)
from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix



confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(random_state=42, max_iter=10000)

logreg.fit(X_train, y_train)

predictions = logreg.predict(X_test)



print(classification_report(y_test, predictions))
features_data = pd.DataFrame({'color': stars_data['star_color'], 'spectral_class': stars_data['spectral_class'], 'star_type': stars_data['star_type']})

X_train, X_test, y_train, y_test = train_test_split(features_data.drop('star_type',axis=1), features_data['star_type'], test_size=0.20, random_state=42)
logreg.fit(X_train, y_train)

predictions = logreg.predict(X_test)



print(classification_report(y_test, predictions))
import plotly.express as px

import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression





# Split data into training and test splits

train_idx, test_idx = train_test_split(stars_data.index, test_size=.2, random_state=42)

stars_data['split'] = 'train'

stars_data.loc[test_idx, 'split'] = 'test'



X = stars_data[['star_color', 'spectral_class']]

y = stars_data['star_type']

X_train = stars_data.loc[train_idx, ['star_color', 'spectral_class']]

y_train = stars_data.loc[train_idx, 'star_type']



# Condition the model on sepal width and length, predict the petal width

model = LinearRegression()

model.fit(X_train, y_train)

stars_data['prediction'] = model.predict(X)



fig = px.scatter(

    stars_data, x='star_color', y='spectral_class',

    marginal_x='histogram', marginal_y='histogram',

    color='split', trendline='ols'

)

fig.update_traces(histnorm='probability', selector={'type':'histogram'})

fig.add_shape(

    type="line", line=dict(dash='dash'),

    x0=y.min(), y0=y.min(),

    x1=y.max(), y1=y.max()

)



fig.show()
features_data['temp'] = pd.Series(stars_data['temperature'], index = features_data.index)

features_data['radius'] = pd.Series(stars_data['radius'], index = features_data.index)



X_train, X_test, y_train, y_test = train_test_split(features_data.drop('star_type',axis=1), features_data['star_type'], test_size=0.20, random_state=42)

logreg.fit(X_train, y_train)

predictions = logreg.predict(X_test)



print(classification_report(y_test, predictions))
features_data['luminosity'] = pd.Series(stars_data['luminosity'], index = features_data.index)

X_train, X_test, y_train, y_test = train_test_split(features_data.drop('star_type', axis=1), features_data['star_type'], test_size=0.20, random_state=42)

logreg.fit(X_train, y_train)

predictions = logreg.predict(X_test)



print(classification_report(y_test, predictions))
features_data['absolute_magnitude'] = pd.Series(stars_data['absolute_magnitude'], index = features_data.index)

del features_data['luminosity']



X_train, X_test, y_train, y_test = train_test_split(features_data.drop('star_type', axis=1), features_data['star_type'], test_size=0.20, random_state=42)

logreg.fit(X_train, y_train)

predictions = logreg.predict(X_test)



print(classification_report(y_test, predictions))