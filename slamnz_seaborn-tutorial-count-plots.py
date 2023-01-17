from pandas import read_csv

data = read_csv("../input/xAPI-Edu-Data.csv")
target = "Class"

features = data.drop(target,1).columns
features_by_dtype = {}



for f in features:

    dtype = str(data[f].dtype)

    if dtype not in features_by_dtype.keys():

        features_by_dtype[dtype] = [f]

    else:

        features_by_dtype[dtype] += [f]
keys = iter(features_by_dtype.keys())

k = next(keys)

l = features_by_dtype[k]

categorical_features = l

k = next(keys)

l = features_by_dtype[k]

numerical_features = l
categorical_features, numerical_features

features, target

pass
data[categorical_features].head()
from seaborn import countplot

from matplotlib.pyplot import figure, show
figure()

countplot(data=data,x=target)

show()
figure()

countplot(data=data,y=target)

show()
width=12

height=6

figure(figsize=(width,height))

countplot(data=data,x=target)

show()
figure(figsize=(12,6))



order=["L","M","H"]



countplot(data=data,x=target,order=order)

show()
descending_order = data[target].value_counts().sort_values(ascending=False).index



figure(figsize=(12,6))

countplot(data=data,x=target,order=descending_order)

show()
figure(figsize=(12,6))

countplot(data=data,x=target,color="tomato")

show()
colours = ["maroon", "navy", "gold"]



figure(figsize=(12,6))

countplot(data=data,x=target,palette=colours)

show()
figure(figsize=(12,6))

countplot(data=data,x=target, hue="StageID")

show()
figure(figsize=(12,6))

ax = countplot(data=data,x=target)

ax.set_xticklabels(["1","2","3"])

show()
figure(figsize=(12,6))

ax = countplot(data=data,x=target)

ax.set_xlabel("X Label Renamed!")

show()
figure(figsize=(12,6))

ax = countplot(data=data,x=target)

ax.set_ylabel("Y Label Renamed!")

show()
from matplotlib.pyplot import suptitle

figure(figsize=(12,6))

suptitle("Enter Title Here")

ax = countplot(data=data,x=target)

show()
fig = figure(figsize=(12,6))

ax = countplot(data=data,x=target)

ax.set_yticks([t*15 for t in range(0,16)])

show()
from seaborn import set



set(font_scale=1.4)

fig = figure(figsize=(12,6))

ax = countplot(data=data,x=target)

show()
from seaborn import despine



fig = figure(figsize=(12,6))

ax = countplot(data=data,x=target)

despine()

show()
from seaborn import axes_style



with axes_style({'axes.grid': False}):

    fig = figure(figsize=(12,6))

    ax = countplot(data=data,x=target)

show()
# Change background colour
from seaborn import axes_style



with axes_style({'axes.facecolor': 'gold'}):

    fig = figure(figsize=(12,6))

    ax = countplot(data=data,x=target)

show()
from seaborn import axes_style



with axes_style({'grid.color': "red"}):

    fig = figure(figsize=(12,6))

    ax = countplot(data=data,x=target)

show()
from matplotlib.pyplot import xticks



figure(figsize=(12,6))

countplot(data=data,x=target)

xticks(rotation=90)