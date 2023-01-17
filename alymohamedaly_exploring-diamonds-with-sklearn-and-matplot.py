import numpy as np

import pandas as pd
df = pd.read_csv('../input/diamonds.csv', index_col=0)

for col in df.columns:

    df.rename(columns={col:col.capitalize()}, inplace=True)   #I just like it capitalized

print(df.shape)

df.head()
df.info()
df.describe()
df[['X','Y','Z']] = df[['X','Y','Z']].replace(0,np.NaN)
df.isnull().sum()
df.dropna(inplace=True)

df.shape
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots(figsize=(12,7))

counts, bins, patches = ax.hist(df["Price"], edgecolor='c', fill=True)

# Set the ticks to be at the edges of the bins.

ax.set_xticks(bins)

# Set the xaxis's tick labels to be formatted with 1 decimal place

ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))



# Change the colors of the quartiles

twentyfifth, seventyfifth = np.percentile(df["Price"], [25, 75])

for patch, MyBin in zip(patches, bins):

    if MyBin < twentyfifth:

        patch.set_facecolor('green')

    elif MyBin > seventyfifth:

        patch.set_facecolor('red')

    else:

        patch.set_facecolor('yellow')

        

# Label the percentages below the x-axis        

bin_centers = 0.5 * np.diff(bins) + bins[:-1]

for count, x in zip(counts, bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

plt.title("Price")

ax.grid(axis='y')
fig, ax = plt.subplots(figsize=(12,7))

counts, bins, patches = ax.hist(df["Carat"], bins=15, edgecolor='c')

ax.set_xticks(bins)

# Set the xaxis's tick labels to be formatted with 1 decimal place

ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

ax.set_xlim(right=3.4)

twentyfifth, seventyfifth = np.percentile(df["Carat"], [25, 75])

for patch, MyBin in zip(patches, bins):

    if MyBin < twentyfifth:

        patch.set_facecolor('green')

    elif MyBin > seventyfifth:

        patch.set_facecolor('red')

    else:

        patch.set_facecolor('yellow')



bin_centers = 0.5 * np.diff(bins[:10]) + bins[:9]

for count, x in zip(counts, bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

plt.title("Carat")

ax.grid(axis='y')
Cut = df[["Cut"]].stack().value_counts()

fig, ax = plt.subplots(figsize=(12,6))

ax.bar(Cut.index,Cut)

plt.title("Cut")

ax.grid(axis='y')
Colors = df[["Color"]].stack().value_counts()

fig, ax = plt.subplots(figsize=(12,6))

ax.bar(Colors.index,Colors)

plt.title("Color")

ax.grid(axis='y')
Clarity = df[["Clarity"]].stack().value_counts()

fig, ax = plt.subplots(figsize=(12,6))

ax.bar(Clarity.index,Clarity)

plt.title("Clarity")

ax.grid(axis='y')
fig, ax = plt.subplots(figsize=(12,6))

counts, bins, patches = ax.hist(df["X"], bins=9, edgecolor='c')

ax.set_xticks(bins)

# Set the xaxis's tick labels to be formatted with 1 decimal place

ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

ax.set_xlim(left=1.1)



# Change the colors of bars at the edges

twentyfifth, seventyfifth = np.percentile(df["X"], [25, 75])

for patch, MyBin in zip(patches[:-1], bins[:-1]):

    if MyBin < twentyfifth:

        patch.set_facecolor('green')

    elif MyBin > seventyfifth:

        patch.set_facecolor('red')

    else:

        patch.set_facecolor('yellow')



bin_centers = 0.5 * np.diff(bins[:-1]) + bins[:-2]

for count, x in zip(counts, bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

plt.title("X")

ax.grid(axis='y')

plt.xlim(3, 10.5)
fig, ax = plt.subplots(figsize=(12,6))

counts, bins, patches = ax.hist(df["Y"],bins=[3,6,6,10], edgecolor='c')

ax.set_xticks(bins)

# Set the xaxis's tick labels to be formatted with 1 decimal place

ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

ax.set_xlim(right=10.5)



patches[0].set_facecolor('#7e8c4e')



bin_centers = 0.5 * np.diff(bins[0:3]) + bins[0:2]

for count, x in zip(counts, bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

plt.title("Y")

ax.grid(axis='y')
fig, ax = plt.subplots(figsize=(12,6))

counts, bins, patches = ax.hist(df["Z"],bins=[1.5,3,5,7], edgecolor='c')

ax.set_xticks(bins)

# Set the xaxis's tick labels to be formatted with 1 decimal place

ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

ax.set_xlim(right=7.5)



patches[0].set_facecolor('#db856f')

patches[1].set_facecolor('#589152')

patches[2].set_facecolor('#d3d875')



bin_centers = 0.5 * np.diff(bins[0:3]) + bins[0:2]

for count, x in zip(counts, bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

plt.title("Z")

ax.grid(axis='Z')
fig, ax = plt.subplots(figsize=(12,6))

counts, bins, patches = ax.hist(df["Depth"], edgecolor='c', bins=15)

ax.set_xticks(bins)

# Set the xaxis's tick labels to be formatted with 1 decimal place

ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

ax.set_xlim(52.6, 71.8)



# Change the colors of bars at the edges

twentyfifth, seventyfifth = np.percentile(df["Depth"], [25, 75])

for patch, MyBin in zip(patches[5:11], bins[5:11]):

    if MyBin < twentyfifth:

        patch.set_facecolor('green')

    elif MyBin > seventyfifth:

        patch.set_facecolor('red')

    else:

        patch.set_facecolor('yellow')



bin_centers = 0.5 * np.diff(bins[5:12]) + bins[5:11]

for count, x in zip(counts[5:11], bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

plt.title("Depth")

ax.grid(axis='y')
fig, ax = plt.subplots(figsize=(12,6))

counts, bins, patches = ax.hist(df["Table"], edgecolor='c', bins=15)

ax.set_xticks(bins)

# Set the xaxis's tick labels to be formatted with 1 decimal place

ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

ax.set_xlim(43.0,74.2)



# Change the colors of bars at the edges

twentyfifth, seventyfifth = np.percentile(df["Table"], [25, 75])

for patch, MyBin in zip(patches[:9], bins[:9]):

    if MyBin < twentyfifth:

        patch.set_facecolor('green')

    elif MyBin > seventyfifth:

        patch.set_facecolor('red')

    else:

        patch.set_facecolor('yellow')



bin_centers = 0.5 * np.diff(bins[:10]) + bins[:9]

for count, x in zip(counts, bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

plt.title("Table")

ax.grid(axis='y')
fig, ax = plt.subplots(figsize=(9,6))

ax.scatter(df["Carat"], df["Price"], marker='.', c='c')

plt.title("Carat vs Price")

plt.xlabel('Carat')

plt.ylabel('Price')
Price = df[["Price", "Cut"]]

Labels = set(df["Cut"])

Values = [Price[df["Cut"] == i].mean().values.item() for i in Labels]

fig, ax = plt.subplots(figsize=(9,6))

Labels = list(Labels)

ax.bar(Labels,Values)

plt.title("Price avarage vs Cut")

ax.grid(axis='y')
Price = df[["Price", "Color"]]

Labels = set(df["Color"])

Values = [Price[df["Color"] == i].mean().values.item() for i in Labels]

fig, ax = plt.subplots(figsize=(9,6))

Labels = list(Labels)

ax.bar(Labels,Values)

plt.title("Price avarage vs Color")

ax.grid(axis='y')
Price = df[["Price", "Clarity"]]

Labels = set(df["Clarity"])

Values = [Price[df["Clarity"] == i].mean().values.item() for i in Labels]

fig, ax = plt.subplots(figsize=(9,6))

Labels = list(Labels)

ax.bar(Labels,Values)

plt.title("Price avarage vs Clarity")

ax.grid(axis='y')
fig, ax = plt.subplots(figsize=(9,6))

ax.scatter(df["X"], df["Price"], marker='.', c='m')

plt.title("X vs Price")

plt.ylabel('Price')
fig, ax = plt.subplots(figsize=(9,6))

ax.scatter(df["Y"], df["Price"], marker='.', c='g')

plt.title("Y vs Price")

plt.ylabel('Price')
fig, ax = plt.subplots(figsize=(9,6))

ax.scatter(df["Z"], df["Price"], marker='.', c='purple')

plt.title("Z vs Price")

plt.ylabel('Price')
fig, ax = plt.subplots(figsize=(9,6))

ax.scatter(df["Depth"], df["Price"], marker='.')

plt.title("Depth vs Price")

plt.xlabel('Depth')

plt.ylabel('Price')
fig, ax = plt.subplots(figsize=(9,6))

ax.scatter(df["Table"], df["Price"], marker='.', c= 'pink')

plt.title("Table vs Price")

plt.xlabel('Table')

plt.ylabel('Price')
# Regression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor



# Modelling Helpers :

from sklearn.model_selection import train_test_split



#preprocessing :

from sklearn.preprocessing import OrdinalEncoder



#evaluation metrics :



# Regression

from sklearn.metrics import mean_squared_error, r2_score
cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

color_categories = ['D','E','F','G','H','I','J']

quality_categories = ['I3', 'I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']



x = df.copy()

x[["Cut", "Color", "Clarity"]] =  OrdinalEncoder([cut_categories, color_categories, quality_categories]).fit_transform(x[["Cut", "Color", "Clarity"]])

y = x.pop('Price')

x.drop('Table', axis = 1, inplace = True)   

xNoDep = x.copy()

xNoDep.drop('Depth', axis = 1, inplace = True)     #Depth improved KNN only

x.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 37)

xNoD_train, xNoD_test, yNoD_train, yNoD_test = train_test_split(xNoDep, y, test_size = 0.3, random_state = 37)
lin_regressor = LinearRegression(normalize=True)

knn_regressor = KNeighborsRegressor(n_neighbors = 5)

rfr_regressor = RandomForestRegressor(random_state = 666, n_estimators = 100)

Bag_regressor = BaggingRegressor(random_state = 222, n_estimators=92)

GradBost_regressor = GradientBoostingRegressor(random_state = 3356)

Ada_regressor = AdaBoostRegressor(random_state = 37)

DT_regressor = DecisionTreeRegressor(criterion='friedman_mse', random_state=12)  # criterion measures the quality of a split
rfr_regressor.fit(x_train,y_train)

rfr_pred = rfr_regressor.predict(x_test)

print("accuracy: "+ str(rfr_regressor.score(x_test,y_test)*100) + "%")

print("Mean squared error: {}".format(mean_squared_error(y_test,rfr_pred)))
Bag_regressor.fit(x_train,y_train)

Bag_pred = Bag_regressor.predict(x_test)

print("accuracy: "+ str(Bag_regressor.score(x_test,y_test)*100) + "%")

print("Mean squared error: {}".format(mean_squared_error(y_test,Bag_pred)))
GradBost_regressor.fit(x_train,y_train)

GradBost_pred = GradBost_regressor.predict(x_test)

print("accuracy: "+ str(GradBost_regressor.score(x_test,y_test)*100) + "%")

print("Mean squared error: {}".format(mean_squared_error(y_test,GradBost_pred)))
knn_regressor.fit(xNoD_train,yNoD_train)

knn_pred = knn_regressor.predict(xNoD_test)

print("accuracy: "+ str(knn_regressor.score(xNoD_test,yNoD_test)*100) + "%")

print("Mean squared error: {}".format(mean_squared_error(yNoD_test,knn_pred)))
DT_regressor.fit(x_train,y_train)

DT_pred = DT_regressor.predict(x_test)

print("accuracy: "+ str(DT_regressor.score(x_test,y_test)*100) + "%")

print("Mean squared error: {}".format(mean_squared_error(y_test,DT_pred)))
Ada_regressor.fit(x_train,y_train)

Ada_pred = Ada_regressor.predict(x_test)

print("accuracy: "+ str(Ada_regressor.score(x_test,yNoD_test)*100) + "%")

print("Mean squared error: {}".format(mean_squared_error(y_test,Ada_pred)))
lin_regressor.fit(x_train,y_train)

lin_pred = lin_regressor.predict(x_test)

print("accuracy: "+ str(lin_regressor.score(x_test,y_test)*100) + "%")

print("Mean squared error: {}".format(mean_squared_error(y_test,lin_pred)))