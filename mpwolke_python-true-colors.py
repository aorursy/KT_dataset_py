# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
print(train.shape)

display(train.head(1))



print(test.shape)

display(test.head(1))
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
train.plot(kind='scatter', x='Aspect', y='Hillshade_3pm', alpha=0.5, color='mediumorchid', figsize = (12,9))

plt.title('Aspect And Hillshade_3pm')

plt.xlabel("Aspect")

plt.ylabel("Hillshade_3pm")

plt.show()
train.Cover_Type.describe()
train.plot(kind='scatter', x='Slope', y='Hillshade_3pm', alpha=0.5, color='chartreuse', figsize = (12,9))

plt.title('Slope And Hillshade_3pm')

plt.xlabel("Slope")

plt.ylabel("Hillshade_3pm")

plt.show()
print ("Skew is:", train.Wilderness_Area1.skew())

plt.hist(train.Wilderness_Area1, color='purple')

plt.show()
train.plot(kind='scatter', x='Aspect', y='Hillshade_9am', alpha=0.5, color='lightcoral', figsize = (12,9))

plt.title('Aspect And Hillshade_9am')

plt.xlabel("Aspect")

plt.ylabel("Hillshade_9am")

plt.show()
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
corr = numeric_features.corr()



print (corr['Cover_Type'].sort_values(ascending=False)[1:11], '\n')

print (corr['Cover_Type'].sort_values(ascending=False)[-10:])
def pivotandplot(data,variable,onVariable,aggfunc):

    pivot_var = data.pivot_table(index=variable,

                                  values=onVariable, aggfunc=aggfunc)

    pivot_var.plot(kind='bar', color='purple')

    plt.xlabel(variable)

    plt.ylabel(onVariable)

    plt.xticks(rotation=0)

    plt.show()
pivotandplot(train,'Soil_Type29','Cover_Type',np.median)
import seaborn as sns
ax = sns.scatterplot(x="Horizontal_Distance_To_Hydrology", y="Vertical_Distance_To_Hydrology", \

                     hue="Cover_Type", legend="full", palette='RdPu', data=train)
ax = sns.scatterplot(x="Horizontal_Distance_To_Hydrology", y="Vertical_Distance_To_Hydrology", \

                     hue="Cover_Type", legend="full", palette='GnBu', data=train)
ax = sns.scatterplot(x="Horizontal_Distance_To_Hydrology", y="Vertical_Distance_To_Hydrology", \

                     hue="Cover_Type", legend="full", palette='BuGn', data=train)
import seaborn as sns
train['log_Cover_Type']=np.log(train['Cover_Type']+1)

Cover_Type=train[['Cover_Type','log_Cover_Type']]



Cover_Type.head(5)
_=sns.regplot(train['Soil_Type37'],Cover_Type['Cover_Type'])
ax = sns.scatterplot(x="Aspect", y="Hillshade_3pm", \

                     hue="Cover_Type", legend="full", palette = "BuGn_r", data=train)
ax = sns.scatterplot(x="Aspect", y="Hillshade_3pm", \

                     hue="Cover_Type", legend="full", palette = "RdPu_r", data=train)
# Save a palette to a variable:

palette = sns.color_palette("bright")



# Use palplot and pass in the variable:

sns.palplot(palette)
# library & dataset

from matplotlib import pyplot as plt

import numpy as np

 

# create data

x = np.random.rand(15)

y = x+np.random.rand(15)

z = x+np.random.rand(15)

z=z*z

 

# Use it with a call in cmap

plt.scatter(x, y, s=z*2000, c=x, cmap="BuPu", alpha=0.4, edgecolors="grey", linewidth=2)

 

# You can reverse it:

plt.scatter(x, y, s=z*2000, c=x, cmap="BuPu_r", alpha=0.4, edgecolors="grey", linewidth=2)

 

# OTHER: viridis / inferno / plasma / magma

plt.scatter(x, y, s=z*2000, c=x, cmap="plasma", alpha=0.4, edgecolors="grey", linewidth=2)
# Then you can pass arguments to each type:

sns.jointplot(x=train["Cover_Type"], y=train["Soil_Type1"], kind='scatter', s=200, color='m', edgecolor="skyblue", linewidth=2)

 

# Custom the color

sns.set(style="white", color_codes=True)

sns.jointplot(x=train["Cover_Type"], y=train["Soil_Type1"], kind='kde', color="skyblue")
sequential_colors = sns.color_palette("RdPu", 10)

sns.palplot(sequential_colors)
# Just load seaborn and the chart looks better:

import seaborn as sns

plt.plot( 'x', 'y', data=train, marker='o', color='mediumvioletred')

plt.show()
sns.palplot(sns.color_palette("Paired"))
# libraries

import matplotlib.pyplot as plt

import numpy as np

 

# create data

x = np.random.normal(size=50000)

y = (x * 3 + np.random.normal(size=50000)) * 5

 

# Make the plot

plt.hexbin(x, y, gridsize=(15,15) )

plt.show()

 

# We can control the size of the bins:

plt.hexbin(x, y, gridsize=(150,150) )

plt.show()


# Libraries

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import kde

 

# Create data: 200 points

data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)

x, y = data.T

 

# Create a figure with 6 plot areas

fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))

 

# Everything sarts with a Scatterplot

axes[0].set_title('Scatterplot')

axes[0].plot(x, y, 'ko')

# As you can see there is a lot of overplottin here!

 

# Thus we can cut the plotting window in several hexbins

nbins = 20

axes[1].set_title('Hexbin')

axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)

 

# 2D Histogram

axes[2].set_title('2D Histogram')

axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)

 

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

k = kde.gaussian_kde(data.T)

xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]

zi = k(np.vstack([xi.flatten(), yi.flatten()]))

 

# plot a density

axes[3].set_title('Calculate Gaussian KDE')

axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)

 

# add shading

axes[4].set_title('2D Density with shading')

axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

 

# contour

axes[5].set_title('Contour')

axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

axes[5].contour(xi, yi, zi.reshape(xi.shape) )
# libraries

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import kde

 

# create data

x = np.random.normal(size=500)

y = x * 3 + np.random.normal(size=500)

 

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

nbins=300

k = kde.gaussian_kde([x,y])

xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]

zi = k(np.vstack([xi.flatten(), yi.flatten()]))

 

# Make the plot

plt.pcolormesh(xi, yi, zi.reshape(xi.shape))

plt.show()

 

# Change color palette

plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)

plt.show()
# Then you can pass arguments to each type:

sns.jointplot(x=train["Cover_Type"], y=train["Soil_Type37"], kind='scatter', s=200, color='m', edgecolor="skyblue", linewidth=2)

 

# Custom the color

sns.set(style="white", color_codes=True)

sns.jointplot(x=train["Cover_Type"], y=train["Soil_Type37"], kind='kde', color="skyblue")
# libraries

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

# Dataset

df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })

 

# plot

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(train['Cover_Type'], train['Horizontal_Distance_To_Hydrology'], train['Vertical_Distance_To_Hydrology'], c='deeppink', s=60)

ax.view_init(30, 185)

plt.show()
# libraries

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

# Dataset

df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })

 

# plot

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(train['Cover_Type'], train['Horizontal_Distance_To_Hydrology'], train['Vertical_Distance_To_Hydrology'], c='darkolivegreen', s=60)

ax.view_init(30, 185)

plt.show()
# libraries

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

# Dataset

df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })

 

# plot

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(train['Cover_Type'], train['Horizontal_Distance_To_Hydrology'], train['Vertical_Distance_To_Hydrology'], c='midnightblue', s=60)

ax.view_init(30, 185)

plt.show()
# libraries

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

# Dataset

df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })

 

# plot

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(train['Cover_Type'], train['Horizontal_Distance_To_Hydrology'], train['Vertical_Distance_To_Hydrology'], c='orangered', s=60)

ax.view_init(30, 185)

plt.show()
sns.palplot(sns.color_palette("Blues"))


ax = sns.violinplot(x="Cover_Type", y="Horizontal_Distance_To_Fire_Points", data=train, 

                    inner=None, color=".8")

ax = sns.stripplot(x="Cover_Type", y="Horizontal_Distance_To_Fire_Points", data=train, 

                   jitter=True)

ax.set_title('Hor Dist to Fire Points vs Cover Type')

ax.set_ylabel('Horz Dist to nearest wildfire ignition points')


# Libraries

import matplotlib.pyplot as plt

import pandas as pd

from math import pi

 

# Set data

df = pd.DataFrame({

'group': ['A','B','C','D'],

'var1': [38, 1.5, 30, 4],

'var2': [29, 10, 9, 34],

'var3': [8, 39, 23, 24],

'var4': [7, 31, 33, 14],

'var5': [28, 15, 32, 14]

})

 

 

 

# ------- PART 1: Create background

 

# number of variable

categories=list(df)[1:]

N = len(categories)

 

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 

# Initialise the spider plot

ax = plt.subplot(111, polar=True)

 

# If you want the first axis to be on top:

ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

 

# Draw one axe per variable + add labels labels yet

plt.xticks(angles[:-1], categories)

 

# Draw ylabels

ax.set_rlabel_position(0)

plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)

plt.ylim(0,40)

 

 

# ------- PART 2: Add plots

 

# Plot each individual = each line of the data

# I don't do a loop, because plotting more than 3 groups makes the chart unreadable

 

# Ind1

values=df.loc[0].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A")

ax.fill(angles, values, 'b', alpha=0.1)

 

# Ind2

values=df.loc[1].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="group B")

ax.fill(angles, values, 'r', alpha=0.1)

 

# Add legend

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
sns.palplot(sns.hls_palette(8, l=.3, s=.8))
sns.palplot(sns.cubehelix_palette(8))
x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.kdeplot(x, y, cmap=cmap, shade=True);
sns.palplot(sns.light_palette((210, 90, 60), input="husl"))
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,8))



sns.boxplot(x='Cover_Type', y='Hillshade_9am', data=train, ax=axis1);

axis1.set_title('Hillshade 9am vs Cover Type')

axis1.set_ylabel('Hillshade index at 9am')

sns.boxplot(x='Cover_Type', y='Hillshade_Noon', data=train, ax=axis2);

axis2.set_title('Hillshade Noon vs Cover Type')

axis2.set_ylabel('Hillshade index at noon')

sns.boxplot(x='Cover_Type', y='Hillshade_3pm', data=train, ax=axis3);

axis3.set_title('Hillshade 3pm vs Cover Type')

axis3.set_ylabel('Hillshade index at 3pm')
sns.palplot(sns.light_palette("green"))
from sklearn.ensemble import RandomForestClassifier
train.columns
#Model with columns without modifications + Soil and Wilderness_Area with Compass

interest_columns = train.columns

interest_columns = interest_columns.drop(['Cover_Type', 'Id'])



X = train[interest_columns]

y = train.Cover_Type

#X_test = test[interest_columns]

my_model = RandomForestClassifier(n_estimators = 100, random_state=42)

my_model = RandomForestClassifier(n_estimators = 719,

                                       max_features = 0.3,

                                       max_depth = 464,

                                       min_samples_split = 2,

                                       min_samples_leaf = 1,

                                       bootstrap = False,

                                       random_state=42)



#CrossValidation

#scores = cross_val_score(my_model, X, y, cv=5, scoring = 'accuracy')

#print(scores)
##PERMUTATION IMPORTANCE

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=1).fit(train_X, train_y)



import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



features_to_plot = ['Elevation', 'Wilderness_Area3']

inter1  =  pdp.pdp_interact(model=my_model, dataset=val_X, model_features=val_X.columns, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour', plot_pdp=True)

plt.show()
cmap = sns.cubehelix_palette(as_cmap=True)



f, ax = plt.subplots(figsize=(20,10))

points = ax.scatter(train['Hillshade_Noon'],train['Elevation'] , c=train['Cover_Type'],label=Cover_Type ,s=20, cmap='rainbow')

#plt.xticks(np.arange(0, 400,20))

#plt.axis('scaled')

f.colorbar(points)

#plt.legend(labels=Cover_Type_Dict.values(),loc='upper left')

plt.show()
#Change data type of Continous variable columns to float

continuos_columns={'Elevation':float,'Aspect':float,'Slope':float,'Horizontal_Distance_To_Hydrology':float,'Vertical_Distance_To_Hydrology':float,

                   'Horizontal_Distance_To_Roadways':float,'Hillshade_9am':float,'Hillshade_Noon':float,'Hillshade_3pm':float,'Horizontal_Distance_To_Fire_Points':float}

train=train.astype(continuos_columns)

train.dtypes
#Function to plot charts of All features

def features_plots(data,continuous_vars,discrete_vars):

    plt.figure(figsize=(15,24.5))

    for i, cv in enumerate(continuous_vars):

        plt.subplot(7, 2, i+1)

        plt.hist(data[cv], bins=len(data[cv].unique()))

        plt.title(cv)

        plt.ylabel('Frequency')

    

    for i, dv in enumerate(discrete_vars):

        plt.subplot(7, 2, i+ 1 + len(continuous_vars))

        data[dv].value_counts().plot(kind='bar', title=dv)

        plt.ylabel('Frequency')
continuous_vars=continuos_columns.keys()

discrete_vars = ['Cover_Type', 'Wilderness_Area1', 'Soil_Type1']

features_plots(train,continuous_vars,discrete_vars)
def pair_scatterplot(data,continuous_vars,y_column,target_column):

    #plt.figure(figsize=(20,10))

    cmap = sns.cubehelix_palette(as_cmap=True)

    for i, cv in enumerate(continuous_vars):

        #plt.subplot(7, 2, i+1)

        #data.boxplot(column=cv,return_type='axes',by=by_column)

        f, ax = plt.subplots(figsize=(20,10))

        ax.set_xlabel(cv)

        ax.set_ylabel(y_column)

        points = ax.scatter(train[cv],train[y_column] , c=train[target_column],label=Cover_Type,s=20, cmap='rainbow')

        f.colorbar(points)

    plt.show()





        #plt.title(cv)

        #plt.ylabel('Value')
pair_scatterplot(train,continuous_vars,'Elevation','Cover_Type')
custom_palette = sns.color_palette("Paired", 9)

sns.palplot(custom_palette)