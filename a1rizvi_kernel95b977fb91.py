import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

sns.set(color_codes=True)

%matplotlib inline
automobile_data = pd.read_csv("../input/autodata-akhtar/Automobile_data.csv")
automobile_data.head()
automobile_data.info()
automobile_data.replace('?', np.nan, inplace=True)
print(automobile_data.isnull().sum())
thresh = len(automobile_data) * .1

automobile_data.dropna(thresh = thresh, axis = 1, inplace = True)
print(automobile_data.isnull().sum())
def impute_median(series):

    return series.fillna(series.median())



#automobile_data['num-of-doors']=automobile_data['num-of-doors'].transform(impute_median) ---String hence ignored

automobile_data.bore=automobile_data['bore'].transform(impute_median)

automobile_data.stroke=automobile_data['stroke'].transform(impute_median)

automobile_data.horsepower=automobile_data['horsepower'].transform(impute_median)

automobile_data.price=automobile_data['price'].transform(impute_median)
automobile_data['num-of-doors'].fillna(str(automobile_data['num-of-doors'].mode().values[0]),inplace=True)

automobile_data['peak-rpm'].fillna(str(automobile_data['peak-rpm'].mode().values[0]),inplace=True)

automobile_data['normalized-losses'].fillna(str(automobile_data['normalized-losses'].mode().values[0]),inplace=True)
print(automobile_data.isnull().sum())
automobile_data.head()
automobile_data.make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))

plt.title("Top 10 Number of vehicles by make (Akhtar)")

plt.ylabel('Number of vehicles')

plt.xlabel('Make');
automobile_data['price']=pd.to_numeric(automobile_data['price'],errors='coerce')

sns.distplot(automobile_data['price']);
print("Skewness: %f" % automobile_data['price'].skew())

print("Kurtosis: %f" % automobile_data['price'].kurt())
plt.figure(figsize=(20,10))

c=automobile_data.corr()

sns.heatmap(c,cmap="BrBG",annot=True)
sns.lmplot('engine-size', # Horizontal axis

           'price', # Vertical axis

           data=automobile_data, # Data source

           fit_reg=False, # Don't fix a regression line

           hue="make", # Set color

           palette="Paired",

           scatter_kws={"marker": "D", # Set marker style

                        "s": 100}) # S marker size
sns.lmplot('city-mpg', # Horizontal axis

           'price', # Vertical axis

           data=automobile_data, # Data source

           fit_reg=False, # Don't fix a regression line

           hue="body-style", # Set color

           palette="Paired",

           scatter_kws={"marker": "D", # Set marker style

                        "s": 100}) # S marker size
sns.boxplot(x="fuel-type", y="price",data = automobile_data)
cars = automobile_data = pd.read_csv("../input/cleaneddata-akhtar/cardata_cleaned.csv")
plot_color = "#dd0033"

title_color = "#333333"

y_title_margin = 1.0 # The amount of space above titles

left   =  0.10  # the left side of the subplots of the figure

right  =  0.95    # the right side of the subplots of the figure

bottom =  0.1    # the bottom of the subplots of the figure

top    =  0.5    # the top of the subplots of the figure

wspace =  0.1     # the amount of width reserved for blank space between subplots

hspace = 0.6 # the amount of height reserved for white space between subplots



plt.subplots_adjust(

    left    =  left, 

    bottom  =  bottom, 

    right   =  right, 

    top     =  top, 

    wspace  =  wspace, 

    hspace  =  hspace

)

sns.set_style("whitegrid") #set seaborn style template
make_hist=sns.countplot(cars['make'], color=plot_color)

make_hist.set_xticklabels(make_hist.get_xticklabels(), rotation=90)

make_hist.set_xlabel('')

make_hist.set_ylabel('Counts', fontsize=14)



ax = make_hist.axes

ax.patch.set_alpha(0)

ax.set_title('Make - Distribution', fontsize=16, color="#333333")

fig = make_hist.get_figure()

fig.figsize=(10,5)

fig.patch.set_alpha(0.5)

fig.savefig('01make_distribution.png',dpi=fig.dpi,bbox_inches='tight')
fig, ax = plt.subplots(figsize=(5,5), ncols=1, nrows=1) # get the figure and axes objects for a 3x2 subplot figure



fig.patch.set_alpha(0.5)

ax.set_title("Symboling - Distribution", y = y_title_margin, color=title_color,fontsize=16)



#Set transparency for individual subplots.

ax.patch.set_alpha(0.5)



symbol_hist=sns.countplot(cars["symboling"], color=plot_color, ax=ax )

#symbol_hist.set_xticklabels(symbol_hist.get_xticklabels(), rotation=90,fontsize=12)

symbol_hist.set_ylabel('Count',fontsize=16 )

symbol_hist.set_xlabel('Symboling - Insurance Risk Factor',fontsize=16)



#plt.show()

fig.savefig('02symboling_distribution.png',dpi=fig.dpi,bbox_inches='tight')
plot_color = "#dd0033"

title_color = "#333333"



fig, ax = plt.subplots(figsize=(20,20), ncols=2, nrows=2)



plt.subplots_adjust(

    left    =  left, 

    bottom  =  bottom, 

    right   =  right, 

    top     =  0.6, 

    wspace  =  0.3, 

    hspace  =  0.5

)



fig.patch.set_alpha(0.5)



ax[0][0].set_title('Body Style Distribution', fontsize=14)

ax[0][0].set_alpha(0)



bstyle_dist=sns.countplot(cars['body-style'],color=plot_color, ax=ax[0][0])

bstyle_dist.set_xticklabels(bstyle_dist.get_xticklabels(),rotation=45, fontsize=14)

bstyle_dist.set_xlabel('')

bstyle_dist.set_ylabel('Counts', fontsize=14)



ax[0][1].set_title('Number of Doors Distribution', fontsize=14)

ax[0][1].set_alpha(0)



numdoors_dist=sns.countplot(cars['num-of-doors'],color=plot_color, ax=ax[0][1])

numdoors_dist.set_xlabel('')

numdoors_dist.set_ylabel('Counts', fontsize=14)



ax[1][0].set_title('Drive Wheels Distribution', fontsize=14)

ax[1][0].set_alpha(0)



drvwheels_dist=sns.countplot(cars['drive-wheels'],color=plot_color, ax=ax[1][0])

drvwheels_dist.set_xticklabels(drvwheels_dist.get_xticklabels(),rotation=45, fontsize=14)

drvwheels_dist.set_xlabel('')

drvwheels_dist.set_ylabel('Counts', fontsize=14)



fig.savefig('03categorical_vars_distribution.png',dpi=fig.dpi,bbox_inches='tight')
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Normalized Losses - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



normloss_hist=sns.distplot(cars['normalized-losses'], color=plot_color, ax=ax[0] )

#symbol_hist.set_xticklabels(symbol_hist.get_xticklabels(), rotation=90,fontsize=12)

#symbol_hist.set_ylabel('Count',fontsize=16 )

symbol_hist.set_xlabel('Normalized Losses',fontsize=16)



ax[1].set_title("Normalized Losses - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

normloss_hist=sns.violinplot(cars['normalized-losses'], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('04normalized_losses_distribution.png',dpi=fig.dpi,bbox_inches='tight')
cars['normalized-losses'].describe()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Wheel Base - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



wbase_hist=sns.distplot(cars["wheel-base"], hist=True, color=plot_color, ax=ax[0] )

#symbol_hist.set_xticklabels(symbol_hist.get_xticklabels(), rotation=90,fontsize=12)

#symbol_hist.set_ylabel('Count',fontsize=16 )

wbase_hist.set_xlabel('Wheel Base',fontsize=16)



ax[1].set_title("Wheel Base - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

wbase_box=sns.violinplot(cars["wheel-base"], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('05wheelbase_distribution.png',dpi=fig.dpi,bbox_inches='tight')
cars['wheel-base'].describe()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Height - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



height_hist=sns.distplot(cars["height"], hist=True, color=plot_color, ax=ax[0] )

height_hist.set_xlabel('Height',fontsize=16)



ax[1].set_title("Height - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

height_box=sns.violinplot(cars["height"], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('06height_distribution.png',dpi=fig.dpi,bbox_inches='tight')
cars['height'].describe()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Engine Size - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



enginesize_hist=sns.distplot(cars["engine-size"], hist=True, color=plot_color, ax=ax[0] )

enginesize_hist.set_xlabel('Engine Size',fontsize=16)



ax[1].set_title("Engine Size - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

enginesize_box=sns.violinplot(cars["engine-size"], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('07enginesize_distribution.png',dpi=fig.dpi,bbox_inches='tight')
print("Mode:" + str(cars["engine-size"].mode()))

print(cars["engine-size"].describe())
print("Mode:" + str(cars["bore"].mode()))

print(cars["bore"].describe())

#cars["bore"].value_counts()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Bore - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



bore_hist=sns.distplot(cars["bore"], hist=True, color=plot_color, ax=ax[0] )

#symbol_hist.set_xticklabels(symbol_hist.get_xticklabels(), rotation=90,fontsize=12)

#symbol_hist.set_ylabel('Count',fontsize=16 )

bore_hist.set_xlabel('Bore',fontsize=16)



ax[1].set_title("Bore - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

bore_box=sns.violinplot(cars["bore"], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('08bore_distribution.png',dpi=fig.dpi,bbox_inches='tight')
print("Mode:" + str(cars["bore"].mode()))

print(cars["bore"].describe())

#cars["bore"].value_counts()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Stroke - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



stroke_hist=sns.distplot(cars["stroke"], hist=True, color=plot_color, ax=ax[0] )

stroke_hist.set_xlabel('stroke',fontsize=16)



ax[1].set_title("Stroke - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

stroke_box=sns.violinplot(cars["stroke"], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('09stroke_distribution.png',dpi=fig.dpi,bbox_inches='tight')
print("Mode:" + str(cars["stroke"].mode()[0]))

print("Median:" + str(cars["stroke"].median()))



print(cars["stroke"].describe())

#cars["stroke"].value_counts()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("horsepower - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



hp_hist=sns.distplot(cars["horsepower"], hist=True, color=plot_color, ax=ax[0] )

hp_hist.set_xlabel('horsepower',fontsize=16)



ax[1].set_title("horsepower - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

hp_box=sns.boxplot(cars["horsepower"], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('10horsepower_distribution.png',dpi=fig.dpi,bbox_inches='tight')
print("Mode:" + str(cars["horsepower"].mode()[0]))

print("Median:" + str(cars["horsepower"].median()))



print(cars["horsepower"].describe())

#cars["stroke"].value_counts()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Fuel Efficiency - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



citympg_hist=sns.distplot(cars["city-mpg"], hist=True, color=plot_color, ax=ax[0] )

citympg_hist.set_xlabel('Fuel Efficiency(City)',fontsize=16)



ax[1].set_title("Fuel Efficiency - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

citympg_box=sns.violinplot(cars["city-mpg"], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('11citympg_distribution.png',dpi=fig.dpi,bbox_inches='tight')
print("Mode:" + str(cars["city-mpg"].mode()[0]))

print("Median:" + str(cars["city-mpg"].median()))



print(cars["city-mpg"].describe())

#cars["city-mpg"].value_counts()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Price - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



price_hist=sns.distplot(cars["price"], hist=True, color=plot_color, ax=ax[0] )

price_hist.set_xlabel('Price',fontsize=16)



ax[1].set_title("Price - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

normloss_hist=sns.violinplot(cars["price"], color=plot_color, ax=ax[1] )



#plt.show()

fig.savefig('12price_distribution.png',dpi=fig.dpi,bbox_inches='tight')
print("Mode:" + str(cars["price"].mode()[0]))

print("Median:" + str(cars["price"].median()))



print(cars["price"].describe())

#cars["price"].value_counts()
ncyl_hist=sns.countplot(cars['num_cylinders'], color=plot_color)

ncyl_hist.set_xlabel('Cylinders')

ncyl_hist.set_ylabel('Counts', fontsize=14)



ax = ncyl_hist.axes

ax.patch.set_alpha(0)

ax.set_title('Number of cylinders - Distribution', fontsize=16, color="#333333")

fig = ncyl_hist.get_figure()

fig.figsize=(10,5)

fig.patch.set_alpha(0.5)

fig.savefig('13numcylinders_distribution.png',dpi=fig.dpi,bbox_inches='tight')
print("Mode:" + str(cars["num_cylinders"].mode()[0]))

print("Median:" + str(cars["num_cylinders"].median()))



print(cars["num_cylinders"].describe())

#cars["num_cylinders"].value_counts()
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 



fig.patch.set_alpha(0.5)

ax[0].set_title("Curb Weight - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[0].patch.set_alpha(0)



cweight_hist=sns.distplot(cars["curb-weight"], hist=True, color=plot_color, ax=ax[0] )

cweight_hist.set_xlabel('Curb Weight',fontsize=16)



ax[1].set_title("Curb Weight - Distribution", y = y_title_margin, color=title_color,fontsize=16)

ax[1].patch.set_alpha(0)

cweight_box=sns.violinplot(cars["curb-weight"], color=plot_color, ax=ax[1] )

cweight_box.set_xlabel('Curb Weight',fontsize=16)



#plt.show()

fig.savefig('12curbweight_distribution.png',dpi=fig.dpi,bbox_inches='tight')
print("Mode:" + str(cars["curb-weight"].mode()[0]))

print("Median:" + str(cars["curb-weight"].median()))



print(cars["curb-weight"].describe())

#cars["curb-weight"].value_counts()