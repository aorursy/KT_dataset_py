#import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#list files and directories in workspace

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_file_path = ('../input/home-data-for-ml-course/train.csv')

test_file_path = ('../input/home-data-for-ml-course/test.csv')

train_raw = pd.read_csv(train_file_path)

test_raw = pd.read_csv(test_file_path)
print(train_raw.shape) #returns a tuple, showing (number_of_rows, number_of_cols)

print(train_raw.columns) #returns a list of all column names

print(train_raw.head()) #returns the head of the dataframe

print(train_raw.info()) #returns a DataFrame with column, number of valid entries, and dtype
col_description = {'SalePrice': "the property's sale price in dollars. This is the target variable that you're trying to predict.",

                   'MSSubClass': "The building class",

                   'MSZoning': "The general zoning classification",

                   'LotFrontage': "Linear feet of street connected to property",

                   'LotArea': "Lot size in square feet",

                   'Street': "Type of road access",

                   'Alley': "Type of alley access",

                   'LotShape': "General shape of property",

                   'LandContour': "Flatness of the property",

                   'Utilities': "Type of utilities available",

                   'LotConfig': "Lot configuration",

                   'LandSlope': "Slope of property",

                   'Neighborhood': "Physical locations within Ames city limits",

                   'Condition1': "Proximity to main road or railroad",

                   'Condition2': "Proximity to main road or railroad (if a second is present)",

                   'BldgType': "Type of dwelling",

                   'HouseStyle': "Style of dwelling",

                   'OverallQual': "Overall material and finish quality",

                   'OverallCond': "Overall condition rating",

                   'YearBuilt': "Original construction date",

                   'YearRemodAdd': "Remodel date",

                   'RoofStyle': "Type of roof",

                   'RoofMatl': "Roof material",

                   'Exterior1st': "Exterior covering on house",

                   'Exterior2nd': "Exterior covering on house (if more than one material)",

                   'MasVnrType': "Masonry veneer type",

                   'MasVnrArea': "Masonry veneer area in square feet",

                   'ExterQual': "Exterior material quality",

                   'ExterCond': "Present condition of the material on the exterior",

                   'Foundation': "Type of foundation",

                   'BsmtQual': "Height of the basement",

                   'BsmtCond': "General condition of the basement",

                   'BsmtExposure': "Walkout or garden level basement walls",

                   'BsmtFinType1': "Quality of basement finished area",

                   'BsmtFinSF1': "Type 1 finished square feet",

                   'BsmtFinType2': "Quality of second finished area (if present)",

                   'BsmtFinSF2': "Type 2 finished square feet",

                   'BsmtUnfSF': "Unfinished square feet of basement area",

                   'TotalBsmtSF': "Total square feet of basement area",

                   'Heating': "Type of heating",

                   'HeatingQC': "Heating quality and condition",

                   'CentralAir': "Central air conditioning",

                   'Electrical': "Electrical system",

                   '1stFlrSF': "First Floor square feet",

                   '2ndFlrSF': "Second floor square feet",

                   'LowQualFinSF': "Low quality finished square feet (all floors)",

                   'GrLivArea': "Above grade (ground) living area square feet",

                   'BsmtFullBath': "Basement full bathrooms",

                   'BsmtHalfBath': "Basement half bathrooms",

                   'FullBath': "Full bathrooms above grade",

                   'HalfBath': "Half baths above grade",

                   'Bedroom': "Number of bedrooms above basement level",

                   'Kitchen': "Number of kitchens",

                   'KitchenQual': "Kitchen quality",

                   'TotRmsAbvGrd': "Total rooms above grade (does not include bathrooms)",

                   'Functional': "Home functionality rating",

                   'Fireplaces': "Number of fireplaces",

                   'FireplaceQu': "Fireplace quality",

                   'GarageType': "Garage location",

                   'GarageYrBlt': "Year garage was built",

                   'GarageFinish': "Interior finish of the garage",

                   'GarageCars': "Size of garage in car capacity",

                   'GarageArea': "Size of garage in square feet",

                   'GarageQual': "Garage quality",

                   'GarageCond': "Garage condition",

                   'PavedDrive': "Paved driveway",

                   'WoodDeckSF': "Wood deck area in square feet",

                   'OpenPorchSF': "Open porch area in square feet",

                   'EnclosedPorch': "Enclosed porch area in square feet",

                   '3SsnPorch': "Three season porch area in square feet",

                   'ScreenPorch': "Screen porch area in square feet",

                   'PoolArea': "Pool area in square feet",

                   'PoolQC': "Pool quality",

                   'Fence': "Fence quality",

                   'MiscFeature': "Miscellaneous feature not covered in other categories",

                   'MiscVal': "$Value of miscellaneous feature",

                   'MoSold': "Month Sold",

                   'YrSold': "Year Sold",

                   'SaleType': "Type of sale",

                   'SaleCondition': "Condition of sale",

                  }
# Testing new column description dictionary.

print(col_description['Fence'])
# getting a summary for all the numerical columns:

train_raw.select_dtypes(exclude = 'object').describe() #select_dtypes returns a df, filtered by dtype #describe returns a summary
# getting a summary for all the categorical and bool columns:

train_raw.select_dtypes(include=['object','bool']).describe()
# getting value counts for specific columns:

print(train_raw.SaleCondition.value_counts())

print(train_raw.GarageType.value_counts(normalize=True)) # if normalize argument is set to True, relative frequencies are shown
#way one

train_raw_top25 = train_raw.copy().sort_values(by='SalePrice', ascending=False).head(25)

#way two

train_raw_top25_2 = train_raw[train_raw['SalePrice'] > 400000]
# to get a quick overview regarding the top25 we can show a simple mean summary statistc

print(train_raw_top25.mean()) # shows a df with the name of the columns and their mean



# to compare specific colums we can also show mean values for specific columns by label subsetting and applying .mean() method

print('Lot Area enthält folgende Informationen: ' + col_description['LotArea'])

print(train_raw_top25['LotArea'].mean())

print(train_raw['LotArea'].mean())
# creating a heatmap plot with matplotlib

import matplotlib.pyplot as plt



df = train_raw.select_dtypes(exclude = 'object').drop('Id',axis=1)

f = plt.figure(figsize=(19, 15))

plt.matshow(df.corr(), fignum=f.number)

plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)

plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
#creating a coloured correlation matrix with pandas

import pandas as pd

import numpy as np



df = train_raw.select_dtypes(exclude = 'object').drop('Id',axis=1)

corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')

# 'RdBu_r' & 'BrBG' are other good diverging colormaps
# including all cols in the list, that are associated to SalePrice with an absolut correlation >.5

subset_cols_2 = [col for col in train_raw.select_dtypes(exclude = 'object') if abs(train_raw[col].corr(train_raw['SalePrice'])) >= 0.5]



# creating drop list:

col_drop_list = set(train_raw.columns) - set(subset_cols_2)



# dropping all elements except our subset_cols_2

train_raw_sub = train_raw.drop(col_drop_list, axis = 1)
train_raw_sub.corr().style.background_gradient(cmap = 'YlOrRd')

import matplotlib.pyplot as plt



# create a figure and axes

fig, ax = plt.subplots()



# scatterplot of SalePrice and OverallQual

ax.scatter(train_raw_sub.OverallQual, # define x axis first

           train_raw_sub.SalePrice) # define y axis then

ax.set_title('Scatterplot of SalePrice by OverallQual')

ax.set_xlabel('Overall Quality')

ax.set_ylabel('Sale Price')



# if we want to add a linear regression curve, we can add:

from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

linear_regressor.fit(train_raw_sub.OverallQual.values.reshape(-1, 1), 

                     train_raw_sub.SalePrice.values.reshape(-1, 1))

SalePricePred = linear_regressor.predict(train_raw_sub.OverallQual.values.reshape(-1, 1))



ax.plot(train_raw_sub.OverallQual,

       SalePricePred,

       color = 'red')
### 1 ) Standardization ###

# import standardization module

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



# create StandardScaler object

sc = StandardScaler()

# create SimpleImputer object, impute by mean

imp = SimpleImputer(strategy = 'mean')



# create a copy of train_raw_sub to prevent overwriting

df = train_raw_sub.copy()

# impute missing values

df_imp = imp.fit_transform(df)

# standardize df with imputed values

df = pd.DataFrame(sc.fit_transform(df_imp))

# create a copy to extract columns later

df2 = df.copy()

# df is now ready for PCA



### 2 )     PCA        ###

# import PCA module

from sklearn.decomposition import PCA



# first we create a scree plot to investigate how many components we should include in the PCA

scree_df = PCA().fit(df)



# visualize a scree plot

# import matplotlib for the graph

import matplotlib.pyplot as plt

# import numpy to calculate cumulated sum

import numpy as np



plt.figure()

plt.plot(np.cumsum(scree_df.explained_variance_ratio_))

plt.xlabel('Number of components')

plt.ylabel('Variance (%)')

plt.show()
# create PCA object with predefined components

pca = PCA(n_components=3)

# create principal components by fitting the object to the preprocessed df

principalComponents = pca.fit_transform(df2)

# create a PCA dataframe

df = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# inspect amount of explained variance

print(pca.explained_variance_ratio_)



#create a covariance matrix

print('NumPy covariance matrix: \n%s' %np.cov(df.T))

#create desciptive index:

ind = [col_description[el] for el in train_raw_sub.columns]

print(ind)

print(pca.components_)

pca_cov = pd.DataFrame(pca.components_.T, index = ind, columns = ['pc1','pc2','pc3'])

print(pca_cov)
# Preprocessing the whole dataset to be PCA'd



df = train_raw.copy()

df_num = df.select_dtypes(exclude = 'object')

df_cat = df.select_dtypes(include = 'object')



# impute cat

imp = SimpleImputer(strategy = 'most_frequent')

df_cat2 = pd.DataFrame(imp.fit_transform(df_cat))

df_cat2.columns = df_cat.columns

print(df_cat2.head(10))



# Encode labels

from sklearn.preprocessing import LabelEncoder



LE = LabelEncoder()



# create list of all columns that should be labelencoded

enc_list = [col for col in df.select_dtypes(include = 'object')]



# encode each column

for col in enc_list:

    df[col] = LE.fit_transform(df_cat2[col])



df_prep = df.copy()

    

# check if everything worked right

print(df.head())

print(df.info())

print(df.describe())
# next we have to standardize all the columns to be able to conduct a PCA

# standardize

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



# apply StandardScaler over whole dataset

df_std = sc.fit_transform(df_prep)

# reattach columns

df_std = pd.DataFrame(df_std, columns = df_prep.columns)

# drop ID column

df_std.drop('Id', axis=1, inplace=True)

# drop still missing cols # WHY DO THEY MISS?

missing_cols = [col for col in df_std if df_std[col].isnull().any()]

df_std.drop(missing_cols, axis = 1, inplace = True)

# inspect results

print(df_std)

print(df_std.describe())
# we are ready to conduct the PCA

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import numpy as np



pca = PCA()

pca.fit(df_std)



plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')



# starting by simple seaborn plots

sns.relplot(x='YearBuilt', #relplot plots 'relationships' between two variables

            y='SalePrice',

            hue = 'FullBath', # hue (Farbton) allows to change the color by mapping to an independent variable

            #style = 'FullBath', # style allows to change the marker of the datapoints, by mapping to a variable. it can help highlighting the hue variable or plot another variable instead

            #size = 'FullBath',# size allows to change the size of the datapoints, by mapping to a variable

            #alpha = 0.5, # alpha allows to make the points transparent (0 = invisible, 1 = full visibility)

            data=train_raw_sub,

            kind = 'scatter',) # the kind argument allows to specify the kind of plot we want to create

graph1 = sns.relplot(x='YearBuilt',y='SalePrice',kind='line',data=train_raw_sub)

graph1.fig.autofmt_xdate() # by using the autoformat method, we can format the axis automatically to make it depict e.g. time data

# as there are multiple measurements at each value of x, seaborn automatically plots the mean and the 95% confidence interval around it to depict the distribution

# the confidence intervals are calculated by bootstrapping and can be disabled by the ci argument:

graph2 = sns.relplot(x='YearBuilt', y='SalePrice', kind = 'line', ci=False, data = train_raw_sub) #by passing ci=False, we inform seaborn to just depict the mean and not show the confidence intervals

graph2.fig.autofmt_xdate()

# we can also plot the std instead of the CI's , which would be particularly interesting for this dataset:

graph3 = sns.relplot(x='YearBuilt', y='SalePrice', kind = 'line', ci = 'sd', data = train_raw_sub)
# again it is also possible to easily split the graph by mapping to a third variable. Wen can use the same arguments as for scatter plots:

sns.relplot(x = 'YearBuilt', y = 'SalePrice', kind = 'line', ci = 'sd', hue = 'FullBath', data = train_raw_sub)

# as we can see, that the plot is a little overloaded now and it is difficult to extract useful information. facetting the plot is a good idea then:

print(col_description['FullBath'])

sns.relplot(x = 'YearBuilt',

            y = 'SalePrice', 

            kind = 'line', 

            ci = 'sd', 

            col = 'FullBath',

            col_wrap = 2, # 'wraps' 2 column per row, so that after 2 columns, the rest of the figures are depicted onto the next row

            data = train_raw_sub)
sns.catplot(x = 'FullBath', y = 'SalePrice', kind = 'strip', hue = 'FullBath', alpha = 0.3, data = train_raw_sub)
# we can also change the axis easily and implement a third variable, however just the hue argument is available, while style and size are not:

sns.catplot(x = 'FullBath', 

            y = 'SalePrice',

            kind = 'strip',

            jitter = True,

            alpha = 1,

            hue = 'OverallQual', 

            palette = sns.color_palette("ch:2.5,-.2,dark=.3", # defines a color palette that is used for the hue argument

                                        n_colors = 10), # defines the number of elements in the palette gradient (should fit to the number of possible data values)

            data = train_raw_sub)
# FullBath seems to be skew distributed, with varying frequencies in each category. 

fb1 = sns.catplot(x = 'FullBath', kind = 'count', palette = 'ch:.25', col = 'OverallQual', col_wrap = 4, data = train_raw)

# rescale SalePrice column

SalePriceK = train_raw['SalePrice'].copy()/100000

# rename pd.Series object name, that will be shown in the fig

SalePriceK.name = 'SalePrice in 100K'

# create fig

dist = sns.distplot(SalePriceK, 

                    #kde = False, # would drop the kernel density estimation curve from the figure

                    #hist = False, # would drop the histogram, leaving just the kde curve

                   )

# a jointplot adds histograms to a scatter plot:

sns.jointplot(x = 'OverallQual', y = 'SalePrice', alpha = 0.5, data = train_raw_sub)
# a hexbin plot shows a bivariate histogram by depicting counts of obversations that fall within hexagonal bins

with sns.axes_style('white'):

    sns.jointplot(x = 'LotArea', y = 'SalePrice', kind = 'hex', color = 'k', data = train_raw)
# so far so good. let's practive some relationshis, e.g. regression lines

sns.regplot(x = 'LotArea', y = 'SalePrice', data = train_raw)
def reduce(x):

    if x < 50000:

        return x

    else:

        return np.nan

TidyLotArea = pd.Series([reduce(row) for row in df['LotArea']], name = 'TidyLotArea')

# now repeat the fig without the outliers in the LotArea:

sns.regplot(x = TidyLotArea, y = 'SalePrice', data = train_raw)
# the same is possible for categorical x data

sns.regplot(y = 'SalePrice', x = 'OverallQual', x_jitter = 0.3, data = train_raw)
# as the plot is quite messy, it might be helpful to change the points for the x data to estimators (mean + variance or ci)

sns.regplot(x = 'OverallQual', 

            y = 'SalePrice',

            data = train_raw, 

            x_jitter = 0.3, 

            x_estimator = np.mean)



sns.regplot(x = 'OverallQual', 

            y = 'SalePrice',

           x_jitter = 0.3,

           order = 3, # order specifys the integer that is passed to np.polyfit, which describes the fitting polynomial

            x_estimator = np.mean,

           data = train_raw)
good_ovrl_qual = df['OverallQual'] >=8

good_ovrl_qual.name = 'Good overall Quality'

sns.regplot(y = good_ovrl_qual, x = 'SalePrice', logistic = True, data = train_raw)
good_ovrl_qual = df['OverallQual'] >=8

good_ovrl_qual.name = 'Good overall Quality'

sns.residplot(y = good_ovrl_qual, x = 'SalePrice', data = train_raw)
f1 = sns.lmplot(x = 'YearBuilt', y = 'SalePrice', data = train_raw)

sns.lmplot(x = 'YearBuilt', y = 'SalePrice', hue = 'FullBath', col = 'FullBath', col_wrap = 2, data = train_raw)
f2 = sns.residplot(x = 'YearBuilt', y = 'SalePrice', data = train_raw)