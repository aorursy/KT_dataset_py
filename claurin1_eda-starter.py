import numpy as np # linear algebra

from scipy.stats import pearsonr

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.api.types import union_categoricals

from matplotlib import pyplot as plt 

import seaborn as sns



from os import listdir

from os import path

from sklearn.decomposition import KernelPCA

train_data = pd.read_csv(path.join("..", "input", "learn-together", "train.csv"))
soil_cols = []

other_cols = []

for x in train_data.columns:

    if x.startswith('Soil'):

        soil_cols.append(x)

    else:

        other_cols.append(x)

train_data.head()
train_descriptives = train_data.describe()

print(train_descriptives.filter(items = other_cols))
#Missingness check 

print('It is {:s} that there are missing data.'.format(

    str(any(train_descriptives.loc['count'] != train_descriptives.loc['count'].Id))))
sns.distplot(train_data.Cover_Type, kde = False, rug = False)
#Are there any patches where there are multiple soil types?

soil_frame_train = train_data.filter(items = soil_cols)

print('It is {:s} that each patch has one and only one soil type.'.format(str(all(soil_frame_train.agg('sum', axis = 'columns') == 1))))
(soil_frame_train.agg('mean')).sort_values()
pearsonr(train_data.Soil_Type32, train_data.Soil_Type33)
#Elevation histogram

sns.distplot(train_data.Elevation, kde = False, rug = True)
wilderness_category = (train_data.filter(like = "Wildern").apply(axis = 1, func = np.flatnonzero) )

wilderness_category = (wilderness_category.astype('int32')).astype('category')

train_data['wilderness_cat'] = wilderness_category
sns.boxplot(data = train_data, x = "wilderness_cat", y = "Elevation", notch = True)
sns.swarmplot(data = train_data, x = "wilderness_cat", y = "Elevation")
#Aspect histogram

sns.distplot(train_data.Aspect, kde = False, rug = False)
#Slope histogram

sns.distplot(train_data.Slope  , kde = True, rug = False)
sns.pairplot(train_data.filter(items = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points', 'Elevation']))
(train_data.filter(items = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Elevation'])).corr()
sns.pairplot(train_data.filter(items = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']))
(train_data.filter(items = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'])).corr()
(train_data.filter(items = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'])).cov()
pca = KernelPCA(

                n_components = 3)

solar_pca = pca.fit_transform(train_data.filter(items = ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']))
pca.lambdas_
sns.pairplot(pd.DataFrame(solar_pca))
def crosstabber(soil_column):#

    ## add try catch for 

    try:

        return(np.ravel(pd.crosstab(train_data[soil_column], train_data.Cover_Type).iloc[1].astype('int')))

    except: 

        return(np.zeros(7).astype('int'))
print(crosstabber("Soil_Type3"))

soil_dictionary = {}

for stype in soil_cols:

    soil_dictionary[stype] = crosstabber(stype)



soil_tabulation  = (pd.DataFrame(soil_dictionary)).T
soil_tabulation