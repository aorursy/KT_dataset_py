# Warnings & Display options

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)





# Load the Data

# We work on a sample to prevent some algorithms from taking too much time. 

path = '../input/mse-pysupport-hackathon/simulation.csv'

df = pd.read_csv(path).sample(frac = 0.1)
# We may work with log SalePrice as that is much less skewed and helps in interpretation

df['TSalePrice'] = np.log1p(df['SalePrice']) 



# Key Variables

df['PastMV'] = df['EstimateLand'] + df['EstimateBuilding']

df['BuildingArea'] = df['BuildingSquareFeet']

df['BedroomShare'] = df['Bedrooms']/df['Rooms']

df['SaleYear'] = df['SaleDate'].str[-4:]



# (Interior) Furnishing Quality - a simple score that is higher if the house has high quality interiors

df['FurnishingQuality'] = 0

df.loc[df.ConstructionQuality == 'Deluxe', 'FurnishingQuality'] += 1

df.loc[df.BasementFinish == 'FormalRecRoom', 'FurnishingQuality'] += 1

df.loc[df.CentralHeating == 'HotWaterStream', 'FurnishingQuality'] += 1

df.loc[df.AtticFinish == 'LivingArea', 'FurnishingQuality'] += 1

df.loc[df.WallMaterial.isin(['Stucco', 'Masonry']), 'FurnishingQuality'] += 1

df.loc[df.RoofMaterial.isin(['Shake', 'Slate', 'Tile']), 'FurnishingQuality'] += 1

df.loc[df.DesignPlan == 'ArchitectPlan', 'FurnishingQuality'] += 1

df.loc[df.RepairCondition == 'AboveAvg', 'FurnishingQuality'] += 1

df.loc[df.Garage1Material.isin(['Stucco', 'Masonry']) |

       df.Garage2Material.isin(['Stucco', 'Masonry']), 'FurnishingQuality'] += 1



# (Interior) Furshining Quantity - a score that captures if the house has a porch, garage, basement, etc.

df['Furnishing'] = 0

df.loc[df.GarageIndicator == 1.0, 'Furnishing'] += 1

df.loc[df.Porch != 'None', 'Furnishing'] += 1

df.loc[df.CathedralCeiling != 'None', 'Furnishing'] += 1

df.loc[df.AtticType != 'None', 'Furnishing'] += 1

df.loc[df.Fireplaces != 0.0, 'Furnishing'] += 1

df.loc[df.Basement.isin(['Full', 'Partial']), 'Furnishing'] += 1
# Both Seem evenly distributed

df.FurnishingQuality.value_counts()
df.Furnishing.value_counts()
# Log transforming SalePrice reveals a bimodal distribution. 

# Some values below k, where log(k) = 5 are too low to be considered. 

import seaborn as sns

import numpy as np

sns.distplot(df.TSalePrice)
# Arms Length Transactions are sales made between family or known members.

# They are more erratic. 

df[['SalePrice', 'ArmsLengthTransaction']].groupby('ArmsLengthTransaction').describe()
# Also remove Condos and Apartments. 

# Condos are close knit shared community living (their price depends on % ownership, and the data for them is lesser).

# Apartments can house multiple families in them (the rooms in them are much larger).

# We retain only Single Family Houses for simplicity. 

df2 = df[(df.TSalePrice > 5) & (df.ArmsLengthTransaction == 1) & 

       (~df.PropertyDesc.isin(['Residential condominium', 

                               'Apartment building with 2 to 6 units, any age', 

                               'Mixed-use commercial/residential building with apartment and commercial area totaling 6 units or less with a square foot area less than 20,000 square feet, any age']))]





#df2 = df2[(df2.TSalePrice < df2.TSalePrice.quantile(.9)) | (df2.TSalePrice > df2.TSalePrice.quantile(.1))]
# Optionally you way want to truncate the data to remove all extreme values based on price

#df2 = df2[(df2.TSalePrice < df2.TSalePrice.quantile(.9)) | (df2.TSalePrice > df2.TSalePrice.quantile(.1))]
# Attributes of Interest - a small set to keep things simple

cols = ['Age', 'BuildingArea', 'BedroomShare', 'Furnishing','FurnishingQuality',

        'OHareNoise','Floodplain', 'RoadProximity', 'SaleYear', 'TownshipName', 'NeighborhoodCode']



#cols = ['BuildingArea', 'Furnishing','FurnishingQuality']

#'OtherRoomsShare','AvgRoomSize', 

x = df2[cols]

#x = df2[list(df2.select_dtypes(exclude = 'object').columns) + ['SaleYear', 'TownshipName', 'NeighborhoodCode']].dropna()



# SalePrice

y = df2[['SalePrice']]



# Filling Nulls - all these are indicator variables with 

na1 = ['OHareNoise','Floodplain', 'RoadProximity']

x[na1] = x[na1].fillna(0)
x.isnull().sum()
# Creating dummies for the three Categorical variables

# Keep the most common categorical value as base

dum = [ 'OHareNoise','Floodplain', 'RoadProximity','TownshipName', 'SaleYear', 'NeighborhoodCode']

dum_base = {}

for i in dum:

    base = str(i) + '_' + str(x[i].value_counts().index[0])

    print(str(i), ' base is ', base)

    x[i] = pd.Categorical(x[i], x[i].value_counts().index)

    x.sort_values(i, inplace = True)

    for col in pd.get_dummies(x[i], drop_first = True, prefix = str(i)).columns:

        dum_base[str(col)] = base

    x = pd.concat([x.drop(i, axis = 1), pd.get_dummies(x[i], drop_first = True, prefix = str(i))], axis = 1, ignore_index = False)

    

x.sort_index(inplace = True)

y.sort_index(inplace = True)
# Sanity Checks



# Zillow claims Cook County area has a market value of 200-250k$

print(y.median())



# Shapes

print(x.shape, y.shape)



x.head()
# We use the GLS to correct for heteroskedasticity, since SalePrice is skewed. 

import numpy as np

import statsmodels.api as sm

from statsmodels.regression.linear_model import GLS

x = sm.add_constant(x, prepend=False)

mod = GLS(y, x)

res = mod.fit()

res.summary()
#res2 = mod.fit_regularized()

numerics = 6

for i in range(len(res.params)-1):

    if i < numerics: 

        var = res.params.index[i]

        coeff = res.params.iloc[i]

        mean = x[var].mean()

        pvalue = res.pvalues[i]

        if pvalue<0.05:

            print(f'Average {var} of a house is {mean:0.2f} and a 10% (about {mean*0.1:0.2f}) increase in {var}, will change house price by {mean*0.1 * coeff:0.2f} dollars')

    if i >= numerics:

        var = res.params.index[i]

        mean = x[var].mean()

        coeff = res.params.iloc[i]

        base = dum_base[var]

        pvalue = res.pvalues[i]

        if pvalue<0.01:

            print(f'For {mean:0.4f} of houses, we have {var}. As compared to {base} the avg house price is greater by {1 * coeff:0.2f} dollars.')

    if i == len(res.params) - 2:

        print('constant')
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

model = CatBoostRegressor(n_estimators = 1000, depth = 8, eval_metric = 'R2', verbose = 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model.fit(x_train, y_train, eval_set =  (x_test, y_test), cat_features = np.where(x.dtypes == 'object')[0])

pd.DataFrame(np.c_[x_train.columns, model.feature_importances_]).sort_values(by = 1, ascending = False)
import shap

shap.initjs()

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(x)
houseNumber = 5450

shap.force_plot(explainer.expected_value, shap_values[houseNumber,:], x.iloc[houseNumber,:])
i = 7000

shap.force_plot(explainer.expected_value, shap_values[i,:], x.iloc[i,:])
# Overall feature importance from Shap

# Similar to catboost native feature importance

shap.summary_plot(shap_values, x, plot_type="bar")
# single feature across the whole dataset

shap.summary_plot(shap_values, x)
import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale, normalize

xPCA = df2[['Age', 'BuildingArea', 'Bedrooms', 'FullBaths','BedroomShare', 'Furnishing','FurnishingQuality']]



# ensure these are lowly correlated variables. 

xPCA.corr()
# Scaling is needed to prevent any feature from dominating the process

x_scaled = normalize(xPCA, axis = 0)



# We compress the data into 3 dimensions

# tSNE can give you a nonlinear decomposition

from sklearn.manifold import TSNE

#pca = TSNE(n_components = 3)

pca = PCA(n_components = 3)



x_transformed = pca.fit_transform(x_scaled)

print('Correl b/w PCA vecs:', np.corrcoef(x_transformed[:, 0], x_transformed[:, 1])[1, 0])
print('Variance explained by pricipal components:', pca.explained_variance_ratio_)
print('Total variance explained:', sum(pca.explained_variance_ratio_))
# Correlation between componants and raw features

# Unable to make out any distinctive details

for j in range(3):

    for i in xPCA.columns: 

        print(f'PCA-{j} with {i}: {np.corrcoef(xPCA[i].values, x_transformed[:, j])[1, 0]}')

    print('\n')
# Scatter plots of features vs components

# Components are uncorrelated

import matplotlib.pyplot as plt

plt.scatter(x_scaled[:, 3], x_scaled[:, 4])

plt.show()



plt.scatter(x_transformed[:, 0], x_transformed[:, 1])

plt.show()
xKMEANS = x_transformed.copy()

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn import cluster, datasets, mixture

from sklearn.metrics import silhouette_samples, silhouette_score as SS

from sklearn.metrics import calinski_harabasz_score as CH, davies_bouldin_score as DB

for i in range(2, 10):

    clusterer = KMeans(n_clusters=i, random_state=10)

    X = normalize(xKMEANS)

    Y = clusterer.fit_predict(X)

    print(i, SS(X, Y), CH(X, Y), DB(X, Y))

# taking the best silouhetter score for clusters = 5

import plotly.express as px

dfPCA = xPCA.copy()

from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=5, random_state=10, algorithm = 'full')

cluster_labels = clusterer.fit_predict(x_transformed)
# Check the Mean of Variables accross clusters, including SalePrice

dfPCA['clusters'] = np.array(cluster_labels)

dfPCA['SalePrice'] = df2['TSalePrice'].values

dfPCA.groupby('clusters').mean()
# Visualise clusters on actual features

from sklearn.cluster import KMeans

fig = px.scatter_3d(dfPCA, x='Age', y='BuildingArea', z='FurnishingQuality', size = 'SalePrice', 

                    color = 'clusters', opacity = 1)

fig.show()
# Visualise clusters on PCA components

# They are more evident here, but less interpretable.

dfPCA_transformed = pd.DataFrame(x_transformed, columns = ['PCA0', 'PCA1', 'PCA2'])

dfPCA_transformed['clusters'] = np.array(cluster_labels)

dfPCA_transformed['SalePrice'] = df2['TSalePrice'].values

from sklearn.cluster import KMeans

fig = px.scatter_3d(dfPCA_transformed, x='PCA0', y='PCA1', z='PCA2', size = 'SalePrice', 

                    color = 'clusters', opacity = 1)

fig.show()
xFACTOR = df2[['Age', 'BuildingArea','LandSquareFeet', 'Bedrooms', 'Rooms', 'FullBaths', 'Furnishing','FurnishingQuality']].dropna()

xFACTOR.head()
import numpy as np

import statsmodels.api as sm

from statsmodels.multivariate.factor import Factor, FactorResults

X = normalize(xFACTOR.values, axis = 0)

mod = Factor(X, 

             n_factor = 2, 

             endog_names = xFACTOR.columns, 

             method = 'pa')

res = mod.fit()



# We "rotate" to enable factors to be correlated with certain features

# https://community.alteryx.com/t5/Data-Science/Ghost-Hunting-Factor-Analysis-with-Python-and-Alteryx/ba-p/566434

FR = FactorResults(mod)



#Rotations available - varimax, quartimax, biquartimax, equamax, oblimin, parsimax, parsimony, biquartimin, promax.

FR.rotate(method = 'oblimin')

print(FR.summary())
# The scree plot tells us how much variance is explained with more and more factors

res.plot_scree(ncomp=4).show()
# We plot the loadings, or the direction of features vis-a-vis 'factors'

# This tells us the direction in which features move

# age, furnishing tend to move together i.e Quality

# area, rooms go together i.e quantity

res.plot_loadings()
# After rotation this is more magnified

FR.plot_loadings()
FACTORS = np.dot(X, mod.loadings)

xFACTOR['FACTOR0'] = FACTORS[:, 0] * 100

xFACTOR['FACTOR1'] = FACTORS[:, 1] * 100

xFACTOR.head()
# Factors 0 correlates to Size variables, while factor 1 correlates to quantity. 

xFACTOR.corr()
xFACTOR.mean()
xFACTOR['HighQuality'] = 0

xFACTOR.loc[xFACTOR.FACTOR1 > 0.592254,  'HighQuality'] = 1

xFACTOR['HighQuantity'] = 0

xFACTOR.loc[xFACTOR.FACTOR0 > 2.663217,  'HighQuantity'] = 1



xFACTOR['Clusters'] = 0

xFACTOR.loc[(xFACTOR.HighQuality == 1) & (xFACTOR.HighQuantity == 0), 'Clusters'] = 1

xFACTOR.loc[(xFACTOR.HighQuality == 0) & (xFACTOR.HighQuantity == 1), 'Clusters'] = 2

xFACTOR.loc[(xFACTOR.HighQuality == 1) & (xFACTOR.HighQuantity == 1), 'Clusters'] = 3

xFACTOR.shape
# Manually create clusters - high and low, quantity and quality

xFACTOR['SalePrice'] = df2['TSalePrice'].values

fig = px.scatter(xFACTOR, x='FACTOR0', y='FACTOR1', color = 'Clusters', opacity = 1)

fig.show()