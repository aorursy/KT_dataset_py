#Imports
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 10

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import seaborn as sns

pd.set_option("display.max_columns",100)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_train = train_data["SalePrice"]

def plot_ann_barh(series, xlim=None, title=None, size=(16,10)):
    """Return axes for a barh chart from pandas Series"""
    #required imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    #setup default values when necessary
    if xlim == None: xlim=series.max()
    if title == None: 
        if series.name == None: title='Title is required'
        else: title=series.name
    
    #create barchart
    ax = series.plot(kind='barh', title=title, xlim=(0,xlim), figsize=size, grid=False)
    sns.despine(left=True)
    
    #add annotations
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+(xlim*0.01), i.get_y()+.38, \
                str(i.get_width()), fontsize=10,
    color='dimgrey')
    
    #the invert will order the data as it is in the provided pandas Series
    plt.gca().invert_yaxis()
    
    return ax
test_data_idx = len(train_data)
combined_df = pd.concat([train_data.set_index("Id").drop("SalePrice", axis=1),test_data.set_index("Id")])
quantitative = [f for f in combined_df.columns if combined_df.dtypes[f] != 'object']
qualitative = [f for f in combined_df.columns if combined_df.dtypes[f] == 'object']
# MSSubClass feature is a float but should be treated as qualitative 
quantitative.remove('MSSubClass')
qualitative.append('MSSubClass')
print("Training set has {:d} rows and {:d} features.".format(train_data.shape[0],train_data.shape[1]))
print("Test set has {:d} rows and {:d} features.".format(test_data.shape[0],test_data.shape[1]))
print("Combined set has {:d} rows and {:d} features.".format(combined_df.shape[0],combined_df.shape[1]))
print("Quantitative features: ")
print(quantitative)
print("Qualitative features: ")
print(qualitative)
valid_na_features = ["Alley", 
                     "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", 
                     "FireplaceQu", 
                     "GarageType","GarageFinish", "GarageQual", "GarageCond", 
                     "PoolQC", 
                     "Fence",
                     "MiscFeature"]
na_feature_replace = ["NoAlley", 
                     "NoBsmt", "NoBsmt", "NoBsmt", "NoBsmt", "NoBsmt", 
                     "NoFireplace", 
                     "NoGarage","NoGarage", "NoGarage", "NoGarage", 
                     "NoPool", 
                     "NoFence",
                     "NoMiscFeature"]
for col,ft_replace in zip(valid_na_features,na_feature_replace):
    combined_df[col].fillna(ft_replace, inplace = True)
    combined_df[col].fillna(ft_replace, inplace = True)
    
#"Wd Shng" should be "WdShing" (maybe could be "Wd Sdng" but guessing other way.)
combined_df.loc[combined_df["Exterior2nd"] == "Wd Shng", "Exterior2nd"] = "WdShing"
# CmentBd should be CemntBd
combined_df.loc[combined_df["Exterior2nd"] == "CmentBd", "Exterior2nd"] = "CemntBd"
# Brk Cmn should be BrkComm
combined_df.loc[combined_df["Exterior2nd"] == "Brk Cmn", "Exterior2nd"] = "BrkComm"

# there's a few other checks need to be done, 
# ex if there's no basement make sure basement other basement features are all "NoBsmt"

# Show distribution of missing data
msno.matrix(combined_df)
missing_cnt = combined_df.isnull().sum()
# Show only columns with non-zero null counts
missing = missing_cnt[missing_cnt > 0] 
# plot series result
ax = plot_ann_barh(missing.sort_values(ascending=False), 
                   xlim=max(missing), title='Count of Null values for each columns')
# BsmtUnfSF, BsmtFinSF2, TotalBsmtSF, BsmtFinSF1
combined_df.loc[combined_df[["BsmtUnfSF","BsmtFinSF2","TotalBsmtSF","BsmtFinSF1","BsmtFullBath","BsmtHalfBath"]].isnull().any(axis=1),:]
# 2 properties with basement features need to be imputed; both don't have a basement, so that's easy!
combined_df["BsmtUnfSF"].fillna(0, inplace=True)
combined_df["BsmtFinSF1"].fillna(0, inplace=True)
combined_df["BsmtFinSF2"].fillna(0, inplace=True)
combined_df["TotalBsmtSF"].fillna(0, inplace=True)
combined_df["BsmtFullBath"].fillna(0, inplace=True)
combined_df["BsmtHalfBath"].fillna(0, inplace=True)
# check the result
#combined_df.loc[combined_df[["BsmtUnfSF","BsmtFinSF2","TotalBsmtSF","BsmtFinSF1","BsmtFullBath","BsmtHalfBath"]].isnull().any(axis=1),:]
# GarageCars, GarageArea
combined_df.loc[combined_df[["GarageCars","GarageArea"]].isnull().any(axis=1),:]
# The property doesn't have a garage but GarageType is set to detached. Need to fix this one also.
# Should scrub the entire dataset and make sure other features aren't labeled incorrectly like this.
combined_df.loc[combined_df[["GarageCars","GarageArea"]].isnull().any(axis=1),"GarageType"] = "NoGarage"
combined_df["GarageCars"].fillna(0, inplace=True)
combined_df["GarageArea"].fillna(0, inplace=True)
#combined_df.loc[combined_df[["GarageCars","GarageArea"]].isnull().any(axis=1),:]
# KitchenQual
combined_df.loc[combined_df["KitchenQual"].isnull(),:]
# look at other properties in the same neighborhood with same overall quality to impute KitchenQual
combined_df.loc[(combined_df["Neighborhood"] == "ClearCr") & 
                (combined_df["OverallQual"].between(4,6)) & 
                (combined_df["YearRemodAdd"].between(1945,1960)),
                ("OverallQual","OverallCond","YearRemodAdd","KitchenQual","GarageQual","ExterQual","BsmtQual","FireplaceQu","PoolQC","Fence")]
# Most properties similar to this based on the matching criteria are in Gd/TA quality. Choose TA since that's more typical.
combined_df["KitchenQual"].fillna("TA", inplace = True)
# SaleType
combined_df.loc[combined_df["SaleType"].isnull(),:]
# Check out other properties sold in same year
combined_df.loc[combined_df["YrSold"] == 2007,"SaleType"].value_counts()
# Check out the "other" saletypes for fun... Maybe the NaN should be Other...
combined_df.loc[(combined_df["YrSold"] == 2007) & (combined_df["SaleType"] == "Oth"),:]
# Both "Other" saletypes are abnormal, while the NaN property has normal salecondition. Are all "other" saletypes abnormal?
combined_df.loc[combined_df["SaleType"] == "Oth","SaleCondition"].value_counts()
# "Other" SaleType is pretty rare and most are "abnormal" SaleCondition. 
# Probably using the most common SaleType with Normal SaleCondition from 2007 would be best guess.
combined_df.loc[(combined_df["YrSold"] == 2007) & (combined_df["SaleCondition"] == "Normal"),"SaleType"].value_counts()
combined_df["SaleType"].fillna("WD", inplace = True)
# Exterior1st, Exterior2nd
combined_df.loc[combined_df[["Exterior1st","Exterior2nd"]].isnull().any(axis=1),:]
# OK, look at other houses in the neighborhood, built around same time, with a recent remodel
combined_df.loc[(combined_df["Neighborhood"] == "Edwards") &
                (combined_df["YearBuilt"].between(1935,1950)) &
                (combined_df["YearRemodAdd"] > 1995),:]
# What about with same rooftype...
#(combined_df["Neighborhood"] == "Edwards") &
#(combined_df["YearBuilt"].between(1935,1950)) &
                                

combined_df.loc[(combined_df["RoofStyle"] == "Flat") &
                (combined_df["RoofMatl"] == "Tar&Grv"),"Exterior2nd"].value_counts()
# Plywood is most common exterior for houses with this type of roof, so will go with that as best guess...
combined_df["Exterior1st"].fillna("Plywood", inplace = True)
combined_df["Exterior2nd"].fillna("Plywood", inplace = True)
# Utilities
combined_df.loc[combined_df[["Utilities"]].isnull().any(axis=1),:]
# Both have gas heating, 2nd has AC, both have electrical. Look at other houses in the neighborhoods.
print(combined_df.loc[(combined_df["Neighborhood"] == "IDOTRR"),"Utilities"].value_counts())
print(combined_df.loc[(combined_df["Neighborhood"] == "Gilbert"),"Utilities"].value_counts())
# both neighborhoods have "AllPub" utilities for all properties, so impute with that value.
combined_df["Utilities"].fillna("AllPub", inplace = True)
# Functional
combined_df.loc[combined_df[["Functional"]].isnull().any(axis=1),:]
# both properties have abnormal salecondition so let's look at other properties w/ this condition.
combined_df.loc[(combined_df["SaleCondition"] == "Abnorml") & 
                (combined_df["OverallQual"].between(4,6)),
                ("Functional","SaleType","GarageQual","BsmtQual","ExterQual","OverallQual","OverallCond")]
combined_df.loc[(combined_df["SaleCondition"] == "Abnorml") & (combined_df["OverallQual"] < 6),"Functional"].value_counts()
# Seems safe to use Typ for Imputing missing "functional" features
combined_df["Functional"].fillna("Typ", inplace = True)
# MsZoning
combined_df.loc[combined_df[["MSZoning"]].isnull().any(axis=1),:]
# Check out the zoning for other properties in the neighborhoods.
combined_df.loc[(combined_df["Neighborhood"] == "IDOTRR","MSZoning")].value_counts()
# These are all residential properties so the missing zoning should be RM since all other residentials in IDOTRR are RM.
combined_df.loc[((combined_df["Neighborhood"] == "IDOTRR") &
                (combined_df[["MSZoning"]].isnull().any(axis=1))), "MSZoning"] = "RM"
# double check the counts. should be 71 now.
combined_df.loc[(combined_df["Neighborhood"] == "IDOTRR","MSZoning")].value_counts()
combined_df.loc[(combined_df["Neighborhood"] == "Mitchel","MSZoning")].value_counts()
# Most properties in the Mitchel neighborhood are zoned as RL. We can check the property size of the few RM properties to see
# if this matches up with what needs to be imputed, otherwise take RL.
combined_df.loc[(combined_df["Neighborhood"] == "Mitchel") & 
                 (combined_df["MSZoning"] == "RM")]
# All the RM have much lower LotFrontage so using RL for our imputed value seems safe.
combined_df.loc[((combined_df["Neighborhood"] == "Mitchel") &
                (combined_df[["MSZoning"]].isnull().any(axis=1))), "MSZoning"] = "RL"
# check the counts... should be 105 RL now
combined_df.loc[(combined_df["Neighborhood"] == "Mitchel","MSZoning")].value_counts()
## MasVnrType, MasVnrArea
# If properties have both of these values missing, presume that the feature doesn't apply. Set Area to 0 and type should be "None".
# There's 1 property that has MasVnrType missing but Area not missing, check this one out first.
combined_df.loc[(combined_df["MasVnrType"].isnull() & combined_df["MasVnrArea"].notnull()),:]
## Since MasVnrArea is non-zero we should impute using similar properties from the neighborhood built around same timeframe.
#combined_df.loc[((combined_df["Neighborhood"] == "Mitchel") & (combined_df["YearBuilt"] == 1961)),:]
combined_df.loc[((combined_df["Neighborhood"] == "Mitchel")& (combined_df["Exterior1st"] == "Plywood")),:]

combined_df.loc[(combined_df["MasVnrType"].isnull() & combined_df["MasVnrArea"].notnull()),"MasVnrType"] = "BrkFace"
## Now fix remaining missing values
combined_df["MasVnrType"] = combined_df["MasVnrType"].fillna("None")
combined_df["MasVnrArea"] = combined_df["MasVnrArea"].fillna(0)
combined_df.loc[combined_df["Electrical"].isnull(),:]
combined_df.loc[(combined_df["YearBuilt"] == 2006) & (combined_df["Neighborhood"] == "Timber"),:]
# All other houses in this neighborhood built in same year had "SBrkr" electrical type. So replace missing value with SBrkr.
combined_df["Electrical"] = combined_df["Electrical"].fillna("SBrkr")
# GarageYrBlt
print(combined_df.loc[combined_df["GarageYrBlt"].isnull(),"GarageType"].value_counts())
print(combined_df.loc[combined_df["GarageYrBlt"].isnull(),"GarageFinish"].value_counts())

# Fill in missing GarageYrBlt with YearBuilt field. 
# Fill with 0 may be another option but I feel that may throw a model off. 
# Further this is missing only when a property doesn't have a garage (feature doesn't apply) 
# and we have a categorical feature that will indicate when there's no garage.
combined_df["GarageYrBlt"].fillna(combined_df["YearBuilt"], inplace = True)
# Fix the property with "detchd" garagetype, that doesn't have a garage
combined_df.loc[((combined_df["GarageType"] == "Detchd") & (combined_df["GarageFinish"] == "NoGarage")),"GarageType"] = "NoGarage"
# Check that the fill is correct...
#train_data.loc[train_data["GarageType"] == "NoGarage",["GarageYrBlt","YearBuilt"]]
combined_df[["LotFrontage","LotArea"]].describe()
combined_df["LotFrontage"].hist(bins=20)
combined_df["LotArea"].hist(bins=20)
combined_df.loc[((combined_df["LotArea"] > 75000) | (combined_df["LotFrontage"] > 175)),:]
# Look at top 5 frontage/area properties in these neighborhoods for reference.
groups = combined_df.groupby("Neighborhood")
ft_list = ["MSZoning","LotFrontage","LotArea","LotShape","LotConfig","GrLivArea","TotRmsAbvGrd"]
for neighborhood in ["ClearCr","Timber","NAmes","Gilbert","Edwards","Mitchel"]:
    print("{}:".format(neighborhood))
    print(groups.get_group(neighborhood)[ft_list].sort_values("LotArea", na_position = "first").tail(5))
# The two properties in ClearCr look anomalous. lotSize is about 10x larger than the other properties in the neighborhood, 
# but the house size is about the same. These could really be huge properties, but also fishy that they are both on a Cul-de-sac....
# There could have been a typo in data entry for LotArea of these properties. Dividing by 10 would bring them more in line with other
# properties in the neighborhood.
# 314 and 336 in Timber look like possible data entry error also.
# ID 314's LotFrontage isn't imputed but its area is about 10x larger than other properties.
# Its frontage is high (150) but there's another property with 149 and about 1/10th the property size.
# 2251 looks out of place also. It's zoned for medium density but is 3x the size of next largest property, which is also RM. 
# It looks like it should be 5600 instead of 56000 and there is another RM zoned property with 5600 plot size also.
# fix these anomalous properties...

combined_df.loc[250,"LotArea"] *= 0.1
combined_df.loc[707,"LotArea"] -= 100000
combined_df.loc[314,"LotArea"] = 21525
combined_df.loc[336,"LotArea"] = 16460
combined_df.loc[2251,"LotArea"] = 5600


# There are some fishy LotFrontage values also but will leave them alone for now.
# Now how do they look.
combined_df[["LotFrontage","LotArea"]].describe()
# Make a new categorical variable indicating LotFrontage was imputed or not. 
# All missing values will be imputed.
combined_df["LotFtg_IsImputed"] = combined_df["LotFrontage"].apply(lambda x: 0 if x > 0 else 1)

# This one was fixed; indicate it was imputed
combined_df.loc[314,"LotFtg_IsImputed"] = 1
# First, lots of same area/shape/config have high confidence that frontage will be same also, so we can copy based on that grouping.
lot_groups = combined_df.groupby(["LotArea","LotShape","LotConfig"])
#for name, group in lot_groups:
#    print(name)
#    print(len(group))
combined_df["LotFrontage"] = lot_groups["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# Next: Create new features - LotWidth, LotDepth, LotAR, LotSize.
# LotWidth is intended to represent width of the lot. (calculated from LotFrontage.)
# LotDepth is intended to represent lot depth from the street.
# LotAR is intended to represent lot aspect ratio (width/depth)
# LotSize - Bin lots by size (lotArea) so we can group similarly, but not exactly, sized lots.
#LotConfig: Lot configuration
#       Inside	Inside lot
#       Corner	Corner lot
#       CulDSac	Cul-de-sac
#       FR2	Frontage on 2 sides of property
#       FR3	Frontage on 3 sides of property
# corner lots should have frontage one 2 sides so need to divide frontage by 2 before calculating.
# Todo: Need to verify this assumption.

# LotWidth
combined_df.loc[combined_df["LotConfig"] == "Inside","LotWidth"] = combined_df.loc[combined_df["LotConfig"] == "Inside","LotFrontage"]
combined_df.loc[combined_df["LotConfig"] == "Corner","LotWidth"] = combined_df.loc[combined_df["LotConfig"] == "Corner","LotFrontage"]/2
combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotWidth"] = combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotFrontage"]
combined_df.loc[combined_df["LotConfig"] == "FR2","LotWidth"] = combined_df.loc[combined_df["LotConfig"] == "FR2","LotFrontage"]/2
combined_df.loc[combined_df["LotConfig"] == "FR3","LotWidth"] = combined_df.loc[combined_df["LotConfig"] == "FR3","LotFrontage"]/3

# LotDepth
# Todo: CulDSac depth could be calculated more accurately using donut area approximation
combined_df.loc[combined_df["LotConfig"] == "Inside","LotDepth"] = combined_df.loc[combined_df["LotConfig"] == "Inside","LotArea"] / combined_df.loc[combined_df["LotConfig"] == "Inside","LotFrontage"]
combined_df.loc[combined_df["LotConfig"] == "Corner","LotDepth"] = combined_df.loc[combined_df["LotConfig"] == "Corner","LotArea"] / (combined_df.loc[combined_df["LotConfig"] == "Corner","LotFrontage"]/2)
combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotDepth"] = combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotArea"] / combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotFrontage"]
combined_df.loc[combined_df["LotConfig"] == "FR2","LotDepth"] = combined_df.loc[combined_df["LotConfig"] == "FR2","LotArea"] / (combined_df.loc[combined_df["LotConfig"] == "FR2","LotFrontage"]/2)
combined_df.loc[combined_df["LotConfig"] == "FR3","LotDepth"] = combined_df.loc[combined_df["LotConfig"] == "FR3","LotArea"] / (combined_df.loc[combined_df["LotConfig"] == "FR3","LotFrontage"]/3)

# LotSize bins
# Will use equal-sized bins and arbitrarily pick bin count as 8. That puts bin size around 8,750 sqft. 
combined_df["LotSize"] = pd.cut(combined_df["LotArea"],8, labels = False)

# round LotWidth down to nearest integer
combined_df["LotWidth"] = np.floor(combined_df["LotWidth"])
# round LotDepth to nearest 5ft resolution
def round_down_to_nearest(self, n):
    return (self // n) * n
# end def
combined_df["LotDepth"] = combined_df["LotDepth"].apply(lambda x: round_down_to_nearest(x, 5))

missing_cnt = combined_df.isnull().sum()
missing = missing_cnt[missing_cnt > 0] 
print(missing)
# Now presumably LotDepth will have less variance across properties within the same neighborhood.
# We can focus on imputing LotDepth then calculate LotFrontage, width, AR based on imputed LotDepth feature.
# We want to first impute LotDepth based on neighborhood, zoning and the size bins, then calculate LotFrontage, width, AR.
# Maybe even impute LotDepth from neighborhood and zoning only...
# Another grouping maybe to group by LotShape instead of neighborhood
combined_df["LotDepth"] = combined_df.groupby(["MSZoning","Neighborhood","LotSize"])["LotDepth"].transform(lambda x: x.fillna(x.median()))
missing_cnt = combined_df.isnull().sum()
missing = missing_cnt[missing_cnt > 0] 
print(missing)
# Inspect the remaining properties...
combined_df.loc[(combined_df["LotDepth"].isnull()),("MSZoning","LotShape","LotFrontage","LotArea","LotDepth","LotAR","LotConfig","LandContour","LandSlope","Neighborhood","HouseStyle")]
# OK now impute remaining properties across neighborhoods by grouping by zoning, lotsize and shape...
combined_df["LotDepth"] = combined_df.groupby(["MSZoning","LotShape","LotSize"])["LotDepth"].transform(lambda x: x.fillna(x.median()))
missing_cnt = combined_df.isnull().sum()
missing = missing_cnt[missing_cnt > 0] 
print(missing)
# Inspect the remaining properties...
combined_df.loc[(combined_df["LotDepth"].isnull()),("MSZoning","LotShape","LotFrontage","LotArea","LotDepth","LotAR","LotConfig","LandContour","LandSlope","Neighborhood","HouseStyle")]
# OK now impute remaining properties across neighborhoods by grouping by zoning and lotsize...
combined_df["LotDepth"] = combined_df.groupby(["MSZoning","LotSize"])["LotDepth"].transform(lambda x: x.fillna(x.median()))
missing_cnt = combined_df.isnull().sum()
missing = missing_cnt[missing_cnt > 0] 
print(missing)
# Finally, calculate missing LotFrontage, width, AR based on imputed LotDepth.
# LotWidth
combined_df.loc[(combined_df["LotWidth"].isnull() & combined_df["LotDepth"].notnull()),"LotWidth"] = combined_df.loc[(combined_df["LotWidth"].isnull() & combined_df["LotDepth"].notnull()),"LotArea"] / combined_df.loc[(combined_df["LotWidth"].isnull() & combined_df["LotDepth"].notnull()),"LotDepth"]
combined_df["LotWidth"] = np.floor(combined_df["LotWidth"])

# Now calculate AR with rounded values., and round it to 2 digits
combined_df.loc[combined_df["LotConfig"] == "Inside","LotAR"] = ((combined_df.loc[combined_df["LotConfig"] == "Inside","LotWidth"]) / combined_df.loc[combined_df["LotConfig"] == "Inside","LotDepth"]).round(2)
combined_df.loc[combined_df["LotConfig"] == "Corner","LotAR"] = ((combined_df.loc[combined_df["LotConfig"] == "Corner","LotWidth"]) / combined_df.loc[combined_df["LotConfig"] == "Corner","LotDepth"]).round(2)
combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotAR"] = ((combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotWidth"]) / combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotDepth"]).round(2)
combined_df.loc[combined_df["LotConfig"] == "FR2","LotAR"] = ((combined_df.loc[combined_df["LotConfig"] == "FR2","LotWidth"]) / combined_df.loc[combined_df["LotConfig"] == "FR2","LotDepth"]).round(2)
combined_df.loc[combined_df["LotConfig"] == "FR3","LotAR"] = ((combined_df.loc[combined_df["LotConfig"] == "FR3","LotWidth"]) / combined_df.loc[combined_df["LotConfig"] == "FR3","LotDepth"]).round(2)

# LotFrontage
combined_df.loc[combined_df["LotConfig"] == "Inside","LotFrontage"] = combined_df.loc[combined_df["LotConfig"] == "Inside","LotWidth"]
combined_df.loc[combined_df["LotConfig"] == "Corner","LotFrontage"] = combined_df.loc[combined_df["LotConfig"] == "Corner","LotWidth"]*2
combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotFrontage"] = combined_df.loc[combined_df["LotConfig"] == "CulDSac","LotWidth"]
combined_df.loc[combined_df["LotConfig"] == "FR2","LotFrontage"] = combined_df.loc[combined_df["LotConfig"] == "FR2","LotWidth"]*2
combined_df.loc[combined_df["LotConfig"] == "FR3","LotFrontage"] = combined_df.loc[combined_df["LotConfig"] == "FR3","LotWidth"]*3

missing_cnt = combined_df.isnull().sum()
missing = missing_cnt[missing_cnt > 0] 
print(missing)
# Display the correlation heatmap
corr = train_data.drop(['Id'], axis=1).corr()

#plt.figure(figsize = (16,10))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax1 = sns.heatmap(corr, mask=mask)

corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)
#plt.figure(figsize = (16,10))
f = pd.melt(combined_df.loc[:test_data_idx,:].reset_index().join(y_train), value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
def violinplot(x, y, **kwargs):
    sns.violinplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(combined_df.loc[:test_data_idx,:].reset_index().join(y_train), id_vars=['SalePrice'], value_vars=qualitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(violinplot, "value", "SalePrice")

import scipy.stats as stats
def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

#plt.figure(figsize = (16,10))
a = anova(combined_df.loc[:test_data_idx,:].reset_index().join(y_train))
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)


# condition1,2 need to be combined when 1-hot encoded
# same with exterior1st/2nd
encoded_df = pd.get_dummies(combined_df.drop(["Condition1","Condition2","Exterior1st","Exterior2nd"], axis = 1))
conditions = combined_df["Condition1"].unique().tolist()
# manually 1-hot encode Condition1/2
for cond in conditions:
    name = "Cond_" + cond
    #print("name: {}".format(name))
    encoded_df[name] = combined_df.loc[:,("Condition1","Condition2")].isin([cond]).any(axis=1)
    # convert bool to 0/1
    encoded_df[name] = encoded_df[name].apply(lambda x: 1 if x else 0)
# end for
#print(combined_df["Condition1"].unique().tolist())
#print(combined_df["Exterior1st"].unique().tolist())
#print(combined_df["Exterior2nd"].unique().tolist())
exteriors = combined_df["Exterior2nd"].unique().tolist()
exteriors.extend(combined_df["Exterior1st"].unique().tolist())
exteriors = list(set(exteriors))
for ext in exteriors:
    name = "Ext_" + ext
    #print("name: {}".format(name))
    encoded_df[name] = combined_df.loc[:,("Exterior1st","Exterior2nd")].isin([ext]).any(axis=1)
    # convert bool to 0/1
    encoded_df[name] = encoded_df[name].apply(lambda x: 1 if x else 0)
# end for
# remove any spaces from column names, replace with '_'
encoded_df.columns = encoded_df.columns.str.replace('\s+', '_')
encoded_df.head()



import scipy.stats as st
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y_train, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y_train, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y_train, kde=False, fit=st.lognorm)
def johnson(y):
    gamma, eta, epsilon, lbda = stats.johnsonsu.fit(y)
    yt = gamma + eta*np.arcsinh((y-epsilon)/lbda)
    return yt, gamma, eta, epsilon, lbda

def johnson_inverse(y, gamma, eta, epsilon, lbda):
    return lbda*np.sinh((y-gamma)/eta) + epsilon

# yt is saleprice transformed using johnsonSU
yt, g, et, ep, l = johnson(y_train)
# yt2 is yt transformed back using inverse function to get original.
yt2 = johnson_inverse(yt, g, et, ep, l)
plt.figure(1); plt.title('SalePrice transformed using JohnsonSU')
sns.distplot(yt)
plt.figure(2); plt.title('Inverse transform function (to verify)')
sns.distplot(yt2)
plt.figure(3); plt.title('Original SalePrice values')
sns.distplot(y_train)

def log_transform(df, col):
    df[col] = np.log1p(df[col].values)

def quadratic(df, col):
    df[col+'2'] = df[col]**2
  
# Transform 
log_transform(encoded_df,'GrLivArea')
log_transform(encoded_df,'1stFlrSF')
log_transform(encoded_df,'2ndFlrSF')
log_transform(encoded_df,'TotalBsmtSF')
log_transform(encoded_df,'LotArea')
log_transform(encoded_df,'LotFrontage')
log_transform(encoded_df,'KitchenAbvGr')
log_transform(encoded_df,'GarageArea')



# quantitative features should have gaussian distribution with 0 mean and unit variance
# for models to perform well.
from sklearn import preprocessing

# use StandardScaler() to normalize data on training set. the same transformation needs to be done later on test set also.
# Need to fit based only on training set, then transform both training and test sets.
# And scaling only quantitative features.
scaler = preprocessing.StandardScaler().fit(encoded_df.loc[:test_data_idx,quantitative])
encoded_df[quantitative] = scaler.transform(encoded_df[quantitative])
# Verify all quantitative features are normalized with range between -1,1
# todo
encoded_df[quantitative].describe()
encoded_df['IsNew'] = encoded_df['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)
encoded_df['HasBasement'] = encoded_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
encoded_df['HasGarage'] = encoded_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
encoded_df['Has2ndFloor'] = encoded_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
encoded_df['HasMasVnr'] = encoded_df['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
encoded_df['HasWoodDeck'] = encoded_df['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
encoded_df['HasPorch'] = encoded_df['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
encoded_df['HasPool'] = encoded_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

# Some utility functions and imports
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LassoLarsCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
import xgboost as xgb

# variables
X_train = encoded_df.loc[:test_data_idx,:].reset_index().drop('Id',axis=1)
X_test = encoded_df.loc[test_data_idx:,:].reset_index().drop('Id',axis=1)
#y = y_train
# yt is saleprice transformed using johnsonSU
#yt, g, et, ep, l = johnson(y_train)
# yt2 is yt transformed back using inverse function to get original.
#yt2 = johnson_inverse(yt, g, et, ep, l)

model_names = ["LassoLarsCV","RidgeCV","DecisionTreeRegressor","RandomForestRegressor","GradientBoostingRegressor","XGBRegressor"]
# models
LassoModel = LassoLarsCV(max_iter=10000) # max_iter=10000
RidgeModel = RidgeCV(cv=10) # cv=10
DTRModel = DecisionTreeRegressor()
RFModel = RandomForestRegressor()
GBMRModel = GradientBoostingRegressor()
XGMRModel = XGBRegressor()
models_list = [LassoModel, RidgeModel, DTRModel, RFModel, GBMRModel, XGMRModel]

for name, model in zip(model_names,models_list):
    score = cross_val_score(model, X_train, yt, n_jobs = -1, scoring = 'neg_mean_squared_error')
    # lower is better
    print("{:25s} RMSE score: {}".format(name,np.sqrt(-1*score).mean()))
def modelfit(model, X, y, useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    '''Helper function to fit XGBoost model parameters.'''
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(X.values, label = y.values)
        cvresult = xgb.cv(xgb_param, 
                          xgtrain, 
                          num_boost_round = model.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          metrics='rmse', # Todo - learn more / parameterize
                          verbose_eval = False,
                          early_stopping_rounds = early_stopping_rounds)
        print("Optimal n_estimators: {}".format(cvresult.shape[0]))
        model.set_params(n_estimators = cvresult.shape[0])
    
    #Fit the modelorithm on the data
    model.fit(X, y,eval_metric='rmse')
        
    #Predict training set:
    X_predictions = model.predict(X)
    #X_predprob = model.predict_proba(X)[:,1]
        
    #Print model report:
    #print("\nModel Report")
    # print("Accuracy : {:4f}".format(metrics.accuracy_score(y.values, X_predictions)))
    # todo - learn more / parameterize scoring
    print("RMSE Score (Train): {:4f}".format(np.sqrt(metrics.mean_squared_error(y, X_predictions))))
#    feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp = pd.Series(model.Booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.figure(figsize = (16,10))
    #xgb.plot_importance(model, max_num_features = 30)

## Step 1: Choose a relatively high learning rate and determine the optimum number of trees for this learning rate. 
#XGB_params = {'booster':('gbtree', 'gblinear'), 
#              'eta':[,], # learning rate. typical values: 0.01-0.2
#              'min_child_weight': [],
#              'max_depth': [], #typical values: 3-10
#              'gamma': [],
##              'max_delta_step': [], # usually not used
#              'subsample': [], # typical values 0.5-1
#              'colsample_bytree': [], # typical values 0.5-1
#              'lambda': [], # L2 regularization term
#              'alpha': [], # L1 regularization term
##              'objective': [], # loss function to be minimized. default: reg:linear
#              'eval_metric': ['rmse'],
#              'seed': [0] # define seed if precisely reproducible result is desired.
#             }

xgb1 = XGBRegressor(learning_rate = 0.1,
                     n_estimators = 1000,
                     max_depth = 5,
                     min_child_weight = 1,
                     gamma = 0,
                     subsample = 0.8,
                     colsample_bytree = 0.8,
                     objective = 'reg:linear',
#                     eval_metric = 'rmse',
                     nthread = 4,
                     scale_pos_weight = 1,
                     seed = 27)

modelfit(xgb1, X_train, yt)
#plt.figure(figsize = (16,10))
#xgb.plot_importance(xgb1, max_num_features = 25)
#print("Optimal number of trees for learning_rate = 0.1: {}".format(xgb1))
# Step 2: Tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,6,1)
}
gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators = 158, max_depth = 5,
                                                 min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8,
                                                 objective = 'reg:linear', nthread = 4, scale_pos_weight = 1, seed = 27), 
                        param_grid = param_test1, 
                        scoring = 'neg_mean_squared_error',
#                        n_jobs = 4, 
                        iid = False, 
                        cv = 5)
gsearch1.fit(X_train,yt)
#gsearch1.cv_results_, 
gsearch1.best_params_, gsearch1.best_score_
gsearch1.best_params_, gsearch1.best_score_
# ({'max_depth': 3, 'min_child_weight': 1}, -0.091174420720999488)
# Step 2b: further tuning max_depth and min_child_weight
param_test2 = {
 'max_depth':range(1,3,1),
 'min_child_weight':range(1,6,1)
}
gsearch2 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators = 158, max_depth = 5,
                                                 min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8,
                                                 objective = 'reg:linear', nthread = 4, scale_pos_weight = 1, seed = 27), 
                        param_grid = param_test2, 
                        scoring = 'neg_mean_squared_error',
#                        n_jobs = 4, 
                        iid = False, 
                        cv = 5)
gsearch2.fit(X_train,yt)
#gsearch1.cv_results_, 
gsearch2.best_params_, gsearch2.best_score_
#({'max_depth': 2, 'min_child_weight': 1}, -0.096348841380856851)

# Step 3: Tune gamma
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators = 158, max_depth = 2,
                                                 min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8,
                                                 objective = 'reg:linear', nthread = 4, scale_pos_weight = 1, seed = 27), 
                        param_grid = param_test3, 
                        scoring = 'neg_mean_squared_error',
#                        n_jobs = 4, 
                        iid = False, 
                        cv = 5)
gsearch3.fit(X_train,yt)
#gsearch1.cv_results_, 
gsearch3.best_params_, gsearch3.best_score_
# ({'gamma': 0.0}, -0.095243329595988871)

# re-calibrate # of boosting rounds based on new parameters.
xgb2 = XGBRegressor(learning_rate = 0.1,
                     n_estimators = 1000,
                     max_depth = 2,
                     min_child_weight = 1,
                     gamma = 0.2,
                     subsample = 0.8,
                     colsample_bytree = 0.8,
                     objective = 'reg:linear',
#                     eval_metric = 'rmse',
                     nthread = 4,
                     scale_pos_weight = 1,
                     seed = 27)
modelfit(xgb2, X_train, yt)
# Optimal n_estimators: 158
#  RMSE Score (Train): 0.086310
# Step 4: Tune subsample and colsample_bytree
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators = 400, max_depth = 2,
                                                 min_child_weight = 1, gamma = 0.2, subsample = 0.8, colsample_bytree = 0.8,
                                                 objective = 'reg:linear', nthread = 4, scale_pos_weight = 1, seed = 27), 
                        param_grid = param_test4, 
                        scoring = 'neg_mean_squared_error',
#                        n_jobs = 4, 
                        iid = False, 
                        cv = 5)
gsearch4.fit(X_train,yt)
#gsearch1.cv_results_, 
gsearch4.best_params_, gsearch4.best_score_
# Step 4b: 2nd level tuning for subsample and colsample_bytree
param_test5 = {
 'subsample':[i/100.0 for i in range(85,100,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators = 400, max_depth = 2,
                                                 min_child_weight = 1, gamma = 0.2, subsample = 0.9, colsample_bytree = 0.8,
                                                 objective = 'reg:linear', nthread = 4, scale_pos_weight = 1, seed = 27), 
                        param_grid = param_test5, 
                        scoring = 'neg_mean_squared_error',
#                        n_jobs = 4, 
                        iid = False, 
                        cv = 5)
gsearch5.fit(X_train,yt)
#gsearch1.cv_results_, 
gsearch5.best_params_, gsearch5.best_score_
# Step 5: Tuning Regularization Parameters
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators = 400, max_depth = 2,
                                                 min_child_weight = 1, gamma = 0.2, subsample = 0.9, colsample_bytree = 0.8,
                                                 objective = 'reg:linear', nthread = 4, scale_pos_weight = 1, seed = 27), 
                        param_grid = param_test6, 
                        scoring = 'neg_mean_squared_error',
#                        n_jobs = 4, 
                        iid = False, 
                        cv = 5)
gsearch6.fit(X_train,yt)
#gsearch1.cv_results_, 
gsearch6.best_params_, gsearch6.best_score_
# Step 5b: 2nd Order Tuning Regularization Parameters
param_test7 = {
 'reg_alpha':[1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
 'reg_lambda':[0.75, 1, 1.3, 1.6, 2, 10, 30]
}
gsearch7 = GridSearchCV(estimator = XGBRegressor(learning_rate = 0.1, n_estimators = 400, max_depth = 2,
                                                 min_child_weight = 1, gamma = 0.2, subsample = 0.9, colsample_bytree = 0.8,
                                                 objective = 'reg:linear', nthread = 4, scale_pos_weight = 1, seed = 27), 
                        param_grid = param_test7, 
                        scoring = 'neg_mean_squared_error',
#                        n_jobs = 4, 
                        iid = False, 
                        cv = 5)
gsearch7.fit(X_train,yt)
#gsearch1.cv_results_, 
gsearch7.best_params_, gsearch7.best_score_
# Update Model
xgb3 = XGBRegressor(learning_rate = 0.1,
                     n_estimators = 400,
                     max_depth = 2,
                     min_child_weight = 1,
                     gamma = 0.2,
                     subsample = 0.9,
                     colsample_bytree = 0.8,
                     reg_alpha = 0.0001,
                     reg_lambda = 1,
                     objective = 'reg:linear',
#                     eval_metric = 'rmse',
                     nthread = 4,
                     scale_pos_weight = 1,
                     seed = 27)
modelfit(xgb3, X_train, yt)
# Optimal n_estimators: 158
#  RMSE Score (Train): 0.086310
# Step 6: Lower Learning Rate and Add More Trees
xgb4 = XGBRegressor(learning_rate = 0.01,
                     n_estimators = 5000,
                     max_depth = 2,
                     min_child_weight = 1,
                     gamma = 0.2,
                     subsample = 0.9,
                     colsample_bytree = 0.8,
                     reg_alpha = 0.0001,
                     reg_lambda = 1,
                     objective = 'reg:linear',
#                     eval_metric = 'rmse',
                     nthread = 4,
                     scale_pos_weight = 1,
                     seed = 27)
modelfit(xgb4, X_train, yt)
# Optimal n_estimators: 158
#  RMSE Score (Train): 0.086310
# Get predictions on test set
X_test = encoded_df.loc[test_data_idx+1:,:].reset_index().drop('Id',axis=1)
y_pred = xgb4.predict(X_test)
# predictions are transformed; need to inverse-transform back .
y_test = johnson_inverse(y_pred, g, et, ep, l)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': y_test})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
