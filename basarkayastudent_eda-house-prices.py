# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

submission_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_df.columns
train_df.info()
train_df.info()
categorical=[]

numerical=[]

for i in range(train_df.columns.size):

    if train_df.iloc[:,i].dtype=="object":

        categorical.append(train_df.columns[i])

    else:

        numerical.append(train_df.columns[i])
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable], bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Distribution with Histogram".format(variable))

    plt.show()
for i in numerical:

    plot_hist(i)
def bar_plot(variable):

    var=train_df[variable]

    varValue=var.value_counts()

    

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}\n{}".format(variable, varValue))
for i in categorical:

    bar_plot(i)
categorical_df=train_df[categorical]

numerical_df=train_df[numerical]

categorical_df["SalePrice"]=train_df.SalePrice
for i in range(0,42,1):

    print(categorical_df[[categorical[i], "SalePrice"]].groupby([categorical[i]], as_index=False).mean().sort_values(by="SalePrice", ascending=False))
train_df.corr()
numerical_df.corr().SalePrice
def detect_outliers(df, features):

    outlier_indices=[]

    

    for c in features:

        # 1st quartile

        Q1=np.percentile(df[c], 25)

        # 2nd quartile

        Q3=np.percentile(df[c], 75)

        # IQR

        IQR=Q3-Q1

        # Outlier step

        outlier_step=IQR*1.5

        # Detect outliers and their indices

        outlier_list_col=df[(df[c]<Q1-outlier_step) | (df[c]>Q3+outlier_step)].index

        

        outlier_indices.extend(outlier_list_col)

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=list(i for i, v in outlier_indices.items() if v>2)

    return multiple_outliers

outlier_index=detect_outliers(train_df, numerical)

outlier_df=[]

for i in outlier_index:

    outlier_df.append(train_df.iloc[i,:])

outlier_df=pd.DataFrame(item for item in outlier_df)



train_df=train_df.drop(detect_outliers(train_df, numerical),axis=0).reset_index(drop=True)
train_df.isnull()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
submission_df.isnull().sum()
submission_df.columns[submission_df.isnull().any()]
miss_cat=categorical_df.columns[categorical_df.isnull().any()]



miss_cat_df_list=[]

for i in range(0,16,1):

    miss_cat_df_list.append(train_df[train_df[miss_cat[i]].isnull()])

miss_cat_df_list
train_df.Alley=train_df.Alley.fillna("NA")

train_df.MasVnrType=train_df.MasVnrType.fillna("None")

train_df.MasVnrArea=train_df.MasVnrArea.fillna(0)

train_df.BsmtQual=train_df.BsmtQual.fillna("NA")

train_df.BsmtCond=train_df.BsmtCond.fillna("NA")

train_df.BsmtExposure=train_df.BsmtExposure.fillna("NA")

train_df.BsmtFinType1=train_df.BsmtFinType1.fillna("NA")

train_df.BsmtFinType2=train_df.BsmtFinType2.fillna("NA")

train_df.FireplaceQu=train_df.FireplaceQu.fillna("NA")

train_df.GarageQual=train_df.GarageQual.fillna("NA")

train_df.GarageType=train_df.GarageType.fillna("NA")

# Think more about this one --> train_df.GarageYrBlt=train_df.GarageYrBlt.fillna("NA")

train_df.GarageFinish=train_df.GarageFinish.fillna("NA")

train_df.GarageCond=train_df.GarageCond.fillna("NA")

train_df.PoolQC=train_df.PoolQC.fillna("NA")

train_df.Fence=train_df.Fence.fillna("NA")

train_df.MiscFeature=train_df.MiscFeature.fillna("NA")

train_df.Electrical.value_counts()

# The counts show that SBrkr is way more likely to be replaced with a nan value

train_df.Electrical=train_df.Electrical.fillna("SBrkr")
miss_cat_df_list=[]

for i in range(0,16,1):

    miss_cat_df_list.append(train_df[train_df[miss_cat[i]].isnull()])

miss_cat_df_list
sub_miss_cat=submission_df[categorical].columns[submission_df[categorical].isnull().any()]

sub_miss_cat
submission_df.Alley=submission_df.Alley.fillna("NA")

submission_df.MasVnrType=submission_df.MasVnrType.fillna("None")

submission_df.MasVnrArea=submission_df.MasVnrArea.fillna(0)

submission_df.BsmtQual=submission_df.BsmtQual.fillna("NA")

submission_df.BsmtCond=submission_df.BsmtCond.fillna("NA")

submission_df.BsmtExposure=submission_df.BsmtExposure.fillna("NA")

submission_df.BsmtFinType1=submission_df.BsmtFinType1.fillna("NA")

submission_df.BsmtFinType2=submission_df.BsmtFinType2.fillna("NA")

submission_df.FireplaceQu=submission_df.FireplaceQu.fillna("NA")

submission_df.GarageQual=submission_df.GarageQual.fillna("NA")

submission_df.GarageType=submission_df.GarageType.fillna("NA")

submission_df.GarageFinish=submission_df.GarageFinish.fillna("NA")

submission_df.GarageCond=submission_df.GarageCond.fillna("NA")

submission_df.PoolQC=submission_df.PoolQC.fillna("NA")

submission_df.Fence=submission_df.Fence.fillna("NA")

submission_df.MiscFeature=submission_df.MiscFeature.fillna("NA")

submission_df.MSZoning.value_counts()

# RL is more likely to be as nan value

submission_df.MSZoning=submission_df.MSZoning.fillna("RL")

submission_df.Utilities.value_counts()

# All of values are AllPub for Utilities feature so we will fill nan values with it

submission_df.Utilities=submission_df.Utilities.fillna("AllPub")

submission_df.Exterior1st.value_counts()

# VinylSd is more likely

submission_df.Exterior1st=submission_df.Exterior1st.fillna("VinylSd")

submission_df.Exterior2nd.value_counts()

submission_df.Exterior2nd=submission_df.Exterior2nd.fillna("VinylSd")

submission_df.KitchenQual.value_counts()

submission_df.KitchenQual=submission_df.KitchenQual.fillna("TA")

submission_df.Functional.value_counts()

submission_df.Functional=submission_df.Functional.fillna("Typ")

submission_df.SaleType.value_counts()

submission_df.SaleType=submission_df.SaleType.fillna("WD")
sub_miss_cat=submission_df[categorical].columns[submission_df[categorical].isnull().any()]

sub_miss_cat
miss_num=numerical_df.columns[numerical_df.isnull().any()]



miss_num_df_list=[]

for i in range(0,3,1):

    miss_num_df_list.append(train_df[train_df[miss_num[i]].isnull()])

miss_num_df_list
train_df.GarageYrBlt=train_df.GarageYrBlt.fillna("NA")

front_area_ratio_list=train_df.LotFrontage/train_df.LotArea

front_area_ratio_list.describe()

front_area_ratio=front_area_ratio_list.median()



index_nan=train_df[train_df.LotFrontage.isnull()].index

for i in index_nan:

    train_df.LotFrontage[i]=train_df.LotArea[i]*front_area_ratio
miss_num_df_list=[]

for i in range(0,3,1):

    miss_num_df_list.append(train_df[train_df[miss_num[i]].isnull()])

miss_num_df_list
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
sub_numerical=[]

for i in range(submission_df.columns.size):

    if submission_df.iloc[:,i].dtype=="float":

        sub_numerical.append(submission_df.columns[i])

    elif submission_df.iloc[:,i].dtype=="int":

        sub_numerical.append(submission_df.columns[i])
sub_numerical.remove("Id")
len(sub_numerical)
sub_miss_num=submission_df[sub_numerical].columns[submission_df[sub_numerical].isnull().any()]

sub_miss_num
front_area_ratio_list=submission_df.LotFrontage/train_df.LotArea

front_area_ratio=front_area_ratio_list.median()



index_nan=submission_df[submission_df.LotFrontage.isnull()].index

for i in index_nan:

    submission_df.LotFrontage[i]=submission_df.LotArea[i]*front_area_ratio



submission_df.GarageYrBlt=submission_df.GarageYrBlt.fillna("NA")

submission_df.BsmtFinSF1=submission_df.BsmtFinSF1.fillna(submission_df.BsmtFinSF1.median())

submission_df.BsmtFinSF2=submission_df.BsmtFinSF2.fillna(submission_df.BsmtFinSF2.median())

submission_df.BsmtUnfSF=submission_df.BsmtUnfSF.fillna(submission_df.BsmtUnfSF.median())

submission_df.TotalBsmtSF=submission_df.TotalBsmtSF.fillna(submission_df.TotalBsmtSF.median())

submission_df.BsmtFullBath=submission_df.BsmtFullBath.fillna(submission_df.BsmtFullBath.median())

submission_df.BsmtHalfBath=submission_df.BsmtHalfBath.fillna(submission_df.BsmtHalfBath.median())

submission_df.GarageCars=submission_df.GarageCars.fillna(submission_df.GarageCars.median())

submission_df.GarageArea=submission_df.GarageArea.fillna(submission_df.GarageArea.median())
sub_miss_num=submission_df[sub_numerical].columns[submission_df[sub_numerical].isnull().any()]

sub_miss_num
submission_df.columns.isnull().any()
f,ax=plt.subplots(figsize=(22,22))

sns.heatmap(train_df.corr(), vmax=1, vmin=-1, annot=True, fmt=".2f")

plt.show()
evaluative_features=["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtExposure",

                     "BsmtCond", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual",

                     "FireplaceQu", "GarageQual", "GarageCond" ,"PoolQC", "Fence"]
train_df.SalePrice=train_df.SalePrice.astype(float)

OQ_List=list(train_df.OverallQual.unique())

price=[]

for i in OQ_List:

    x=train_df[train_df.OverallQual==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

OQ_data=pd.DataFrame({"quality_point":OQ_List, "price_average":price})

new_index=(OQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=OQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.quality_point, y=sorted_data.price_average)

plt.xlabel("Overall Quality Evaluation Point (Over 10)", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Overall Quality Evaluation and Price", fontsize=17)

plt.show()
train_df.SalePrice=train_df.SalePrice.astype(float)

OC_List=list(train_df.OverallCond.unique())

price=[]

for i in OC_List:

    x=train_df[train_df.OverallCond==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

OC_data=pd.DataFrame({"condition_point":OC_List, "price_average":price})

new_index=(OC_data.price_average.sort_values(ascending=True)).index.values

sorted_data=OC_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.condition_point, y=sorted_data.price_average)

plt.xlabel("Overall Condition Evaluation Point (Over 10)", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Overall Condition Evaluation and Price", fontsize=17)

plt.show()
evaluative_features[2]
train_df.SalePrice=train_df.SalePrice.astype(float)

EQ_List=list(train_df.ExterQual.unique())

price=[]

for i in EQ_List:

    x=train_df[train_df.ExterQual==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

EQ_data=pd.DataFrame({"eq_point":EQ_List, "price_average":price})

new_index=(EQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=EQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.eq_point, y=sorted_data.price_average)

plt.xlabel("Exterior Quality Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Exterior Quality Evaluation and Price", fontsize=17)

plt.text(-0.3,280000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor", fontsize=17)

plt.show()
evaluative_features[3]
train_df.SalePrice=train_df.SalePrice.astype(float)

EC_List=list(train_df.ExterCond.unique())

price=[]

for i in EC_List:

    x=train_df[train_df.ExterCond==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

EC_data=pd.DataFrame({"ec_point":EC_List, "price_average":price})

new_index=(EC_data.price_average.sort_values(ascending=True)).index.values

sorted_data=EC_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.ec_point, y=sorted_data.price_average)

plt.xlabel("Exterior Condition Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Exterior Condition Evaluation and Price", fontsize=17)

plt.text(-0.3,140000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor", fontsize=17)

plt.show()
evaluative_features[4]
train_df.SalePrice=train_df.SalePrice.astype(float)

BQ_List=list(train_df.BsmtQual.unique())

price=[]

for i in BQ_List:

    x=train_df[train_df.BsmtQual==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

BQ_data=pd.DataFrame({"bq_point":BQ_List, "price_average":price})

new_index=(BQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=BQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.bq_point, y=sorted_data.price_average)

plt.xlabel("Height Evaluation for Basement", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Height Evaluation for Basement and Price", fontsize=17)

plt.text(-0.3,220000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor\nNA: No Basement", fontsize=17)

plt.show()
evaluative_features[5]
train_df.SalePrice=train_df.SalePrice.astype(float)

BE_List=list(train_df.BsmtExposure.unique())

price=[]

for i in BE_List:

    x=train_df[train_df.BsmtExposure==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

BE_data=pd.DataFrame({"be_point":BE_List, "price_average":price})

new_index=(BE_data.price_average.sort_values(ascending=True)).index.values

sorted_data=BE_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.be_point, y=sorted_data.price_average)

plt.xlabel("Basement Exposure Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Basement Exposure Evaluation and Price", fontsize=17)

plt.text(-0.3,180000,"Gd: Good Exposure\nAv: Average Exposure\nMn: Minimum Exposure\nNo: No Exposure\nNA: No Basement", fontsize=17)

plt.show()
evaluative_features[6]
train_df.SalePrice=train_df.SalePrice.astype(float)

BC_List=list(train_df.BsmtCond.unique())

price=[]

for i in BC_List:

    x=train_df[train_df.BsmtCond==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

BC_data=pd.DataFrame({"bc_point":BC_List, "price_average":price})

new_index=(BC_data.price_average.sort_values(ascending=True)).index.values

sorted_data=BC_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.bc_point, y=sorted_data.price_average)

plt.xlabel("Basement Condition Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Basement Condition Evaluation and Price", fontsize=17)

plt.text(-0.3,180000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor\nNA: No Basement", fontsize=17)

plt.show()
evaluative_features[7]
train_df.SalePrice=train_df.SalePrice.astype(float)

BF1_List=list(train_df.BsmtFinType1.unique())

price=[]

for i in BF1_List:

    x=train_df[train_df.BsmtFinType1==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

BF1_data=pd.DataFrame({"bf1_point":BF1_List, "price_average":price})

new_index=(BF1_data.price_average.sort_values(ascending=True)).index.values

sorted_data=BF1_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.bf1_point, y=sorted_data.price_average)

plt.xlabel("Basement Finished Area Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Basement Finished Area Evaluation and Price", fontsize=17)

plt.text(-0.3,170000,"GLQ: Good Living Quarters\nALQ: Average Living Quarters\nBLQ: Below Average Living Quarters\nRec: Average Rec Room\nLwQ: Low Quality\nUnf: Unfinshed\nNA: No Basement", fontsize=17)

plt.show()
evaluative_features[8]
train_df.SalePrice=train_df.SalePrice.astype(float)

BF2_List=list(train_df.BsmtFinType2.unique())

price=[]

for i in BF2_List:

    x=train_df[train_df.BsmtFinType2==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

BF2_data=pd.DataFrame({"bf2_point":BF2_List, "price_average":price})

new_index=(BF2_data.price_average.sort_values(ascending=True)).index.values

sorted_data=BF2_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.bf2_point, y=sorted_data.price_average)

plt.xlabel("Basement Finished Area Evaluation (if multiple)", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Basement Finished Area Evaluation and Price", fontsize=17)

plt.text(-0.3,143000,"GLQ: Good Living Quarters\nALQ: Average Living Quarters\nBLQ: Below Average Living Quarters\nRec: Average Rec Room\nLwQ: Low Quality\nUnf: Unfinshed\nNA: No Basement", fontsize=17)

plt.show()
evaluative_features[9]
train_df.SalePrice=train_df.SalePrice.astype(float)

HQ_List=list(train_df.HeatingQC.unique())

price=[]

for i in HQ_List:

    x=train_df[train_df.HeatingQC==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

HQ_data=pd.DataFrame({"hq_point":HQ_List, "price_average":price})

new_index=(HQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=HQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.hq_point, y=sorted_data.price_average)

plt.xlabel("Heating Quality & Condition Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Heating Quality & Condition Evaluation and Price", fontsize=17)

plt.text(-0.3,162000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor", fontsize=17)

plt.show()
evaluative_features[10]
train_df.SalePrice=train_df.SalePrice.astype(float)

KQ_List=list(train_df.KitchenQual.unique())

price=[]

for i in KQ_List:

    x=train_df[train_df.KitchenQual==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

KQ_data=pd.DataFrame({"kq_point":KQ_List, "price_average":price})

new_index=(KQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=KQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.kq_point, y=sorted_data.price_average)

plt.xlabel("Kitchen Quality Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Kitchen Quality Evaluation and Price", fontsize=17)

plt.text(-0.3,232000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor", fontsize=17)

plt.show()
evaluative_features[11]
train_df.SalePrice=train_df.SalePrice.astype(float)

FQ_List=list(train_df.FireplaceQu.unique())

price=[]

for i in FQ_List:

    x=train_df[train_df.FireplaceQu==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

FQ_data=pd.DataFrame({"fq_point":FQ_List, "price_average":price})

new_index=(FQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=FQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.fq_point, y=sorted_data.price_average)

plt.xlabel("Fireplace Quality Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Fireplace Quality Evaluation and Price", fontsize=17)

plt.text(-0.3,232000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor\nNA: No Fireplace", fontsize=17)

plt.show()
evaluative_features[12]
train_df.SalePrice=train_df.SalePrice.astype(float)

GQ_List=list(train_df.GarageQual.unique())

price=[]

for i in GQ_List:

    x=train_df[train_df.GarageQual==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

GQ_data=pd.DataFrame({"gq_point":GQ_List, "price_average":price})

new_index=(GQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=GQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.gq_point, y=sorted_data.price_average)

plt.xlabel("Garage Quality Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Garage Quality Evaluation and Price", fontsize=17)

plt.text(-0.3,152000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor\nNA: No Garage", fontsize=17)

plt.show()
evaluative_features[13]
train_df.SalePrice=train_df.SalePrice.astype(float)

GC_List=list(train_df.GarageCond.unique())

price=[]

for i in GC_List:

    x=train_df[train_df.GarageCond==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

GC_data=pd.DataFrame({"gc_point":GC_List, "price_average":price})

new_index=(GC_data.price_average.sort_values(ascending=True)).index.values

sorted_data=GC_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.gc_point, y=sorted_data.price_average)

plt.xlabel("Garage Condition Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Garage Condition Evaluation and Price", fontsize=17)

plt.text(-0.3,136000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor\nNA: No Garage", fontsize=17)

plt.show()
evaluative_features[14]
train_df.SalePrice=train_df.SalePrice.astype(float)

PQ_List=list(train_df.PoolQC.unique())

price=[]

for i in PQ_List:

    x=train_df[train_df.PoolQC==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

PQ_data=pd.DataFrame({"pq_point":PQ_List, "price_average":price})

new_index=(PQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=PQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.pq_point, y=sorted_data.price_average)

plt.xlabel("Pool Quality Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Pool Quality Evaluation and Price", fontsize=17)

plt.text(-0.3,122000,"Ex: Excellent\nGd: Good\nTA: Typical/Average\nFA: Fair\nPo: Poor\nNA: No Pool", fontsize=17)

plt.show()
evaluative_features[15]
train_df.SalePrice=train_df.SalePrice.astype(float)

FQ_List=list(train_df.Fence.unique())

price=[]

for i in FQ_List:

    x=train_df[train_df.Fence==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

FQ_data=pd.DataFrame({"fq_point":FQ_List, "price_average":price})

new_index=(FQ_data.price_average.sort_values(ascending=True)).index.values

sorted_data=FQ_data.reindex(new_index)





plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.fq_point, y=sorted_data.price_average)

plt.xlabel("Fence Quality Evaluation", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Fence Quality Evaluation and Price", fontsize=17)

plt.text(-0.3,152000,"GdPrv: Good Privacy\nMnPrv: Minimum Privacy\nGdWo: Good Wood\nMnWw: Minimum Wood/Wire\nNA: No Fence", fontsize=17)

plt.show()
type_features=[]

for i in categorical:

    if not i in evaluative_features:

        type_features.append(i)
len(type_features)
train_df.MSSubClass=train_df.MSSubClass.astype(str)
type_features.append("MSSubClass")
type_features[29]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.MSSubClass.unique())

price=[]

for i in List:

    x=train_df[train_df.MSSubClass==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Type and Price", fontsize=17)

plt.text(-0.3, 101000, "20: 1-STORY 1946 & NEWER ALL STYLES\n30: 1-STORY 1945 & OLDER\n40: 1-STORY W/FINISHED ATTIC ALL AGES\n45: 1-1/2 STORY - UNFINISHED ALL AGES\n50: 1-1/2 STORY FINISHED ALL AGES\n60: 2-STORY 1946 & NEWER\n70: 2-STORY 1945 & OLDER\n75: 2-1/2 STORY ALL AGES\n80: SPLIT OR MULTI-LEVEL\n85: SPLIT FOYER\n90: DUPLEX - ALL STYLES AND AGES\n120: 1-STORY PUD (Planned Unit Development) - 1946 & NEWER\n150: 1-1/2 STORY PUD - ALL AGES\n160: 2-STORY PUD - 1946 & NEWER\n180: PUD - MULTILEVEL - INCL SPLIT LEV/FOYER\n190: 2 FAMILY CONVERSION - ALL STYLES AND AGES", fontsize=14)

plt.show()
type_features[0]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.MSZoning.unique())

price=[]

for i in List:

    x=train_df[train_df.MSZoning==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Zoning Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Zoning Type and Price", fontsize=17)

plt.text(-0.3, 151000, "A: Agriculture\nC: Commercial\nFV: Floating Village Residential\nI: Industrial\nRH: Residential High Density\nRL: Residential Low Density\nRP: Residential Low Density Park\nRM: Residential Medium Density", fontsize=14)

plt.show()
type_features[1]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Street.unique())

price=[]

for i in List:

    x=train_df[train_df.Street==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Street Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Street Type and Price", fontsize=17)

plt.text(-0.4, 153000, "Grvl: Gravel\nPave: Paved", fontsize=14)

plt.show()
type_features[2]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Alley.unique())

price=[]

for i in List:

    x=train_df[train_df.Alley==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Alley Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Alley Type and Price", fontsize=17)

plt.text(-0.4, 153000, "Grvl: Gravel\nPave: Paved\nNA: No Alley Access", fontsize=14)

plt.show()
type_features[3]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.LotShape.unique())

price=[]

for i in List:

    x=train_df[train_df.LotShape==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Lot Shape of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Lot Shape and Price", fontsize=17)

plt.text(-0.4, 179000, "Reg: Regular\nIR1: Slightly irregular\nIR2:Moderately Irregular\nIR3: Irregular", fontsize=14)

plt.show()
type_features[4]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.LandContour.unique())

price=[]

for i in List:

    x=train_df[train_df.LandContour==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Flatness of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Flatness and Price", fontsize=17)

plt.text(-0.4, 189000, "Lvl: Near Flat/Level\nBnk: Banked - Quick and significant rise from street grade to building\nHLS: Hillside - Significant slope from side to side\nLow: Depression", fontsize=14)

plt.show()
type_features[5]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Utilities.unique())

price=[]

for i in List:

    x=train_df[train_df.Utilities==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Utilities of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Utilities and Price", fontsize=17)

plt.text(-0.4, 189000, "AllPub: All public Utilities (E,G,W,& S)\nNoSewr: Electricity, Gas, and Water (Septic Tank)\nNoSeWa: Electricity and Gas Only\nELO: Electricity only", fontsize=14)

plt.show()
type_features[6]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.LotConfig.unique())

price=[]

for i in List:

    x=train_df[train_df.LotConfig==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Lot Configuration of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Lot Configuration and Price", fontsize=17)

plt.text(-0.4, 179000, "Inside: Inside lot\nCorner: Corner lot\nCulDSac: Cul-de-sac\nFR2: Frontage on 2 sides of property\nFR3: Frontage on 3 sides of property", fontsize=14)

plt.show()
type_features[7]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.LandSlope.unique())

price=[]

for i in List:

    x=train_df[train_df.LandSlope==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Land Slope of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Land Slope and Price", fontsize=17)

plt.text(-0.4, 177000, "Gtl: Gentle slope\nMod: Moderate Slope\nSev: Severe Slope", fontsize=14)

plt.show()
type_features[8]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Neighborhood.unique())

price=[]

for i in List:

    x=train_df[train_df.Neighborhood==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Neighborhood of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Neighborhood and Price", fontsize=17)

plt.show()
type_features[9]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Condition1.unique())

price=[]

for i in List:

    x=train_df[train_df.Condition1==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Proximity to Various Conditions of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Proximity to Various Conditions and Price", fontsize=17)

plt.text(-0.4, 177000, "", fontsize=14)

plt.show()
type_features[10]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Condition2.unique())

price=[]

for i in List:

    x=train_df[train_df.Condition2==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Proximity to Various Conditions of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Proximity to Various Conditions (if multiple) and Price", fontsize=17)

plt.text(-0.4, 177000, "", fontsize=14)

plt.show()
type_features[11]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.BldgType.unique())

price=[]

for i in List:

    x=train_df[train_df.BldgType==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Type of House and Price", fontsize=17)

plt.text(-0.4, 150000, "1Fam: Single-family Detached\n2FmCon: Two-family Conversion; originally built as one-family dwelling\nDuplx: Duplex\nTwnhsE: Townhouse End Unit\nTwnhsI: Townhouse Inside Unit", fontsize=14)

plt.show()
type_features[12]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.HouseStyle.unique())

price=[]

for i in List:

    x=train_df[train_df.HouseStyle==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Style of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Style of House and Price", fontsize=17)

plt.text(-0.4, 150000, "1Story: One story\n1.5Fin: One and one-half story: 2nd level finished\n1.5Unf: One and one-half story: 2nd level unfinished\n2Story: Two story\n2.5Fin: Two and one-half story: 2nd level finished\n2.5Unf: Two and one-half story: 2nd level unfinished\nSFoyer: Split Foyer\nSLvl: Split Level", fontsize=14)

plt.show()
type_features[13]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.RoofStyle.unique())

price=[]

for i in List:

    x=train_df[train_df.RoofStyle==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Roof Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Roof Type of House and Price", fontsize=17)

plt.text(-0.4, 210000, "Flat: Flat\nGable: Gable\nGambrel: Gabrel (Barn)\nHip: Hip\nMansard: Mansard\nShed: Shed", fontsize=14)

plt.show()
len(type_features)
type_features[14]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.RoofMatl.unique())

price=[]

for i in List:

    x=train_df[train_df.RoofMatl==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Roof Material of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Roof Material of House and Price", fontsize=17)

plt.text(-0.4, 200000, "ClyTile: Clay or Tile\nCompShg: Standard (Composite) Shingle\nMetal: Metal\nTar&Grv: Gravel & Tar\nWdShake: Wood Shakes\nWdShngl: Wood Shingles", fontsize=14)

plt.show()
type_features[15]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Exterior1st.unique())

price=[]

for i in List:

    x=train_df[train_df.Exterior1st==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Exterior Covering of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Exterior Covering of House and Price", fontsize=17)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
type_features[16]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Exterior2nd.unique())

price=[]

for i in List:

    x=train_df[train_df.Exterior2nd==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Exterior Covering of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Exterior Covering (if multiple) of House and Price", fontsize=17)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
type_features[17]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.MasVnrType.unique())

price=[]

for i in List:

    x=train_df[train_df.MasVnrType==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Masonry Veneer Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Masonry Veneer Type of House and Price", fontsize=17)

plt.text(-0.4, 201000, "BrkCmn: Brick Common\nBrkFace: Brick Face\nCBlock: Cinder Block\nNone: None\nStone: Stone", fontsize=14)

plt.show()
type_features[18]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Foundation.unique())

price=[]

for i in List:

    x=train_df[train_df.Foundation==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Foundation Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Foundation Type of House and Price", fontsize=17)

plt.text(-0.4, 171000, "BrkTil: Brick & Tile\nCBlock: Cinder Block\nPConc: Poured Contrete\nSlab: Slab\nStone: Stone\nWood: Wood", fontsize=14)

plt.show()
type_features[19]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Heating.unique())

price=[]

for i in List:

    x=train_df[train_df.Heating==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Heating Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Heating Type of House and Price", fontsize=17)

plt.text(-0.4, 141000, "Floor: Floor Furnace\nGasA: Gas forced warm air furnace\nGasW: Gas hot water or steam heat\nGrav: Gravity furnace\nOthW: Hot water or steam heat other than gas\nWall: Wall furnac", fontsize=14)

plt.show()
type_features[20]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.CentralAir.unique())

price=[]

for i in List:

    x=train_df[train_df.CentralAir==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Central Air Conditioning of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Central Air Conditioning House and Price", fontsize=17)

plt.text(-0.4, 161000, "N: No\nY:Yes", fontsize=14)

plt.show()
type_features[21]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Electrical.unique())

price=[]

for i in List:

    x=train_df[train_df.Electrical==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Electrical System of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Electrical System of House and Price", fontsize=17)

plt.text(-0.4, 151000, "SBrkr: Standard Circuit Breakers & Romex\nFuseA: Fuse Box over 60 AMP and all Romex wiring (Average)\nFuseF: 60 AMP Fuse Box and mostly Romex wiring (Fair)\nFuseP: 60 AMP Fuse Box and mostly knob & tube wiring (poor)\nMix: Mixed", fontsize=14)

plt.show()
type_features[22]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.Functional.unique())

price=[]

for i in List:

    x=train_df[train_df.Functional==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Functionality of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Functionality of House and Price", fontsize=17)

plt.text(-0.4, 135000, "Typ: Typical Functionality\nMin1: Minor Deductions 1\nMin2: Minor Deductions 2\nMod: Moderate Deductions\nMaj1: Major Deductions 1\nMaj2: Major Deductions 2\nSev: Severely Damaged\nSal: Salvage only", fontsize=14)

plt.show()
type_features[23]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.GarageType.unique())

price=[]

for i in List:

    x=train_df[train_df.GarageType==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Garage Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Garage Type of House and Price", fontsize=17)

plt.text(-0.4, 175000, "2Types: More than one type of garage\nAttchd: Attached to home\nBasment: Basement Garage\nBuiltIn: Built-In (Garage part of house - typically has room above garage)\nCarPort: Car Port\nDetchd: Detached from home\nNA: No Garage", fontsize=14)

plt.show()
type_features[24]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.GarageFinish.unique())

price=[]

for i in List:

    x=train_df[train_df.GarageFinish==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Interior Finish of the Garage", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Interior Finish of the Garage and Price", fontsize=17)

plt.text(-0.4, 195000, "Fin: Finished\nRFn: Rough Finished\nUnf: Unfinished\nNA: No Garage", fontsize=14)

plt.show()
type_features[25]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.PavedDrive.unique())

price=[]

for i in List:

    x=train_df[train_df.PavedDrive==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Paved Driveaway of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Paved Driveaway of House and Price", fontsize=17)

plt.text(-0.4, 155000, "Y: Paved\nP: Partial Pavement\nN: Dirt/Gravel", fontsize=14)

plt.show()
type_features[26]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.MiscFeature.unique())

price=[]

for i in List:

    x=train_df[train_df.MiscFeature==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Miscellaneous Features of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Miscellaneous Features of House and Price", fontsize=17)

plt.text(-0.4, 145000, "Elev: Elevator\nGar2: 2nd Garage (if not described in garage section)\nOthr: Other\nShed: Shed (over 100 SF)\nTenC: Tennis Court\nNA: None", fontsize=14)

plt.show()
type_features[27]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.SaleType.unique())

price=[]

for i in List:

    x=train_df[train_df.SaleType==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Sale Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Sale Type of House and Price", fontsize=17)

plt.text(-0.4, 145000, "", fontsize=14)

plt.show()
type_features[28]
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.SaleCondition.unique())

price=[]

for i in List:

    x=train_df[train_df.SaleCondition==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Sale Condition of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Sale Condition of House and Price", fontsize=17)

plt.text(-0.4, 205000, "Normal: Normal Sale\nAbnorml: Abnormal Sale -  trade, foreclosure, short sale\nAdjLand: Adjoining Land Purchase\nAlloca: Allocation - two linked properties with separate deeds, typically condo with a garage unit	\nFamily: Sale between family members\nPartial: Home was not completed when last assessed (associated with New Homes)", fontsize=14)

plt.show()
train_index=train_df.index.values

df=pd.concat([train_df, submission_df], axis=0).reset_index(drop=True)
categorical_features=type_features

categorical_features
numerical.remove("Id")

numerical.remove("MSSubClass")

numerical.remove("OverallQual")

numerical.remove("OverallCond")

numerical_features=numerical

numerical_features
evaluative_features
evaluative_features
df.OverallQual.unique()
g=sns.factorplot(x="OverallQual", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
df.OverallCond.unique()
g=sns.factorplot(x="OverallCond", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
df.ExterQual.unique()
eq_list=[]

for i in range(0, len(df.ExterQual), 1):

    if df.ExterQual[i]=="Ex":

        eq_list.append(5)

    elif df.ExterQual[i]=="Gd":

        eq_list.append(4)

    elif df.ExterQual[i]=="TA":

        eq_list.append(3)

    elif df.ExterQual[i]=="Fa":

        eq_list.append(2)

    else:

        eq_list.append(1)

df.ExterQual=eq_list
g=sns.factorplot(x="ExterQual", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
df.ExterCond.unique()
ec_list=[]

for i in range(0, len(df.ExterCond), 1):

    if df.ExterCond[i]=="Ex":

        ec_list.append(5)

    elif df.ExterCond[i]=="Gd":

        ec_list.append(4)

    elif df.ExterCond[i]=="TA":

        ec_list.append(3)

    elif df.ExterCond[i]=="Fa":

        ec_list.append(2)

    else:

        ec_list.append(1)

df.ExterCond=ec_list
g=sns.factorplot(x="ExterCond", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[4]
df.BsmtQual.unique()
bq_list=[]

for i in range(0, len(df.BsmtQual), 1):

    if df.BsmtQual[i]=="Ex":

        bq_list.append(5)

    elif df.BsmtQual[i]=="Gd":

        bq_list.append(4)

    elif df.BsmtQual[i]=="TA":

        bq_list.append(3)

    elif df.BsmtQual[i]=="Fa":

        bq_list.append(2)

    elif df.BsmtQual[i]=="Po":

        bq_list.append(1)

    elif df.BsmtQual[i]=="NA":

        bq_list.append(0)

df.BsmtQual=bq_list
g=sns.factorplot(x="BsmtQual", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[5]
df.BsmtExposure.unique()
be_list=[]

for i in range(0, len(df.BsmtExposure), 1):

    if df.BsmtExposure[i]=="Gd":

        be_list.append(4)

    elif df.BsmtExposure[i]=="Av":

        be_list.append(3)

    elif df.BsmtExposure[i]=="Mn":

        be_list.append(2)

    else:

        be_list.append(0)

df.BsmtExposure=be_list
g=sns.factorplot(x="BsmtExposure", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[6]
df.BsmtCond.unique()
bc_list=[]

for i in range(0, len(df.BsmtCond), 1):

    if df.BsmtCond[i]=="Ex":

        bc_list.append(5)

    elif df.BsmtCond[i]=="Gd":

        bc_list.append(4)

    elif df.BsmtCond[i]=="TA":

        bc_list.append(3)

    elif df.BsmtCond[i]=="Fa":

        bc_list.append(2)

    elif df.BsmtCond[i]=="Po":

        bc_list.append(1)

    else:

        bc_list.append(0)

df.BsmtCond=bc_list
g=sns.factorplot(x="BsmtCond", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[7]
df.BsmtFinType1.unique()
bf1_list=[]

for i in range(0, len(df.BsmtFinType1), 1):

    if df.BsmtFinType1[i]=="GLQ":

        bf1_list.append(6)

    elif df.BsmtFinType1[i]=="ALQ":

        bf1_list.append(5)

    elif df.BsmtFinType1[i]=="BLQ":

        bf1_list.append(4)

    elif df.BsmtFinType1[i]=="Rec":

        bf1_list.append(3)

    elif df.BsmtFinType1[i]=="LwQ":

        bf1_list.append(2)

    elif df.BsmtFinType1[i]=="Unf":

        bf1_list.append(1)

    else:

        bf1_list.append(0)

set(bf1_list)

df.BsmtFinType1=bf1_list
g=sns.factorplot(x="BsmtFinType1", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[8]
df.BsmtFinType2.unique()
bf2_list=[]

for i in range(0, len(df.BsmtFinType2), 1):

    if df.BsmtFinType2[i]=="GLQ":

        bf2_list.append(6)

    elif df.BsmtFinType2[i]=="ALQ":

        bf2_list.append(5)

    elif df.BsmtFinType2[i]=="BLQ":

        bf2_list.append(4)

    elif df.BsmtFinType2[i]=="Rec":

        bf2_list.append(3)

    elif df.BsmtFinType2[i]=="LwQ":

        bf2_list.append(2)

    elif df.BsmtFinType2[i]=="Unf":

        bf2_list.append(1)

    else:

        bf2_list.append(0)

set(bf2_list)

df.BsmtFinType2=bf2_list
g=sns.factorplot(x="BsmtFinType2", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[9]
df.HeatingQC.unique()
hq_list=[]

for i in range(0, len(df.HeatingQC), 1):

    if df.HeatingQC[i]=="Ex":

        hq_list.append(5)

    elif df.HeatingQC[i]=="Gd":

        hq_list.append(4)

    elif df.HeatingQC[i]=="TA":

        hq_list.append(3)

    elif df.HeatingQC[i]=="Fa":

        hq_list.append(2)

    elif df.HeatingQC[i]=="Po":

        hq_list.append(1)

    else:

        hq_list.append(0)

set(hq_list)

df.HeatingQC=hq_list
df[evaluative_features]
g=sns.factorplot(x="HeatingQC", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[10]
df.KitchenQual.unique()
kq_list=[]

for i in range(0, len(df.KitchenQual), 1):

    if df.KitchenQual[i]=="Ex":

        kq_list.append(5)

    elif df.KitchenQual[i]=="Gd":

        kq_list.append(4)

    elif df.KitchenQual[i]=="TA":

        kq_list.append(3)

    elif df.KitchenQual[i]=="Fa":

        kq_list.append(2)

    elif df.KitchenQual[i]=="Po":

        kq_list.append(1)

set(kq_list)

df.KitchenQual=kq_list
g=sns.factorplot(x="KitchenQual", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[11]
df.FireplaceQu.unique()
fq_list=[]

for i in range(0, len(df.FireplaceQu), 1):

    if df.FireplaceQu[i]=="Ex":

        fq_list.append(5)

    elif df.FireplaceQu[i]=="Gd":

        fq_list.append(4)

    elif df.FireplaceQu[i]=="TA":

        fq_list.append(3)

    elif df.FireplaceQu[i]=="Fa":

        fq_list.append(2)

    elif df.FireplaceQu[i]=="Po":

        fq_list.append(1)

    else:

        fq_list.append(0)

set(fq_list)

df.FireplaceQu=fq_list
g=sns.factorplot(x="FireplaceQu", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[12]
df.GarageQual.unique()
gq_list=[]

for i in range(0, len(df.GarageQual), 1):

    if df.GarageQual[i]=="Ex":

        gq_list.append(5)

    elif df.GarageQual[i]=="Gd":

        gq_list.append(4)

    elif df.GarageQual[i]=="TA":

        gq_list.append(3)

    elif df.GarageQual[i]=="Fa":

        gq_list.append(2)

    elif df.GarageQual[i]=="Po":

        gq_list.append(1)

    else:

        gq_list.append(0)

set(gq_list)

df.GarageQual=gq_list
g=sns.factorplot(x="GarageQual", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[13]
df.GarageCond.unique()
gc_list=[]

for i in range(0, len(df.GarageCond), 1):

    if df.GarageCond[i]=="Ex":

        gc_list.append(5)

    elif df.GarageCond[i]=="Gd":

        gc_list.append(4)

    elif df.GarageCond[i]=="TA":

        gc_list.append(3)

    elif df.GarageCond[i]=="Fa":

        gc_list.append(2)

    elif df.GarageCond[i]=="Po":

        gc_list.append(1)

    else:

        gc_list.append(0)

set(gc_list)

df.GarageCond=gc_list
g=sns.factorplot(x="GarageCond", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[14]
df.PoolQC.unique()
pq_list=[]

for i in range(0, len(df.PoolQC), 1):

    if df.PoolQC[i]=="Ex":

        pq_list.append(5)

    elif df.PoolQC[i]=="Gd":

        pq_list.append(4)

    elif df.PoolQC[i]=="TA":

        pq_list.append(3)

    elif df.PoolQC[i]=="Fa":

        pq_list.append(2)

    else:

        pq_list.append(0)

set(pq_list)

df.PoolQC=pq_list
g=sns.factorplot(x="PoolQC", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
evaluative_features[15]
df.Fence.unique()
f_list=[]

for i in range(0, len(df.Fence), 1):

    if df.Fence[i]=="GdWo":

        f_list.append(5)

    elif df.Fence[i]=="GdPrv":

        f_list.append(5)

    elif df.Fence[i]=="MnWw":

        f_list.append(2)

    elif df.Fence[i]=="MnPrv":

        f_list.append(2)

    else:

        f_list.append(0)

set(f_list)

df.Fence=f_list
g=sns.factorplot(x="Fence", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
df[evaluative_features]
type_features
type_features[0]
df.MSZoning.unique()
g=sns.factorplot(x="MSZoning", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
mz_list=[]

for i in range(0, len(df.MSZoning), 1):

    if df.MSZoning[i]=="FV":

        mz_list.append("FVRL")

    elif df.MSZoning[i]=="RL":

        mz_list.append("FVRL")

    elif df.MSZoning[i]=="RH":

        mz_list.append("RHRM")

    elif df.MSZoning[i]=="RM":

        mz_list.append("RHRM")

    else:

        mz_list.append("C")

set(mz_list)

df.MSZoning=mz_list
g=sns.factorplot(x="MSZoning", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[1]
df.Street.unique()
g=sns.factorplot(x="Street", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[2]
df.Alley.unique()
g=sns.factorplot(x="Alley", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[3]
df.LotShape.unique()
g=sns.factorplot(x="LotShape", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
ls_list=[]

for i in range(0, len(df.LotShape), 1):

    if df.LotShape[i]=="IR3":

        ls_list.append("IR")

    elif df.LotShape[i]=="IR2":

        ls_list.append("IR")

    elif df.LotShape[i]=="IR1":

        ls_list.append("IR")

    else:

        ls_list.append("R")

set(ls_list)

df.LotShape=ls_list
g=sns.factorplot(x="LotShape", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[4]
df.LandContour.unique()
g=sns.factorplot(x="LandContour", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
lc_list=[]

for i in df.LandContour:

    if i=="Low":

        lc_list.append("LH")

    elif i=="HLS":

        lc_list.append("LH")

    elif i=="Lvl":

        lc_list.append("Lvl")

    else:

        lc_list.append("Bnk")

df.LandContour=lc_list
set(lc_list)
g=sns.factorplot(x="LandContour", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[5]
df.Utilities.unique()
type_features[6]
df.LotConfig.unique()
g=sns.factorplot(x="LotConfig", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
loc_list=[]

for i in df.LotConfig:

    if i=="FR3":

        loc_list.append("CF")

    elif i=="CulDSac":

        loc_list.append("CF")

    elif i=="Inside":

        loc_list.append("IFC")

    elif i=="FR2":

        loc_list.append("IFC")

    elif i=="Corner":

        loc_list.append("IFC")

set(loc_list)

df.LotConfig=loc_list
g=sns.factorplot(x="LotConfig", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[7]
df.LandSlope.unique()
g=sns.factorplot(x="LandSlope", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[8]
df.Neighborhood.unique()
List=list(df.Neighborhood.unique())

price=[]

for i in List:

    x=df[df.Neighborhood==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Neighborhood of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Neighborhood and Price", fontsize=17)

plt.show()
n_list=[]

for i in df.Neighborhood:

    if i=="NridgHt":

        n_list.append("A")

    elif i=="NoRidge":

        n_list.append("A")

    elif i=="StoneBr":

        n_list.append("A")

    elif i=="Timber":

        n_list.append("B")

    elif i=="Veenker":

        n_list.append("B")

    elif i=="Somerst":

        n_list.append("B")

    elif i=="ClearCr":

        n_list.append("B")

    elif i=="Crawfor":

        n_list.append("C")

    elif i=="CollgCr":

        n_list.append("C")

    elif i=="Blmngtn":

        n_list.append("C")

    elif i=="Gilbert":

        n_list.append("C")

    elif i=="NWAmes":

        n_list.append("C")

    elif i=="SawyerW":

        n_list.append("C")

    elif i=="Mitchel":

        n_list.append("D")

    elif i=="NAmes":

        n_list.append("D")

    elif i=="NPkVill":

        n_list.append("D")

    elif i=="SWISU":

        n_list.append("D")

    elif i=="Blueste":

        n_list.append("D")

    elif i=="Sawyer":

        n_list.append("D")

    elif i=="BrkSide":

        n_list.append("E")

    elif i=="Edwards":

        n_list.append("E")

    elif i=="OldTown":

        n_list.append("E")

    elif i=="BrDale":

        n_list.append("F")

    elif i=="IDOTRR":

        n_list.append("F")

    elif i=="MeadowV":

        n_list.append("F")

set(n_list)

df.Neighborhood=n_list
g=sns.factorplot(x="Neighborhood", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[9]
df.Condition1.unique()
g=sns.factorplot(x="Condition1", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
c1_list=[]

for i in df.Condition1:

    if i=="RRNn":

        c1_list.append("RP")

    elif i=="PosA":

        c1_list.append("RP")

    elif i=="PosN":

        c1_list.append("POR")

    elif i=="RRNe":

        c1_list.append("POR")

    elif i=="RRAn":

        c1_list.append("NR")

    elif i=="Norm":

        c1_list.append("NR")

    elif i=="Feedr":

        c1_list.append("FR")

    elif i=="RRAe":

        c1_list.append("FR")

    elif i=="Artery":

        c1_list.append("A")

set(c1_list)

df.Condition1=c1_list
g=sns.factorplot(x="Condition1", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[10]
g=sns.factorplot(x="Condition2", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
c2_list=[]

for i in df.Condition2:

    if i=="Norm":

        c2_list.append("N")

    elif i=="Feedr":

        c2_list.append("FR")

    elif i=="RRAn":

        c2_list.append("FR")

    elif i=="Artery":

        c2_list.append("AR")

    elif i=="RRNn":

        c2_list.append("AR")

    else:

        c2_list.append("POS")

set(c2_list)

df.Condition2=c2_list
g=sns.factorplot(x="Condition2", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
df[df.Condition2=="POS"]
df[type_features]
type_features[11]
df.BldgType.unique()
g=sns.factorplot(x="BldgType", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
bt_list=[]

for i in df.BldgType:

    if i=="1Fam":

        bt_list.append("FT")

    elif i=="TwnhsE":

        bt_list.append("FT")

    elif i=="Twnhs":

        bt_list.append("DTF")

    elif i=="2fmCon":

        bt_list.append("DTF")

    elif i=="Duplex":

        bt_list.append("DTF")

set(bt_list)

df.BldgType=bt_list
g=sns.factorplot(x="BldgType", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[12]
df.HouseStyle.unique()
g=sns.factorplot(x="HouseStyle", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
hs_list=[]

for i in df.HouseStyle:

    if i=="1Story":

        hs_list.append("SF1")

    elif i=="SLvl":

        hs_list.append("SF1")

    elif i=="2.5Fin":

        hs_list.append("SF1")

    elif i=="1.5Unf":

        hs_list.append("1.5U")

    elif i=="2Story":

        hs_list.append("2S")

    else:

        hs_list.append("UFS")

set(hs_list)

df.HouseStyle=hs_list
g=sns.factorplot(x="HouseStyle", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[13]
df.RoofStyle.unique()
g=sns.factorplot(x="RoofStyle", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
rs_list=[]

for i in df.RoofStyle:

    if i=="Shed":

        rs_list.append("S")

    elif i=="Flat":

        rs_list.append("FH")

    elif i=="Hip":

        rs_list.append("FH")

    else:

        rs_list.append("GMG")

set(rs_list)

df.RoofStyle=rs_list
g=sns.factorplot(x="RoofStyle", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[14]
df.RoofMatl.unique()
g=sns.factorplot(x="RoofMatl", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
rm_list=[]

for i in df.RoofMatl:

    if i=="Roll":

        rm_list.append("R")

    elif i=="Membran":

        rm_list.append("WMW")

    elif i=="WDShake":

        rm_list.append("WMW")

    elif i=="WdShngl":

        rm_list.append("WMW")

    else:

        rm_list.append("CMT")

set(rm_list)

df.RoofMatl=rm_list
g=sns.factorplot(x="RoofMatl", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[15]
df.Exterior1st.unique()
df.SalePrice=df.SalePrice.astype(float)

List=list(df.Exterior1st.unique())

price=[]

for i in List:

    x=df[df.Exterior1st==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Exterior Covering of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Exterior Covering of House and Price", fontsize=17)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
e1_list=[]

for i in df.Exterior1st:

    if i=="ImStucc":

        e1_list.append("A")

    elif i=="Stone":

        e1_list.append("A")

    elif i=="CemntBd":

        e1_list.append("A")

    elif i=="VinylSd":

        e1_list.append("A")

    elif i=="BrkFace":

        e1_list.append("B")

    elif i=="Plywood":

        e1_list.append("B")

    elif i=="AsbShng":

        e1_list.append("D")

    elif i=="Cblock":

        e1_list.append("D")

    elif i=="AsphShn":

        e1_list.append("D")

    elif i=="BrkComm":

        e1_list.append("D")

    else:

        e1_list.append("C")

set(e1_list)

df.Exterior1st=e1_list
df.SalePrice=df.SalePrice.astype(float)

List=list(df.Exterior1st.unique())

price=[]

for i in List:

    x=df[df.Exterior1st==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Exterior Covering of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Exterior Covering of House and Price", fontsize=17)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
type_features[16]
df.Exterior2nd.unique()
df.SalePrice=df.SalePrice.astype(float)

List=list(df.Exterior2nd.unique())

price=[]

for i in List:

    x=df[df.Exterior2nd==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Exterior Covering of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Exterior Covering of House and Price", fontsize=17)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
e2_list=[]

for i in df.Exterior2nd:

    if i=="Other":

        e2_list.append("O")

    elif i=="CmentBd":

        e2_list.append("A")

    elif i=="VnylSd":

        e2_list.append("A")

    elif i=="ImStucc":

        e2_list.append("A")

    elif i=="BrkFace":

        e2_list.append("A")

    elif i=="PlyWood":

        e2_list.append("A")

    elif i=="HdBoard":

        e2_list.append("A")

    elif i=="Brk Cmn":

        e2_list.append("C")

    elif i=="AsbShng":

        e2_list.append("C")

    elif i=="CBlock":

        e2_list.append("C")

    else:

        e2_list.append("B")

set(e2_list)

df.Exterior2nd=e2_list
df.SalePrice=df.SalePrice.astype(float)

List=list(df.Exterior2nd.unique())

price=[]

for i in List:

    x=df[df.Exterior2nd==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.ylabel("Average Price ($)", fontsize=14)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
type_features[17]
df.MasVnrType.unique()
g=sns.factorplot(x="MasVnrType", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[18]
df.Foundation.unique()
g=sns.factorplot(x="Foundation", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[19]
df.Heating.unique()
g=sns.factorplot(x="Heating", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
h_list=[]

for i in df.Heating:

    if i=="GasA":

        h_list.append("A")

    elif i=="GasW":

        h_list.append("A")

    elif i=="OthW":

        h_list.append("A")

    else:

        h_list.append("B")

set(h_list)

df.Heating=h_list
g=sns.factorplot(x="Heating", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[20]
df.CentralAir.unique()
g=sns.factorplot(x="CentralAir", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[21]
g=sns.factorplot(x="Electrical", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
e_list=[]

for i in df.Electrical:

    if i=="SBrkr":

        e_list.append("A")

    elif i=="FuseF":

        e_list.append("B")

    elif i=="FuseA":

        e_list.append("B")

    else:

        e_list.append("C")

set(e_list)

df.Electrical=e_list
g=sns.factorplot(x="Electrical", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[22]
df.Functional.unique()
g=sns.factorplot(x="Functional", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
fu_list=[]

for i in df.Functional:

    if i=="Typ":

        fu_list.append("A")

    elif i=="Maj2":

        fu_list.append("C")

    else:

        fu_list.append("B")

set(fu_list)

df.Functional=fu_list
g=sns.factorplot(x="Functional", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[23]
df.GarageType.unique()
g=sns.factorplot(x="GarageType", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
gt_list=[]

for i in df.GarageType:

    if i=="Attchd":

        gt_list.append("A")

    elif i=="BuiltIn":

        gt_list.append("A")

    elif i=="NA":

        gt_list.append("C")

    elif i=="CarPort":

        gt_list.append("C")

    else:

        gt_list.append("B")

set(gt_list)

df.GarageType=gt_list
g=sns.factorplot(x="GarageType", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[24]
df.GarageFinish.unique()
g=sns.factorplot(x="GarageFinish", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
gf_list=[]

for i in df.GarageFinish:

    if i=="RFn":

        gf_list.append("RF")

    elif i=="Fin":

        gf_list.append("RF")

    else:

        gf_list.append("UN")

set(gf_list)

df.GarageFinish=gf_list
g=sns.factorplot(x="GarageFinish", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[25]
df.PavedDrive.unique()
g=sns.factorplot(x="PavedDrive", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[26]
df.MiscFeature.unique()
g=sns.factorplot(x="MiscFeature", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
mf_list=[]

for i in df.MiscFeature:

    if i=="NA":

        mf_list.append("NA")

    else:

        mf_list.append("GSO")

set(mf_list)

df.MiscFeature=mf_list
g=sns.factorplot(x="MiscFeature", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
type_features[27]
df.SaleType.unique()
df.SalePrice=df.SalePrice.astype(float)

List=list(df.SaleType.unique())

price=[]

for i in List:

    x=df[df.SaleType==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.ylabel("Average Price ($)", fontsize=14)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
st_list=[]

for i in df.SaleType:

    if i=="New":

        st_list.append("NC")

    elif i=="Con":

        st_list.append("NC")

    elif i=="CWD":

        st_list.append("CC")

    elif i=="ConLI":

        st_list.append("CC")

    else:

        st_list.append("O")

set(st_list)

df.SaleType=st_list
df.SalePrice=df.SalePrice.astype(float)

List=list(df.SaleType.unique())

price=[]

for i in List:

    x=df[df.SaleType==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.ylabel("Average Price ($)", fontsize=14)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
type_features[28]
df.SaleCondition.unique()
df.SalePrice=df.SalePrice.astype(float)

List=list(df.SaleCondition.unique())

price=[]

for i in List:

    x=df[df.SaleCondition==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.ylabel("Average Price ($)", fontsize=14)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
sc_list=[]

for i in df.SaleCondition:

    if i=="Partial":

        sc_list.append("P")

    elif i=="AdjLand":

        sc_list.append("A")

    else:

        sc_list.append("O")

set(sc_list)

df.SaleCondition=sc_list
df.SalePrice=df.SalePrice.astype(float)

List=list(df.SaleCondition.unique())

price=[]

for i in List:

    x=df[df.SaleCondition==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.ylabel("Average Price ($)", fontsize=14)

plt.text(-0.4, 200000, "", fontsize=14)

plt.show()
type_features[29]
df.MSSubClass.unique()
train_df.SalePrice=train_df.SalePrice.astype(float)

List=list(train_df.MSSubClass.unique())

price=[]

for i in List:

    x=train_df[train_df.MSSubClass==i]

    priceav=x.SalePrice.mean()

    price.append(priceav)

data=pd.DataFrame({"type":List, "price_average":price})

new_index=(data.price_average.sort_values(ascending=True)).index.values

sorted_data=data.reindex(new_index)



plt.figure(figsize=(18,9))

sns.barplot(x=sorted_data.type, y=sorted_data.price_average)

plt.xlabel("Type of House", fontsize=14)

plt.ylabel("Average Price ($)", fontsize=14)

plt.title("Relation Between Type and Price", fontsize=17)

plt.text(-0.3, 101000, "20: 1-STORY 1946 & NEWER ALL STYLES\n30: 1-STORY 1945 & OLDER\n40: 1-STORY W/FINISHED ATTIC ALL AGES\n45: 1-1/2 STORY - UNFINISHED ALL AGES\n50: 1-1/2 STORY FINISHED ALL AGES\n60: 2-STORY 1946 & NEWER\n70: 2-STORY 1945 & OLDER\n75: 2-1/2 STORY ALL AGES\n80: SPLIT OR MULTI-LEVEL\n85: SPLIT FOYER\n90: DUPLEX - ALL STYLES AND AGES\n120: 1-STORY PUD (Planned Unit Development) - 1946 & NEWER\n150: 1-1/2 STORY PUD - ALL AGES\n160: 2-STORY PUD - 1946 & NEWER\n180: PUD - MULTILEVEL - INCL SPLIT LEV/FOYER\n190: 2 FAMILY CONVERSION - ALL STYLES AND AGES", fontsize=14)

plt.show()
mc_list=[]

for i in df.MSSubClass:

    if i=="120":

        mc_list.append("A")

    elif i=="60":

        mc_list.append("A")

    elif i=="20":

        mc_list.append("B")

    elif i=="40":

        mc_list.append("B")

    elif i=="70":

        mc_list.append("B")

    elif i=="80":

        mc_list.append("B")

    elif i=="160":

        mc_list.append("C")

    elif i=="75":

        mc_list.append("C")

    elif i=="85":

        mc_list.append("C")

    elif i=="90":

        mc_list.append("C")

    elif i=="50":

        mc_list.append("C")

    else:

        mc_list.append("D")

set(mc_list)

df.MSSubClass=mc_list
g=sns.factorplot(x="MSSubClass", y="SalePrice", data=df, kind="bar")

g.set_ylabels("Price")

plt.show()
df[type_features]
df=pd.get_dummies(df, columns=type_features)
df.head()
numerical_features
f,ax=plt.subplots(figsize=(22,22))

sns.heatmap(df[numerical_features].corr(), vmax=1, vmin=-1, annot=True, fmt=".2f")

plt.show()
len(list(df[df.GarageYrBlt=="NA"].index.values))
df.GarageYrBlt=df.GarageYrBlt.replace("NA", 1979)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len=len(train_df)
train_data=df[:train_df_len]
sub_data=df[train_df_len:]
train_data.drop(["Id"], axis=1, inplace=True)

train_data
x=train_data.drop(["SalePrice"], axis=1)

y=train_data.SalePrice
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=42, test_size=0.3)
logreg=LogisticRegression()

logreg.fit(x_train, y_train)

train_acc=round(logreg.score(x_train, y_train)*100,3)

test_acc=round(logreg.score(x_test, y_test)*100,3)

print("Training Accuracy:", train_acc, "%")

print("Test Accuracy:", test_acc, "%")
y_pred=logreg.predict(x_test)
predd=pd.DataFrame({"pred":y_pred, "test":y_test})
predd[predd.pred==predd.test]
pred100=[]

for i in predd.pred:

    pred100.append(100*(int(i/100)))

predd["pred100"]=pred100
predd[predd.pred100==predd.test]
pred1000=[]

for i in predd.pred:

    pred1000.append(1000*(int(i/1000)))

predd["pred1000"]=pred1000
predd[predd.pred1000==predd.test]
pred10000=[]

for i in predd.pred:

    pred10000.append(10000*(int(i/10000)))

predd["pred10000"]=pred10000
predd[predd.pred10000==predd.test]
predd.head(30)
#######################################
rs=42

classifier=[DecisionTreeClassifier(random_state=rs),

           SVC(random_state=rs),

           RandomForestClassifier(random_state=rs),

           KNeighborsClassifier(),

           LogisticRegression(random_state=rs)]



dt_param_grid={"min_samples_split": range(10,500,20), 

               "max_depth": range(1,20,2)}



svc_param_grid={"kernel": ["rbf"], 

               "gamma": [0.001, 0.01, 0.1, 1],

               "C": [1,10,50,100,200,300,1000]}



rf_param_grid={"max_features": [1,3,10],

              "min_samples_split": [2,3,10],

              "min_samples_leaf": [1,3,10],

              "bootstrap": [False],

              "n_estimators": [100,300],

              "criterion": ["gini"]}



knn_param_grid={"n_neighbors": np.linspace(1,19,10, dtype=int).tolist(),

               "weights": ["distance", "uniform"],

               "metric": ["euclidean", "manhattan"]}



logreg_param_grid={"C": np.logspace(-3,3,7),

                  "penalty": ["l1", "l2"]}



classifier_param=[dt_param_grid, svc_param_grid, rf_param_grid, knn_param_grid, logreg_param_grid]
cv_result=[]

best_estimators=[]

for i in range(len(classifier)):

    clf=GridSearchCV(classifier[i], param_grid=classifier_param[i], cv=StratifiedKFold(n_splits=10), scoring="accuracy", n_jobs=-1, verbose=1)

    clf.fit(x_train, y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_result=[100*each for each in cv_result]
results=pd.DataFrame({"Cross Validation Best Scores": cv_result, "ML Models": ["DecisionTreeClassifier", "SVM", "RandomForestClassifier", "KNeighborsClassifier", "LogisticRegression"]})

f,ax=plt.subplots(figsize=(12,7))

g = sns.barplot(data=results, y="ML Models", x="Cross Validation Best Scores")

g.set_ylabel("")

g.set_xlabel("Accuracy %")

plt.show()

for i in range(len(results)):

    print(results["ML Models"][i], "Accuracy:", results["Cross Validation Best Scores"][i], "%")
voting_c=VotingClassifier(estimators=[("dt", best_estimators[0]), ("rf", best_estimators[2])],

                         voting="soft", n_jobs=-1)

voting_c=voting_c.fit(x_train, y_train)

print("Accuracy:", 100*accuracy_score(voting_c.predict(x_test), y_test), "%")
sub_data.drop(["Id", "SalePrice"], axis=1, inplace=True)
sub_price=pd.Series(voting_c.predict(sub_data), name="SalePrice").astype(float)

result=pd.concat([submission_df["Id"], sub_price], axis=1)
result.to_csv("submission.csv", index=False)