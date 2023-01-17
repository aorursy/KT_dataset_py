import warnings

warnings.filterwarnings("ignore")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(rc={'figure.figsize':(15,8)})

plt.figure(figsize=(15,8))
df = pd.read_csv("../input/house-price-continuation/house_price_continuation.csv",index_col = [0])
df.head()
df.shape
df.columns
columns = list(df.columns)
columns
colmns_small = [x.lower() for x in columns]
colmns_small
def rename(dataframe,deflt_col_name,col_rename):

    return dataframe.rename(columns = {deflt_col_name:col_rename}, inplace = True)
for i in range(len(columns)):

    rename(df,columns[i],colmns_small[i])
df.columns
# categorical data EDA func

def catvar(var,tar):

    print("---------------Head of Data------------")

    print(var.head(20))

    print("---------------Tail of Data------------")

    print(var.tail(20))

    print("-------Describing data------")

    print(var.describe())

    print("-------Percentage of data points-------")

    print(var.value_counts(normalize = True))

    print("-------NULL Values check------")

    print(var.isnull().sum())

    print("-------Percentage of NULL Values ------")

    print(var.isnull().sum()/var.shape[0])

    print("------Shape of Data Frame------")

    print(var.shape)

    print("-------BOX PLOT-------")

    sns.boxplot(var,tar)

    

    
# Numerical data EDA func

def numvar(var,tar):

    print("---------------Head of Data------------")

    print(var.head(20))

    print("---------------Tail of Data------------")

    print(var.tail(20))

    print("-------Describing data------")

    print(var.describe())

    print("-------NULL Values check------")

    print(var.isnull().sum())

    print("-------Percentage of NULL Values ------")

    print(var.isnull().sum()/var.shape[0])

    print("-------Skewness check-------")

    print(var.skew())

    print("------correlation check------")

    print(var.corr(tar))

    print("------Shape of Data Frame------")

    print(var.shape)

    #print("-------DIST PLOT-------")

    #sns.distplot(var,bins = 30)

    print("-------REG PLOT---------")

    sns.regplot(var,tar)

    

    
catvar(df.bsmtfintype2,df.saleprice)
# we infer that most of 88% data is unfinished
# there are 2 percentage null values 

# we can treat them
is_nan = df[['bsmtfintype2']].isnull()

rows_has_nan = is_nan.any(axis=1)

rows_with_nan = df[['foundation','bsmtqual', 'bsmtcond', 'bsmtexposure', 'bsmtfintype1',

       'bsmtfinsf1', 'bsmtfintype2']][rows_has_nan]
print(rows_with_nan)
# this says that No Basement section of other columns is related with NAN values in this column
df.bsmtfintype2.fillna("NoBs",inplace = True)
catvar(df.bsmtfintype2,df.saleprice)
# na values are no basement values.

# so its filled with "NoBs" values
#numvar(df.bsmtfinsf2,df.saleprice)
print("-------REG PLOT---------")

sns.regplot(df.bsmtfinsf2,df.saleprice)
df1 = df[df.bsmtfinsf2 != 0]
df1.shape
print("-------REG PLOT---------")

sns.regplot(df1.bsmtfinsf2,df1.saleprice)
df1.bsmtfinsf2.corr(df1.saleprice)
df.bsmtfinsf2.corr(df.saleprice)
# this shows that after removing 0 values corr comes to 2%

# before remvoing it was negatively correlated
numvar(df.bsmtunfsf,df.saleprice)
df[df.bsmtunfsf == 0].shape[0]/df.shape[0]
# basement unfinished sqfeet is 8%

# so 8% of houses have finished the basement
sns.distplot(df.bsmtunfsf,bins = 30)
df2 = df[df.bsmtunfsf != 0]
numvar(df2.bsmtunfsf,df2.saleprice)
sns.distplot(df2.bsmtunfsf,bins = 30)
# this variable doesnot have any impact on target variable

# so we can drop this variables

def drop(var):

    return df.drop(var, inplace = True, axis = 1)
#drop(["bsmtunfsf",'bsmtfinsf2'])
df.shape
numvar(df.totalbsmtsf,df.saleprice)
def distplot(var):

    return sns.distplot(var,bins = 30)
distplot(df.totalbsmtsf)
df3 = df[df.totalbsmtsf !=0]
numvar(df3.totalbsmtsf,df3.saleprice)
def box(var):

    return sns.boxplot(var)
box(df.totalbsmtsf)
box(df3.totalbsmtsf)
# no of zero values

((df.shape[0] - df3.shape[0])/df.shape[0])*100
# 2.5 % zero values
# 2.5% houses didnot have basement still
catvar(df.heating,df.saleprice)
# this shows most of the houses have GasA
catvar(df.heatingqc,df.saleprice)
# around most of the houses have excellent heating system
catvar(df.centralair, df.saleprice)
# around 93% houses have central air system
catvar(df.electrical,df.saleprice)
df.dropna(subset = ['electrical'],inplace = True, axis = 0)
df.shape
numvar(df['1stflrsf'],df.saleprice)
distplot(df['1stflrsf'])
numvar(df['2ndflrsf'],df.saleprice)
distplot(df['2ndflrsf'])
df4 = df[df['2ndflrsf']!=0]
numvar(df4['2ndflrsf'],df4.saleprice)
distplot(df4['2ndflrsf'])
# the skewness is reduced and correlation bw variables have been increased

# this variable influences the target data
numvar(df.lowqualfinsf,df.saleprice)
df5 = df[df.lowqualfinsf!=0]
numvar(df5.lowqualfinsf,df5.saleprice)
# if we remove 0 values the correlation goes to 30%

# this 0 values indicates most of the finished square feet are of high quality
(df.shape[0]-df5.shape[0])/df.shape[0]
# 98% values are 0 , so most of the values are high quality finished square feet
# so the data doesnot have more variance

df.lowqualfinsf.var()
np.var(df.lowqualfinsf)
# so we can ignore the column
numvar(df.grlivarea, df.saleprice)
distplot(df.grlivarea)
# this data is highly skewed

# and also highly correlated
df.bsmtfullbath.head()
catvar(df.bsmtfullbath,df.saleprice)
# this data shows 58% of 0 full bathrooms

# 40% of 1 full bathrooms
catvar(df.bsmthalfbath,df.saleprice)
# 94 of the data shows that there is zero half bathrooms
catvar(df.fullbath, df.saleprice)
# this shows the pattern in the no of full bathrooms
catvar(df.halfbath, df.saleprice)
# this variable also shows some pattern with the target variable
catvar(df.bedroomabvgr, df.saleprice)
# this graph shows most of the houses have 3 bedrooms above grade.

# second is 2 bedrooms above grade

#third is 4 bathrooms above grade.
catvar(df.kitchenabvgr, df.saleprice)
# maximum all the house have 1 kitchens in above grade.

# so this shows dont have much variance
# this variable doesnot have much impact in target
catvar(df.kitchenqual, df.saleprice)
# this graph shows that most of the data are average quality kitchens

# second is good quality kitchens
catvar(df.totrmsabvgrd, df.saleprice)
# this graph indicates that no of rooms above grade has some pattern with sale price
catvar(df.functional,df.saleprice)
# this graph says most of the data are typical functionality

#second is minor deductions 2

# third is minor deductions 3
catvar(df.fireplaces, df.saleprice)
# this graph has, the target variable is influenced by independant variable
catvar(df.fireplacequ,df.saleprice)
# null values percentage

df.fireplacequ.isnull().sum()/df.shape[0]
# 47% null values
df[['fireplaces','fireplacequ']].head(20)
df[['fireplaces','fireplacequ']].tail(20)
df.fireplacequ.fillna("No",inplace = True)
catvar(df.fireplacequ,df.saleprice)
# this data shows some patterns

# this data has some influence in target
catvar(df.garagetype,df.saleprice)
# this data has some NA values

# no Garage 

df.garagetype.fillna("No",inplace = True)
catvar(df.garagetype,df.saleprice)
# this value has some patterns in data
catvar(df.garageyrblt,df.saleprice)
df.garageyrblt.isnull().sum()/df.shape[0]
# there are 5% missing values
df[['garagetype','garageyrblt']].tail(50)
df.garageyrblt.fillna(np.inf,inplace = True)
df[['garagetype','garageyrblt']].tail(50)
catvar(df.garagefinish,df.saleprice)
df.garagefinish.fillna("No",inplace =True)
catvar(df.garagefinish,df.saleprice)
# this variable has some patterns with the target variable
df.garagecars.head(30)
catvar(df.garagecars, df.saleprice)
# this graph says some patterns according to target variable

# 0 says no car capacity

# 1 2 3 says number of car capacity
df.garagearea.head(20)
numvar(df.garagearea,df.saleprice)
df[df.garagearea == 0].shape
# here 0 has a value because 0 means there is no Garage
distplot(df.garagearea)
# almost the skewness value is very low
df6 = df[df.garagearea != 0]
numvar(df6.garagearea,df6.saleprice)
distplot(df6.garagearea)
# this data has more correlation with tha target as of 61%

# this data influences the target var
catvar(df.garagequal,df.saleprice)
df.garagequal.fillna("No",inplace = True)
catvar(df.garagequal,df.saleprice)
# this data has some patterns with the target variable
df.garagecond.head(20)
catvar(df.garagecond,df.saleprice)
df.garagecond.fillna("No",inplace = True)
catvar(df.garagecond,df.saleprice)
# this data doesnot have much patterns with the target variable
catvar(df.paveddrive,df.saleprice)
# most of the house have the paved drive way about 91%
numvar(df.wooddecksf,df.saleprice)
# this data has correlated with the target var for about 30%

# here 0 indicates the house have no wood deck
distplot(df.wooddecksf)
numvar(df.openporchsf,df.saleprice)
# this data is correlated for about 30% with the target variable

# 0 values says that house have no open porch
distplot(df.openporchsf)
numvar(df.enclosedporch,df.saleprice)
# the 0 values indicates that the house has no enclosed porch
df7 = df[df.enclosedporch !=0]

numvar(df7.enclosedporch,df7.saleprice)
# after removing the 0 values the corr of var increases with the target variable

# first it was about negatively correlated and after removal it was positively corr
df['3ssnporch'].head()
numvar(df['3ssnporch'],df.saleprice)
# this value 0 indicates the houses dont have 3 season porch
df['3ssnporch'].groupby(df['3ssnporch']).count()
# percentage of 0 values

1422/1446
#about 98.3% values are 0 so this variable is not much influencing the target.

# dropping this variable is better

# the variance is not high.
df.screenporch.head()
numvar(df.screenporch,df.saleprice)
df.screenporch.groupby(df.screenporch).count()
1332/1446
# about 92% values are 0 , 0 indicates that the house dont have screen porch

# this variable has not much more impact in target variable
numvar(df.poolarea,df.saleprice)
df.poolarea.groupby(df.poolarea).count()
1440/1446
# 99.5% of values are 0

# because the house dont have poool

# the variation among data is very low

# so this variable has not much impact in target variable
catvar(df.poolqc,df.saleprice)
df.poolqc.fillna("No",inplace= True)
catvar(df.poolqc,df.saleprice)
# this variable says that about all the houses have no pools
catvar(df.fence,df.saleprice)
df.fence.fillna("No",inplace = True)
catvar(df.fence,df.saleprice)
# about 80% houses have no fences

# this variable has not much impact in target variable
catvar(df.miscfeature,df.saleprice)
df.miscfeature.fillna("No",inplace = True)
catvar(df.miscfeature,df.saleprice)
# about 96% of the homes dont have misc features

# this variable has the variation at low

# so this variable has not much impact in target variable
df.miscval.head(20)
df[['miscfeature','miscval']].head(20)
numvar(df.miscval,df.saleprice)
df.miscval.groupby(df.miscval).count()
1396/1446
# around 96.5% of the values are about 0

# this variable doesnot have much variation
df.mosold.head()
catvar(df.mosold,df.saleprice)
# this variable has no variation with the target variable

# all the values are about same range
df.yrsold.head()
catvar(df.yrsold,df.saleprice)
# this variable also has no impact with the target variable

# all the year price ranges are same
df.saletype.head(30)
catvar(df.saletype,df.saleprice)
# this varible does not show much variation with the target variable

# new house has higher rate
df.salecondition.head(50)
catvar(df.salecondition,df.saleprice)
# this variable does not show much variation with the target variable
df.to_csv("house_data_eda")