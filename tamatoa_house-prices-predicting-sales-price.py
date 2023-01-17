

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

from scipy.stats import norm

import seaborn as sns;sns.set(style="whitegrid",color_codes=True) #data visualization and tune set params

import matplotlib.pyplot as plt;plt.figure(figsize=(15,15)) #data plotting/visualization

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



print ("Train dataset has {0} rows and {1} columns".format(train.shape[0],train.shape[1]))

print ("Test dataset has {0} rows and {1} columns".format(test.shape[0],test.shape[1]))

#lets check for the missing values in the train dataset as missing values can really spoil your predictions

missing_values = train.columns[train.isnull().any()]

#we shall be using bash style color codes ***\033[1;36m*** to differentiate the ouput

print ("\033[1;36m"+"There are {0} missing values in the dataset Train\n---------------------------------------------------------".format(len(missing_values))+"\033[0m")

#We see that there are only 19 missing values only we can go ahead and see the column names with missing values

for missing_vals in missing_values:#loop througth the list and print all the column names

    print ("\033[1;32m"+missing_vals+"\033[0m")

print ("\033[1;36m"+"---------------------------------------------------------"+"\033[0m")
#Then we shall be getting the percentage of the missing values in columns of our dataset like below

percentage_missing = train.isnull().sum()/len(train)

percentage_missing = percentage_missing[percentage_missing > 0]

percentage_missing.sort_values(inplace=True)#we use inplace=True to make changes to our columns

print(percentage_missing)

#lets plot to visualize the missing values

percentage_missing = percentage_missing.to_frame()

percentage_missing.columns=['Count']

percentage_missing.index.names = ['Name']

percentage_missing['Name'] = percentage_missing.index

plt.figure(figsize=(15,15))

sns.barplot(x="Name",y="Count",data=percentage_missing)

plt.xticks(rotation=90)

###Now lets check the distribution of the target variable SalePrice

plt.figure(figsize=(15,15))

sns.distplot(train["SalePrice"])

print ("SalePrice right-skewenes = {0}".format(train["SalePrice"].skew()))

#We see that the target variable SalePrice is right-skewed we need to log transform this variable so that it can be normally distributed. as Normally distributed variables help in better modelling the relationship between the target variable and independent variables

normal_dist = np.log(train["SalePrice"])

"""Then we can see that atleast it has been nearly normally distributed"""

print ("SalePrice skew now = {0}".format(normal_dist.skew()))

plt.figure(figsize=(15,15))

sns.distplot(normal_dist)
#Now we need to separate Numerical and categorical data from the dataset each will be attacked and analyzed from defferent angles

numerical_data = train.select_dtypes(include=[np.number])

categorical_data = train.select_dtypes(exclude=[np.number])

print("The dataset has {0} numerical data and {1} categrical data".format(numerical_data.shape[1],categorical_data.shape[1]))

#we are interested to learn about the correlation behavior of numeric variables. Out of 38 variables, I presume some of them must be correlated. If found, we can later remove these correlated variables as they won't provide any useful information to the model.

corr = numerical_data.corr()

plt.figure(figsize=(15,15))

sns.heatmap(corr)
#a numerical correlation data 

print ("\033[1;34m"+"The Correlation of the first 15 variables\n-------------------------------")

print(corr["SalePrice"].sort_values(ascending=True)[:15])

print("\033[0m")

print ("\033[1;35m"+"The Correlation of the last 5 variables\n--------------------------------")

print(corr["SalePrice"].sort_values(ascending=True)[-5:])

print("\033[0m")
print(train['OverallQual'].unique())

#OverallQual is between a scale of 1 t0 10, 

#which we can fairly treat as an ordinal variable order.An ordinal value has an inherent order

#finally lets compute our median and visualize it in a plot

pivot = train.pivot_table(index="OverallQual",values="SalePrice",aggfunc=np.median)

#show a more sorted table

print(pivot.sort_values)

pivot.plot(kind='bar',color="magenta")
plt.figure(figsize=(15,15))

sns.jointplot(x=train['GrLivArea'],y=train['SalePrice'])
#lets first understand whats in the categorical features

categorical_data.describe()
#calculate median

cat_pivot = train.pivot_table(index="SaleCondition",values="SalePrice",aggfunc=np.median)

cat_pivot.sort_values
#we go ahead and visualize the median cat_pivot

cat_pivot.plot(kind="bar",color="blue")
#we use list compression to assign our return values to cat and for simplicity

#Read more about this technique and you will enjoy its power

cat=[f for f in train.columns if train.dtypes[f]=="object"]

#print(cat)

def anova(frame):

    anv = pd.DataFrame()

    anv['features'] = cat

    pvals=[]

    for c in cat:

        samples=[]

        for cls in frame[c].unique():

            s=frame[frame[c]==cls]['SalePrice'].values

            samples.append(s)

        pval=stats.f_oneway(*samples)[1]

        pvals.append(pval)

    anv["pval"]=pvals

    return anv.sort_values("pval")
plt.figure(figsize=(15,15))

categorical_data['SalePrice']=train.SalePrice.values

k=anova(categorical_data)

k['disparity']=np.log(1./k['pval'].values)

sns.barplot(data=k,x="features",y="disparity")

plt.xticks(rotation=90)

plt.show()
#lets create numeric data plots

num=[f for f in train.columns if train[f].dtypes != "object"]

num.remove("Id")

c = {'color': ['r']}

nd = pd.melt(train,value_vars=num)

n1 = sns.FacetGrid(nd,col="variable", hue_kws=c,col_wrap=4,sharex=False,sharey=False)

n1.map(sns.distplot,"value")

n1
#lets plot categorical data

def boxplot(x,y,**kwargs):

    sns.boxplot(x=x,y=y)

    x=plt.xticks(rotation=90)

cate = [f for f in train.columns if train[f].dtypes=="object"]

nd =pd.melt(train,id_vars="SalePrice",value_vars=cate)

np1 = sns.FacetGrid(nd,col="variable",col_wrap=2,sharex=False,sharey=False,size=5)

np1.map(boxplot,"value","SalePrice")
#Guys lets remove outliers

train.drop(train[train["GrLivArea"]>4000].index,inplace=True)

train.shape #seems and its true we remove 4 rows compared to our original data
#imputing using mode

test.loc[666, 'GarageQual'] = "TA" #stats.mode(test['GarageQual']).mode

test.loc[666, 'GarageCond'] = "TA" #stats.mode(test['GarageCond']).mode

test.loc[666, 'GarageFinish'] = "Unf" #stats.mode(test['GarageFinish']).mode

test.loc[666, 'GarageYrBlt'] = "1980" #np.nanmedian(test['GarageYrBlt'])
#let us mark it as a NA variable

test.loc[1116,'GarageType']=np.nan
# import func LabelEncoder

from sklearn.preprocessing import LabelEncoder

lben = LabelEncoder()

def vectorize(data,var,fill_na=None):

    if fill_na is not None:

        data[var].fill_na(fill_na,inplace=True)

    lben.fit(data[var])

    data[var]=lben.transform(data[var])

    return data
#combine the dataset

alldata = train.append(test)

#alldata.shape



#now lets impute LotFrontage by the median of Neighbourhood

lotf_by_neig=train['LotFrontage'].groupby(train["Neighborhood"])

for key,group in lotf_by_neig:

    idx=(alldata["Neighborhood"]==key)&(alldata['LotFrontage'].isnull())

    alldata.loc[idx,'LotFrontage'] = group.median()
#impute missing values

alldata["MasVnrArea"].fillna(0, inplace=True)

alldata["BsmtFinSF1"].fillna(0, inplace=True)

alldata["BsmtFinSF2"].fillna(0, inplace=True)

alldata["BsmtUnfSF"].fillna(0, inplace=True)

alldata["TotalBsmtSF"].fillna(0, inplace=True)

alldata["GarageArea"].fillna(0, inplace=True)

alldata["BsmtFullBath"].fillna(0, inplace=True)

alldata["BsmtHalfBath"].fillna(0, inplace=True)

alldata["GarageCars"].fillna(0, inplace=True)

alldata["GarageYrBlt"].fillna(0.0, inplace=True)

alldata["PoolArea"].fillna(0, inplace=True)
qual_dict={np.nan:0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}

name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond',\

                 'HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])

for i in name:

    alldata.head()