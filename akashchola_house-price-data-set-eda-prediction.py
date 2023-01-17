#data preprocessing
import pandas as pd         #for dataframe 
import numpy as np          #for linear algebra

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#Showing Working directory for files
import os
for dirname,_,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname,filename))
f = open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt','r')
#print(f.read())
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
print("Total rows :" , df.shape[0])
print("Total columns :" , df.shape[1])
#Setting the no of rows to displays in result
pd.set_option('display.max_rows',df.shape[1]+1)   

#Head of the data in transpose form
df.head().T 
#Columns name of the data
df.columns
#For Basic summary statistics :
pd.set_option('display.max_rows',df.shape[1]+1)
df.describe().T

#Summary statistics in long form.
#Info about the data : variables names and their data type and non null count
df.info()
int_to_obj = []
for i in df.columns:
    if df.loc[:,i].dtype == 'int64':
        if len(df.loc[:,i].unique()) < 15:
            print(i)
            print(len(df.loc[:,i].unique()))
            int_to_obj.append(i)

date_var = ['MoSold','YrSold','YearRemodAdd','YearBuilt'] 
int_to_obj = set(int_to_obj) - set(date_var)   #check this
int_to_obj = ['MSSubClass','OverallQual','OverallCond','GarageCars']
for i in int_to_obj:
    df[i] = df[i].astype(object)
#droping the id columns
df = df.drop('Id',axis = 1)
#Total no of missing values
df.isna().sum().sum()
#Variables which have missing values
dict = {"variable" : [],"Count" : []}
for var in df.columns:
   if df.loc[:,var].isna().sum() > 0:
    #print(var,":",df.loc[:,var].isna().sum())
    dict["variable"].append(var)
    dict["Count"].append(df.loc[:,var].isna().sum())
#Making data frame and sorting them
mis_data = pd.DataFrame(dict).sort_values('Count',ascending =  False)
mis_data.style.background_gradient(cmap = 'Blues')
#Visualization
mis_plot = px.bar(mis_data.sort_values('Count'),x = 'Count',y = 'variable',
            color= 'Count',orientation= 'h',height=700,width=850,title= "Missing Value")
mis_plot.show()
df[list(mis_data['variable'])].info()
#Missing inputation : 
dict1 = {"variable" : [],"Miss_count" : [],"Miss_prop" : [],"mean" : [],"median" : [],"skew":[]}
for var in mis_data.variable:
    if df[var].dtype == 'float64':
        dict1["variable"].append(var)
        dict1["Miss_count"].append(df[var].isna().sum())
        dict1["Miss_prop"].append(df[var].isna().sum()/df.shape[0])
        dict1["mean"].append(df[var].mean())
        dict1["median"].append(df[var].median())
        dict1['skew'].append(df[var].skew())
print(pd.DataFrame(dict1))

#Adjustment for gridSpace
from matplotlib.gridspec import GridSpec

#Size of plots
fig = plt.figure(constrained_layout = True,figsize = (15,5))
gs = GridSpec(2, 3, figure=fig)     #No of grid and plots positions

#Distribution plot
i = 0
col = ['red','blue','green']
for var in list(dict1["variable"]):
    plt.subplot(gs[0,i])
    sns.distplot(df[var],color= col[i])
    i= i+1

#Boxplot
i = 0
for var in list(dict1["variable"]):
    plt.subplot(gs[1,i])
    sns.boxplot(df[var],color = col[i],orient='v')
    i= i+1
#Repalcement of NA's
for var in mis_data.variable:
    if df[var].dtype == 'float64':
        df[var] = df[var].fillna(df[var].median())
        
#Missing inputation : 
dict2 = {"variable" : [],"Miss_count" : [],"Miss_prop" : [],"mode" : [],"mode_freq" : []}
for var in mis_data.variable:
    if df[var].dtype == 'object':
        dict2["variable"].append(var)
        dict2["Miss_count"].append(df[var].isna().sum())
        dict2["Miss_prop"].append(df[var].isna().sum()/df.shape[0])
        dict2["mode"].append(df[var].value_counts().index[0])
        dict2["mode_freq"].append(df[var].value_counts().values[0])
print(pd.DataFrame(dict2))

df.drop(["PoolQC","MiscFeature","Alley","Fence"],axis = 1,inplace=True)
#Checking the Basement fetures
df[df["BsmtCond"].isnull()][["BsmtExposure","BsmtFinType2","BsmtFinType1","BsmtCond","BsmtQual",
   'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']]
basement_features= ["BsmtExposure","BsmtFinType2","BsmtFinType1","BsmtCond","BsmtQual"]
for var in basement_features:
    df[var] = df[var].fillna("No Basement")
#Checking the garage features
df[df["GarageType"].isnull()][["GarageType","GarageFinish","GarageQual","GarageCond",
             'GarageYrBlt','GarageCars', 'GarageArea']]
garage_features = ["GarageType","GarageFinish","GarageQual","GarageCond"]
for var in garage_features:
    df[var] = df[var].fillna("No Garage")
#For fire features
df[df["FireplaceQu"].isna()][["Fireplaces","FireplaceQu"]]
df['FireplaceQu'].fillna("No Fire",inplace = True)
df["MasVnrType"].fillna(df["MasVnrType"].value_counts().index[0],inplace = True)
df["Electrical"].fillna(df["Electrical"].value_counts().index[0],inplace = True)
#Sanity Check : Whether there is any remaining NA's are not. 
df.isna().sum().sum()
print("Total Columns in Train data set:",df.shape[1])
#Loading Test data set
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print("Total rows :" , test_df.shape[0])
print("Total columns :" , test_df.shape[1])
#First will take only those columns presnt in train data set.
col_in_train = df.columns.tolist()
test_df = test_df[list(set(col_in_train) - set(["SalePrice"]))]
print("train shape",df.shape)
print("test shape",test_df.shape)
#Displaying Side By Side Data frame
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
#Over all 
print("Over all NA's Count in test = ",test_df.isna().sum().sum())
mis_data_test = {"Variables" : [],"Count" : [],"dtype":[]}
for var in test_df.columns:
    if test_df.loc[:,var].isna().sum() > 0:
        mis_data_test["Variables"].append(var)
        mis_data_test["Count"].append(test_df.loc[:,var].isna().sum())
        mis_data_test["dtype"].append(test_df[var].dtype)
mis_data_test = pd.DataFrame(mis_data_test).sort_values("Count",ascending= False)

#Missing Info in train and test
print("No of variables having Missing values in Train data set = ",mis_data.shape[0])
print("No of variables having Missing values in Test data set = ",mis_data_test.shape[0])
com_var = list(set(mis_data['variable']).intersection(set(mis_data_test['Variables'])))
print("No of comman variables in both = ",len(com_var))
new_var = list(set(mis_data_test['Variables']) - set(mis_data['variable']))
print("No of new variable having NA's = ",len(new_var))

#will show side by side variables missing df for train and test
display_side_by_side(mis_data.reset_index(),mis_data_test.reset_index(),
                     mis_data_test.loc[mis_data_test["Variables"].isin(new_var),].reset_index())

mis_data_test.dtype.value_counts()
#[1] For numerical variables
var_done = []
for var in mis_data_test.Variables:
    if test_df[var].dtype == 'float64':
        test_df[var] = test_df[var].fillna(test_df[var].median())
        var_done.append(var)
print(len(var_done))
#[2] Same treatment for matching categorical variables.
#[a] basement features
for var in list(set(basement_features).intersection(set(mis_data_test.Variables))):
    test_df[var] = test_df[var].fillna("No Basement")
    var_done.append(var)
print(len(var_done))
#[b] garage features
for var in list(set(garage_features).intersection(set(mis_data_test.Variables))):
    test_df[var] = test_df[var].fillna("No Garage")
    var_done.append(var)
print(len(var_done))
#[c] others
test_df['FireplaceQu'].fillna("No Fire",inplace = True)
test_df["MasVnrType"].fillna(test_df["MasVnrType"].value_counts().index[0],inplace = True)
var_done.extend(['FireplaceQu',"MasVnrType"])
print(len(var_done))
#Remaining Variables
remain_var = set(mis_data_test.Variables) - set(list(var_done))

print("Remaing variables : ","\n",mis_data_test[mis_data_test.Variables.isin(remain_var)])
print(len(var_done))
for var in remain_var:
    test_df[var].fillna(test_df[var].value_counts().index[0],inplace= True)
    var_done.append(var)
print("Total no of var for which replacement is done : ",len(var_done))
print("Total no of remainning var having NA's :",mis_data_test.shape[0]- len(var_done))
print("\n","Total no of NA's in Train : ",df.isna().sum().sum())
print("\n","Total no of NA's in Test : ",test_df.isna().sum().sum())
#Converting some  of int to object
int_to_obj = ['MSSubClass','OverallQual','OverallCond','GarageCars']
for var in int_to_obj:
    test_df[var] = test_df[var].astype(object)
print("Summary : \n",df['SalePrice'].describe())
print("\n Skewness of Sale Price Data = ",df['SalePrice'].skew())
print("\n Median of Sale Price Data = ",df['SalePrice'].median())
#Visualization.
from matplotlib.gridspec import GridSpec
fig = plt.figure(constrained_layout = True, figsize = (15,7))
gs = GridSpec(2,2,figure=fig)
plt.subplot(gs[0,:])
sns.distplot(df['SalePrice'],color="red",)      #Distribution plot
plt.title("Distrution Plot")
plt.subplot(gs[1,0])
sns.boxplot(df['SalePrice'],color="orange",orient='h')   #Boxplot
plt.title("Bar Plot")
plt.subplot(gs[1,1])
sns.violinplot(df['SalePrice'],color = "pink")           #Violin
plt.title("Violin Plot")
plt.show()
print("Total No of Var in Train = ",df.shape[1])
#Categorical Variables
cat_var = df.select_dtypes(include = "object")
print("Total No of categorical variables = ", cat_var.shape[1])

#Numerical Variables
num_int_date_var = df.select_dtypes(exclude = "object")
print("Total No of num_int_date variables = ", num_int_date_var.shape[1])

#Date variables
date_var = df[['MoSold', 'YrSold', 'YearRemodAdd', 'YearBuilt','GarageYrBlt']]
print("Total No of date variables = ",date_var.shape[1])

#Integer variables which have unique less than 15 values : descrete
desc_var = []
for var in num_int_date_var:
    if df[var].dtypes == 'int64' and var not in list(date_var.columns) and len(df[var].unique()) < 15:
        desc_var.append(var)
desc_var = df[desc_var]
print("Total No of descrete Variables = ",desc_var.shape[1])

#Actual Continuous variables
con_var = set(num_int_date_var.columns) - set(date_var.columns) - set(desc_var.columns)
con_var = df[con_var]
print("Total No of Countinuous Variables = ",con_var.shape[1])
cat_var.columns
#First convert the meaning description of int values:
print("Unique Values: ", df["MSSubClass"].unique())
replace_dict = {
        20 : '1-STORY 1946 & NEWER ALL STYLES',
        30: '1-STORY 1945 & OLDER',
        40: '1-STORY W/FINISHED ATTIC ALL AGES',
        45: '1-1/2 STORY - UNFINISHED ALL AGES',
        50: '1-1/2 STORY FINISHED ALL AGES',
        60: '2-STORY 1946 & NEWER',
        70: '2-STORY 1945 & OLDER',
        75: '2-1/2 STORY ALL AGES',
        80: 'SPLIT OR MULTI-LEVEL',
        85: 'SPLIT FOYER',
        90: 'DUPLEX - ALL STYLES AND AGES',
       120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
       150: '1-1/2 STORY PUD - ALL AGES',
       160: '2-STORY PUD - 1946 & NEWER',
       180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
       190: '2 FAMILY CONVERSION - ALL STYLES AND AGES'
}
cat_var["MSSubClass"] = cat_var["MSSubClass"].replace(replace_dict)
print("Unique Values: ", cat_var["MSSubClass"].unique())
#For over all data frame
df["MSSubClass"] = df["MSSubClass"].replace(replace_dict)
#For test
test_df["MSSubClass"] = test_df["MSSubClass"].replace(replace_dict)
fig = plt.figure(figsize = (15,5))
#to order this need refrence table
tab = df.groupby(['MSSubClass'],as_index = False)['SalePrice'].median().sort_values("SalePrice",ascending=False)
mx = sns.boxplot(x = cat_var['MSSubClass'], y = df['SalePrice'],order = tab.MSSubClass)
mx.set_xticklabels(mx.get_xticklabels(),rotation = 75)
fig.suptitle("Category : Type of dewelling",fontsize = 15)
fig.show()
import warnings
warnings.filterwarnings('ignore')
tab = df.groupby(['MSSubClass'])['MSSubClass','SalePrice'].agg({'MSSubClass':'count','SalePrice':'median'})
tab.sort_values('SalePrice',ascending = False).style.background_gradient(cmap = 'Reds')

cat_b = ['MSZoning','Street','LotShape','LandContour','LotConfig','LandSlope']

#Will try with box plot,boxen plot and distribution plot.
fig = plt.figure(constrained_layout = True,figsize = (20,28))
gs = GridSpec(4,2,figure = fig)
for var in cat_b:
    print(var)
    print(df[var].unique())
#Box plots
i=0
for var in  ['MSZoning','LotConfig']:
    plt.subplot(gs[0,i])
    table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
    plot = sns.boxenplot(data = df, x = var,y = 'SalePrice',order = table.iloc[:,0])
    plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)
    i = i+1
i=0
for var in  ['LandContour','LotShape']:
    plt.subplot(gs[1,i])
    table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
    plot = sns.boxplot(data = df, x = var,y = 'SalePrice',order = table.iloc[:,0])
    plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)
    i = i+1
i=2
for var in ['Street','LandSlope']:
    plt.subplot(gs[i,:])
    i = i+1
    unique_vals = df[var].unique()
    targets = [df.loc[df[var] == val] for val in unique_vals]
    for target in targets:
        plot = sns.distplot(target[['SalePrice']])
    plot.legend('upper right',labels = unique_vals)
    plt.xlabel("SalePrice")
    plt.title(var)

plt.suptitle("Category_B : Structure of Land and Property",fontsize = 20)
plt.show()
print("Street \n",df['Street'].value_counts())
print("LandSlop \n",df['LandSlope'].value_counts())
cat_c =  ['Neighborhood','Condition1','Condition2','Utilities','BldgType','HouseStyle','PavedDrive']
for var in cat_c:
    print(var)
    print(df[var].unique())
#will plot box plot, violin plot, boxen plot and strip plot
#Box plot
fig = plt.figure(constrained_layout= True, figsize = (20,20))
gs = GridSpec(4,3, figure = fig)
i = 0 
for var in ['Neighborhood','HouseStyle']:
    plt.subplot(gs[i,:])
    if i == 1:
        plt.subplot(gs[i,:-1])
    table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
    plot = sns.boxplot(data = df , x = var, y = 'SalePrice',orient = 'v',order = table.iloc[:,0])
    i = i+1
i = 0
for var in ['Condition1','Condition2']:
    plt.subplot(gs[2:4,i])
    table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
    plot = sns.boxenplot(data = df , y = var, x = 'SalePrice',orient = 'h',order = table.iloc[:,0])
    i = i+1
i = 1
for var in ['Utilities','BldgType','PavedDrive']:
    plt.subplot(gs[i,-1])
    if i == 1:
        table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
        plot = sns.stripplot(x = var, y = 'SalePrice',order = table.iloc[:,0],data = df )
    else:
        table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
        plot = sns.violinplot(data = df , y = var, x = 'SalePrice',orient = 'h',order = table.iloc[:,0])
    i = i+1
plt.suptitle("CAT_C : Style, location and Outdoors",fontsize = 25)
cat_d =  ['ExterQual','ExterCond','KitchenQual','FireplaceQu','Functional','Heating','HeatingQC','CentralAir','Electrical','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
         'MasVnrType','Foundation']
for var in cat_d:
    print(var,":",len(df[var].unique()))
    
#will plot box , violin and stripplot 
fig = plt.figure(constrained_layout = True, figsize = (20,24))
gs = GridSpec(6,4,figure = fig)

#distribution plot
i = 0
for var in ['ExterQual','KitchenQual','MasVnrType','CentralAir']:
    plt.subplot(gs[0,i])
    unique_vals = df[var].unique()
    targets = [df.loc[df[var] == val] for val in unique_vals]
    for target in targets:
        plot = sns.distplot(target[['SalePrice']])
    plot.legend('upper right',labels = unique_vals)
    plt.xlabel("SalePrice")
    plt.title(var)
    i = i+1

i = 3
for var in ['Exterior1st','Exterior2nd']:
    plt.subplot(gs[1:,i])
    i = i - 1
    table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
    plot = sns.boxenplot(y = var, x = 'SalePrice',order = table.iloc[:,0],data = df ,orient='h')

i = 1
for var in ['ExterCond','HeatingQC','Electrical','FireplaceQu','RoofMatl']:
    if i == 5:
        plt.subplot(gs[i,:-2])
        table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
        plot = sns.boxplot(x = var, y = 'SalePrice',order = table.iloc[:,0],data = df )
    else:
        plt.subplot(gs[i,0])
        table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
        plot = sns.boxplot(x = var, y = 'SalePrice',order = table.iloc[:,0],data = df )
    i = i+1
i = 1        
for var in ['Heating','RoofStyle','Foundation','Functional']:
    plt.subplot(gs[i,1])
    table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
    plot = sns.violinplot(x = var, y = 'SalePrice',order = table.iloc[:,0],data = df )
    i = i+1
    
plt.suptitle("CAT_C : Style, location and Outdoors",fontsize = 25)
cat_e =  ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageCars','GarageQual','GarageCond']
for var in cat_e:
    print(var,":",len(df[var].unique()))
    print(df[var].unique())

fig = plt.figure(constrained_layout = True,figsize = (20,15))
gs = GridSpec(6,4, figure = fig)

plt.subplot(gs[0,:-2])
table = df.groupby(['BsmtFinType1'],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
plot = sns.violinplot(x = 'BsmtFinType1', y = 'SalePrice',order = table.iloc[:,0],data = df )
plot.set_xticklabels(plot.get_xticklabels(),rotation = 90);

plt.subplot(gs[0,2:])
table = df.groupby(['BsmtFinType2'],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
plot = sns.violinplot(x = 'BsmtFinType2', y = 'SalePrice',order = table.iloc[:,0],data = df )
plot.set_xticklabels(plot.get_xticklabels(),rotation = 90);

plt.subplot(gs[1:3,0:2])
var = 'BsmtQual'
unique_vals = df[var].unique()
targets = [df.loc[df[var] == val] for val in unique_vals]
for target in targets:
    plot = sns.distplot(target[['SalePrice']])
plot.legend('upper right',labels = unique_vals);
plt.xlabel("SalePrice");
plt.title(var);

plt.subplot(gs[1:3,2:3])
table = df.groupby(['BsmtCond'],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
plot = sns.boxenplot(x = 'BsmtCond', y = 'SalePrice',order = table.iloc[:,0],data = df )
plot.set_xticklabels(plot.get_xticklabels(),rotation = 90);

plt.subplot(gs[1:3,3:])
table = df.groupby(['BsmtExposure'],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
plot = sns.boxenplot(x = 'BsmtExposure', y = 'SalePrice',order = table.iloc[:,0],data = df )
plot.set_xticklabels(plot.get_xticklabels(),rotation = 90);
fig = plt.figure(constrained_layout=True,figsize=(20,15))
gs = GridSpec(3, 3, figure=fig)

plt.subplot(gs[0,:])
a1 = sns.boxenplot(data=df,x="GarageType",y="SalePrice")
a1.set_xticklabels(a1.get_xticklabels(), rotation=90);

plt.subplot(gs[1,:-1])
var = "GarageFinish"
unique_vals = df[var].unique()
targets = [df.loc[df[var] == val] for val in unique_vals]
for target in targets:
    plot = sns.distplot(target[['SalePrice']])
plot.legend('upper right',labels = unique_vals);
plt.xlabel("SalePrice");
plt.title("GarageFinish")

plt.subplot(gs[1:,-1])
a1 = sns.boxplot(data=df,x="GarageCars",y="SalePrice")
a1.set_xticklabels(a1.get_xticklabels(), rotation=90);

plt.subplot(gs[-1,0])
a1 = sns.stripplot(data=df,x="GarageQual",y="SalePrice")
a1.set_xticklabels(a1.get_xticklabels(), rotation=90);

plt.subplot(gs[-1,-2])
a1 = sns.stripplot(data=df,x="GarageCond",y="SalePrice")
a1.set_xticklabels(a1.get_xticklabels(), rotation=90);
cat_f =  ['SaleType','SaleCondition','OverallQual','OverallCond']
for var in cat_f:
    print(var,":",len(df[var].unique()))
    print(df[var].unique())

rep_dict = {10:'Very Exc',9:'Exc',8:'VG',7:'Good',6:'Abv Avg',5:'Avg',4:'Bel Avg',3:'Fair',2:'Poor',1:'Very Poor'}
for var in ['OverallQual','OverallCond']:
    df[var] = df[var].replace(rep_dict)
    cat_var[var] = df[var].replace(rep_dict)
    test_df[var] = test_df[var].replace(rep_dict)

fig = plt.figure(constrained_layout= True,figsize = (20,16))
gs = GridSpec(4,1,figure=fig)
i = 0
for var in cat_f:
    plt.subplot(gs[i,:])
    if i > 1:
        table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
        plot = sns.boxenplot(x = var, y = 'SalePrice',order = table.iloc[:,0],data = df )
    else:
        table = df.groupby([var],as_index=False)['SalePrice'].median().sort_values('SalePrice',ascending = False)
        plot = sns.boxplot(x = var, y = 'SalePrice',order = table.iloc[:,0],data = df )
    i = i+1
for var in list(desc_var.columns):
    print(df[var].unique())
desc_var.columns

from numpy import median
fig = plt.figure(constrained_layout = True,figsize = (20,15))
gs = GridSpec(5,3,figure = fig)
i = 0
for var in ['BsmtFullBath', 'BsmtHalfBath', 'FullBath']:
    plt.subplot(gs[0,i])
    plot = sns.barplot(data = df, x =var,y = 'SalePrice',estimator = median)
    i = i+1
i = 0
for var in ['HalfBath','KitchenAbvGr', 'Fireplaces']:
    plt.subplot(gs[1,i])
    plot = sns.barplot(data = df, x =var,y = 'SalePrice',estimator = median)
    i = i+1
i = 2
for var in ['PoolArea','TotRmsAbvGrd','BedroomAbvGr']:
    plt.subplot(gs[i,:])
    plot = sns.barplot(data = df, x =var,y = 'SalePrice',estimator = median)
    i = i+1
for var in date_var.columns:
    plot_tab = df.groupby([var])['SalePrice'].median().plot()
    plt.ylabel("Median Sale Price")
    plt.show()
df[list(con_var.columns) + ['SalePrice']].corr()
#creating a duplicate data set for this section.
import copy
data = df.copy()
date_var.columns.tolist()
#getting skewed features
skewed_var = []
for var in con_var.columns:
    if data[var].skew() > 0 or data[var].skew() < 0:
        skewed_var.append(var)

#final list of variables for log transformation
skewed_var_1 = []
for var in skewed_var:
    if 0 not in data[var].unique() and var not in date_var.columns.tolist():
        skewed_var_1.append(var)
        print(var)

#Applying the log transformation.
values_check = {"Variables" : [],"Before_Skewness" : [],"After_Skewness" : []}
for var in skewed_var_1:
    values_check["Variables"].append(var)
    values_check["Before_Skewness"].append(data[var].skew())
    data[var] = np.log(data[var])
    values_check["After_Skewness"].append(data[var].skew())
pd.DataFrame(values_check)
#Check the districution of above 5 variables
fig = plt.figure(constrained_layout = True,figsize = (15,15))
from matplotlib.gridspec import GridSpec
gs =  GridSpec(5,1,figure = fig)
i = 0
for var in skewed_var_1:
    plt.subplot(gs[i,0])
    sns.distplot(data[var])
    i = i+1
from sklearn.preprocessing import LabelEncoder
labelenco = LabelEncoder()

for var in cat_var.columns:
    data[var] = labelenco.fit_transform(data[var])                  #for train
    test_df[var] = labelenco.fit_transform(test_df[var])            #for test
#Now dependent and independent variables
y = data['SalePrice']
X = data.drop(['SalePrice'],axis = 1)

col_x = X.columns

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X = pd.DataFrame(scale.fit_transform(X),columns = [col_x])            # for train data set

col_test = test_df.columns                                            # for test data set
test_df = pd.DataFrame(scale.fit_transform(test_df),columns = [col_test])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 100)
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

feat_sel = SelectFromModel(Lasso(alpha = 0.005,random_state = 100))
feat_sel.fit(X_train,y_train)

#Getting selected features
sel_feat = X_train.columns[(feat_sel.get_support())]
print(sel_feat)
#Dropping all others remainnig features
X_train = X_train[sel_feat].reset_index(drop = True)
X_train.head()
#Similarly for testing dataset
X_test = X_test[sel_feat]
X_test.head()
#for Final test data set for submission
test_df = test_df[sel_feat]
test_df.head()
#Importing the files
from sklearn.linear_model import LinearRegression         #for modeling
from sklearn import metrics                               #for metrices

lm = LinearRegression()

#Fitting the model
lm.fit(X_train,y_train)

#Prediction for test data set
lm_predict = lm.predict(X_test)

#Scatter Plot : Between Y_test and Y_test_prediction
fig = plt.figure(figsize = (15,10))
plt.scatter(y_test,lm_predict)
plt.xlabel("Y_test",fontsize = 15); plt.ylabel("Y_test_predict",fontsize = 15)
plt.show()

#Model Evaluation : will check with RMSE, MSE, MAE and Accuracy
print("Mean Absolute Error : ",metrics.mean_absolute_error(y_test,lm_predict))
print("Mean Squared Error : ",metrics.mean_squared_error(y_test,lm_predict))
print("Root Mean Absolute Error : ",np.sqrt(metrics.mean_squared_error(y_test,lm_predict)))

#Model Accuracy : with lm.score()
print("\nAccuracy of Linear Regression Model : ",round(lm.score(X_test,y_test)*100),'%')

from sklearn.ensemble import GradientBoostingRegressor             #GBM Algorithm
from sklearn.model_selection import cross_val_score,GridSearchCV   

#For Metrices 
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

#Definning User Define fuction for the Model simulation
def model_fit(algorithm,train_set,train_y,test_set,test_y,
             perform_CV = True,print_FeaturesImportance = True,cv_folds = 5):     #setting some of by default values
    #Fitting the Model
    algorithm.fit(train_set,train_y)     
    
    #Prediction
    train_pred = algorithm.predict(train_set)
    test_pred = algorithm.predict(test_set)
    
    #Model Evaluation Reports
    print("Model Evaluation Reports/Metrices : ")
    model_eval = pd.DataFrame({"Evaluation Metrices" : [],"Train" :[],"Test":[]})
    model_eval["Evaluation Metrices"] = ["MAE","MSE","RMSE","Accuracy"]
    model_eval["Train"] = [MAE(train_y,train_pred),MSE(train_y,train_pred),np.sqrt(MSE(train_y,train_pred)),algorithm.score(train_set,train_y)]
    model_eval["Test"] = [MAE(test_y,test_pred),MSE(test_y,test_pred),np.sqrt(MSE(test_y,test_pred)),algorithm.score(test_set,test_y)]
    print(model_eval)
    
    #Cross Validation
    if perform_CV :
        cv_score = cross_val_score(algorithm,train_set,train_y,cv = cv_folds)
        print("\nCV Scores :",cv_score)
        print("Min: ",np.min(cv_score));print("Max: ",np.max(cv_score));print("Mean: ",np.mean(cv_score));print("Std: ",np.std(cv_score))
    
    #Printing features Importance
    if print_FeaturesImportance :
        feat_imp = algorithm.feature_importances_
        feat_col = list(train_set.columns)
        feat_df = pd.DataFrame({"Features" : feat_col,"Importance" : feat_imp})
        feat_df = feat_df.sort_values("Importance",ascending = False)
        
        #Plot them
        fig = plt.figure(figsize = (15,15))
        fig = sns.barplot(data = feat_df,x = 'Features',y = 'Importance')
        plt.xlabel("Features",fontsize = 15);plt.ylabel("Importance",fontsize=15);plt.title("Features Importance",fontsize = 20)
        plt.xticks(rotation = 90)
        plt.show()
    
#First Model : Baseline Model
gbm_1 = GradientBoostingRegressor(random_state=5)
model_fit(gbm_1,X_train,y_train,X_test,y_test)
#Tuning the number of estimators
param1 = {'n_estimators': range(1000,4000,1000)}

#Putting the parameters
G_search1 = GridSearchCV(estimator= GradientBoostingRegressor(learning_rate = 0.05,min_samples_split = 10,min_samples_leaf = 15,
                                                             max_depth = 4,max_features = 'sqrt', random_state = 5),
                        param_grid = param1,iid = False,cv = 5)

#Fitting the Model
G_search1.fit(X_train,y_train)

#Printing results
G_search1.cv_results_
#Printing Best Parameters
print(G_search1.best_params_)

#Printing Best Score
print(G_search1.best_score_)
#We can tune others parameter to get good accuracy
gbm_2 = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.05,max_depth=9,min_samples_split=17,max_features='sqrt',
                                 min_samples_leaf=13,loss='huber',random_state=5)

#Check this model
model_fit(gbm_2,X_train,y_train,X_test,y_test)
#Submission with Gradient Boosting Regressor
gbm_prediction = gbm_2.predict(test_df)
gbm_prediction
from xgboost import XGBRegressor

xgb_1 = XGBRegressor(n_estimators=1000,learning_rate=0.05,gamma=0,subsample=0.75,
                    max_depth=7,random_state=5,min_child_weight=1,colsample_bytree=0.8)

model_fit(xgb_1,X_train,y_train,X_test,y_test)
#Submission with XGBoosting 
xgb_prediction = xgb_1.predict(test_df)
xgb_prediction = np.exp(xgb_prediction)
xgb_prediction
sample_submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
sample_submission['SalePrice'] = xgb_prediction
sample_submission.to_csv("final_submission.csv",index = False)
sample_submission
test_df.shape