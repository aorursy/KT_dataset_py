# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import missingno as ms
#importing train and test dataset

test=pd.read_csv("/kaggle/input/big-mart-sales-prediction/Test.csv")

train=pd.read_csv("/kaggle/input/big-mart-sales-prediction/Train.csv")
#first five entry of train data_set

train.head()
#first five entry of test data_set

test.head()
#shape of datasets

print(train.shape)

print(test.shape)
#findind missing values

def missing_values(data1,data2,col_name="missning_val"):

    missing_train=pd.DataFrame(train.isna().sum()/len(train)*100,columns=[col_name])

    missing_test=pd.DataFrame(test.isna().sum()/len(test)*100,columns=[col_name])

    miss_data=pd.DataFrame({"Train":missing_train.iloc[:,0],"Test":missing_test.iloc[:,0]})

    return miss_data
missing_values(train,test)
print(ms.matrix(train))

print(ms.matrix(test))


train.describe()
test.describe()
train.describe(include="O")
test.describe(include="O")
data_train=train.copy()
data_train.head()
data_train.describe(include="O")
cat_column=data_train.describe(include="O").columns
cat_column
def count_plots(cat_column):

    sns.countplot(data_train[cat_column]).set_title=cat_column

    print(data_train[cat_column].value_counts(normalize=True))

    plt.xticks(rotation=90)

    plt.show()
count_plots("Item_Fat_Content")
count_plots("Outlet_Identifier")
count_plots("Outlet_Type")
count_plots("Outlet_Size")
count_plots("Outlet_Location_Type")
count_plots("Item_Type")
con_column=data_train.describe().columns
con_column
def dist_plot(con_column):

    sns.distplot(data_train[con_column])

    plt.show()
dist_plot("Item_Weight")
dist_plot("Item_MRP")
dist_plot("Outlet_Establishment_Year")
dist_plot("Item_Visibility")
dist_plot("Item_Outlet_Sales")
sns.scatterplot(data_train["Item_MRP"],data_train["Item_Outlet_Sales"],hue="Outlet_Type",data=data_train)
sns.jointplot(data_train["Item_Outlet_Sales"],data_train["Item_MRP"],kind="reg")
data_train.corr()
plt.figure(figsize=(12,8))

sns.heatmap(data_train.corr(),annot=True,fmt=".2f")

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(data_train["Item_Type"],hue="Outlet_Location_Type",data=data_train)

plt.xticks(rotation=90)

plt.show()
data_train["Item_Fat_Content"].unique()
data_train["Item_Fat_Content"]=data_train["Item_Fat_Content"].map({"Low Fat":"low_fat","low fat":"low_fat","LF":"low_fat",

                                                                  "Regular":"regular","reg":"regular"})
plt.figure(figsize=(12,8))

sns.countplot(data_train["Item_Type"],hue="Item_Fat_Content",data=data_train)

plt.xticks(rotation=90)

plt.show()
sales=data_train.pivot_table(values="Item_Outlet_Sales",index=["Outlet_Type","Outlet_Location_Type","Outlet_Identifier"

                                                        ],aggfunc=np.sum)
sales.style.background_gradient(cmap="Reds")
sales.plot(kind="bar",figsize=(12,8))
data_train["Item_Identifier_typ"]=data_train["Item_Identifier"].apply(lambda x: x[0:2])
data_train.insert(0,'Item_Identifier_typ', data_train.pop("Item_Identifier_typ"))
sls=data_train.pivot_table(values="Item_Outlet_Sales",index=["Item_Identifier_typ","Item_Type","Item_Fat_Content"],aggfunc=np.sum)
sls.plot(kind="bar",figsize=(12,8))
sns.pairplot(data_train)
data_train.columns
data_train.pivot_table(values="Item_Outlet_Sales",index=["Item_Identifier_typ","Outlet_Identifier","Item_Fat_Content"],aggfunc=np.sum)
plt.figure(figsize=(15,6))

sns.boxplot(data_train["Item_Type"],data_train["Item_Outlet_Sales"])

plt.xticks(rotation=90)

plt.show()
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))

plt.subplot(121)

sns.boxplot(data_train["Outlet_Size"],data_train["Item_Outlet_Sales"])

plt.subplot(122)

sns.boxplot(data_train["Item_Fat_Content"],data_train["Item_Outlet_Sales"])

plt.show()



plt.figure(figsize=(15,6))

sns.boxplot(data_train["Outlet_Establishment_Year"],data_train["Item_Outlet_Sales"])

plt.show()
plt.figure(figsize=(16,5))

plt.subplot(121)

sns.boxplot(data_train["Outlet_Type"],data_train["Item_Outlet_Sales"])

plt.subplot(122)

sns.boxplot(data_train["Outlet_Location_Type"],data_train["Item_Outlet_Sales"])

plt.show()

plt.figure(figsize=(16,5))

sns.boxplot(data_train["Outlet_Identifier"],data_train["Item_Outlet_Sales"])

plt.show()
data_train["Outlet_years"]=2020-data_train["Outlet_Establishment_Year"]
sns.countplot(data_train["Outlet_years"])
plt.figure(figsize=(20,5))

sns.boxplot(data_train["Outlet_years"],data_train["Item_Outlet_Sales"],data=data_train).set_title("Item_Sales VS Outlet_year")

plt.show()