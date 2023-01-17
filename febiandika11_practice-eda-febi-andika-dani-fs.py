# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

dir=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        dir.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df1=pd.read_csv(dir[0])

df2=pd.read_csv(dir[1])
#Preview Dataset

df1.head(10)
#Descriptive Statistics

df1.describe()
#Preview shape dataset

df1.shape
#Preview info dataset

df1.info()
# Make a list for both of the data type 

categorical_list = []

numerical_list = []

def check_dtypes(df):

    #Looping 

    for col in df.columns.tolist():

        if df[col].dtype=='object':

            categorical_list.append(col)

        else:

            numerical_list.append(col)

    

    #make dataframe that have two feature, that is categorical and numerical feature

    categorical = pd.Series(categorical_list, name='Categorical Feature')

    numerical = pd.Series(numerical_list, name='Numerical Feature')

    df_dtypes = pd.concat([categorical,numerical], axis=1)

    

    return df_dtypes
#Using separate function

check_dtypes(df1)
def unique(df):

    for col in df.columns:

        if df[col].dtypes == 'object':

            unique_columns = len(df[col].unique())

            print("Feature '{col}' has {unique_columns} unique categories".format(col=col, unique_columns=unique_columns))
#To see the numbers of unique values in each column

unique(df1)
def missing_value(df):

    #count the number of missing value 

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)

    missing  = pd.concat([total, percent], axis=1, keys=['Total_Missing', 'Percent(%)'])

    

    return missing.head(26)
#To check 

missing_value(df1)
#drop size and VIN columns

df=df1.drop(columns=['size','vin'])
def fill_missing(df, varlist = None , vartype = None ):

    # filling numerical data with median 

    if vartype == 'numerical' :

        for col in varlist:

            df[col] = df[col].fillna(df[col].median())

    

    # filling categorical data with modus  

    if vartype == 'categorical' :

        for col in varlist:

            df[col] = df[col].fillna(df[col].mode().iloc[0])
#Re-separate features 

categorical_list = []

numerical_list = []

check_dtypes(df)
# Filling numerical type variables

fill_missing(df, numerical_list, 'numerical')
# Filling categorical type variables

fill_missing(df, categorical_list, 'categorical')
# Recheck missing 

missing_value(df)
def Univariate_plot(df, column, vartype, hue = None ):

    sns.set(style="darkgrid")

    

    if vartype == 'numerical':

        fig, ax=plt.subplots(nrows = 3, ncols=1,figsize=(12,12))

        # Distribution Plot

        ax[0].set_title("Distribution Plot",fontsize = 10)

        sns.distplot(df[column], kde=False, fit=stats.gamma, color='darkblue', label = column, ax=ax[0])

        

        # Violinplot 

        ax[1].set_title("Violin Plot",fontsize = 10)

        sns.violinplot(data= df, x=column, color = 'limegreen', inner="quartile", orient='h', ax=ax[1])

        

        #Boxplot

        ax[2].set_title("Box Plot",fontsize = 10)

        sns.boxplot(data =df, x=column,color='cyan',orient="h",ax=ax[2])

        

        fig.tight_layout()

        

    if vartype == 'categorical' :

        #Count plot 

        fig = plt.figure(figsize=(12,6))

        plt.title('Count Plot',fontsize = 20)

        ax=sns.countplot(data=df, x=column, palette="YlOrRd_r")

        ax.set_xlabel(column, fontsize = 15)

        ax.tick_params(labelsize=12)
pr=df[df['price'] <= 140000]

Univariate_plot(pr,'price','numerical')
# condition variable visualization

Univariate_plot(df, 'condition', 'categorical' )

plt.title('The Numbers of Car Conditions')

plt.show()
# type variable visualization

Univariate_plot(df, 'type', 'categorical' )

plt.xticks(rotation = 'vertical')

plt.title('The Numbers of Car Types')
group=pd.DataFrame(df['fuel'].value_counts()).reset_index()

f=plt.figure(figsize=(4,4))

plt.pie(group['fuel'],labels=group['index'],radius=2,rotatelabels=True,autopct='%.2f%%')

plt.show()
group=pd.DataFrame(df['transmission'].value_counts()).reset_index()

f=plt.figure(figsize=(4,4))

plt.pie(group['transmission'],labels=group['index'],radius=2,rotatelabels=True,autopct='%1.1f%%')

plt.show()
#create correlation with hitmap



#create correlation

corr = df[numerical_list].corr(method = 'pearson')



#convert correlation to numpy array

mask = np.array(corr)



#to mask the repetitive value for each pair

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots(figsize = (20,20))

fig.set_size_inches(20,20)

sns.heatmap(corr,mask=mask, vmax = 0.9, square = True, annot = True)
#Barplot of average car prices by brands (manufacture)

ave=pd.DataFrame(df.groupby(df['manufacturer']).mean().sort_values(by='price',ascending=False).reset_index().head(10))

f=plt.figure(figsize=(10,10))

x=range(10)

plt.bar(x,ave['price'],width=0.8,align='center')

plt.xticks(x,ave['manufacturer'],rotation='vertical')

plt.title('Top-10 Average Car Prices by Brands')

plt.xlabel('Manufacture')

plt.ylabel('Average car prices')

plt.show()
#Barplot of average car odometer by brands (manufacture)

ave=pd.DataFrame(df.groupby(df['manufacturer']).mean().sort_values(by='odometer',ascending=False).reset_index().head(10))

f=plt.figure(figsize=(10,10))

x=range(10)

plt.bar(x,ave['odometer'],width=0.8,align='center')

plt.xticks(x,ave['manufacturer'],rotation='vertical')

plt.title('Top-10 Average Car Odometer by Brands')

plt.xlabel('Manufacture')

plt.ylabel('Average car odometer')

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(x='condition',y='price',data=pr)
plt.figure(figsize=(10,10))

sns.boxplot(x='paint_color',y='price',data=pr)
plt.figure(figsize=(10,10))

sns.boxplot(x='transmission',y='price',data=pr)