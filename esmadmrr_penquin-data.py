# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import missingno as msno

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



from collections import Counter



import warnings

warnings.filterwarnings('ignore') #python kaynaklı hataları gösterme demek



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Not: Style 'ın içindeki fonsiyonları göreviliriz.'

#plt.style.available
penqui=pd.read_csv('/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_lter.csv')
df=penqui.copy()

df.head(50)
df.columns
df.describe()
df.drop(['Sample Number','Comments'],inplace=True,axis=1)

df
df.info()
def bar_plot(variable):

    #****

    #    input: variable ex: 'Sex'

    #   output: bar plot & value count

    #****

    # get feature

    var = df[variable]

    #count number of categorical variable(value/sample)

    varValue=var.value_counts()

    

    #visualize

    plt.figure(figsize=(10,5))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel('Frequency')

    plt.title(variable)

    plt.show()

    print('{}: \n {}'.format(variable,varValue))
df.info()
df.columns
category1=['studyName','Species', 'Region', 'Island', 'Stage',

       'Individual ID', 'Clutch Completion', 'Date Egg','Sex']

for c in category1:

    bar_plot(c)
df.drop(["Stage","Region"],inplace=True,axis=1)
df_Sex=df.Sex[df.Sex=="."]
df[df["Date Egg"]=='12/1/09']
df.Sex[(df["Sex"]=='.')]="FEMALE"
#use the dataframe.nunique() function to find the unique values

unique_counts = pd.DataFrame.from_records([(col, df[col].nunique(),df[col].value_counts()) for col in df.columns],columns=['Column_Name', 'Num_Unique','value_counts']).sort_values(by=['Num_Unique'])

unique_counts
def plot_hist(variable):

    plt.figure(figsize=(10,5))

    plt.hist(df[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar=['Culmen Length (mm)', 'Culmen Depth (mm)',

       'Flipper Length (mm)', 'Body Mass (g)', 'Delta 15 N (o/oo)',

       'Delta 13 C (o/oo)']
for n in numericVar:

    plot_hist(n)
sns.countplot(x ='Body Mass (g)', hue = 'Clutch Completion', data = df);
df.columns
def relationship(variable):

    g = sns.catplot(y=variable, hue='Clutch Completion',

                data=df, kind="count",

                height=4, aspect=.7);

    

  
relation=['Sex','Species','Island']

for r in relation:

    relationship(r)
df.columns


variable=['Culmen Length (mm)', 'Culmen Depth (mm)',

       'Flipper Length (mm)', 'Body Mass (g)','Delta 15 N (o/oo)',

       'Delta 13 C (o/oo)']

for v in variable:

    a=df[[v,'Clutch Completion']].groupby(['Clutch Completion'],as_index=False).mean()

    print(a)
sns.pairplot( df,hue='Clutch Completion',diag_kind="hist",corner=True);
df.columns
df.columns
def outlierr(df,variable):

    outlier_indices=[]

    

    for c in variable:

        #first quartile

        Q1=np.percentile(df[c],25)

        

        #third quartile

        Q3=np.percentile(df[c],75)

        

        #IQR

        IQR = Q3 - Q1

        

         

        #Detect outlier and their indeces

        outlier_list_col=df[(df[c]< Q1- IQR *1.5) | (df[c]> Q3 + IQR *1.5)].index

        

        #store indeces

        outlier_indices.extend(outlier_list_col)

    

    #Listenin içindeki aykırı değerlerden kaçar tane var onu hesaplatıyoruz.

    outlier_indices=Counter(outlier_indices)

    

    multiple_outliers= list(i for i, v in outlier_indices.items() if v>2)    

    

    print(multiple_outliers)

    

    return   multiple_outliers



        

    
df.loc[outlierr(df,['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Delta 15 N (o/oo)','Delta 13 C (o/oo)'])]
def outlier_boxplot(variable):

    g=sns.catplot(y=variable,x='Clutch Completion',data=df,kind="box", palette="Set1",linewidth=2.5,height=4, aspect=.7);

    
variable=['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Delta 15 N (o/oo)','Delta 13 C (o/oo)']



for v in variable:

    outlier_boxplot(v)
df.isnull().sum()
df[df.Sex.isnull()]
msno.bar(df);
male_sex=df["Individual ID"].str.contains("A2")

female_sex=df["Individual ID"].str.contains("A1")
print(Counter(df.Sex[male_sex]))

print(Counter(df.Sex[female_sex]))



df.Sex[male_sex]="MALE"

df.Sex[female_sex]="FEMALE"
df.isnull().sum()
msno.matrix(df,color=(0.5,0.3,0.7));
def missingValue(variable):

    missing_list=[]

    

    for m in variable:

    

        a=df[df[variable].isnull()].index

        missing_list.extend(a)

        df[variable] = df[variable].fillna(df.groupby(['Species', 'Island'])[variable].transform('mean'))

   

    

    return missing_list
missingValue(['Culmen Length (mm)', 'Culmen Depth (mm)',

       'Flipper Length (mm)', 'Body Mass (g)', 'Delta 15 N (o/oo)',

       'Delta 13 C (o/oo)'])
df.isnull().sum()
msno.matrix(df,color=(0.5,0.3,0.2));
variable=['Culmen Length (mm)', 'Culmen Depth (mm)',

       'Flipper Length (mm)', 'Body Mass (g)', 'Delta 15 N (o/oo)',

       'Delta 13 C (o/oo)']

sns.heatmap(df[variable].corr(),annot=True);