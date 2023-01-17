# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', None)
raw_train=pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

#Drop column ID

raw_train = raw_train.iloc[:,1:]

raw_train.head()
raw_train.info()
list(set(raw_train.dtypes.tolist()))
raw_train.isnull().sum()
raw_train.shape
raw_train.describe()
raw_train[raw_train.duplicated()]
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
plt.figure(figsize=(9, 8))

sns.distplot(raw_train['Cover_Type'], color='g', hist_kws={'alpha': 0.4});
raw_train.hist(figsize=(20, 30), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations
def plot_distribution(col):

    #plt.figure(figsize=(6,4))

    ax = sns.countplot(raw_train[col])

    height = sum([p.get_height() for p in ax.patches])

    for p in ax.patches:

            ax.annotate(f'{100*p.get_height()/height:.2f} %', (p.get_x()+0.3, p.get_height()+5000),animated=True)
plot_distribution("Cover_Type")
plt.figure(figsize=(10,7))

size = 10

corr = raw_train.iloc[:,:size].corr()



#num_cols = raw_train.select_dtypes(exclude=['object']).columns  # Without this also, it will generate the same result by selecting only numeric columns

#corr = raw_train[num_cols].corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,annot=True)
corr = raw_train.drop('Cover_Type', axis=1).corr() # We already examined SalePrice correlations

plt.figure(figsize=(20, 15))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);




df_num_corr = raw_train.corr()

df_num_corr

#golden_features_list = df_num_corr[abs(df_num_corr) > 0.2].sort_values(ascending=False)

#print("There is {} strongly correlated values with Covet_Type:\n{}".format(len(golden_features_list), golden_features_list))



sns.pairplot(raw_train.iloc[:,:size])
for i in range(0, len(raw_train.columns), 5):

    sns.pairplot(data=raw_train,

                x_vars=raw_train.columns[i:i+5],

                y_vars=['Cover_Type'])
raw_train.columns[0:5]
bin_cols = [f'Wilderness_Area{i+1}' for i in range(4)]

print(bin_cols)



fig, ax = plt.subplots(1,6, figsize=(30,10))



for i, col in enumerate(bin_cols):

     ax0 = plt.subplot(1,4,i+1)

     

     #data[col].value_counts().plot.bar(color='pink')

     sns.countplot(f'{col}', data= raw_train)

     #print(ax0.patches)

     height = sum([p.get_height() for p in ax0.patches])

     for p in ax0.patches:

         #get_x : Return the left coord of the rectangle

         ax0.text(p.get_x()+p.get_width()/2., p.get_height(), f'{100*p.get_height()/height:.2f} %', ha='center') 

     plt.xlabel(f'{col}')

plt.suptitle('Distribution over binary feature of train data')
# Plot wrt Cover_Type for binary features

fig, ax = plt.subplots(1,4, figsize=(30, 8))

for i in range(4): 

    sns.countplot(f'Wilderness_Area{i+1}', hue='Cover_Type', data=raw_train, ax=ax[i])

    ax[i].set_ylim([0, 3000])

    ax[i].set_title(f'Wilderness_Area{i+1}', fontsize=15)

fig.suptitle("Binary Feature Distribution Wildnerness (Train Data)", fontsize=20)

plt.show()
featuresToPlot=["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",

                "Horizontal_Distance_To_Roadways","Horizontal_Distance_To_Fire_Points","Hillshade_9am",

                "Hillshade_Noon","Hillshade_3pm"]



fig, ax = plt.subplots(3,4 ,figsize=(30,10))

for i, col in enumerate(featuresToPlot):

    #print(i,col)

    plt.subplot(2,5,i+1)

    raw_train.boxplot( column =[col])

    
#Need to work on Outliers

Q1=raw_train.quantile(0.25)

Q3=raw_train.quantile(0.75)

IQR=Q3-Q1

#print(IQR)

Q1['Elevation']
#Find the outlier value for Vertical_Distance_To_Hydrology

print(len(raw_train[(raw_train['Vertical_Distance_To_Hydrology']<(Q1['Vertical_Distance_To_Hydrology']-1.5*IQR['Vertical_Distance_To_Hydrology']))]))

print(len(raw_train[(raw_train['Vertical_Distance_To_Hydrology']>(Q3['Vertical_Distance_To_Hydrology']+1.5*IQR['Vertical_Distance_To_Hydrology']))]))







#df = df[~((df < (Q1–1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df = raw_train[~((raw_train<(Q1-1.5 * IQR)) |(raw_train>(Q3+1.5*IQR))).all(axis=1)]

df.shape
print((raw_train<(Q1-1.5 * IQR)) | (raw_train>(Q3+1.5*IQR)))
fig, ax = plt.subplots(2,5 ,figsize=(30,10))

for i, col in enumerate(featuresToPlot):

    #print(i,col)

    #plt.subplot(2,5,i+1)

    raw_train.boxplot(by='Cover_Type', column =[col])
#Merge 4 category Wilderness Type into One column and check for the distribution

newLabels =["Rawah","Neota","ComanchePeak","CachePoudre"]

oldCols =["Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4"]

df_one_hot=raw_train[oldCols]

df_one_hot.head()

df_one_hot.idxmax(axis=1)

raw_train['Wild']=df_one_hot.idxmax(axis=1)

di = {"Wilderness_Area1":"Rawah","Wilderness_Area2":"Neota","Wilderness_Area3":"ComanchePeak","Wilderness_Area4":"CachePoudre"}

raw_train['Wild']=raw_train['Wild'].map(di)  



sns.countplot(raw_train['Wild'])
sns.countplot(f'Wild', hue='Cover_Type', data=raw_train)
from sklearn.feature_selection import VarianceThreshold

sel_variance_threshold = VarianceThreshold() 

X_train_remove_variance = sel_variance_threshold.fit_transform(raw_train)

print(X_train_remove_variance.shape)
raw_train.describe()
X = raw_train.drop('Cover_Type',axis=1)

y = raw_train['Cover_Type']