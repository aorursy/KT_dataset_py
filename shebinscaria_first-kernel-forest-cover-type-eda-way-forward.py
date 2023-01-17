import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns

import numpy as np
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


train_dataset = pd.read_csv("/kaggle/input/learn-together/train.csv")

test_dataset = pd.read_csv("/kaggle/input/learn-together/test.csv")
train_dataset.head()
def evaluate_metric_score(y_true, y_pred):

    if y_true.shape[0] != y_pred.shape[0]:

        raise Exception("Sizes do not match")

        return 0

    else:

        size = y_true.shape[0]

        matches = 0

        y_true_array = np.array(list(y_true))

        y_pred_array = np.array(list(y_pred))

        for i in range(0,size):

            if y_true_array[i]==y_pred_array[i]:

                matches = matches + 1

        return matches/size
train_dataset.shape
test_dataset.shape
X = train_dataset.copy()

X = X.drop(columns=['Cover_Type'])

y = train_dataset[['Cover_Type']]
X.head()
X.columns
train_dataset.describe()
train_dataset.info()
test_dataset.describe()
test_dataset.info()
y['Cover_Type'].value_counts()
len(X['Id'].unique()) == len(X)
#List of continuous numeric data

list_contfeatures=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
len(list_contfeatures)
check = train_dataset.copy()

check_test = test_dataset.copy()
#Setting values to string type

check['Cover_Type']=check['Cover_Type'].astype(str)
for col in list_contfeatures:

    sns.catplot(x='Cover_Type',y=col,data=check,kind='box')
list_feat_wildarea=[]

str1 = "Wilderness_Area"

for i in range(1,5):

    str2=str1+str(i)

    list_feat_wildarea.append(str2)

check['Wilderness_Area_sum']=check[list_feat_wildarea].sum(axis=1)

check_test['Wilderness_Area_sum']=check_test[list_feat_wildarea].sum(axis=1)
#Checking in train data whether it contains only 1 unique value and it should be 1

check['Wilderness_Area_sum'].unique()
#Checking in test data whether it contains only 1 unique value and it should be 1

check_test['Wilderness_Area_sum'].unique()
list_feat_soiltype=[]

str1 = "Soil_Type"

for i in range(1,41):

    str2=str1+str(i)

    list_feat_soiltype.append(str2)

check['Soil_Type_sum']=check[list_feat_soiltype].sum(axis=1)

check_test['Soil_Type_sum']=check_test[list_feat_soiltype].sum(axis=1)
#Checking in train data whether it contains only 1 unique value and it should be 1

check['Soil_Type_sum'].unique()
#Checking in test data whether it contains only 1 unique value and it should be 1

check_test['Soil_Type_sum'].unique()
str1="Wilderness_Area"

for i in range(1,5):

    str2=str1+str(i)

    check.loc[(check[str2]==1),str1]=str2

    check_test.loc[(check_test[str2]==1),str1]=str2
#Lets check the uniqueness - should have only 4 categories

check['Wilderness_Area'].unique()
check_test['Wilderness_Area'].unique()
str1="Soil_Type"

not_present =[]

for i in range(1,41):

    str2=str1+str(i)

    check.loc[(check[str2]==1),str1] = str2

    if len(check.loc[(check[str2]==1),str1]) ==0:

        not_present.append(str2)

    check_test.loc[(check_test[str2]==1),str1]=str2
check['Soil_Type'].unique()
not_present
len(check_test['Soil_Type'].unique())==40
sns.catplot(x='Wilderness_Area',data=check,kind='count',order=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])

plt.xticks(rotation=90)
plt.figure(figsize=(50,10))

g= sns.catplot(x='Soil_Type',data=check,kind='count',height=4,aspect=2)

plt.xticks(rotation=90)
fig,ax = plt.subplots()

ax.hist(check['Elevation'])

fig.set_size_inches([5,5])

plt.show()
for col in list_contfeatures:

    fig,ax=plt.subplots()

    ax.hist(check[col])

    ax.set_title(col)

    ax.set_xlabel(col)

    ax.set_ylabel("Number of occurrences")

    fig.set_size_inches([5,5])

    plt.show()
check[list_contfeatures].corr()
check_corr=check[list_contfeatures].corr().abs()
# Generate a mask for the upper triangle

mask = np.zeros_like(check_corr, dtype=np.bool)

# To extract values corresponding to upper(u) triangle(tri) of mask use np.triu_indices_from

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(check_corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5)
xdf=check_corr.mask(check_corr<0.4).mask(check_corr==1)
xdf
xdf.to_csv("correlation_values_with0_5mask.csv")
sns.regplot(x="Elevation",y="Horizontal_Distance_To_Roadways",data=check)
sns.relplot(x="Elevation",y="Horizontal_Distance_To_Roadways",data=check,row="Cover_Type",kind='scatter')
sns.regplot(x="Aspect",y="Hillshade_3pm",data=check)
sns.relplot(x="Aspect",y="Hillshade_3pm",data=check,row="Cover_Type",kind='scatter')
sns.regplot(x='Slope',y='Hillshade_Noon',data=check)
sns.relplot(x='Slope',y='Hillshade_Noon',data=check,row='Cover_Type')
sns.regplot(x='Horizontal_Distance_To_Hydrology',y='Vertical_Distance_To_Hydrology',data=check)
sns.relplot(x='Horizontal_Distance_To_Hydrology',y='Vertical_Distance_To_Hydrology',data=check,row='Cover_Type')
sns.regplot(x='Hillshade_9am',y='Hillshade_3pm',data=check)
sns.relplot(x='Hillshade_9am',y='Hillshade_3pm', data=check,kind='scatter',row='Cover_Type')
sns.regplot(x='Horizontal_Distance_To_Fire_Points',y='Hillshade_9am',data=check)
sns.relplot(x='Horizontal_Distance_To_Fire_Points',y='Hillshade_9am',data=check,row='Cover_Type')
for col in list_contfeatures:

    fig1,ax1 = plt.subplots(figsize=(20,10))

    g=sns.boxplot(ax=ax1,x='Cover_Type',y=col,data=check,hue='Wilderness_Area',hue_order=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])

    ax1.set_title(col,fontsize=24)

    plt.show()
for col in list_contfeatures:

    fig1,ax1 = plt.subplots(figsize=(20,10))

    g=sns.swarmplot(ax=ax1,x='Cover_Type',y=col,data=check,hue='Wilderness_Area',hue_order=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])

    ax1.set_title(col,fontsize=24)

    plt.show()
for col in list_contfeatures:

    fig1,ax1 = plt.subplots(figsize=(20,10))

    g=sns.violinplot(ax=ax1,x='Cover_Type',y=col,data=check,hue='Wilderness_Area',hue_order=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])

    ax1.set_title(col,fontsize=24)

    plt.show()
for col in list_contfeatures:

    fig1,ax1 = plt.subplots(figsize=(20,10))

    g=sns.boxplot(ax=ax1,x='Wilderness_Area',y=col,data=check,hue='Cover_Type',hue_order=['1','2','3','4','5','6','7'])

    ax1.set_title(col,fontsize=24)

    plt.show()
check.groupby(['Wilderness_Area','Cover_Type']).size()
check.groupby(['Wilderness_Area','Soil_Type']).size()
check_test.shape
wilderness_area1_dataset = pd.concat([check.loc[check['Wilderness_Area']=="Wilderness_Area1",check.columns!='Cover_Type'],check_test.loc[check_test['Wilderness_Area']=="Wilderness_Area1"]],axis=0)
wilderness_area1_dataset.describe()
wilderness_area2_dataset = pd.concat([check.loc[check['Wilderness_Area']=="Wilderness_Area2",check.columns!='Cover_Type'],check_test.loc[check_test['Wilderness_Area']=="Wilderness_Area2"]],axis=0)
wilderness_area2_dataset.describe()
wilderness_area3_dataset = pd.concat([check.loc[check['Wilderness_Area']=="Wilderness_Area3",check.columns!='Cover_Type'],check_test.loc[check_test['Wilderness_Area']=="Wilderness_Area3"]],axis=0)
wilderness_area3_dataset.describe()
wilderness_area4_dataset = pd.concat([check.loc[check['Wilderness_Area']=="Wilderness_Area4",check.columns!='Cover_Type'],check_test.loc[check_test['Wilderness_Area']=="Wilderness_Area4"]],axis=0)
wilderness_area4_dataset.describe()