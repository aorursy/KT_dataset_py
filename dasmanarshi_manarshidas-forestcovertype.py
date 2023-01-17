import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Importing necessary libraries



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
train_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

test_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
#Checking first five rows

train_df.head()
#Total number of records

print('Size of train data',train_df.shape)

print('Size of test data',test_df.shape)
#Column Details

print('Column Details : \n',train_df.columns)

print(train_df.info())
for column in train_df.columns:

    print(column, train_df[column].nunique())
#print(train_df.dtypes.value_counts())

#print(test_df.dtypes.value_counts())



## Checking formissing values 

print(f"Missing Values in train: {train_df.isna().any().any()}")

print(f"Missing Values in test: {test_df.isna().any().any()}")
train_df.describe()
target_classes = range(1,8)

target_class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', \

                      'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']



numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', \

                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', \

                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']



categorical_features = [ 'Wilderness_Area', 'Soil_Type' ]
# Extract target from data

y = train_df['Cover_Type']

#train_df = train_df.drop('Cover_Type', axis=1)

train_df = train_df.drop(["Id"], axis = 1)

test_ids = test_df["Id"]

test_df = test_df.drop(["Id"], axis = 1)
print("percent of negative values (training): " + '%.3f' % ((train_df.loc[train_df['Vertical_Distance_To_Hydrology'] < 0].shape[0] / train_df.shape[0])*100))

print("percent of negative values (testing): " + '%.3f' % ((test_df.loc[test_df['Vertical_Distance_To_Hydrology'] < 0].shape[0]/ test_df.shape[0])*100))
plt.figure(figsize=(12,8))

sns.boxplot(train_df['Vertical_Distance_To_Hydrology'])

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(test_df['Vertical_Distance_To_Hydrology'])

plt.show()
plt.figure(figsize=(12,5))

plt.title("Distribution of forest categories(Target Variable)")

ax = sns.distplot(y)
## pie chart of 7 Cover_Type

typeCount = list(y.value_counts().values)

fig1, ax1 = plt.subplots()

ax1.pie(typeCount, labels=target_class_names, autopct='%1.1f%%', shadow=True)

ax1.axis('equal')

plt.show()
# plot target var

plt.hist(y, bins='auto')

plt.title('Cover_Type')

plt.xlabel('Class')

plt.ylabel('# Instances')

plt.show()

print(y.value_counts())
sns.FacetGrid(train_df, hue="Cover_Type", height=10).map(plt.scatter, "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology").add_legend()
for feature_name in numerical_features:

        plt.figure()

        sns.distplot(train_df[feature_name], label='train')

        sns.distplot(test_df[feature_name], label='test')

        plt.legend()

        plt.show()
# categorical distributions btw train and test set

train_wilderness_categorical = train_df['Wilderness_Area1'].copy().rename('Wilderness_Area')

train_wilderness_categorical[train_df['Wilderness_Area2'] == 1] = 2

train_wilderness_categorical[train_df['Wilderness_Area3'] == 1] = 3

train_wilderness_categorical[train_df['Wilderness_Area4'] == 1] = 4



test_wilderness_categorical = test_df['Wilderness_Area1'].copy().rename('Wilderness_Area')

test_wilderness_categorical[test_df['Wilderness_Area2'] == 1] = 2

test_wilderness_categorical[test_df['Wilderness_Area3'] == 1] = 3

test_wilderness_categorical[test_df['Wilderness_Area4'] == 1] = 4



plt.figure()

sns.countplot(train_wilderness_categorical, label='train')

plt.title('Wilderness_Area in Train')



plt.figure()

sns.countplot(test_wilderness_categorical, label='test')

plt.title('Wilderness_Area in Test')



plt.show()
soil_classes = range(1,41)



train_soiltype_categorical = train_df['Soil_Type1'].copy().rename('Soil_Type')

for cl in soil_classes:

    train_soiltype_categorical[train_df['Soil_Type'+str(cl)] == 1] = cl



test_soiltype_categorical = test_df['Soil_Type1'].copy().rename('Soil_Type')

for cl in soil_classes:

    test_soiltype_categorical[test_df['Soil_Type'+str(cl)] == 1] = cl



plt.figure(figsize=(10, 5))

sns.countplot(train_soiltype_categorical, label='train')

plt.title('Soil_Type in Train')



plt.figure(figsize=(10, 5))

sns.countplot(test_soiltype_categorical, label='test')

plt.title('Soil_Type in Test')



plt.show()
sns.swarmplot(data=train_df,y="Elevation" ,x="Cover_Type")
sns.violinplot(data=train_df,y="Aspect" ,x="Cover_Type")
sns.stripplot(data=train_df,y="Slope" ,x="Cover_Type")
sns.lvplot(data=train_df,y="Hillshade_9am" ,x="Cover_Type")
## Calculates correlations between columns in a dataframe

# and converted to heatmap

sns.heatmap(train_df.corr())
sns.pairplot(train_df,vars=["Elevation","Aspect"],hue='Cover_Type',palette='husl',plot_kws={'alpha':0.5})