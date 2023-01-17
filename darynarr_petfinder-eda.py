import pandas as pd

import numpy as np



import os

import math



import seaborn as sns

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 50)
df_orig = pd.read_csv('../input/train/train.csv', index_col = "PetID")
df_orig.head()
print("Columns that contains NA:", list(df_orig.columns[df_orig.isna().any()]))
df = df_orig.copy()
breeds = pd.read_csv('../input/breed_labels.csv')

colors = pd.read_csv('../input/color_labels.csv')

states = pd.read_csv('../input/state_labels.csv')
display(breeds.head())

display(colors.head())

display(states.head())
breed_labels = dict(zip(breeds.BreedID,

                       breeds.BreedName))



color_labels = dict(zip(colors.ColorID,

                       colors.ColorName))



state_labels = dict(zip(states.StateID,

                       states.StateName))
type_labels = {1 : 'Dog', 

               2 : 'Cat'}



gender_labels = {1 : 'Male', 

                 2 : 'Female', 

                 3 : 'Mixed (group of pets)'}



maturity_size_labels = {1 : 'Small', 

                        2 : 'Medium', 

                        3 : 'Large', 

                        4 : 'Extra Large', 

                        0 : 'Not Specified'}



fur_length_labels = {1 : 'Short', 

                     2 : 'Medium', 

                     3 : 'Long', 

                     0 : 'Not Specified'}



# for columns 'Vaccinated', 'Dewormed', 'Sterilized'

treatment_labels = {1 : 'Yes', 

                    2 : 'No', 

                    3 : 'Not Sure'}



health_labels = {1 : 'Healthy', 

                 2 : 'Minor Injury', 

                 3 : 'Serious Injury', 

                 0 : 'Not Specified'}
for i in [1,2]:

    df['Breed{}'.format(i)] = df['Breed{}'.format(i)].map(breed_labels)

    

for i in [1,2,3]:

    df['Color{}'.format(i)] = df['Color{}'.format(i)].map(color_labels)



df['State'] = df['State'].map(state_labels)

df['Type'] = df['Type'].map(type_labels)

df['Gender'] = df['Gender'].map(gender_labels)

df['MaturitySize'] = df['MaturitySize'].map(maturity_size_labels)

df['FurLength'] = df['FurLength'].map(fur_length_labels)



for col in ['Vaccinated', 'Dewormed', 'Sterilized']:

    df[col] = df[col].map(treatment_labels)

    

df['Health'] = df['Health'].map(health_labels)
df.head()
print("Number of rows:", df.shape[0])

print("Number of columns:", df.shape[1])
df.info()
print("Columns that contains NA:", list(df_orig.columns[df.isna().any()]))
missing_Breed1 = df[df[['Breed1']].isnull().any(axis=1)]

print("Number of rows that have missing values in Breed1: ", 

      missing_Breed1.shape[0])
df.drop(list(missing_Breed1.index), axis =0, inplace = True)
# define AdoptionSpeed as categorical variable

df['AdoptionSpeed'] = pd.Categorical(df['AdoptionSpeed'], 

                                     categories=[0,1,2,3,4],

                                    ordered = True)
print(df.AdoptionSpeed.describe())

plt.figure(figsize=(9, 8))

sns.countplot(df.AdoptionSpeed, palette = 'winter');
print(df.Type.describe())

plt.figure(figsize=(8, 5))

sns.countplot(df.Type, palette = 'winter');
sns.catplot(x="Type", hue ="AdoptionSpeed", kind='count', data=df, palette = 'winter');
sns.boxplot(x="Type", y =df.AdoptionSpeed.astype(int), data=df, palette = 'winter');
print(df.Gender.describe())

plt.figure(figsize=(8, 5))

sns.countplot(df.Gender, palette = 'winter');
sns.catplot(x="Gender", hue ="AdoptionSpeed", kind='count', data=df, palette = 'winter');
sns.boxplot(x="Gender", y =df.AdoptionSpeed.astype(int), data=df, palette = 'winter');
cats = df[df['Type'] == 'Cat']

top20_cat_breeds = cats.Breed1.value_counts().sort_values(ascending = False).iloc[:20].index.tolist()



print(cats.Breed1.describe())

plt.figure(figsize=(13,10))

ax1 = sns.countplot(y='Breed1', palette = 'winter', 

              data=cats[cats['Breed1'].isin(top20_cat_breeds)])

ax1.set(ylabel = 'Cat breeds')

ax1.set_title('Top 20 cat breeds')

plt.show()
dogs = df[df['Type'] == 'Dog']

top20_dog_breeds = dogs.Breed1.value_counts().sort_values(ascending = False).iloc[:20].index.tolist()



print(dogs.Breed1.describe())

plt.figure(figsize=(13,10))

ax2 = sns.countplot(y='Breed1', palette = 'winter', 

              data=dogs[dogs['Breed1'].isin(top20_dog_breeds)])

ax2.set(ylabel = 'Dog breeds')

ax2.set_title('Top 20 dog breeds')



plt.show()
# create the feature PureBreed

breedless_labels = ['Mixed Breed',

                    'Domestic Medium Hair',

                    'Domestic Long Hair',

                    'Domestic Short Hair']

df['PureBreed'] = df.apply(lambda row: "Breedlees" if ((row['Breed1'] != row['Breed2']) & (row['Breed2'] == row['Breed2'])) or 

                                                        (row['Breed1'] in breedless_labels) else "Pure", axis =1)
print(df.PureBreed.describe())

plt.figure(figsize=(8, 5))

sns.countplot(df.Type, palette = 'winter', hue = df.PureBreed);
plt.figure(figsize=(8, 5))



sns.boxplot(x="Type", y =df.AdoptionSpeed.astype(int), hue = 'PureBreed', data=df, palette = 'winter');
sns.countplot(y='Color1', palette = 'winter', data=df);
plt.figure(figsize=(8, 5))



sns.boxplot(x="Color1", y =df.AdoptionSpeed.astype(int), data=df, palette = 'winter');
print(df.Age.describe())

plt.figure(figsize=(10, 5))

sns.distplot(df['Age'], kde = True);
age_bins = {

        (0, 6): '0 to 5 months',

        (6, 12): '6 to 11 months',

        (12, 36): '1 to 2 years',

        (36, 60): '3 to 4 years',

        (60, 96): '5 to 7 years',

        (96, np.inf): '8 and more years'}



df['Age_bins'] = None

for age_inter in age_bins.keys():

    df.loc[(age_inter[0] <= df['Age']) & (df['Age'] < age_inter[1]), 

           ['Age_bins']] = age_bins[age_inter]

fig, axs = plt.subplots(ncols=2,figsize=(17,7))



ax1 = sns.countplot(x='Age_bins', palette = 'winter', data=df, ax=axs[0]);

plt.sca(ax1)

plt.xticks(rotation=45)

ax2 = sns.boxplot(x='Age_bins', y = df.AdoptionSpeed.astype(int), palette = 'winter', data=df, ax=axs[1]);

plt.sca(ax2)

plt.xticks(rotation=45);
print(df.MaturitySize.describe())

fig, axs = plt.subplots(ncols=2,figsize=(20,7))

sns.countplot(x='MaturitySize', hue = 'Type',  palette = 'winter', data=df, ax=axs[0]);

sns.boxplot(x='Type', y = df.AdoptionSpeed.astype(int),hue = 'MaturitySize', palette = 'winter', data=df, ax=axs[1]);

plt.show()
print(df.FurLength.describe())

fig, axs = plt.subplots(ncols=2,figsize=(20,7))

sns.countplot(x='FurLength', hue = 'Type',  palette = 'winter', data=df, ax=axs[0]);

sns.boxplot(x='Type', y = df.AdoptionSpeed.astype(int),hue = 'FurLength', palette = 'winter', data=df, ax=axs[1]);

plt.show()
df.Vaccinated.describe()
df.Dewormed.describe()
df.Sterilized.describe()
fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20,20))

sns.countplot(x='Vaccinated', palette = 'winter', data=df, ax=axs[0, 0]);

sns.boxplot(x='Vaccinated', y = df.AdoptionSpeed.astype(int), palette = 'winter', data=df, ax=axs[0,1]);

sns.countplot(x='Dewormed', palette = 'winter', data=df, ax=axs[1, 0]);

sns.boxplot(x='Dewormed', y = df.AdoptionSpeed.astype(int), palette = 'winter', data=df, ax=axs[1,1]);

sns.countplot(x='Sterilized', palette = 'winter', data=df, ax=axs[2, 0]);

sns.boxplot(x='Sterilized', y = df.AdoptionSpeed.astype(int), palette = 'winter', data=df, ax=axs[2,1]);
fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(20,20))

sns.countplot(x='Vaccinated', hue='Age_bins',  palette = 'winter', data=df, ax=axs[0]);

sns.countplot(x='Dewormed',hue='Age_bins',  palette = 'winter', data=df, ax=axs[1]);

sns.countplot(x='Sterilized', hue='Age_bins', palette = 'winter', data=df, ax=axs[2]);
print(df.Health.describe())

fig, axs = plt.subplots(ncols=2,figsize=(20,7))

sns.countplot(x='Health',  palette = 'winter', data=df, ax=axs[0]);

sns.boxplot(x='Health', y = df.AdoptionSpeed.astype(int), palette = 'winter', data=df, ax=axs[1]);

plt.show()
print(df.Quantity.describe())

fig, axs = plt.subplots(ncols=2,figsize=(13,5))

sns.countplot(x='Quantity',  palette = 'winter', data=df, ax=axs[0]);

sns.stripplot(x='Quantity', y = df['AdoptionSpeed'].astype(int), data =df, palette = 'winter',ax=axs[1])

plt.show()
print(df.Fee.describe())

fig, axs = plt.subplots(ncols=2, nrows =1,figsize=(13,5))

sns.distplot(df['Fee'],  color = 'blue', kde = False, ax=axs[0]);

sns.catplot(y='Fee', x = 'AdoptionSpeed', data =df, palette = 'winter', kind = 'strip',ax=axs[1]);

plt.close(2)

plt.show()
fee_outliers_IDs = list(df[df['Fee'] > 1500].index)

df[df.index.isin(fee_outliers_IDs)]
# removing outliers

df.drop(fee_outliers_IDs, axis =0, inplace=True)
print(df.Fee.describe())

fig, axs = plt.subplots(ncols=2, nrows =1,figsize=(13,5))

sns.distplot(df['Fee'],  color = 'blue', kde = False, ax=axs[0]);

sns.catplot(y='Fee', x = 'AdoptionSpeed', data =df, palette = 'winter', kind = 'strip',ax=axs[1]);

plt.close(2)

plt.show()
df.State.describe()

# fig, axs = plt.subplots(ncols=1, nrows =2,figsize=(13,17))

plt.figure(figsize=(13, 5))

ax1=sns.countplot(x='State',  palette = 'winter', data=df);

# plt.sca(ax1)

plt.xticks(rotation=45)

# ax2=sns.catplot(y='State', x = 'AdoptionSpeed', data =df, palette = 'winter', kind = 'box',ax = axs[1]);

# plt.close(2);

plt.show()
sns.catplot(y='State', x = 'AdoptionSpeed', data =df, palette = 'winter', kind = 'box');
df.RescuerID.describe()
rescuers = pd.DataFrame(df.RescuerID.value_counts()).reset_index()

rescuers.columns = ['RescuerID', 'Number of rescued animals']

rescuers.head()
rescuers['Number of rescued animals'].describe()
rescuers.columns
df['RescuerNumber'] = df['RescuerID'].map(dict(zip(rescuers.RescuerID,

                                                  rescuers['Number of rescued animals'])))
df['RescuerNumber'].describe()
sns.distplot(df['RescuerNumber'], kde = False, bins = 30);
print(df['VideoAmt'].describe())
print("{0:.2%} of pets don't have videos".format(

    df[df['VideoAmt'] == 0].shape[0] / float(df.shape[0])))
print(df['PhotoAmt'].describe())

plt.figure(figsize=(10, 5))

sns.countplot(df['PhotoAmt'].astype(int), palette='winter_r');