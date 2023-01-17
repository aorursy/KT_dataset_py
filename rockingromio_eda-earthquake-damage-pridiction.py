import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
X_train = pd.read_csv("../input/richters-predictor-modeling-earthquake-damage/train_values.csv")

y_train = pd.read_csv("../input/richters-predictor-modeling-earthquake-damage/train_labels.csv")

X_test = pd.read_csv("../input/richters-predictor-modeling-earthquake-damage/test_values.csv")

print(X_train.shape)

print(X_test.shape)
X_train.columns
X_train.head()
X_train.info()
fig, axes = plt.subplots(ncols = 3, figsize = (20, 5))

sns.distplot(X_train['geo_level_1_id'], rug=True, ax = axes[0])

sns.distplot(X_train['geo_level_2_id'], rug=True, ax = axes[1])

sns.distplot(X_train['geo_level_3_id'], rug=True, ax = axes[2])
X_train['geo_level_1_id'] = X_train['geo_level_1_id'] / X_train['geo_level_1_id'].max()

X_train['geo_level_2_id'] = X_train['geo_level_2_id'] / X_train['geo_level_2_id'].max()

X_train['geo_level_3_id'] = X_train['geo_level_3_id'] / X_train['geo_level_3_id'].max()
y_train.info()

y_train['damage_grade'] = y_train['damage_grade'].astype('object')
y_train.info()
fig, axes = plt.subplots(ncols = 3, figsize = (20, 5))



sns.boxplot(y= X_train['geo_level_1_id'], x= y_train['damage_grade'], ax = axes[0])

sns.boxplot(y= X_train['geo_level_2_id'], x= y_train['damage_grade'], ax = axes[1])

sns.boxplot(y= X_train['geo_level_3_id'], x= y_train['damage_grade'], ax = axes[2])
X_train['count_floors_pre_eq'] = X_train['count_floors_pre_eq'].astype("int64")
fig, axes = plt.subplots(ncols = 2, nrows = 2, figsize = (20, 20))

sns.countplot(X_train['count_floors_pre_eq'], ax = axes[0][0])

sns.countplot(X_train['age'], ax = axes[0][1])

sns.countplot(X_train['area_percentage'], ax = axes[1][0])

sns.countplot(X_train['height_percentage'], ax = axes[1][1])
print(f"Floor Count Unique Values : {X_train['count_floors_pre_eq'].unique()}")

print(f"Age Unique Values : {X_train['age'].unique()}")

print(f"Area Percentage Unique Values : {X_train['area_percentage'].unique()}")

print(f"Height Percentage Unique Values : {X_train['height_percentage'].unique()}")
fig, axes = plt.subplots(ncols = 2, nrows = 2, figsize = (20, 20))



sns.boxplot(y= X_train['age'], x= y_train['damage_grade'], ax = axes[0][0])

sns.boxplot(y= X_train['count_floors_pre_eq'], x= y_train['damage_grade'], ax = axes[0][1])

sns.boxplot(y= X_train['area_percentage'], x= y_train['damage_grade'], ax = axes[1][0])

sns.boxplot(y= X_train['height_percentage'], x= y_train['damage_grade'], ax = axes[1][1])
X_train['damage_grade'] = y_train['damage_grade']
plt.figure(figsize=(15,8))

sns.countplot(x=X_train["count_floors_pre_eq"],hue=X_train["damage_grade"],palette="viridis")
plt.figure(figsize=(15,8))

sns.countplot(x=X_train["age"],hue=X_train["damage_grade"],palette="viridis")
X_train['damage_grade'] = X_train['damage_grade'].astype(int)
X_train[['count_floors_pre_eq','age', 'area_percentage', 'height_percentage', 'damage_grade']].corr()
## log(age + 1) because age can be zero and log of zero does not exist

X_train['log_age'] = (X_train['age'] + 1).apply(np.log)
sns.boxplot(y= X_train['log_age'], x= y_train['damage_grade'])
X_train[['log_age', 'damage_grade']].corr()
X_train['log_area_per'] = (X_train['area_percentage']).apply(np.log)

X_train[['log_area_per', 'damage_grade']].corr()

## We can not get any useful information from log of areas
X_train['log_height_per'] = (X_train['height_percentage']).apply(np.log)

X_train[['log_height_per', 'damage_grade']].corr()

## We can not get any useful information from log of height
del X_train['log_area_per']

del X_train['log_height_per']
print(f"Land Surface Condition Catagories : {X_train['land_surface_condition'].unique()}")

print(f"Foundation Type Catagories : {X_train['foundation_type'].unique()}")

print(f"Roof Type Catagories : {X_train['roof_type'].unique()}")

print(f"Ground Floor Types Catagories : {X_train['ground_floor_type'].unique()}")

print(f"Other Floor Type Catagories : {X_train['other_floor_type'].unique()}")

print(f"Positions Catagories : {X_train['position'].unique()}")

print(f"Plan Coniguration Catagories : {X_train['plan_configuration'].unique()}")

print(f"Legal Ownership Catagories : {X_train['legal_ownership_status'].unique()}")
fig, axes = plt.subplots(ncols = 2, nrows = 4, figsize = (20, 20))

sns.countplot(X_train['land_surface_condition'], ax = axes[0][0])

sns.countplot(X_train['foundation_type'], ax = axes[0][1])

sns.countplot(X_train['roof_type'], ax = axes[1][0])

sns.countplot(X_train['ground_floor_type'], ax = axes[1][1])

sns.countplot(X_train['other_floor_type'], ax = axes[2][0])

sns.countplot(X_train['position'], ax = axes[2][1])

sns.countplot(X_train['plan_configuration'], ax = axes[3][0])

sns.countplot(X_train['legal_ownership_status'], ax = axes[3][1])
fig, axes = plt.subplots(ncols = 2, nrows = 4, figsize = (20, 20))

sns.countplot(X_train['land_surface_condition'], hue = X_train['damage_grade'], ax = axes[0][0])

sns.countplot(X_train['foundation_type'], hue = X_train['damage_grade'], ax = axes[0][1])

sns.countplot(X_train['roof_type'], hue = X_train['damage_grade'], ax = axes[1][0])

sns.countplot(X_train['ground_floor_type'], hue = X_train['damage_grade'], ax = axes[1][1])

sns.countplot(X_train['other_floor_type'], hue = X_train['damage_grade'], ax = axes[2][0])

sns.countplot(X_train['position'], hue = X_train['damage_grade'], ax = axes[2][1])

sns.countplot(X_train['plan_configuration'], hue = X_train['damage_grade'], ax = axes[3][0])

sns.countplot(X_train['legal_ownership_status'], hue = X_train['damage_grade'], ax = axes[3][1])
cols = [['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone'], ['has_superstructure_stone_flag', 

        'has_superstructure_cement_mortar_stone'], ['has_superstructure_mud_mortar_brick', 

        'has_superstructure_cement_mortar_brick'], ['has_superstructure_timber', 'has_superstructure_bamboo'], 

        ['has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered']]



fig, axes = plt.subplots(ncols = 2, nrows = 5, figsize = (20,20))

for i, c in enumerate(cols):

    X_train[c[0]].value_counts().plot.pie(autopct="%.1f%%", ax = axes[i][0])

    X_train[c[1]].value_counts().plot.pie(autopct="%.1f%%", ax = axes[i][1])

plt.show()

X_train['has_superstructure_other'].value_counts().plot.pie(autopct = "%.1f%%")

plt.show()
cols = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 

        'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 

        'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 

        'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other'

       ]



for c in cols:

    plt.figure(figsize=(15,4))

    total = float(len(X_train[c])) 

    ax = sns.countplot(x = X_train[c], hue=X_train.damage_grade, palette='Paired')

    plt.title(f"{c} VS Damage Grade")

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}'.format(height/total),

                ha="center") 

    plt.show()
cols=[["has_secondary_use","has_secondary_use_agriculture"],

             ["has_secondary_use_hotel","has_secondary_use_rental"],

             ["has_secondary_use_institution","has_secondary_use_school"],

             ["has_secondary_use_industry", "has_secondary_use_health_post"],

             ["has_secondary_use_gov_office","has_secondary_use_use_police"]]



fig, axes = plt.subplots(ncols = 2, nrows = 5, figsize = (20,20))

for i, c in enumerate(cols):

    X_train[c[0]].value_counts().plot.pie(autopct="%.2f%%", ax = axes[i][0])

    X_train[c[1]].value_counts().plot.pie(autopct="%.2f%%", ax = axes[i][1])

plt.show()

X_train['has_secondary_use_other'].value_counts().plot.pie(autopct = "%.2f%%")

plt.show()
cols=["has_secondary_use","has_secondary_use_agriculture","has_secondary_use_hotel",

      "has_secondary_use_rental","has_secondary_use_institution","has_secondary_use_school",

      "has_secondary_use_industry","has_secondary_use_health_post","has_secondary_use_gov_office",

      "has_secondary_use_use_police","has_secondary_use_other"]



for c in cols:

    plt.figure(figsize=(15,4))

    total = float(len(X_train[c])) 

    ax = sns.countplot(x = X_train[c], hue=X_train.damage_grade, palette='Paired')

    plt.title(f"{c} VS Damage Grade")

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}'.format(height/total),

                ha="center") 

    plt.show()
X_train['count_families'].unique()
fig = plt.figure(figsize=(15,4))

ax = sns.countplot(X_train['count_families'])

totals = []



# find the values and append to list

for i in ax.patches:

    totals.append(i.get_height())



# set individual bar lables using above list

total = sum(totals)



# set individual bar lables using above list

for i in ax.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax.text(i.get_x()+.12, i.get_height()+5, \

            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15, color='black')
plt.figure(figsize=(10,8))

ax=X_train.groupby("damage_grade")["count_families"].sum().sort_values().plot.bar(color=["mediumturquoise","turquoise","aquamarine"],

                                                                               )

# create a list to collect the plt.patches data

totals = []



# find the values and append to list

for i in ax.patches:

    totals.append(i.get_height())



# set individual bar lables using above list

total = sum(totals)



# set individual bar lables using above list

for i in ax.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax.text(i.get_x()+.12, i.get_height()+5, \

            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,

                color='black')

plt.title("Families Affected due to earthquake")

plt.ylabel("No. of families")

plt.xlabel("Damage Grade")

plt.show()