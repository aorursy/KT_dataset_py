# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder 

#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# read the train dataset

train_ds = pd.read_csv("../input/train.csv")
# show the train dataset informations

train_ds.info()
# show 10 sample rows of the train dataset

train_ds.sample(10)
# show descriptive statistics of the dataset

train_ds.describe()
# read the test dataset

test_ds = pd.read_csv("../input/test.csv")
# show the test dataset informations

test_ds.info()
# show 10 sample rows of the test dataset

test_ds.sample(10)
# show descriptive statistics of the test dataset

test_ds.describe()
dss = [train_ds, test_ds]
# count the number of missing value per column in the datasets

for ds in dss:

    ds.isnull().sum()

    # show distinct values for each column

    for col in ds:

        print(ds[col].value_counts())
# cleaning dataset ds by filling column where missing value has been observed

# return a new cleaned dataset

#  - Age, 263 values are missing in the full dataset, 

#    fill with median

#  - Embarked, 2 values are missing in the full dataset, 

#    meaning that 2 persons didn't embarked, hence we remove this 2 rows

#  - Fare, 1 value is missing in the full dataset, 

#    fill with median

#  - Cabin, 1014 values are missing in the full dataset, 

#    as there is more than 77% values missing we drop this column

#  - PassengerId and Ticket doesn't bring any values as they're as many differents values as tickets

#    drop the columns

def clean_ds(ds):

    ds_copy = ds.copy()

    ds_copy["Age"].fillna(ds_copy["Age"].median(), inplace = True)

    ds_copy.dropna(subset=["Embarked"], inplace=True)

    ds_copy["Fare"].fillna(ds_copy["Fare"].median(), inplace = True)

    ds_copy.drop(columns=['PassengerId','Cabin', 'Ticket'], inplace=True)

    return ds_copy
# clean datasets

dss_cleaned = [clean_ds(ds) for ds in dss]

for ds_cleaned in dss_cleaned:

    # double check that they're no more missing values

    ds_cleaned.isnull().sum()
# feature engineering

# return a new enriched dataset

#  - Name, add a column title that just containe the title of the person

#  - Fare, add FareCat column that binerize the Fare values into 4 group

#  - Age, add AgeCat column that binerize the Age values into 5 group

def enrich_dataset(ds):

    enriched = ds.copy()

    enriched["Title"] = enriched["Name"].str.extract(" ([A-Za-z]+)\.", expand=True)

    enriched['FareCat'] = pd.qcut(enriched['Fare'], 4)

    enriched['AgeCat'] = pd.cut(enriched['Age'].astype(int), 5)

    return enriched
# enrich the clean dataset

enriched_dss = [enrich_dataset(ds) for ds in dss_cleaned]

# show sample

enriched_dss[0].sample(10)
enriched_dss[1].sample(10)
for enriched_ds in enriched_dss:

    # show distinct values for Sex, title and embared

    print(enriched_ds["Sex"].value_counts())

    print(enriched_ds["Title"].value_counts())

    print(enriched_ds["Embarked"].value_counts())
# handle categorical feature

#  - sex is a string with 2 values male and female lets encode the values

#  - Embarked can takes 3 values S, C, Q, let encode the values

#  - Title code can takes differents values lets encode them

def encode_categorical_feature(ds):

    label = LabelEncoder()

    encoded = ds.copy()

    encoded['Sex_Coded'] = label.fit_transform(encoded['Sex'])

    encoded['Embarked_Coded'] = label.fit_transform(encoded['Embarked'])

    encoded['Title_Coded'] = label.fit_transform(encoded['Title'])

    encoded['AgeBin_Coded'] = label.fit_transform(encoded['AgeCat'])

    encoded['FareBin_Coded'] = label.fit_transform(encoded['FareCat'])

    return encoded
# encod categorical features

encoded_dss = [encode_categorical_feature(enriched_ds) for enriched_ds in enriched_dss]

# show sample

encoded_dss[0].sample(10)
encoded_dss[1].sample(10)
#graph distribution of qualitative data: Pclass

#we know class mattered in survival, now let's compare class and a 2nd feature

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = encoded_dss[0], ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = encoded_dss[0], split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')
#graph distribution of qualitative data: Sex

#we know sex mattered in survival, now let's compare sex and a 2nd feature

fig, qaxis = plt.subplots(1,2,figsize=(14,12))



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=encoded_dss[0], ax = qaxis[0])



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=encoded_dss[0], ax  = qaxis[1])
# how does embark port factor with class, sex, and survival compare

# facetgrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html

e = sns.FacetGrid(encoded_dss[0], col = "Embarked")

e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')

e.add_legend()
# plot distributions of age of passengers who survived or did not survive

a = sns.FacetGrid(encoded_dss[0], hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , encoded_dss[0]['Age'].max()))

a.add_legend()
h = sns.FacetGrid(encoded_dss[0], row = 'Sex', col = 'Pclass', hue = 'Survived')

h.map(plt.hist, 'Age', alpha = .75)

h.add_legend()
#correlation heatmap of dataset

def correlation_heatmap(ds):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        ds.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(encoded_dss[0])
def create_final_ds_only_coded(ds):

    return ds[["Sex_Coded", "Embarked_Coded", "Title_Coded", "AgeBin_Coded", "FareBin_Coded"]]
from sklearn.model_selection import train_test_split



encoded_train_ds = encoded_dss[0]

encoded_test_ds = encoded_dss[1]



targets = encoded_train_ds["Survived"]

features = create_final_ds_only_coded(encoded_train_ds)

X_test = create_final_ds_only_coded(encoded_test_ds)

X_train, X_cv, Y_train, Y_cv = train_test_split(features, targets, test_size=0.2)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
def create_final_ds_with_family(ds):

        return ds[["Sex_Coded", "Embarked_Coded", "Title_Coded", "AgeBin_Coded", "FareBin_Coded", "SibSp", "Parch"]]
targets = encoded_train_ds["Survived"]

features = create_final_ds_with_family(encoded_train_ds)

X_test = create_final_ds_with_family(encoded_test_ds)

X_train, X_cv, Y_train, Y_cv = train_test_split(features, targets, test_size=0.2)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# feature engineering

# enrich dataset by computing the number of family member size of the person

def add_family_feature_engineering(ds):

    ds["NumberOfFamilyMember"] = ds["SibSp"] + ds["Parch"] + 1

    ds["HasOtherFamilyMemberOnBoard"] = 1 #initialize to yes/1 is alone

    ds["HasOtherFamilyMemberOnBoard"].loc[ds["NumberOfFamilyMember"] > 1] = 0
add_family_feature_engineering(features)

add_family_feature_engineering(X_test)

X_train, X_cv, Y_train, Y_cv = train_test_split(features, targets, test_size=0.2)
X_train.head(5)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# try to augment the number of tree

random_forest = RandomForestClassifier(n_estimators=200)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
res = pd.DataFrame({

        "PassengerId": test_ds["PassengerId"],

        "Survived": Y_pred

    })

res.to_csv('res_rf.csv', index=False)
import tensorflow as tf

from tensorflow.keras import layers
# train model with validation set

model = tf.keras.Sequential()

# Adds a densely-connected layer with 10 units to the model

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train.values, Y_train.values, epochs=200, batch_size=16, validation_data=(X_cv.values, Y_cv.values))
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.preprocessing import Binarizer



predictions = model.predict(X_test.values)

threshold = 0.5

binarizer = Binarizer(threshold)

final_preds = binarizer.fit_transform(predictions)

final_preds = final_preds.astype(np.int32)
ds_res = test_ds[["PassengerId"]].copy()

ds_res["Survived"] = final_preds



ds_res.to_csv('res_nn.csv', index=False)