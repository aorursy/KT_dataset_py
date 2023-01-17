import numpy as np

import pandas as pd

import math as math

# Checking for kaggle/input/titanic/****.csv

import os

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv");

train_data.name = "Training Set"

test_data.name = "Test Set"

train_data.sample(10)
# Column datatypes

train_data.info()

print('+'*40)

test_data.info()
# Columns

columns = train_data.columns

print(columns)
def concat_df(train_data, test_data):

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(data):

    return data.loc[:890], data.loc[891:].drop(['Survived'], axis=1)
# Combine dataset for exploratory analytics

df = concat_df(train_data, test_data)

df.name = "Total Data"

df.info()
embarked_labels={"S": "Southampton", "C": "Chernboug", "Q": "Queenstown"}
# Change Pclass, Sex, Embarked to type category

for col in ['Pclass', 'Sex', 'Embarked']:

    train_data[col] = train_data[col].astype('category')

    test_data[col] = test_data[col].astype('category')

train_data.info()

print("-"*40)

test_data.info()
# Check for null count

print("Check for nulls in train data: ")

print(train_data.isnull().sum())

print("+"*40)

print("Check for nulls in test data: ")

print(test_data.isnull().sum())
# Describe gives a very quick brief 5-point summary of the data. Take a peek look into numerical data.

train_data.describe(include="all").transpose()
# changing back to original data types.

train_data["Pclass"] = train_data["Pclass"].astype("int")

train_data["Sex"] = train_data["Sex"].astype("object")

train_data["Embarked"] = train_data["Embarked"].astype("object")

test_data["Pclass"] = test_data["Pclass"].astype("int")

test_data["Sex"] = test_data["Sex"].astype("object")

test_data["Embarked"] = test_data["Embarked"].astype("object")

train_data.info()

print("+"*40)

test_data.info()
_ = len(train_data[train_data['Survived'] == 1])/len(train_data) * 100

print('Survived %: ', _)
import matplotlib.pyplot as plt

import seaborn as sns

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sns.color_palette(flatui)

# Categorical plots (Bar)

fig, ax = plt.subplots(2, 3, figsize=(20,15))

ax = ax.flatten()

unique_sibsp = train_data["SibSp"].unique()

unique_sibsp.sort()

unique_parch = train_data["Parch"].unique()

unique_parch.sort()



cols = [

    {"col": "Pclass", "x_labels": ["First", "Second", "Thrid"], "title": "Class of Passenger"},

    {"col": "Embarked", "x_labels": ["Chernboug", "Queenstown", "Southampton"], "title": "Where Passenger Boarded?"},

    {"col": "Survived", "x_labels": ["Not Survived", "Survived"], "title": "Who many passengers survived?"},

    {"col": "Sex", "x_labels": ["Male", "Female"], "title": "Sex of person"},

    {"col": "SibSp", "x_labels": unique_sibsp, "title": "Siblings & Spouses count"},

    {"col": "Parch", "x_labels": unique_parch, "title": "Parents & Children count"}

]

for i in range(0, 6):

    _ = train_data[cols[i]['col']].value_counts()

    _ = sns.barplot(x=_.index, y=_, ax=ax[i]) # returns ax of matplotlib

    _.set_xticklabels(cols[i]['x_labels'])

    _.set_ylabel('Count')

    _.set_title(cols[i]['title'])

    for patch in _.patches:

        label_x = patch.get_x() + patch.get_width()/2 # Mid point in x

        label_y = patch.get_y() + patch.get_height() + 10 # Mid point in y

        _.text(label_x,

               label_y,

               "{} ({:.1%})".format(int(patch.get_height()), patch.get_height()/len(train_data[cols[i]['col']])),

               horizontalalignment='center',

               verticalalignment='center')
fig, ax = plt.subplots(2, 3, figsize=(20,15))

ax = ax.flatten()



cols = [

    {"col": "Pclass", "x_labels": ["First", "Second", "Thrid"], "title": "Average age of passengers in each class"},

    {"col": "Embarked", "x_labels": ["Chernboug", "Queenstown", "Southampton"], "title": "Average age of passengers based on boarding location"},

    {"col": "Sex", "x_labels": ["Male", "Female"], "title": "Average age of person based on gender"},

    {"col": "SibSp", "x_labels": unique_sibsp, "title": "Average age of Siblings & Spouses"},

    {"col": "Parch", "x_labels": unique_parch, "title": "Average age of Parents & Children"}

]



for i in range(0, 5):

    ax_ = sns.barplot(x=cols[i]["col"], y="Age", ax=ax[i], data=train_data) # returns ax of matplotlib

    ax_.set_xticklabels(cols[i]["x_labels"])

    ax_.set_ylabel("Average Age")

    ax_.set_title(cols[i]["title"])

    for patch in ax_.patches:

        ax_.text(

            patch.get_x() + patch.get_width()/2,

            patch.get_y() + (0 if math.isnan(patch.get_height()) else patch.get_height()/2),

            "{:.2f}".format(patch.get_height()),

            horizontalalignment="center",

            verticalalignment="center")
fig, ax = plt.subplots(2, 3, figsize=(20,15))

ax = ax.flatten()



cols = [

    {"col": "Pclass", "x_labels": ["First", "Second", "Thrid"], "title": "% of survivors in each passenger class"},

    {"col": "Embarked", "x_labels": ["Chernboug", "Queenstown", "Southampton"], "title": "% of survivors based on boarding location"},

    {"col": "Sex", "x_labels": ["Male", "Female"], "title": "% of survivors based on gender"},

    {"col": "SibSp", "x_labels": unique_sibsp, "title": "% of survivors in siblings & spouses"},

    {"col": "Parch", "x_labels": unique_parch, "title": "% of surviviors in parent & children"}

]



for i in range(0, 5):

    ax_ = sns.barplot(x=cols[i]["col"], y="Survived", ax=ax[i], data=train_data) # returns ax of matplotlib

    ax_.set_xticklabels(cols[i]["x_labels"])

    ax_.set_ylabel("Average Age")

    ax_.set_title(cols[i]["title"])

    for patch in ax_.patches:

        ax_.text(

            patch.get_x() + patch.get_width()/2,

            patch.get_y() + (0 if math.isnan(patch.get_height()) else patch.get_height()/2),

            "{:.2f}%".format(patch.get_height() * 100),

            horizontalalignment="center",

            verticalalignment="center")
fig, ax = plt.subplots(2, 2, figsize=(20,15))

ax = ax.flatten()



cols = [

    {"col": "Pclass", "x_labels": ["First", "Second", "Thrid"], "title": "% of survivors grouped by gender in each passenger class"},

    {"col": "Embarked", "x_labels": ["Chernboug", "Queenstown", "Southampton"], "title": "% of survivors grouped by gender in each boarding class"},

    {"col": "SibSp", "x_labels": unique_sibsp, "title": "% of survivors grouped by gender in siblings and spouses"},

    {"col": "Parch", "x_labels": unique_parch, "title": "% of survivors grouped by gender in parents and children"}

]



for i in range(0, 4):

    ax_ = sns.barplot(x=cols[i]["col"], y="Survived", hue="Sex", ax=ax[i], data=train_data) # returns ax of matplotlib

    ax_.set_xticklabels(cols[i]["x_labels"])

    ax_.set_ylabel("Average Age")

    ax_.set_title(cols[i]["title"])

    for patch in ax_.patches:

        ax_.text(

            patch.get_x() + patch.get_width()/2,

            patch.get_y() + (0 if math.isnan(patch.get_height()) else patch.get_height()/2),

            "{:.2f}%".format(patch.get_height() * 100),

            horizontalalignment="center",

            verticalalignment="center")
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

ax = ax.flatten()

cols = [

    {"col": "Age", "x_label": "Age", "title": "Age Distribution", "color": "green"},

    {"col": "Fare", "x_label": "Fare", "title": "Fare Distribution", "color": "violet"}

]

for i in range(0, len(cols)):

    _col = cols[i]["col"]

    data_ = train_data[_col][pd.notnull(train_data[_col])]

    _ = sns.distplot(data_, kde=True, hist=False, ax=ax[i], color=cols[i]["color"])

    _.set_title(cols[i]["title"])

    _.set_xlabel(cols[i]["x_label"])

    mean_ = data_.mean()

    median_ = data_.median()

    _.axvline(mean_, linestyle="--", color="red")

    _.axvline(median_, linestyle="--", color="orange")

    _.legend({"Mean": mean_, "Median": median_})
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax = ax.flatten()

cols = [

    {"col": "Age", "x_label": "Age", "title": "Age Distribution"},

    {"col": "Fare", "x_label": "Fare", "title": "Fare Distribution"}

]

for i in range(0, len(cols)):

    _col = cols[i]["col"]

    data_ = train_data[_col][pd.notnull(train_data[_col])]

    _ = sns.boxplot(data_, ax=ax[i])

    _.set_title(cols[i]["title"])

    _.set_xlabel(cols[i]["x_label"])
fig, ax = plt.subplots(3, 2, figsize=(20, 15))

ax = ax.flatten()

cols = [

    {'x': 'Pclass', 'y': 'Age', 'x_label': 'Pclass', 'title': 'Age Distribution based on Pclass'},

    {'x': 'Pclass', 'y': 'Fare', 'x_label': 'Pclass', 'title': 'Fare Distribution based on Pclass'},

    {'x': 'Embarked', 'y': 'Age', 'x_label': 'Embarked', 'title': 'Age Distribution based on Embarked'},

    {'x': 'Embarked', 'y': 'Fare', 'x_label': 'Embarked', 'title': 'Fare Distribution based on Embarked'},

    {'x': 'Sex', 'y': 'Age', 'x_label': 'Sex', 'title': 'Age Distribution based on Sex'},

    {'x': 'Sex', 'y': 'Fare', 'x_label': 'Sex', 'title': 'Fare Distribution based on Sex'}

]

for i in range(0, len(cols)):

    data_ = train_data[train_data[cols[i]['y']].notnull()]

    _ = sns.boxplot(x=cols[i]['x'], y=cols[i]['y'], data=data_, ax=ax[i])

    _.set_title(cols[i]['title'])

    _.set_xlabel(cols[i]['x_label'])
fig, ax = plt.subplots(3, 2, figsize=(20, 15))

ax = ax.flatten()

cols = [

    {'x': 'Pclass', 'y': 'Age', 'x_label': 'Pclass', 'title': 'Age Distribution based on Pclass'},

    {'x': 'Pclass', 'y': 'Fare', 'x_label': 'Pclass', 'title': 'Fare Distribution based on Pclass'},

    {'x': 'Embarked', 'y': 'Age', 'x_label': 'Embarked', 'title': 'Age Distribution based on Embarked'},

    {'x': 'Embarked', 'y': 'Fare', 'x_label': 'Embarked', 'title': 'Fare Distribution based on Embarked'},

    {'x': 'Sex', 'y': 'Age', 'x_label': 'Sex', 'title': 'Age Distribution based on Sex'},

    {'x': 'Sex', 'y': 'Fare', 'x_label': 'Sex', 'title': 'Fare Distribution based on Sex'}

]

for i in range(0, len(cols)):

    data_ = train_data[train_data[cols[i]['y']].notnull()]

    _ = sns.boxplot(x=cols[i]['x'], y=cols[i]['y'], hue='Survived', data=data_, ax=ax[i])

    _.set_title(cols[i]['title'])

    _.set_xlabel(cols[i]['x_label'])
ax = sns.FacetGrid(train_data, col="Sex", row="Pclass", legend_out=True)

ax.map(plt.hist, "Age", color="blue")
ax = sns.FacetGrid(train_data, col="Sex", row="Pclass", hue="Survived", legend_out=True)

ax.map(sns.kdeplot, "Age")
ax = sns.FacetGrid(train_data, col="Pclass", hue="Survived")

ax.map(sns.scatterplot, "Age", "Fare")
# ax = sns.FacetGrid(train_data, col="Sex", row="Pclass", hue="Survived", legend_out=True)

# ax.map(sns.kdeplot, "Age")

sns.factorplot(x="Sex", y="Survived", col="Pclass", data=train_data, saturation=.5, kind="bar")
_ = sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=train_data, kind='bar')

for patch in _.ax.patches:

    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle

    label_y = patch.get_y() + patch.get_height()/2

    _.ax.text(label_x,

              label_y,

              '{:.3%}'.format(patch.get_height()),

              horizontalalignment='center',

              verticalalignment='center')
_ = sns.factorplot(x='Embarked', y='Survived', hue='Sex', data=train_data, kind='bar')

for patch in _.ax.patches:

    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle

    label_y = patch.get_y() + patch.get_height()/2

    _.ax.text(label_x,

              label_y,

              '{:.3%}'.format(patch.get_height()),

              horizontalalignment='center',

              verticalalignment='center')
sns.catplot(data=train_data, x="Sex", y="Age", col="Pclass", hue="Survived", kind="swarm")
_ = sns.catplot(x="Sex", y="Survived", col="Pclass", data=train_data, kind='bar')

_.set_axis_labels("", "Survival Rate")

_.set_titles("{col_name}-{col_var}")

for ax in _.axes[0]:

    for patch in ax.patches:

        label_x = patch.get_x() + patch.get_width()/2

        label_y = patch.get_y() + patch.get_height()/2

        ax.text(label_x,

                label_y,

                "{0:.2f}%".format(patch.get_height()*100),

                horizontalalignment='center',

                verticalalignment='center')
_ = sns.catplot(x="Sex", y="Survived", col="Embarked", data=train_data, kind='bar')

_.set_axis_labels("", "Survival Rate")

_.set_titles("{col_name}")

for ax in _.axes[0]:

    for patch in ax.patches:

        label_x = patch.get_x() + patch.get_width()/2

        label_y = patch.get_y() + patch.get_height()/2

        ax.text(label_x,

                label_y,

                "{0:.2f}%".format(patch.get_height()*100),

                horizontalalignment='center',

                verticalalignment='center')
sns.catplot(x="Parch", y="Survived", col="Sex", data=train_data, kind="bar")
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

ax = ax.flatten()

cols = [

    {"col": "Pclass", "title": "Passenger Class", "x_labels": ["1", "2", "3"]},

    {"col": "Embarked", "title": "Boarded from", "x_labels": ["S", "Q", "C"]},

    {"col": "Sex", "title": "Gender", "x_labels": ["M", "F"]}

]

for i in range(0, len(cols)):

    _ = sns.countplot(x=cols[i]['col'], data=train_data, hue="Survived", ax=ax[i])

    _.set_title(cols[i]['title'])

    _.set_xticklabels(cols[i]['x_labels'])

    for patch in _.patches:

        x_label = patch.get_x() + patch.get_width()/2

        y_label = patch.get_y() + patch.get_height() + 7

        

        _.text(x_label,

               y_label,

               "{0}".format(patch.get_height()),

               horizontalalignment='center',

               verticalalignment='center')
sns.countplot(x="Parch", hue="Sex", data=train_data)
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

ax = ax.flatten()

pd.crosstab(train_data['Survived'], train_data['Pclass']).plot.bar(stacked=True, ax=ax[0])

pd.crosstab(train_data['Survived'], train_data['Embarked']).plot.bar(stacked=True, ax=ax[1])

pd.crosstab(train_data['Survived'], train_data['Sex']).plot.bar(stacked=True, ax=ax[2])
fig, ax = plt.subplots(figsize=(10, 5))

sns.kdeplot(

    data=train_data['Age'][(train_data['Survived'] == 0) & (train_data['Age'].notnull())],

    ax=ax,

    color='Red',

    shade=True)

sns.kdeplot(

    data=train_data['Age'][(train_data['Survived'] == 1) & (train_data['Age'].notnull())],

    ax=ax,

    color='Blue',

    shade=True)

ax.legend(["Not Survived", "Survived"])

ax.set_title("Superimposed KDE plot for age of Survived and Not Survived")
fig, ax = plt.subplots(figsize=(10, 5))

sns.kdeplot(data=train_data.loc[train_data['Survived'] == 0, 'Fare'], color='Red', shade=True, legend=True)

sns.kdeplot(data=train_data.loc[train_data['Survived'] == 1, 'Fare'], color='Blue', shade=True, legend=True)

ax.legend(["Not Survived", "Survived"])

ax.set_title("Superimposed KDE plot for fare of Survived and Not Survived")
corr = train_data.corr()

plt.figure(figsize=(15, 10))

sns.heatmap(corr, annot=True)
# Check for survival rate against each category

def survival_rate(data, column):

    categories_ = data[column].unique()

    categories_.sort()

    print('{0} based survival rate:'.format(column))

    print('+'*40)

    for cat in categories_:

        _ = data.loc[data[column] == cat]

        print('{0} - {1} survival rate: {2:.3f}'.format(column, cat, (_['Survived'].sum()/len(_)) * 100))
for d_ in [train_data, test_data]:

    print(d_.name)

    print("+"*40)

    for col in d_.columns:

        missing_ = d_[col].isnull().sum()

        if missing_ > 0:

            print("'{0}' column missing value count {1}({2:.2f}%)".format(col, missing_, missing_/len(d_)*100))

    print("+"*40)
# Correlation Coefficents

_ = df.corr().abs().unstack().sort_values().reset_index().rename(

    columns={"level_0": "F1", "level_1": "F2", 0: "Coef_"})

_ = _[_["F1"] == "Age"]

print(_)
# _ = df.groupby(["Sex", "Pclass"]).describe()

_ = df.groupby(["Sex", "Pclass", "SibSp"]).describe()

_ = _["Age"].loc[:, ["mean", "50%"]]

print(_)

# When grouped by sex and by pclass mean and median are most consistent and they values are pretty much closer.

# So replacing the values based on these stats should be easy.
df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].apply(lambda x: x.fillna(x.median()))
# Comparing Age distributions before and after imputing the values.

df_age_old = concat_df(train_data, test_data)

df_age_old = df_age_old["Age"][pd.notnull(df_age_old["Age"])]

plt.figure(figsize=(10, 5))

sns.kdeplot(df_age_old, color="red", shade=True)

sns.kdeplot(df["Age"], color="green", shade=True)

# Data distribution remains almost same before and after imputing values.
df[df['Embarked'].isnull()]
df["Embarked"] = df["Embarked"].fillna("S")
df[df["Fare"].isnull()]

_ = df.groupby(["Pclass", "Parch", "SibSp"]).describe()

_ = _["Fare"].loc[:, ["mean", "50%"]]

print(_)

df["Fare"] = df.groupby(["Pclass", "Parch", "SibSp"])["Fare"].apply(lambda x: x.fillna(x.mean()))
print("Passengers without deck: ", df["Cabin"].isnull().sum())

df["Deck"] = df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else "M")

df_decks = df.groupby(["Deck", "Pclass"]).count().drop(

    columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={"Name": "Count"}).transpose()

df_decks
deck_x_pclass = pd.crosstab(df["Deck"], df["Pclass"])

plt.title = "passengers in each deck"

# deck_x_pclass.plot.bar(stacked=True)

plt.figure(figsize=(20, 10))

# now stack and reset

stacked = deck_x_pclass.stack().reset_index().rename(columns={0:'value'})

stacked["percent"] = 0

plt.figure(figsize=(20, 8))

# plot grouped bar chart

sns.barplot(x="Deck", y="value", hue="Pclass", data=stacked)

# Plot percentage of passengers

for index in deck_x_pclass.index:

    total_ = deck_x_pclass.loc[index].sum()

    for pclass in deck_x_pclass.columns:

        val_ = stacked.loc[(stacked["Deck"] == index) & (stacked["Pclass"] == pclass), 'value']

        stacked.loc[(stacked["Deck"] == index) & (stacked["Pclass"] == pclass), 'percent'] = val_/total_*100
stacked[stacked["percent"] > 0]
# Replace Deck T with A as it has only one passenger

_ = df[df["Deck"] == "T"].index

if (_.size > 0):

    df.loc[_, "Deck"] = "A"
df_survived = df.groupby(["Deck", "Survived"]).count().drop(

    columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']

    ).rename(columns={'Name':'Count'}).transpose()

print(df_survived)

surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}

decks = df_survived.columns.levels[0]

for deck in decks:

    for survive in range(0, 2):

        surv_counts[deck][survive] = df_survived[deck][survive][0]

df_surv_counts = pd.DataFrame(surv_counts)

surv_percentages = {}

for col in df_surv_counts:

    surv_percentages[col] = [(count / df_surv_counts[col].sum()) * 100 for count in df_surv_counts[col]]

print(surv_percentages)

df_survived_percentages = pd.DataFrame(surv_percentages).transpose()

deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')

bar_count = np.arange(len(deck_names))  

bar_width = 0.85

not_survived = df_survived_percentages[0]

survived = df_survived_percentages[1]

plt.figure(figsize=(20, 10))

plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")

plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")

plt.xlabel('Deck', size=15, labelpad=20)

plt.ylabel('Survival Percentage', size=15, labelpad=20)

plt.xticks(bar_count, deck_names)    

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

# plt.title('Survival Percentage in Decks')
df["Deck"] = df["Deck"].replace(["A", "B", "C"], "ABC").replace(["D", "E"], "DE").replace(["F", "G"], "FG")

df["Deck"].value_counts()
df.drop(["Cabin"], inplace=True, axis=1)
train, test = divide_df(df)

print(train.info())

print("+"*40)

print(test.info())
df["Fare"] = pd.qcut(df["Fare"], 13)
plt.figure(figsize=(20, 5))

sns.countplot(x="Fare", hue="Survived", data=df)

plt.xlabel('Fare')

plt.ylabel('Passenger Count')

plt.tick_params(axis='x')

plt.tick_params(axis='y')



plt.legend(['Not Survived', 'Survived'], loc='upper right')
df["Age"] = pd.qcut(df["Age"], 10)
plt.figure(figsize=(20, 5))

sns.countplot(x="Age", hue="Survived", data=df)

plt.xlabel('Age')

plt.ylabel('Passenger Count')

plt.tick_params(axis='x')

plt.tick_params(axis='y')



plt.legend(['Not Survived', 'Survived'], loc='upper right')
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

sns.countplot(x="FamilySize", data=df, ax=ax[0])

sns.countplot(x="FamilySize", hue="Survived", data=df, ax=ax[1])
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

df['FamilySizeGrouped'] = df['FamilySize'].map(family_map)
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

sns.countplot(x="FamilySizeGrouped", data=df, ax=ax[0])

sns.countplot(x="FamilySizeGrouped", hue="Survived", data=df, ax=ax[1])
df["TicketFrequency"] = df.groupby("Ticket")["Ticket"].transform("count")
fig, axs = plt.subplots(figsize=(12, 5))

sns.countplot(x='TicketFrequency', hue='Survived', data=df)



plt.xlabel('Ticket Frequency', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
df["Title"] = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
df["IsMarried"] = 0

df.loc[df["Title"] == 'Mrs', "IsMarried"] = 1
plt.figure(figsize=(20, 7))

ax_ = sns.countplot(x="Title", data=df)

for patch in ax_.patches:

    label_x = patch.get_x() + patch.get_width()/2

    label_y = patch.get_y() + patch.get_height()+ 10

    ax_.text(label_x, label_y, patch.get_height(), horizontalalignment='center', verticalalignment='center')

sns.countplot(x="Title", hue="Survived", data=df)
df['Title'] = df['Title'].replace(

    ['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

df['Title'] = df['Title'].replace(

    ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
plt.figure(figsize=(20, 7))

ax_ = sns.countplot(x="Title", data=df)

for patch in ax_.patches:

    label_x = patch.get_x() + patch.get_width()/2

    label_y = patch.get_y() + patch.get_height()+ 10

    ax_.text(label_x, label_y, patch.get_height(), horizontalalignment='center', verticalalignment='center')

plt.figure(figsize=(20, 7))

sns.countplot(x="Title", hue="Survived", data=df)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train, test = divide_df(df)

dfs = [train, test]
non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'FamilySizeGrouped', 'Age', 'Fare']

for df_ in dfs:

    for feature in non_numeric_features:        

        df_[feature] = LabelEncoder().fit_transform(df_[feature])
cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'FamilySizeGrouped']

encoded_features = []



for df in dfs:

    for feature in cat_features:

        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

        n = df[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df.index

        encoded_features.append(encoded_df)



train = pd.concat([train, *encoded_features[:6]], axis=1)

test = pd.concat([test, *encoded_features[6:]], axis=1)
drop_cols = ['Deck', 'Embarked', 'FamilySize', 'FamilySizeGrouped', 'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title', 'TicketFrequency', 'IsMarried']

train = train.drop(columns=drop_cols)

test = test.drop(columns=drop_cols)
X_train = train.loc[:, train.columns != "Survived"]

y_train = train.loc[:, "Survived"]

X_test = test
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

X_train_scaled = std_scaler.fit_transform(X_train)

X_test_scaled = std_scaler.fit_transform(X_test)

print(X_train.shape, X_test.shape)
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(criterion="gini",

                                  n_estimators=1750,

                                  max_depth=7,

                                  min_samples_split=6,

                                  min_samples_leaf=6,

                                  max_features="auto",

                                  oob_score=True,

                                  random_state=40,

                                  n_jobs=-1)
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc

N = 5

oob = 0

prob_df = pd.DataFrame(np.zeros((len(X_test), N*2)))

prob_df.columns = ["KFold{}_{}".format(i, j) for i in range(1, N+1) for j in range(2)]

prob_df

imp_df = pd.DataFrame(np.zeros((X_train.shape[1], N)))

imp_df.columns = ["KFold{}".format(i) for i in range(1, N+1)]

imp_df.index = X_train.columns
fprs, tprs, scores = [], [], []

skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    print("Fold {}".format(fold))

    print("+"*40)

    rfc_model.fit(X_train.loc[trn_idx], y_train.loc[trn_idx])

    predict_proba_ = rfc_model.predict_proba(X_train.loc[trn_idx])

    trn_fpr, trn_tpr, trn_thresholds = roc_curve(

        y_train[trn_idx],

        predict_proba_[:, 1])

    trn_auc_score = auc(trn_fpr, trn_tpr)

    predict_proba_val_ = rfc_model.predict_proba(X_train.loc[val_idx])

    val_fpr, val_tpr, val_thresholds = roc_curve(

        y_train.loc[val_idx],

        predict_proba_val_[:, 1])

    val_auc_score = auc(val_fpr, val_tpr)

    scores.append((trn_auc_score, val_auc_score))

    fprs.append(val_fpr)

    tprs.append(val_tpr)

    

    prob_df.loc[:, "KFold{}_0".format(fold)] = rfc_model.predict_proba(X_test)[:, 0]

    prob_df.loc[:, "KFold{}_1".format(fold)] = rfc_model.predict_proba(X_test)[:, 1]

    

    imp_df.iloc[:, fold - 1] = rfc_model.feature_importances_

    oob += rfc_model.oob_score_ / N

    print('Fold {} OOB Score: {}\n'.format(fold, rfc_model.oob_score_)) 



print('Average OOB Score: {}'.format(oob))
imp_df["Mean"] = imp_df.mean(axis = 1)
imp_df.sort_values(by='Mean', inplace=True, ascending=False)

plt.figure(figsize=(15, 10))

sns.barplot(x='Mean', y=imp_df.index, data=imp_df)



plt.xlabel('')

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)

# plt.title('Random Forest Classifier Mean Feature Importance Between Folds')
class_survived = [col for col in prob_df.columns if col.endswith('1')]

prob_df['1'] = prob_df[class_survived].sum(axis=1) / N

prob_df['0'] = prob_df.drop(columns=class_survived).sum(axis=1) / N

prob_df['pred'] = 0

pos = prob_df[prob_df['1'] >= 0.5].index

prob_df.loc[pos, 'pred'] = 1



y_pred = prob_df['pred'].astype(int)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = test_data['PassengerId']

submission_df['Survived'] = y_pred.values

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)