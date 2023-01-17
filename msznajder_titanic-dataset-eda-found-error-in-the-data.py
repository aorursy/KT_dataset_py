import numpy as np



from scipy import stats

from statsmodels.formula.api import ols



import pandas as pd

from pandas.tools.plotting import scatter_matrix



import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline 



import warnings

warnings.filterwarnings('ignore')
titanic_train = pd.read_csv("../input/train.csv")
titanic_train.head()
titanic_train = titanic_train.set_index("PassengerId")

titanic_train.head()
titanic_train.info()
titanic_train.describe()
titanic_train.describe()
titanic_train.hist(bins=20, figsize=(18, 16), color="#f1b7b0");
titanic_train["Name"].value_counts()
titanic_train["Ticket"].value_counts()
titanic_train["Cabin"].value_counts()
titanic_train["Sex"].value_counts()
titanic_train["Sex"].value_counts().plot(kind='bar', figsize=(6, 4), grid=True, color="#f1b7b0", title="Sex")
titanic_train["Embarked"].value_counts()
titanic_train["Embarked"].value_counts().plot(kind='bar', figsize=(6, 4), grid=True, color="#f1b7b0", title="Embarked")
scatter_matrix(titanic_train, figsize=(18, 16), c="#f1b7b0", hist_kwds={'color':['#f1b7b0']});
corr_matrix = titanic_train.corr()

corr_matrix
fig, axes = plt.subplots(figsize=(8, 8))

cax = axes.matshow(corr_matrix, vmin=-1, vmax=1, cmap=plt.cm.pink)

fig.colorbar(cax)

ticks = np.arange(0, len(corr_matrix), 1)

axes.set_xticks(ticks)

axes.set_yticks(ticks)

axes.set_xticklabels(corr_matrix)

axes.set_yticklabels(corr_matrix)

plt.show()
titanic_class_counts = titanic_train["Pclass"].value_counts(sort=False)

titanic_class_counts.index = ["First class", "Second class", "Third class"]

titanic_class_counts
titanic_sex_counts = titanic_train["Sex"].value_counts()

titanic_sex_counts
titanic_age_groups_counts = pd.cut(titanic_train["Age"], bins=[0, 14, 24, 34, 44, 54, 64, 80]).value_counts().sort_index()

titanic_age_groups_counts
titanic_train["Age"].describe()
titanic_train.loc[titanic_train["Age"].argmin()]
titanic_train.loc[titanic_train["Age"].argmax()]
titanic_train = titanic_train.drop(titanic_train["Age"].argmax(), axis=0)
titanic_train["Age"].describe()
titanic_age_groups_counts = pd.cut(titanic_train["Age"], bins=[0, 14, 24, 34, 44, 54, 64, 80]).value_counts().sort_index()

titanic_age_groups_counts
titanic_train.loc[titanic_train["Age"].argmax()]
titanic_sibsp_counts = titanic_train["SibSp"].value_counts()

titanic_sibsp_counts
titanic_parch_counts = titanic_train["Parch"].value_counts()

titanic_parch_counts
titanic_fare_groups_counts = pd.cut(titanic_train["Fare"], bins=[0, 20, 40, 60, 80, 100, 300, 600]).value_counts().sort_index()

titanic_fare_groups_counts
titanic_embarked_counts = titanic_train["Embarked"].value_counts()

titanic_embarked_counts.index = ["Southampton", "Cherbourg", "Queenstown"]

titanic_embarked_counts
titanic_survived_counts = titanic_train["Survived"].value_counts(sort=False)

titanic_survived_counts.index = ["Not survived", "Survived"]

titanic_survived_counts
def get_survival_ratio(passengers_df):

    return passengers_df["Survived"].sum() / passengers_df["Survived"].count()
overall_survival_ratio = get_survival_ratio(titanic_train)

overall_survival_ratio
titanic_pclass_group = titanic_train.groupby("Pclass")

titanic_pclass_group.groups
titanic_sex_group = titanic_train.groupby("Sex")

titanic_sex_group.groups
titanic_age_group = titanic_train.groupby(pd.cut(titanic_train["Age"], bins=[0, 14, 24, 34, 44, 54, 64, 80]))

titanic_age_group.groups
titanic_sibsp_group = titanic_train.groupby("SibSp")

titanic_sibsp_group.groups
titanic_parch_group = titanic_train.groupby("Parch")

titanic_parch_group.groups
titanic_fare_group = titanic_train.groupby(pd.cut(titanic_train["Fare"], bins=[0, 20, 40, 60, 80, 100, 300, 600]))

titanic_fare_group.groups
titanic_embarked_group = titanic_train.groupby("Embarked")

titanic_embarked_group.groups
def group_age(age):

    if age <= 14:

        return "(0, 14]"

    elif age > 14 and age <= 24:

        return "(14, 24]"

    elif age > 24 and age <= 34:

        return "(24, 34]"

    elif age > 34 and age <= 44:

        return "(34, 44]"

    elif age > 44 and age <= 54:

        return "(44, 54]"

    elif age > 54 and age <= 64:

        return "(54, 64]"

    elif age > 64:

        return "(64, 80]"

    else:

        return np.nan

    

def group_fare(fare):

    if fare <= 20:

        return "(0, 20]"

    elif fare > 20 and fare <= 40:

        return "(20, 40]"

    elif fare > 40 and fare <= 60:

        return "(40, 60]"

    elif fare > 60 and fare <= 80:

        return "(60, 80]"

    elif fare > 80 and fare <= 100:

        return "(80, 100]"

    elif fare > 100 and fare <= 300:

        return "(100, 300]"

    elif fare > 300 and fare <= 600:

        return "(300, 600]"

    else:

        return np.nan
titanic_train["AgeGroup"] = titanic_train.apply(lambda row: group_age(row["Age"]), axis=1)

titanic_train["FareGroup"] = titanic_train.apply(lambda row: group_fare(row["Fare"]), axis=1)
titanic_train.head()
titanic_pclass_survival_ratio = titanic_pclass_group.apply(get_survival_ratio)

titanic_pclass_survival_ratio.index = ["Pclass: " + str(idx) for idx in titanic_pclass_survival_ratio.index]

titanic_pclass_survival_ratio
titanic_sex_survival_ratio = titanic_sex_group.apply(get_survival_ratio)

titanic_sex_survival_ratio.index = ["Sex: " + str(idx) for idx in titanic_sex_survival_ratio.index]

titanic_sex_survival_ratio
titanic_age_survival_ratio = titanic_age_group.apply(get_survival_ratio)

titanic_age_survival_ratio.index = ["Age: " + str(idx) for idx in titanic_age_survival_ratio.index]

titanic_age_survival_ratio
titanic_sibsp_survival_ratio = titanic_sibsp_group.apply(get_survival_ratio)

titanic_sibsp_survival_ratio.index = ["Sibsp: " + str(idx) for idx in titanic_sibsp_survival_ratio.index]

titanic_sibsp_survival_ratio
titanic_parch_survival_ratio = titanic_parch_group.apply(get_survival_ratio)

titanic_parch_survival_ratio.index = ["Parch: " + str(idx) for idx in titanic_parch_survival_ratio.index]

titanic_parch_survival_ratio
titanic_fare_survival_ratio = titanic_fare_group.apply(get_survival_ratio)

titanic_fare_survival_ratio.index = ["Fare: " + str(idx) for idx in titanic_fare_survival_ratio.index]

titanic_fare_survival_ratio
titanic_embarked_survival_ratio = titanic_embarked_group.apply(get_survival_ratio)

titanic_embarked_survival_ratio.index = ["Embarked: " + str(idx) for idx in titanic_embarked_survival_ratio.index]

titanic_embarked_survival_ratio
survival_ratios = pd.concat([titanic_pclass_survival_ratio,

                             titanic_sex_survival_ratio,

                             titanic_age_survival_ratio,

                             titanic_sibsp_survival_ratio,

                             titanic_parch_survival_ratio,

                             titanic_fare_survival_ratio,

                             titanic_embarked_survival_ratio], axis=0)

survival_ratios["Overall"] = overall_survival_ratio

survival_ratios
titanic_pclass_survival_count = titanic_pclass_group.apply(len)

titanic_sex_survival_count = titanic_sex_group.apply(len)

titanic_age_survival_count = titanic_age_group.apply(len)

titanic_sibsp_survival_count = titanic_sibsp_group.apply(len)

titanic_parch_survival_count = titanic_parch_group.apply(len)

titanic_fare_survival_count = titanic_fare_group.apply(len)

titanic_embarked_survival_count = titanic_embarked_group.apply(len)



groups_counts = pd.concat([titanic_pclass_survival_count,

                           titanic_sex_survival_count,

                           titanic_age_survival_count,

                           titanic_sibsp_survival_count,

                           titanic_parch_survival_count,

                           titanic_fare_survival_count,

                           titanic_embarked_survival_count

])

groups_counts["Overall"] = len(titanic_train)

groups_counts
def get_groups_survival_ratio_plot(survival_ratios, groups_counts, labels, figsize=(20, 10)):

    idx = np.arange(len(survival_ratios))

    width = len(survival_ratios) / 50



    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

    axes[0].bar(idx, groups_counts, width, color='#f9d9ac', label="Passengers number")

    axes[0].set_title("Atributes groups passengers counts")

    axes[0].set_ylabel("Number of passengers")

    axes[0].legend(loc=2)

    

    axes[1].bar(idx, survival_ratios, width, color='#f1b7b0', label="Survived")

    axes[1].bar(idx, 1 - survival_ratios, width, bottom=survival_ratios, color='#f0f0f0', label="Not survived")

    axes[1].set_title("Survival ratios for atributes groups")

    axes[1].set_ylabel("Survival ratio")

    axes[1].legend(loc=2)

    

    plt.xticks(idx, labels, rotation='vertical', fontsize=12)

    plt.tight_layout()

    plt.show()
get_groups_survival_ratio_plot(survival_ratios.values, groups_counts.values, survival_ratios.index)
model = ols("Survived ~ C(Pclass)", titanic_train).fit()

model.summary()
model = ols("Survived ~ C(Sex)", titanic_train).fit()

model.summary()
model = ols("Survived ~ AgeGroup", titanic_train).fit()

model.summary()
titanic_train[titanic_train["SibSp"] == 8]
model = ols("Survived ~ C(SibSp)", titanic_train).fit()

model.summary()
model = ols("Survived ~ C(Parch)", titanic_train).fit()

model.summary()
model = ols("Survived ~ C(FareGroup)", titanic_train).fit()

model.summary()
titanic_embarked_group.get_group("S")["Pclass"].value_counts(sort=False)
titanic_embarked_group.get_group("Q")["Pclass"].value_counts(sort=False)
titanic_embarked_group.get_group("C")["Pclass"].value_counts(sort=False)
model = ols("Survived ~ Embarked", titanic_train).fit()

model.summary()