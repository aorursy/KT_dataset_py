# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt # visualizations

import seaborn as sns # visualizations



import sklearn.cluster as cluster

from scipy.cluster.hierarchy import dendrogram, linkage



from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

import sklearn.model_selection as mod_sel

from sklearn.feature_selection import RFECV

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegressionCV

import re



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing dataset

df = pd.read_csv('../input/train.csv')



# Creating train dataset

train_df = df.copy(deep=True)



# Importing test dataset

test_df = pd.read_csv('../input/test.csv')



# Keeping both datasets in list so we can clean both easily

data_cleaner = [train_df, test_df]
# Head of the train dataset

train_df.head()
# Shape of the dataset

train_df.shape, test_df.shape
# See data types

train_df.dtypes
# Describe dataset (summary statistics)

train_df.describe(include='all')
# Missing values over columns

print(train_df.isna().sum(), "\n\n\n", test_df.isna().sum())
# Numerical and categorical variables for different types of visualizations

numer_vars = ['Age','SibSp','Parch','Fare']

categ_vars = ['Pclass','Sex','Embarked']
sns.set() # setting visualizations' aspect



# Scatter Matrix Figure

sns.pairplot(train_df, vars=numer_vars, hue="Survived", plot_kws={"alpha":0.8}, height=3, aspect=1.3)



# Layout

plt.subplots_adjust(top=0.93)

plt.suptitle("Pairwise relationship of metric variables", fontsize=20)



plt.show()
sns.set()



# Prepare figure

fig, axes = plt.subplots(1, len(numer_vars), figsize=(14,7), constrained_layout=True)

    

# Plot data

for ax, f in zip(axes, numer_vars):

    sns.boxplot(x='Survived', y=f, data=train_df, ax=ax)



# Layout

plt.suptitle("Metric variables' box plots", fontsize=20)



plt.show()
sns.set()



# Features to plot

plot_features = ["SibSp", "Parch"]



# Prepare figure

fig, axes = plt.subplots(1, len(plot_features), figsize=(14,7))



# Plot data

for ax, f in zip(axes, plot_features):

    sns.countplot(x=f, hue='Survived', data=train_df, ax=ax)



# Layout

axes[0].set_title("Survived X SibSp", size=15)

axes[1].set_title("Survived X Parch", size=15)

plt.subplots_adjust(wspace=0.3) # adjust width between subplots



plt.show()
sns.set()



# Prepare figure

fig, axes = plt.subplots(1, len(categ_vars), figsize=(14,7))



# Plot data

for ax, f in zip(axes, categ_vars):

    sns.countplot(x=f, hue='Survived', data=train_df, ax=ax)



# Layout

plt.subplots_adjust(wspace=0.3) # adjust width between subplots

plt.suptitle("Categorical variables' absolute frequencies", fontsize=20)



plt.show()
sns.set(style="white")



# Compute the correlation matrix

corr = train_df[numer_vars].corr()



# Prepare figure

fig, ax = plt.subplots(figsize=(14, 7))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True) # Make a diverging palette between two HUSL colors. Return a matplotlib colormap object.



# Plot data

sns.heatmap(corr, annot=corr, cmap=cmap, vmax=0.5, square=True, center=0, linewidths=.5, ax=ax)



# Layout

plt.subplots_adjust(top=0.93)

plt.suptitle("Correlation matrix", fontsize=20)



plt.show()
# High Fare values

train_df["Fare"].describe()
# Eliminating extreme Fare values ((512-32.2)/49.7 = 9.65 std away from mean!) so the model performs correcly in the test set

train_df.drop(train_df.loc[train_df['Fare'] > 300].index, inplace=True) # this also updates the data_cleaner 1st element!
# Visualizing if there are any multivariate outliers



# Standardizing numerical variables as we will use distances. First we will need to impute missing values temporarily before we actually impute them.

scaler = StandardScaler()

imputer = SimpleImputer(strategy="median")

X = imputer.fit_transform(train_df[numer_vars])

X_scaled = scaler.fit_transform(X)



# Hierarchical clustering assessment using scipy

Z = linkage(X_scaled, method="ward")



sns.set()



# Figure

fig = plt.figure(figsize=(16,7))



# Dendrogram plot

dendrogram(Z, color_threshold=0, orientation='top', no_labels=True, above_threshold_color='k')



# Layout

plt.title('Hierarchical Clustering Dendrogram', fontsize=23)

plt.xlabel('Passengers', fontsize=13)

plt.ylabel('Euclidean Distance', fontsize=13)



plt.show()
sns.set()



# Prepare figure

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23,6))



# Plot data

ax1_data = train_df.isna().sum().sort_values(ascending=False)

ax1.bar(x=ax1_data.index, height=ax1_data)



ax2_data = train_df.isna().apply(lambda x:sum(x), axis=1).value_counts().sort_index().drop(0, axis=0)

ax2.bar(x=ax2_data.index, height=ax2_data)



ax3_data = test_df.isna().sum().sort_values(ascending=False)

ax3.bar(x=ax3_data.index, height=ax3_data)



# Layout

plt.suptitle("Missing value distribution", fontsize=20)



ax1.set_ylabel("Missing value count")

ax1.spines['right'].set_visible(False)

ax1.spines['top'].set_visible(False)

ax1.set_title("Missing Values per Variable - Train", fontsize=15)

for tick in ax1.get_xticklabels():

    tick.set_rotation(-45)



ax2.set_xlabel("Missing values")

ax2.set_ylabel("Row count")

ax2.spines['right'].set_visible(False)

ax2.spines['top'].set_visible(False)

ax2.set_xticklabels(["","",1,"","","",2])

ax2.set_title("Number of Rows / Number of Missing Values - Train", fontsize=15)



ax3.set_ylabel("Missing value count")

ax3.spines['right'].set_visible(False)

ax3.spines['top'].set_visible(False)

ax3.set_title("Missing Values per Variable - Test", fontsize=15)

for tick in ax3.get_xticklabels():

    tick.set_rotation(-45)



plt.show()
# Missing values over columns

print(train_df.isna().sum(), "\n\n\n", test_df.isna().sum())
# Imputing Missing values

numer_vars_imp = ["Age", "Fare"]

categ_vars_imp = ["Embarked"] # we don't impute Cabin here as we will treat it ahead



numer_imputer = SimpleImputer(strategy="mean")

categ_imputer = SimpleImputer(strategy="most_frequent")



# Imputing missing values in both train and test set

for i in data_cleaner:

    i[numer_vars_imp] = numer_imputer.fit_transform(i[numer_vars_imp])

    i[categ_vars_imp] = categ_imputer.fit_transform(i[categ_vars_imp])



# Checking if missing values were imputed

print(train_df.isna().sum(), "\n\n\n", test_df.isna().sum())
# Cabin

def cabin_fe(df):

    cabin_imputer = SimpleImputer(strategy="constant", fill_value="NK") # imputing missing values by "NK" (Not Known)

    df["Cabin"] = cabin_imputer.fit_transform(df[["Cabin"]])



    temp = df["Cabin"].str.extractall("([a-zA-Z]+)").reset_index(level=1).drop("match", axis=1) # extracting the deck of each cabin each passenger stayed



    ohe = OneHotEncoder(sparse=False) # onehot encode the decks

    decks = pd.DataFrame(ohe.fit_transform(temp[[0]]), index=temp.index, columns=ohe.categories_[0])



    decks = decks.reset_index().groupby(by="index").max() # binary for each deck marks if passenger has at least one cabin in each deck

    

    return decks





decks = cabin_fe(train_df)

train_df = train_df.merge(decks, left_index=True, right_index=True)

train_df.head()
sns.set()



# data

datatotal = train_df[["A", "B", "C", "D", "E", "F", "G", "NK", "T"]].sum().sort_values(ascending=False)

datasurvived = train_df.loc[train_df["Survived"]==1, ["A", "B", "C", "D", "E", "F", "G", "NK", "T"]].sum()[datatotal.index]



# figure

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))



# axes

# absolute values

sns.barplot(x=datatotal.index, y=datatotal.values, label="Total", color=(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), ax=ax1)

sns.barplot(x=datasurvived.index, y=datasurvived.values, label="Survived", color=(0.8666666666666667, 0.5176470588235295, 0.3215686274509804), ax=ax1)



# proportions

sns.barplot(x=datatotal.index, y=np.ones(datatotal.shape), color=(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), ax=ax2)

sns.barplot(x=(datasurvived / datatotal).index, y=(datasurvived / datatotal).values, color=(0.8666666666666667, 0.5176470588235295, 0.3215686274509804), ax=ax2)



# layout

plt.suptitle("Survived X Decks", fontsize=20)

ax1.legend(ncol=2, loc="upper right")

ax1.set_xlabel("Decks")

ax1.set_ylabel("Counts")

ax2.set_xlabel("Decks")

ax2.set_ylabel("Proportions")



plt.show()
# Creating the Deck dummy variables in test_df

decks = cabin_fe(test_df)

print(decks.columns, "\n\n\n", train_df["T"].value_counts()) # There's no "T" column in test_df. What should we do?
# Removing "T" column from train_df and assigning single value to NK

train_df.loc[train_df["T"]==1, "NK"] = 1

train_df.drop("T", axis=1, inplace=True)

# Creating the Deck dummy variables in test_df

test_df = test_df.merge(decks, left_index=True, right_index=True)
# Title

train_df["Title"] = train_df["Name"].str.extract(", (\w*\s*\w*)\.")

train_df["Title"].value_counts() # high cardinality feature. what can we do?
# Keeping highest frequency titles and labeling remaining as "Other"

train_df["Title"] = train_df["Title"].apply(lambda x: x if x in ["Mr", "Miss", "Mrs", "Master"] else "Other")



sns.set()



# Prepare figure

fig, axes = plt.subplots(figsize=(14,6))



# Plot data

sns.countplot(x="Title", hue='Survived', data=train_df)



# Layout

plt.title("Survived X Title", size=20)



plt.show()
# Creating Title column for test_df

test_df["Title"] = test_df["Name"].str.extract(", (\w*\s*\w*)\.")

test_df["Title"] = test_df["Title"].apply(lambda x: x if x in ["Mr", "Miss", "Mrs", "Master"] else "Other")

test_df.head()
# Women and Children first policy

train_df["WCF_policy"] = train_df.apply(lambda x: 1 if (x["Sex"]=="female") | (x["Age"]<16) else 0, axis=1)

test_df["WCF_policy"] = test_df.apply(lambda x: 1 if (x["Sex"]=="female") | (x["Age"]<16) else 0, axis=1)
sns.set()



# Features to plot

plot_features = ["WCF_policy", "Sex"]



# Prepare figure

fig, axes = plt.subplots(1, len(plot_features), figsize=(17,7))



# Plot data

for ax, f in zip(axes, plot_features):

    sns.countplot(x=f, hue='Survived', data=train_df, ax=ax)



# Layout

axes[0].set_title("Survived X WCF_policy", size=15)

axes[1].set_title("Survived X Sex", size=15)

plt.subplots_adjust(wspace=0.3) # adjust width between subplots



plt.show()
# Family size

train_df["Family_size"] = train_df.loc[:,['SibSp','Parch']].sum(axis=1)

# Family aboard

train_df["Family_aboard"] = train_df["Family_size"].apply(lambda x: 1 if x!=0 else 0)
sns.set()



# Features to plot

plot_features = ["Family_size", "Family_aboard"]



# Prepare figure

fig, axes = plt.subplots(1, len(plot_features), figsize=(17,7))



# Plot data

for ax, f in zip(axes, plot_features):

    sns.countplot(x=f, hue='Survived', data=train_df, ax=ax)



# Layout

axes[0].set_title("Survived X Family Size", size=15)

axes[1].set_title("Survived X Family Aboard", size=15)

plt.subplots_adjust(wspace=0.3) # adjust width between subplots



plt.show()
# Creating Family Size and Family aboard in test set

test_df["Family_size"] = test_df.loc[:,['SibSp','Parch']].sum(axis=1)

test_df["Family_aboard"] = test_df["Family_size"].apply(lambda x: 1 if x!=0 else 0)

test_df.head()
# Dropping useless features and updating data_cleaner

train_df = train_df.drop(["Name", "Ticket", "Cabin"], axis=1)

test_df = test_df.drop(["Name", "Ticket", "Cabin"], axis=1)

data_cleaner[0] = train_df

data_cleaner[1] = test_df

train_df.head()
# Categorical Variables

cat_vars = ["Pclass", "Sex", "Embarked", "Title"]

ohe = OneHotEncoder(drop="first", sparse=False, dtype="int") # why do we drop one of the categories?



for i in data_cleaner:

    ohe.fit(i.loc[:, cat_vars])

    new_columns = list(ohe.get_feature_names())

    i[new_columns] = pd.DataFrame(ohe.transform(i[cat_vars]), index=i.index)

    i.drop(cat_vars, axis=1, inplace=True)

    

train_df.head()
# Metric variables

metr_vars = ["Age", "SibSp", "Parch", "Fare"]

std_labels = list(map(lambda x: "std_" + x, metr_vars))

std = StandardScaler()



for i in data_cleaner:

    std.fit(i.loc[:, metr_vars])

    i[std_labels] = pd.DataFrame(std.transform(i[metr_vars]), index=i.index)

    i.drop(metr_vars, axis=1, inplace=True)

    

train_df.head()
test_df.head()
# Choosing best features to include in the model

estimator = DecisionTreeClassifier(min_samples_leaf=45)

selector = RFECV(estimator, cv=mod_sel.StratifiedKFold(10), scoring="accuracy")



X = train_df.drop(["PassengerId", "Survived"], axis=1)

y = train_df["Survived"]



selector.fit(X, y)



print("Optimal number of features : %d" % selector.n_features_)



sns.set()



# Figure

fig, ax = plt.subplots(figsize= (10, 6))



# Plot number of features VS. cross-validation scores

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)



# Layout

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")



plt.show()
feature_rk = pd.Series(selector.ranking_, index=X.columns)

feature_rk
X_sel = X[["std_Fare", "WCF_policy", "std_Age", "x0_3", "x2_S", "Family_size", "NK"]]



estimator = DecisionTreeClassifier(random_state=0)

param_grid = {

    "min_samples_leaf":list(range(5, 100, 5))

}

gs = mod_sel.GridSearchCV(estimator, param_grid, "accuracy", cv=mod_sel.StratifiedKFold(10))



gs.fit(X_sel, y)

gs.best_params_, gs.best_score_
# Visualizing the Decision Tree

dot_data = export_graphviz(gs.best_estimator_, out_file=None, 

                           feature_names=X_sel.columns.to_list(),

                           class_names=["Died", "Survived"],

                           filled=True,

                           rounded=True,

                           special_characters=True)  

graphviz.Source(dot_data)
lr = LogisticRegressionCV(penalty="l1", solver="saga", cv=mod_sel.StratifiedKFold(10), scoring="accuracy", random_state=0)

lr.fit(X, y)

scores = lr.scores_[1]

print("{} iterations were performed. Begins with {:.2f} accuracy, ends with {:.2f} accuracy.".format(scores.size, scores[0][0], scores[-1][-1]))
feature_rk = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False)



sns.set()



fig, ax = plt.subplots(figsize=(10,6))



sns.barplot(x=feature_rk.values, y=feature_rk.index, palette="vlag", ax=ax)



plt.show()
X_sel = X[["WCF_policy", "E", "D", "x3_Mr", "G", "x3_Other", "x0_3", "x3_Miss", "NK", "std_Fare", "std_Age", "x2_S", "Family_size"]]



mlp = MLPClassifier(solver="sgd", learning_rate="invscaling", random_state=0)



param_grid = {

    "hidden_layer_sizes":((50,50), (75, 75)),

    "activation":("logistic", "tanh", "relu"),

    "alpha":10.0 ** -np.arange(1, 7)

}

gs = mod_sel.GridSearchCV(mlp, param_grid, "accuracy", cv=mod_sel.StratifiedKFold(10))



gs.fit(X_sel, y)

gs.best_params_, gs.best_score_
# Applying model on test_df

X_test = test_df.drop("PassengerId", axis=1)

prediction = lr.predict(X_test)



test_df["Survived"] = prediction



# Submit file

submit = test_df[['PassengerId','Survived']]

submit.to_csv("../working/submit.csv", index=False)



print('Test Data Distribution: \n', test_df['Survived'].value_counts(normalize = True))

submit.sample(10)