import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.rcParams.update({'font.size': 12})
maths = pd.read_csv("../input/student-mat.csv")
portugese = pd.read_csv("../input/student-por.csv")
set(maths.columns).symmetric_difference(portugese.columns)
#label the data before concatenation
maths["subject"] = "maths"
portugese["subject"] = "portugese"

#record shape before concatenation
print("Shape of 'maths' dataset: ", maths.shape)
print("Shape of 'portugese' dataset: ", portugese.shape)

#concatenate and record shape
data = pd.concat([maths, portugese])
print("Shape of total dataset: ", data.shape)
maths = data[data["subject"]=="maths"]
print("The Math Class dataset has", maths.shape[0], "rows with ", maths.shape[1], " columns") 
print("number of rows with missing values in Maths Class dataset: ", maths[maths.isnull().any(axis=1)].shape[0], "\n")

portugese = data[data["subject"]=="portugese"]
print("The Portugese Class dataset has", portugese.shape[0], "rows with ", portugese.shape[1], " columns") 
print("number of rows with missing values in Portugese Class dataset: ", portugese[portugese.isnull().any(axis=1)].shape[0])
%reset_selective -f "^maths$"
%reset_selective -f "^portugese$"
data.isnull().any().any()
describe = data.describe(include='all')
describe = describe.transpose()
binary_variables = describe.loc[describe["unique"] == 2, ].index

binary_values = [(var, data[var].unique()) for var in  binary_variables]

#pulling aside famsize variable for special treatment
#famsize_before_binary = data["famsize"]#.copy(deep=True)
#binary_values[3] = ("famsize", ["LE3", "GT3"]) #switching this around so LE3 gets codes as 0, GT3 as 1
#binary_values[3]

binary_values = [
    {"school" : ["GP", "MS"]},
    {"sex" : ["F", "M"]},
    {"subject" : ["maths", "portugese"]},
    {"address" : ["U", "R"]},
    {"famsize" : ["LE3", "GT3"]},
    {"Pstatus" : ["A", "T"]}, #"cohabitation status" - so 1 is "True - cohabitating", 0 is "False - not cohabitating"
    {"schoolsup" : ["no", "yes"]},
    {"famsup" : ["no", "yes"]},
    {"paid" : ["no", "yes"]},
    {"activities" : ["no", "yes"]},
    {"internet" : ["no", "yes"]},
    {"nursery" : ["no", "yes"]},
    {"higher" : ["no", "yes"]},
    {"romantic" : ["no", "yes"]}
]

print("There are ", len(binary_variables), " binary variables")
print("There are ", len(binary_values), " binary codes specified")

#binary code variables with only 2 unique values
#[col for spec in binary_values for col, codes in spec.items()]
for spec in binary_values:
    for col, codes in spec.items():
        
        data.loc[data[col] == codes[0], col] = 0
        data.loc[data[col] == codes[1], col] = 1
        zeroes = data.loc[data[col] == 0, col].shape[0]
        ones = data.loc[data[col] == 1, col].shape[0]
        
        print("\nVariables for ", col, " encoded to 0: ", str(zeroes))
        print("Variables for ", col, " encoded to 1: ", str(ones))

        data[col] = pd.to_numeric(data[col])
        
categorical_variables = describe.loc[describe["unique"] > 2, ].index
categorical_variables
#categorical_values = [(var, data[var].unique()) for var in  categorical_variables]
#categorical_values
data = pd.get_dummies(data, columns=categorical_variables)
data.columns
data.loc[(data["Walc"].isin([4,5]) & data["Dalc"].isin([1,2])), "binge_drinker"] = 1
data.loc[(data["Walc"].isin([4,5]) & data["Dalc"].isin([4,5])), "heavy_drinker"] = 1
data[["binge_drinker", "heavy_drinker"]] = data[["binge_drinker", "heavy_drinker"]].fillna(0)

fig, ax = plt.subplots()
sns.countplot("binge_drinker", hue="heavy_drinker", data=data, ax=ax)

l = ax.legend()
l.set_title("")
new_legend_labels = ['light-moderate drinker', 'heavy drinker']
for t, l in zip(l.texts, new_legend_labels): t.set_text(l)

ax.set_xlabel("")
ax.set_xticklabels(['consistent drinking behaviour', 'binge drinking'])
fig.suptitle("Distribution of extreme drinking behaviour");

#for t, l in zip(l.texts, new_legend_labels): t.set_text(l)
data["overall_grade"] = (data["G1"] + data["G2"] + (data["G3"]*2)) / 4
data["overall_grade"] = pd.to_numeric(data["overall_grade"])
from matplotlib.lines import Line2D

cmap = plt.cm.coolwarm
legend = [("G3", 0.25), ("G2", 0.5), ("G3", 0.75), ("overall_grade", 1.)]

fig, ax = plt.subplots()

def set_colour_and_plot_kde(label_info):
    
    sns.distplot(data[label_info[0]], hist=False, ax=ax, color=cmap(label_info[1]))
    return Line2D([0], [0], color=cmap(label_info[1]), lw=2)

custom_lines = [set_colour_and_plot_kde(item) for item in legend]
custom_labels = [item[0] for item in legend]
ax.legend(custom_lines, custom_labels)
fig.suptitle("distribution of academic measures");
data.isnull().any().any()
cols = data.columns.tolist()
cols = cols[:30] + cols [-3:]
corr_data = data[cols] 

f, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 25))

#calculates correlation matrix for a given method, plots a heatmap and then returns the correlation matrix
def corr_matrix(dataset, i, method):

    corr = dataset.corr(method=method)
    row = {"pearson" : 0, "spearman" : 1}

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axes[row[method], i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 16)
    ax.title.set_fontsize(20)
    return {(dataset.loc[0, "subject"], method) : corr}

maths = corr_data[corr_data["subject"] == 0]
portugese = corr_data[corr_data["subject"] == 1]

pearson_corr = [corr_matrix(data, i, "pearson") for i, data in enumerate([maths, portugese])] 
spearman_corr = [corr_matrix(data, i, "spearman") for i, data in enumerate([maths, portugese])]

axes[0,0].set_title("Maths Class (Pearson Correlation)", size=18)
axes[0,1].set_title("Portugese Class (Pearson Correlation)", size=18)
axes[1,0].set_title("Maths Class (Spearman Correlation)", size=18)
axes[1,1].set_title("Portugese Class (Spearman Correlation)", size=18)
plt.tight_layout()
subset = corr_data[["overall_grade", "absences", "failures", "Medu", "Fedu", "nursery","higher", "sex", "age", "studytime", "freetime",
                       "health", "Dalc", "Walc", "subject"]]

f, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 25))

#calculates correlation matrix for a given method, plots a heatmap and then returns the correlation matrix
def corr_matrix(dataset, i, method):

    corr = dataset.corr(method=method).round(1)
    row = {"pearson" : 0, "spearman" : 1}

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axes[row[method], i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 16)
    ax.title.set_fontsize(20)
    return {(dataset.loc[0, "subject"], method) : corr}

maths = subset[subset["subject"] == 0]
portugese = subset[subset["subject"] == 1]

pearson_corr = [corr_matrix(data, i, "pearson") for i, data in enumerate([maths, portugese])] 
spearman_corr = [corr_matrix(data, i, "spearman") for i, data in enumerate([maths, portugese])]

axes[0,0].set_title("Maths Class (Pearson Correlation)", size=18)
axes[0,1].set_title("Portugese Class (Pearson Correlation)", size=18)
axes[1,0].set_title("Maths Class (Spearman Correlation)", size=18)
axes[1,1].set_title("Portugese Class (Spearman Correlation)", size=18)
plt.tight_layout()

#stash all the created matrices for the comparison in the next section
corrs = [item for sublist in [pearson_corr, spearman_corr] for item in sublist]

data.to_csv('student_math_por_formatted.csv')