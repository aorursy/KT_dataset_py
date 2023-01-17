import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolours
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read the training data into a DataFrame
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.isna().any()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.isna().any()
train.describe()
# Produces two stacked histogram plots, side-by-side, separating passengers by
# survival
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)

ax1.hist((train.loc[train["Survived"] == 0, "Age"].dropna(),train.loc[train["Survived"] == 1, "Age"].dropna()),
         bins = int(train["Age"].max()-train["Age"].min()), histtype = 'barstacked',
        color=['darkred','green'], label=['Perished','Survived'], edgecolor='black')

ax1.set_xlabel('Age'); ax1.set_ylabel('Number of Occurances')
ax1.set_xlim(int(train["Age"].min()), int(train["Age"].max()))
ax1.legend(loc='upper right', frameon=False, fontsize=12)

ax2.hist([train.loc[train["Survived"]==0,"Fare"].dropna(),train.loc[train["Survived"]==1,"Fare"].dropna()], 
         bins = 50, color=['darkred','green'], histtype='barstacked',label=['Perished','Survived'], log=True,
        edgecolor='black')

ax2.set_xlabel('Fare'); ax2.set_ylabel('Number of Occurances')
ax2.set_xlim(0.0,train["Fare"].max()); ax2.set_ylim(1.0,train["Fare"].max())
ax2.legend(loc='upper right', frameon=False, fontsize=12)

plt.show()
# Organise data into survived and perished by feature
sex_survived = [sum(train.loc[(train["Sex"]=="female"),"Survived"]),
            sum(train.loc[(train["Sex"]=="male"),"Survived"])]
sex_perished = [len(train.loc[train["Sex"]=="female"]) - sex_survived[0], 
                              len(train.loc[train["Sex"]=="male"]) - sex_survived[1]]

class_survived = [sum(train.loc[(train["Pclass"]==1),"Survived"]),
                  sum(train.loc[(train["Pclass"]==2),"Survived"]),
                  sum(train.loc[(train["Pclass"]==3),"Survived"])]
class_perished = [len(train.loc[(train["Pclass"]==1)]) - class_survived[0],
                  len(train.loc[(train["Pclass"]==2)]) - class_survived[1],
                  len(train.loc[(train["Pclass"]==3)]) - class_survived[2]]

embarked_survived = [sum(train.loc[(train["Embarked"]=="S"),"Survived"]),
                    sum(train.loc[(train["Embarked"]=="Q"),"Survived"]), 
                   sum(train.loc[(train["Embarked"]=="C"),"Survived"]) ]
embarked_perished = [len(train.loc[train["Embarked"]=="S"]) - embarked_survived[0], 
                     len(train.loc[train["Embarked"]=="Q"]) - embarked_survived[1],
                     len(train.loc[train["Embarked"]=="C"]) - embarked_survived[2]]

# Plot three stacked graphs, side-by-side, which indicate survival based on
# discrete features
fig = plt.figure(figsize=(16,5))
ax1 = fig.add_subplot(131); ax2 = fig.add_subplot(132); ax3 = fig.add_subplot(133); 

# Gender survival
ax1.bar([0,1],sex_perished, color='darkred', label='Perished', edgecolor='black')
ax1.bar([0,1], sex_survived, color='green', label='Survived', bottom = sex_perished, edgecolor='black')
ax1.set_xticks(np.arange(2))
ax1.set_xticklabels(("female","male"), fontsize=12)
ax1.set_ylabel("Number of Passengers")
ax1.set_xlabel('Sex of Passenger')

# Class survival
ax2.bar([0,1,2],class_perished, color='darkred', label='Perished', edgecolor='black')
ax2.bar([0,1,2], class_survived, color='green', label='Survived', bottom = class_perished, edgecolor='black')
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(("First","Second", "Third"), fontsize=12)
ax2.set_xlabel('Class of Travel')

# Embarked survival
ax3.bar([0,1,2],embarked_perished, color='darkred', label='Perished', edgecolor='black')
ax3.bar([0,1,2], embarked_survived, color='green', label='Survived', bottom = embarked_perished, edgecolor='black')
ax3.set_xticks(np.arange(3))
ax3.set_xticklabels(("S","Q", "C"), fontsize=12)
ax3.set_xlabel('Place of Embarkment')

plt.show()
# Separate survived and perished by sex
female_survived = train.loc[(train['Sex']=='female') & (train['Survived']==1)]
female_perished = train.loc[(train['Sex']=='female') & (train['Survived']==0)]
male_survived = train.loc[(train['Sex']=='male') & (train['Survived']==1)]
male_perished = train.loc[(train['Sex']=='male') & (train['Survived']==0)]

# Bin the ages and fares. The ages are distributed fairly evenly, so we
# choose even bin sizes. Because the distribution of fares is uneven, we
# choose bins by quartiles.
train["Fare Bin"], fare_bins = pd.qcut(train["Fare"], q=5, labels = [1,2,3,4,5], retbins = True)
train["Age Bin"], age_bins = pd.cut(train["Age"], bins=8, labels = [1,2,3,4,5,6,7,8], retbins = True)

# Calculate the widths and centres of the bins
age_widths = age_bins[1:] - age_bins[:-1]
age_bins = (age_bins[:-1] + age_bins[1:])/2.

fare_widths = fare_bins[1:] - fare_bins[:-1]
fare_bins = (fare_bins[:-1] + fare_bins[1:])/2.

# Organise the histograms
survived_age_bar = train.loc[train["Survived"] == 1, "Age Bin"].dropna().value_counts().sort_index()
perished_age_bar = train.loc[train["Survived"] == 0, "Age Bin"].dropna().value_counts().sort_index()

survived_fare_bar = train.loc[train["Survived"] == 1, "Fare Bin"].dropna().value_counts().sort_index()
perished_fare_bar = train.loc[train["Survived"] == 0, "Fare Bin"].dropna().value_counts().sort_index()

# Now make a histogram scatter plot. This largely follows the matplotlib tutorial.
left, width = 0.1, 0.7
bottom, height = 0.1, 0.7
spacing = 0.005

fig = plt.figure(figsize=(8,8))

# Set up axes
ax_scatter = plt.axes([left, bottom, width, height])
ax_scatter.tick_params(direction='in', which='both', top=True, right=True, labelsize=12)
ax_xbar = plt.axes([left, bottom + height + spacing, width, 0.2])
ax_xbar.tick_params(direction='in', which='both', labelbottom=False)
ax_ybar = plt.axes([left + width + spacing, bottom, 0.2, height])
ax_ybar.tick_params(direction='in', which='both',labelleft=False)

# Plot the scatter data:
ax_scatter.semilogy(female_survived['Age'], female_survived['Fare'], marker='s',
            markerfacecolor='None', markeredgecolor='green', ls='None', markersize=7)
ax_scatter.semilogy(female_perished['Age'], female_perished['Fare'], marker='s',
          markerfacecolor='None', markeredgecolor='darkred', ls='None', markersize=7)
ax_scatter.semilogy(male_survived['Age'], male_survived['Fare'], marker='^',
          markerfacecolor='None', markeredgecolor='green', ls='None', markersize=7)
ax_scatter.semilogy(male_perished['Age'], male_perished['Fare'], marker='^',
          markerfacecolor='None', markeredgecolor='darkred', ls='None', markersize=7)

# We plot these points off the scatter axis in order to plot a legend describing the
# meaning of the marker shapes
ax_scatter.semilogy(-1,0.01, marker='^', markeredgecolor='black', markerfacecolor='None',
                    ls='None',label='Male')
ax_scatter.semilogy(-1,0.01, marker='s', markeredgecolor='black', markerfacecolor='None',
                    ls='None', label='Female')

# Set the scatter axis limits
ax_scatter.set_xlim(0.0, 81)
ax_scatter.set_ylim(1.0, 1.5*train["Fare"].max())
ax_scatter.set_xlabel('Age', fontsize=14); ax_scatter.set_ylabel('Fare', fontsize=14)
ax_scatter.legend(loc = 'lower right', fontsize=14, frameon=False)

# Plot text to describe colours
ax_scatter.text(0.8,0.95,'Survived', color='green', transform=ax_scatter.transAxes, fontsize=14)
ax_scatter.text(0.8,0.9,'Perished', color='darkred', transform=ax_scatter.transAxes, fontsize=14)

# Plot the ages histogram
ax_xbar.bar(age_bins,perished_age_bar, color='darkred', label='Perished', 
            width = age_widths, edgecolor='black')
ax_xbar.bar(age_bins,survived_age_bar, color='green', label='Survived',
            bottom = perished_age_bar, width = age_widths, edgecolor='black')
ax_xbar.set_xlim(ax_scatter.get_xlim()); ax_xbar.axis('off')

# Plot the fares histogram
ax_ybar.barh(fare_bins,perished_fare_bar,color='darkred', label='Perished', 
             edgecolor='black', align='center', height = fare_widths)
ax_ybar.barh(fare_bins,survived_fare_bar, color='green', label='Survived',
            left = perished_fare_bar, edgecolor='black', align='center',
              height = fare_widths)
ax_ybar.set_yscale('log'); 
ax_ybar.set_ylim(ax_scatter.get_ylim()); ax_ybar.axis('off')

plt.show()
# Define new family feature
train["Family"] = train["SibSp"] + train["Parch"]

fig = plt.figure(figsize=(16,4))
ax1 = fig.add_subplot(131); ax2 = fig.add_subplot(132); ax3 = fig.add_subplot(133);

# Parch survival
ax1.hist((train.loc[train["Survived"] == 0, "Parch"].dropna(),train.loc[train["Survived"] == 1, "Parch"].dropna()),
         bins = int(train["Parch"].max()-train["Parch"].min()), histtype = 'barstacked',
        color=['darkred','green'], label=['Perished','Survived'], log=True, edgecolor='black')
ax1.set_xlabel('Number of Travelling Parents/Children')
ax1.set_ylabel('Number of Occurances')

# SibSp survival
ax2.hist((train.loc[train["Survived"] == 0, "SibSp"].dropna(),train.loc[train["Survived"] == 1, "SibSp"].dropna()),
         bins = int(train["SibSp"].max()-train["SibSp"].min()), histtype = 'barstacked',
        color=['darkred','green'], label=['Perished','Survived'], log=True, edgecolor='black')
ax2.set_xlabel('Number of Travelling Siblings/Spouses')

# Family survival
ax3.hist((train.loc[train["Survived"] == 0, "Family"].dropna(),train.loc[train["Survived"] == 1, "Family"].dropna()),
         bins = int(train["Family"].max()-train["Family"].min()), histtype = 'barstacked',
        color=['darkred','green'], label=['Perished','Survived'], log=True, edgecolor='black')
ax3.set_xlabel('Number of Travelling Family Members')

plt.show()
fig = plt.figure(figsize=(16,4))
ax1 = fig.add_subplot(131); ax2 = fig.add_subplot(132); ax3 = fig.add_subplot(133);

# Family survival full
ax1.hist((train.loc[train["Survived"] == 0, "Family"].dropna(),train.loc[train["Survived"] == 1, "Family"].dropna()),
         bins = int(train["Family"].max()-train["Family"].min()), histtype = 'barstacked',
        color=['darkred','green'], label=['Perished','Survived'], edgecolor='black')
ax1.set_xlabel('Number of Travelling Family Members')
ax1.set_ylabel('Number of Occurances')

# Family survival grouping 1
train["Family Bins"] = train["Family"]

mask = train["Family"] == 0
train.loc[mask, "Family Bins"] = 0
mask = train["Family"] ==1
train.loc[mask, "Family Bins"] = 1
mask = train["Family"] == 2
train.loc[mask,"Family Bins"] = 1
mask = train["Family"] == 3
train.loc[mask,"Family Bins"] = 2
mask = train["Family"] > 3
train.loc[mask, "Family Bins"] = 3

family_bins_survived = [sum(train.loc[(train["Family Bins"]==0),"Survived"]),
                        sum(train.loc[(train["Family Bins"]==1),"Survived"]),
                        sum(train.loc[(train["Family Bins"]==2),"Survived"]),
                        sum(train.loc[(train["Family Bins"]==3),"Survived"])]

family_bins_perished = [len(train.loc[(train["Family Bins"]==0)]) - family_bins_survived[0],
                        len(train.loc[(train["Family Bins"]==1)]) - family_bins_survived[1],
                        len(train.loc[(train["Family Bins"]==2)]) - family_bins_survived[2],
                        len(train.loc[(train["Family Bins"]==3)]) - family_bins_survived[3]]

ax2.bar([0,1,2,3],family_bins_perished,
        color='darkred', label='Perished', edgecolor='black')
ax2.bar([0,1,2,3], family_bins_survived, color='green', 
        label='Survived', bottom = family_bins_perished, edgecolor='black')

ax2.set_xlabel('Size of Family ')
ax2.set_xticks(np.arange(4))
ax2.set_xticklabels(['Alone','Small','Medium','Large'])

# Family survival group 2
train["Family Bins"] = train["Family"]
        
mask = train["Family"] > 4
train.loc[mask, "Family Bins"] = 5

family_bins_survived = [sum(train.loc[(train["Family Bins"]==0),"Survived"]),
                        sum(train.loc[(train["Family Bins"]==1),"Survived"]),
                        sum(train.loc[(train["Family Bins"]==2),"Survived"]),
                        sum(train.loc[(train["Family Bins"]==3),"Survived"]),
                        sum(train.loc[(train["Family Bins"]==4),"Survived"]),
                        sum(train.loc[(train["Family Bins"]==5),"Survived"])]

family_bins_perished = [len(train.loc[(train["Family Bins"]==0)]) - family_bins_survived[0],
                        len(train.loc[(train["Family Bins"]==1)]) - family_bins_survived[1],
                        len(train.loc[(train["Family Bins"]==2)]) - family_bins_survived[2],
                        len(train.loc[(train["Family Bins"]==3)]) - family_bins_survived[3],
                        len(train.loc[(train["Family Bins"]==4)]) - family_bins_survived[4],
                        len(train.loc[(train["Family Bins"]==5)]) - family_bins_survived[5]]

ax3.bar(range(6),family_bins_perished,
        color='darkred', label='Perished', edgecolor='black')
ax3.bar(range(6), family_bins_survived, color='green', 
        label='Survived', bottom = family_bins_perished, edgecolor='black')

ax3.set_xlabel('Size of Family')
ax3.set_xticks(np.arange(6))
ax3.set_xticklabels(['0','1','2','3','4', '>4'])


plt.show()
# Collect deck information from Cabin feature. Replace missing values with M.
decks = []

for cabin in train["Cabin"]:
    if pd.isnull(cabin):
        decks.append('M')
    else:
        decks.append(cabin[0])
                
train["Deck"] = decks
print("Full Deck Information:")
print(train["Deck"].value_counts())
deck_survived = []; deck_perished = []

# Separate passengers by survival
for deck in train["Deck"].unique():
    deck_survived.append(sum(train.loc[(train['Deck']==deck),"Survived"]))
    deck_perished.append(len(train.loc[(train['Deck']==deck),"Survived"]) - deck_survived[-1])

# First plot each different category. M stands for missing data.
fig = plt.figure(figsize=(16,6))
ax1 = fig.add_subplot(111); 
ax1.bar(range(len(train["Deck"].unique())),deck_perished,
        color='darkred', label='Perished', edgecolor='black')
ax1.bar(range(len(train["Deck"].unique())),deck_survived,color='green', 
        label='Survived', bottom = deck_perished, edgecolor='black')

ax1.set_xlabel('Deck')
ax1.set_xticks(np.arange(len(train["Deck"].unique())))
ax1.set_xticklabels(train["Deck"].unique())
ax1.set_ylabel('Number of Occurances')

plt.show()
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131); ax2 = fig.add_subplot(132); ax3 = fig.add_subplot(133)

# First group by those who have cabins and those who don't

train["Has Cabin"] = train["Cabin"]
mask = train["Deck"] == 'M'
train.loc[mask,"Has Cabin"] = 0
train.loc[np.invert(mask), "Has Cabin"] = 1

deck_survived = []; deck_perished = []

for deck in range(2):
    deck_survived.append(sum(train.loc[(train['Has Cabin']==deck),"Survived"]))
    deck_perished.append(len(train.loc[(train['Has Cabin']==deck),"Survived"]) - deck_survived[-1])

ax1.bar(range(2),deck_perished,
        color='darkred', label='Perished', edgecolor='black')
ax1.bar(range(2),deck_survived,color='green', 
        label='Survived', bottom = deck_perished, edgecolor='black')

ax1.set_xticks(np.arange(2))
ax1.set_xticklabels(['No Cabin', "Cabin"])
ax1.set_ylabel('Number of Occurances')

# Next group by first, second and third class decks.
deck_mapping = {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 2, 'G': 2, 'T':0, 'M':3}
train["Deck Group 1"] = train["Deck"]
train = train.replace({"Deck Group 1":deck_mapping})

deck_survived = []; deck_perished = []

for deck in range(4):
    deck_survived.append(sum(train.loc[(train['Deck Group 1']==deck),"Survived"]))
    deck_perished.append(len(train.loc[(train['Deck Group 1']==deck),"Survived"]) - deck_survived[-1])

    
ax2.bar(range(4),deck_perished,
        color='darkred', label='Perished', edgecolor='black')
ax2.bar(range(4),deck_survived,color='green', 
        label='Survived', bottom = deck_perished, edgecolor='black')

ax2.set_xlabel('Deck Class')
ax2.set_xticks(np.arange(4))
ax2.set_xticklabels(['Upper', 'Middle', 'Lower', 'Missing'])

# Finally replace missing values with expected decks by passengers class
deck_mapping = {'A': 0, 'B': 1, 'C': 0, 'D': 1, 'E': 1, 'F': 1, 'G': 0, 'T':0, 'M':2}
train["Deck Group 2"] = train["Deck"]
train = train.replace({"Deck Group 2":deck_mapping})

deck_survived = []; deck_perished = []

for deck in range(3):
    deck_survived.append(sum(train.loc[(train['Deck Group 2']==deck),"Survived"]))
    deck_perished.append(len(train.loc[(train['Deck Group 2']==deck),"Survived"]) - deck_survived[-1])

    
ax3.bar(range(3),deck_perished,
        color='darkred', label='Perished', edgecolor='black')
ax3.bar(range(3),deck_survived,color='green', 
        label='Survived', bottom = deck_perished, edgecolor='black')

ax3.set_xlabel('Deck Class')
ax3.set_xticks(np.arange(3))
ax3.set_xticklabels(['ACGT', 'BDEF', 'M'])

plt.show()
names = train["Name"]
titles = []
        
# Split name and find title
for name in names:
    split_name = name.split(',')[1]
    titles.append(split_name.split()[0][:-1])
        
train["Title"] = titles
unique = train["Title"].unique()
        
# These are the most common titles, and therefore the ones we care most about
titles_mapping = {'Mr':'Mr', 'Mrs': 'Mrs', 'Miss':'Miss', 'Master': 'Master', 
                          'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss'}
        
# If the title is not one of the common ones, register it as rare
for title in unique:
    if not title in titles_mapping:
        titles_mapping[title] = 'Rare' 
        
# Replace titles to simplify
train = train.replace({"Title":titles_mapping})
title_survived = []; title_perished = []

# Divide passengers into survived and perished
for title in train["Title"].unique():
    title_survived.append(sum(train.loc[(train['Title']==title),"Survived"]))
    title_perished.append(len(train.loc[(train['Title']==title),"Survived"]) - title_survived[-1])

# Plot survival by title
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)

ax.bar(range(5),title_perished,
        color='darkred', label='Perished', edgecolor='black')
ax.bar(range(5),title_survived,color='green', 
        label='Survived', bottom = title_perished, edgecolor='black')

ax.set_xticks(np.arange(5))
ax.set_xticklabels(train["Title"].unique())
ax.set_xlabel('Title'); ax.set_ylabel('Number of Occurances')

plt.show()
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Create a validation set
simple_train, simple_val = train_test_split(train, test_size=0.2, random_state=42)

# Set aside labels
simple_train_labels = np.array(simple_train["Survived"])
simple_val_labels = np.array(simple_val["Survived"])


# Sort out missing values
num_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),])

# Define the full data cleaning pipeline
full_pipeline  = ColumnTransformer([("numerical",num_pipeline,["Pclass","Age","Fare"]),
                                   ("categorical",OneHotEncoder(),["Sex"]),])

# Finally, clean the data
simple_train = full_pipeline.fit_transform(simple_train)
simple_val = full_pipeline.fit_transform(simple_val)

from sklearn import tree

# Define simple tree and use training set to fit
simple_tree =  tree.DecisionTreeClassifier()
simple_tree.fit(simple_train,simple_train_labels)

# Use validation set to get an idea of how good the model is
train_score = simple_tree.score(simple_train, simple_train_labels)
val_score = simple_tree.score(simple_val,simple_val_labels)
print('% correct for training set: ', round(100.*train_score,1))
print('% correct for validation set: ', round(100.*val_score,1))
# Train a simplified model
simple_tree_depth = tree.DecisionTreeClassifier(max_depth=5)
simple_tree_depth.fit(simple_train, simple_train_labels)

# Use validation set to get an idea of how good the model is
train_score = simple_tree_depth.score(simple_train, simple_train_labels)
val_score = simple_tree_depth.score(simple_val,simple_val_labels)
print('% correct for training set: ', round(100.*train_score,1))
print('% correct for validation set: ', round(100.*val_score,1))
from sklearn.model_selection import cross_val_score

# Use cross validation to see how well it does
scores = cross_val_score(simple_tree_depth, simple_train, simple_train_labels, cv=5)
print('Mean Score (% correct): ', round(100.*scores.mean(),1))
print('Standard Deviation (% correct): ',round(100.*scores.std(),1))

from sklearn.ensemble import RandomForestClassifier

rnd_tree = RandomForestClassifier(n_estimators=20, max_depth=5)
scores = cross_val_score(rnd_tree, simple_train, simple_train_labels, cv=5)

print('Mean Score (% correct): ', round(100.*scores.mean(),1))
print('Standard Deviation (% correct): ',round(100.*scores.std(),1))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
scores = cross_val_score(knn, simple_train, simple_train_labels, cv=5)

print('Mean Score (% correct): ', round(100.*scores.mean(),1))
print('Standard Deviation (% correct): ',round(100.*scores.std(),1))
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

# First we need the predictions from each model.
# We use the validation set for this.
simple_tree_predictions = simple_tree_depth.predict(simple_val)

# These models haven't been fitted yet, so we do that first
rnd_tree.fit(simple_train, simple_train_labels)
rnd_tree_predictions = rnd_tree.predict(simple_val)

knn.fit(simple_train, simple_train_labels)
knn_predictions = knn.predict(simple_val)

# Calculate the confusion matrices 
simple_tree_confusion = confusion_matrix(simple_val_labels,simple_tree_predictions)
rnd_tree_confusion = confusion_matrix(simple_val_labels,rnd_tree_predictions)
knn_confusion = confusion_matrix(simple_val_labels,knn_predictions)

# Plotting function - tad ott but I'm a PhD students so pretty plots are
# a source of pride for me...
def plot_confusion_comparison(predictions, titles):
    
    """
    Plots a set of confusion matrices.
    """
    
    plt_labels = ["Perished", "Survived"]
    props = dict(boxstyle='round', facecolor='aliceblue', alpha=1.0, ec='aliceblue')
    
    fig = plt.figure(figsize=(len(predictions)*4,5))
    plt_dim = int(100 + 10*len(predictions))
    axes = []
    
    for iax in range(len(predictions)):
        axes.append(fig.add_subplot(plt_dim+1+iax))

    for iax,ax in enumerate(axes):

        ax.matshow(predictions[iax], cmap='Blues')
        ax.set_title(titles[iax])
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(plt_labels)
        ax.xaxis.set_ticks_position('bottom');

        if iax <1:
            ax.set_yticks(np.arange(2))
            ax.set_yticklabels(plt_labels, fontsize=12)
        else:
            ax.set_yticks([])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, round(predictions[iax][j,i],3),
                           ha="center", va="center", color="midnightblue", bbox=props, fontsize=12)

    plt.tight_layout()
    plt.show()

# Scale the matrices by the number of samples in each column and plot
predictions = [simple_tree_confusion, rnd_tree_confusion, knn_confusion]

for ipred, pred in enumerate(predictions):
    row_sum = pred.sum(axis=1, keepdims=True)
    predictions[ipred] = predictions[ipred]/row_sum
    
titles = ['Decision Tree','Random Forest','K Nearest Neighbours']

plot_confusion_comparison(predictions,titles)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Now we need the probability predictions from each model. The 'score'
# is the probability of picking survive
simple_tree_predictions = cross_val_predict(simple_tree_depth, simple_val, 
                                            simple_val_labels, cv=3,
                                           method = "predict_proba")
simple_tree_scores = simple_tree_predictions[:,1] 

# These models haven't been fitted yet, so we do that first
rnd_tree_predictions = cross_val_predict(rnd_tree, simple_val, 
                                            simple_val_labels, cv=3,
                                           method = "predict_proba")
rnd_tree_scores = rnd_tree_predictions[:,1]

knn_predictions = cross_val_predict(knn, simple_val, 
                                            simple_val_labels, cv=3,
                                           method = "predict_proba")
knn_scores = knn_predictions[:,1]

# Then we collect the true and false positive rates
tree_fpr, tree_tpr, tree_thresholds = roc_curve(simple_val_labels,simple_tree_scores)
rnd_tree_fpr, rnd_tree_tpr, rnd_thresholds = roc_curve(simple_val_labels,rnd_tree_scores)
knn_fpr, knn_tpr, knn_thresholds = roc_curve(simple_val_labels,knn_scores)

# Then we plot them, along with the random baseline
def plot_roc_comparison(predictions, labels):
    
    """
    Plots the ROC curves for a given set of predictions (max 5 reasonably),
    against the random baseline.
    """
    
    c = ['cornflowerblue','mediumvioletred','seagreen','rosybrown', 'darkorange']
    plt.figure(figsize=(6,6))
    
    for ipred, pred in enumerate(predictions):
        plt.plot(pred[0], pred[1], label=labels[ipred], color=c[ipred%len(c)])

    plt.plot([0,1],[0,1], label='Random Baseline', color='gray', ls='--')

    plt.legend(frameon=False, loc='lower right', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlim(0.,1.); plt.ylim(0.,1.)
    plt.tick_params(direction='in', which='both')
    plt.show()

predictions = [[tree_fpr, tree_tpr],[rnd_tree_fpr, rnd_tree_tpr],[knn_fpr, knn_tpr]]
plot_roc_comparison(predictions, titles)

print(f'Decision Tree AUC Score: ', roc_auc_score(simple_val_labels, simple_tree_scores))
print(f'Random Forest AUC Score: ', roc_auc_score(simple_val_labels, rnd_tree_scores))
print(f'K Nearest Neighbours AUC Score: ', roc_auc_score(simple_val_labels, knn_scores))
# Define a new class to prepare the data.

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class PrepareJourneyData(BaseEstimator, TransformerMixin):
    
    def __init__(self,):
        pass
    
    def fit(self,X):
        return self
    
    def transform(self,data):
        
        # Combine SibSp and Parch into Family
        data["Family"] = data["SibSp"] + data["Parch"]
        data["Family Bins"] = data["Family"]
        
        # Bin family data into small, medium and large families
        mask = data["Family"] < 1
        data.loc[mask, "Family Bins"] = 0
        mask = data["Family"] == 1
        data.loc[mask, "Family Bins"] = 1
        mask = data["Family"] == 2
        data.loc[mask, "Family Bins"] = 1
        mask = data["Family"] == 3
        data.loc[mask, "Family Bins"] = 1
        mask = data["Family"] > 3
        data.loc[mask,"Family Bins"] = 2
        
        # Replace any missing values in Embarked with the most common place of embarkment
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode(dropna=True)[0])
        embarked_mapping = {'S':0, 'C':1,'Q':2}
        data = data.replace({"Embarked":embarked_mapping})
    
        # Obtain deck information from cabin. Replace missing cabins with M
        decks = []

        for cabin in data["Cabin"]:
  
            if pd.isnull(cabin):
                decks.append('M')
            else:
                decks.append(cabin[0])
        
       
        # Group decks into numerical values
        data["Deck"] = decks
        deck_mapping = {'A': 0, 'B':0, 'C':0, 'D': 1,'E':1, 'F':2, 'G':2, 'T':0, 'M':3}
        data = data.replace({"Deck":deck_mapping})
        data = pd.get_dummies(data, columns = ["Deck"])
        
        # Replace missing fares with median from class of travel
        classes = data["Pclass"].unique()
        for pclass in classes:
            mask = data["Pclass"] == pclass
            data.loc[mask, "Fare"] = data.loc[mask,"Fare"].fillna(data.loc[mask,"Fare"].median())
            data.loc[mask & (data["Fare"]<1),"Fare"] = data.loc[mask,"Fare"].median()
        
        # Finally, bin the fare data by quartile
        data["Fare Bin"], bins = pd.qcut(data["Fare"], q=8, labels = [0,1,2,3,4,5,6,7], retbins = True)
        
        # Drop features no longer needed. Only certain decks had any effect on the model,
        # so I dropped other deck information
        data = data.drop(["SibSp","Parch","Fare", "Cabin", "Family", "Deck_2"], axis=1)
        
        return data


class PreparePersonalData(BaseEstimator, TransformerMixin):
    
    def __init__(self,):
        pass
    
    def fit(self, X):
        return self
    
    def transform(self, data):
        
        # Convert Sex to numerical value
        encoder = OrdinalEncoder()
        data["Sex"] = encoder.fit_transform(data[["Sex"]])
        
        names = data["Name"]
        titles = []
        
        # This will split the title from the name. It works in all
        # but one instance (the countess), however we'll deal with 
        # that in a minute
        for name in names:
            split_name = name.split(',')[1]
            titles.append(split_name.split()[0][:-1])
        
        data["Title"] = titles
        unique = data["Title"].unique()
        
        # These are the most common titles, and therefore the ones we care most about
        titles_mapping = {'Mr':'Mr', 'Mrs': 'Mrs', 'Miss':'Miss', 'Master': 'Master', 
                          'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss'}
        
        # If the title is not one of the common ones, register it as rare
        for title in unique:
            if not title in titles_mapping:
                titles_mapping[title] = 'Rare'     

        # Replace titles to simplify
        data = data.replace({"Title":titles_mapping})
        
        # Passengers with missing ages can then have it assigned by their title, class and whether
        # they were travelling with parent/children
        unique_titles = data["Title"].unique()
        unique_classes = data["Pclass"].unique()
        
        for title in unique_titles:

            for pclass in unique_classes:
                
                length_a = len(data.loc[(data["Title"]==title) & (data["Pclass"]==pclass) & (data["Parch"]>0)])
                length_b = len(data.loc[(data["Title"]==title) & (data["Pclass"]==pclass) & (data["Parch"]==0)])
                
                if (length_a > 2 and length_b >2):
                    mean_age = data.loc[(data["Title"]==title) & (data["Pclass"]==pclass) & (data["Parch"]>0),"Age"].mean()
                    data.loc[(data["Title"]==title) & (data["Pclass"]==pclass) & (data["Parch"]>0),"Age"] = \
                            data.loc[(data["Title"]==title) & (data["Pclass"]==pclass) & (data["Parch"]>0),"Age"].fillna(mean_age)

                    mean_age = data.loc[(data["Title"]==title) & (data["Pclass"]==pclass) & (data["Parch"]==0),"Age"].mean()
                    data.loc[(data["Title"]==title) & (data["Pclass"]==pclass) & (data["Parch"]==0),"Age"] = \
                                data.loc[(data["Title"]==title) & (data["Pclass"]==pclass) & (data["Parch"]==0),"Age"].fillna(mean_age)
                else:
                    mean_age = data.loc[(data["Title"]==title) & (data["Pclass"]==pclass),"Age"].mean()
                    data.loc[(data["Title"]==title) & (data["Pclass"]==pclass),"Age"] = \
                            data.loc[(data["Title"]==title) & (data["Pclass"]==pclass),"Age"].fillna(mean_age)


        # Now bin ages
        data["Age Bin"] = pd.cut(data["Age"], bins=7, labels = [0,1,2,3,4,5,6])

        # Now encode titles
        titles_mapping = {'Mr': 1, "Mrs": 2, "Master": 3, "Miss": 4, "Rare": 5}
        data = data.replace({"Title":titles_mapping})
        data = pd.get_dummies(data, columns = ["Title"])
        
        # Finally drop unused features
        data = data.drop(["Age", "Name","Parch","Pclass"], axis = 1)

        return data

# Read in clean training dataset
train = pd.read_csv('/kaggle/input/titanic/train.csv')

# Define the full data cleaning pipeline
full_pipeline  = ColumnTransformer([("family",PrepareJourneyData(),["Pclass","SibSp","Parch", "Fare","Cabin","Embarked"]),
                                   ("personal", PreparePersonalData(),["Sex","Name","Age", "Pclass", "Parch"]),
                                   ])


# Set aside labels
train_labels = np.array(train["Survived"])

# Finally, clean the data
full_train = full_pipeline.fit_transform(train)

rnd_forest = RandomForestClassifier(random_state=42)
scores = cross_val_score(rnd_forest, full_train, train_labels, cv=5)

print('Mean Score (% correct): ', round(100.*scores.mean(),1))
print('Standard Deviation (% correct): ',round(100.*scores.std(),1))
from sklearn.model_selection import GridSearchCV

param_grid = [{'max_depth': [8,12,None], 'max_features':[8,12,'auto'], 
               'n_estimators':[100,200,400], 'bootstrap': [True,False],
              'min_samples_split': [2,10,50],'min_samples_leaf': [2,6,12]}]

grid_search = GridSearchCV(rnd_forest, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(full_train, train_labels)

curves = grid_search.cv_results_
print(f'Highest Score: ', round(100.*max(curves["mean_test_score"]),1), '%')
print(f'Corresponding Parameters: ', curves["params"][np.argmax(curves["mean_test_score"])])
# Model with best combination of parameters
rnd_forest = RandomForestClassifier(max_depth = 12, max_features=8,  n_estimators = 100, min_samples_leaf = 2, 
                                   min_samples_split=2, bootstrap = True, oob_score = True,random_state=42)

predictions = cross_val_predict(rnd_forest, full_train, 
                                            train_labels, cv=5,
                                           method = "predict_proba")
scores = predictions[:,1]
fpr, tpr, rnd_thresholds = roc_curve(train_labels, scores)

plot_roc_comparison([[fpr,tpr]], ['Random Forest'])
roc_auc_score(train_labels, scores)
predictions = cross_val_predict(rnd_forest, full_train, 
                                            train_labels, cv=3)
rnd_tree_confusion = confusion_matrix(train_labels,predictions)
row_sum = rnd_tree_confusion.sum(axis=1, keepdims=True)
predictions = rnd_tree_confusion/row_sum
plot_confusion_comparison([predictions], ['Random Forest'])
rnd_forest.fit(full_train, train_labels)

features = ["Class", "Family_Bin", "Embarked","ABCT","DE","M", "Fare Bin",
            "Sex","Age Bin",'Mr',"Mrs","Master","Miss","Rare"]

for feature, importance in zip(features, rnd_forest.feature_importances_):
    print(feature,'\t', round(100.*importance,2),'%')
    
print(f'OOB Score: {rnd_forest.oob_score_}')
# Predict the survival and output to a submission file
test_data = full_pipeline.fit_transform(test)

predictions =  rnd_forest.predict(test_data)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('titanic_submission.csv', index=False)
output.head()