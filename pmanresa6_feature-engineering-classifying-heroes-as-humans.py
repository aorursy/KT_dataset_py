import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import cm
import seaborn as sns
import tqdm
import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
sns.set(rc={"figure.figsize": (10, 12)})
np.random.seed(sum(map(ord, "palettes")))
metadata = pd.read_csv("../input/heroes_information.csv", index_col=0)
powers = pd.read_csv("../input/super_hero_powers.csv")
print("Heroes information data shape: ", metadata.shape)
print("Hero super powers data shape: ", powers.shape)
metadata.head()
powers.head()
def clean_repeated_heroes(metadata, powers):
    
    print("Initial shape of metadata and powers: ")
    print("Powers:", powers.shape)
    print("Metadata", metadata.shape)
    
    print("\nStart cleaning...")
    
    powers.drop_duplicates(inplace=True)
    metadata.drop_duplicates(inplace=True)
    
    # Handle Goliath
    goliath_idxs_to_drop = [100, 289, 290] # not dropping Goliath IV, it will be used to join powers
    metadata.drop(goliath_idxs_to_drop, inplace=True)
    metadata.loc[metadata.name == "Goliath IV", "Race"] = "Human"
    
    # Avoid outersected entries. i.e. appearing in metadata, but not in powers. And viceversa.
    metadata = metadata[metadata.name.isin(powers.hero_names)]
    powers = powers[powers.hero_names.isin(metadata.name)]
    
    # Spider-Man
    metadata.loc[metadata.name.str.contains("Spider-Man")] = metadata[metadata.name.str.contains("Spider-Man")].mode().values[0]
    metadata.drop(623, inplace=True)
    metadata.drop(624, inplace=True)

    # Nova
    metadata.drop(497, inplace=True)

    # Angel
    metadata.loc[metadata.name == "Angel", "Race"] = "Vampire"
    metadata.drop(23, inplace=True)

    # Blizzard
    metadata.loc[metadata.name == "Blizzard"] = metadata.loc[metadata.name == "Blizzard II"].values
    metadata.at[115, 'name'] = "Blizzard"
    metadata.at[116, 'Race'] = "Human"
    metadata.at[115, 'Race'] = "Human"
    metadata.drop(117, inplace=True)

    # Black Canary
    metadata.drop(97, inplace=True)

    # Captain Marvel
    metadata.at[156, 'Race'] = "Human"
    metadata.drop(155, inplace=True)

    # Blue Beettle
    metadata.at[122, 'Race'] = "Human"
    metadata.at[124, 'Race'] = "Human"
    metadata.at[122, 'Height'] = 183.0
    metadata.at[125, 'Height'] = 183.0
    metadata.at[122, 'Weight'] = 86.0
    metadata.at[125, 'Weight'] = 86.0
    metadata.drop(123, inplace=True)

    # Vindicator
    metadata.drop(696, inplace=True)

    # Atlas
    metadata.drop(48, inplace=True)

    # Speedy
    metadata.drop(617, inplace=True)

    # Firestorm
    metadata.drop(260, inplace=True)

    # Atom
    metadata.drop(50, inplace=True)
    metadata.at[49, 'Race'] = "Human"
    metadata.at[53, 'Race'] = "Human"
    metadata.at[54, 'Race'] = "Human"
    metadata.at[49, 'Race'] = "Human"
    metadata.at[54, 'Height'] = 183.0
    metadata.at[49, 'Height'] = 183.0
    metadata.at[53, "Weight"] = 72.0

    # Batman
    metadata.drop(69, inplace=True)

    # Toxin
    metadata.drop(673, inplace=True)

    # Namor
    metadata.drop(481, inplace=True)

    # Batgirl
    metadata.drop(62, inplace=True)
    
    print("Final shape of metadata and powers: ")
    print("Powers:", powers.shape)
    print("Metadata", metadata.shape)
    
    print("\nCleaning done")
    
    return metadata, powers
# if you run it twice, it won't work due to hard-coded indexers won't match.
# you need to get the data and run it again

# metadata, powers = clean_repeated_heroes(metadata, powers)
powers.drop_duplicates(inplace=True)
metadata.drop_duplicates(inplace=True)
print("Number of rows with more than 1 entry per hero name in metadata ", (metadata.name.value_counts() > 1).sum()  )
print("Number of rows with more than 1 entry per hero name in powers ", (powers.hero_names.value_counts() > 1).sum() )
mask = metadata.name.value_counts() > 1
metadata.name.value_counts()[mask].sum() - mask.sum() # get excessive number of rows from repeated names
# Does it match with difference in table length?
metadata.shape[0] - powers.shape[0]
repeated_heroes = mask.index[mask]
repeated_heroes[~repeated_heroes.isin(powers.hero_names)]
powers[powers.hero_names.str.contains("Goliath")]
metadata[metadata.name.str.contains("Goliath")]
goliath_idxs_to_drop = [100, 289, 290] # not dropping Goliath IV, it will be used to join powers
metadata.drop(goliath_idxs_to_drop, inplace=True)

# modify Goliath IV row
metadata.loc[metadata.name == "Goliath IV", "Race"] = "Human"
metadata[metadata.name.str.contains("Goliath")]
# How many superheroes that appear in metadata, do not have an entry in powers?
(~metadata.name.isin(powers.hero_names)).sum()
# How many superheroes that appear in powers, do not have an entry in metadata?
(~powers.hero_names.isin(metadata.name)).sum()
metadata = metadata[metadata.name.isin(powers.hero_names)]
powers = powers[powers.hero_names.isin(metadata.name)]
metadata.shape
powers.shape
(metadata.name.value_counts() > 1).sum() 
repeated_heroes = repeated_heroes.drop("Goliath")

# let's go 1 by 1
for rh in repeated_heroes:
    print("*********** ", rh, " **************")
    print(metadata[metadata.name.str.contains(rh)])
    print("\n\n")
# Spider-Man
# As all three instances seem similar, let's just take the mode of each column as new values
metadata.loc[metadata.name.str.contains("Spider-Man")] = metadata[metadata.name.str.contains("Spider-Man")].mode().values[0]
metadata.drop(623, inplace=True)
metadata.drop(624, inplace=True)

# Nova
# They are different superheroes, but with the same name. It is not clear which one is represented in powers df
# We cannot keep both as both would have same superpowers and thus confuse the classifier. For simplicity,
# let's only keep the human Nova. And to choose between male or female, let's keep the female.
metadata.drop(497, inplace=True)

# Angel
# There are 2 Angels. Both rows seemed to have been split from one single row. Let's merge it back.
metadata.loc[metadata.name == "Angel", "Race"] = "Vampire"
metadata.drop(23, inplace=True)

# Blizzard
# Blizzard II has only 1 difference in superpower, therefore we claim both Blizzard and Blizzard II will have same 
# characteristics. And it's considered Human, according to Wikipedia.
metadata.loc[metadata.name == "Blizzard"] = metadata.loc[metadata.name == "Blizzard II"].values
metadata.at[115, 'name'] = "Blizzard"
metadata.at[116, 'Race'] = "Human"
metadata.at[115, 'Race'] = "Human"
metadata.drop(117, inplace=True)

# Black Canary
# Let's take only one of them, as they're practically similar
metadata.drop(97, inplace=True)

# Captain Marvel
# All are same, in exception of Captain Marvel II, which is showed in powers df. Let's keep CM and CM II, but in case 
# of Captain Marvel we will keep the original one by DC (the one from Marvel is copied)
metadata.at[156, 'Race'] = "Human"
metadata.drop(155, inplace=True)

# Blue Beettle
# In powers df, there are the three blue beetles, and they are indeed different in terms of powers.
# Let's keep all of them but they will all have the same characteristics as they are really similar.
metadata.at[122, 'Race'] = "Human"
metadata.at[124, 'Race'] = "Human"
metadata.at[122, 'Height'] = 183.0
metadata.at[125, 'Height'] = 183.0
metadata.at[122, 'Weight'] = 86.0
metadata.at[125, 'Weight'] = 86.0
metadata.drop(123, inplace=True)

# Vindicator
# keep only the one that does not have null values
metadata.drop(696, inplace=True)

# Atlas
# Keep only one of them
metadata.drop(48, inplace=True)

# Speedy
# Searched in Google, they are mainly the same, but male version introduced 1941 and female on 2001. Let's 
# just keep the female as it has more characteristics
metadata.drop(617, inplace=True)

# Firestorm
# keep the one that doesn't have null values
metadata.drop(260, inplace=True)

# Atom
# All atoms shown there are covered in powers df. Let's drop the row that has Atom and few null values. And add Human
# as race, plus other characteristics (all will have similar ones)
metadata.drop(50, inplace=True)
metadata.at[49, 'Race'] = "Human"
metadata.at[53, 'Race'] = "Human"
metadata.at[54, 'Race'] = "Human"
metadata.at[49, 'Race'] = "Human"
metadata.at[54, 'Height'] = 183.0
metadata.at[49, 'Height'] = 183.0
metadata.at[53, "Weight"] = 72.0

# Batman
# let's only drop the short and skinny Batman. Because he is just not.
metadata.drop(69, inplace=True)

# Toxin
# let's just keep one of them, as the second, for example
metadata.drop(673, inplace=True)

# Namor
# keep the one without null values
metadata.drop(481, inplace=True)

# Batgirl
# drop the one with null values
metadata.drop(62, inplace=True)
metadata.shape, powers.shape
metadata = metadata.replace('-', np.nan) 
metadata = metadata.replace(-99, np.nan)

metadata.isnull().sum()
metadata.dropna(subset=['Race'], inplace=True)
metadata.isnull().sum() 
# drop Skin color because it has too many null values
metadata.drop("Skin color", axis=1, inplace=True)
# transform Human- race into Human (as they are not mutations)
metadata.loc[:, "Race"] = metadata.apply(lambda x: "Human" if(x.Race.startswith("Human-")) else x.Race, axis=1)
# add label for modeling
metadata['label'] = metadata.apply(lambda x: "No-Human" if(x.Race != "Human") else x.Race, axis=1)
fig, ax = pyplot.subplots(figsize=(14,8))
sns.boxplot(x="Weight", y="label", hue="Gender", data=metadata, ax=ax)

fig, ax = pyplot.subplots(figsize=(14,8))
sns.boxplot(x="Height", y="label",  hue="Gender",data=metadata, ax=ax)
fig, ax = pyplot.subplots(figsize=(12,14))
sns.boxplot(x="Height", y="Race", data=metadata[metadata.Gender == 'Male'])
fig, ax = pyplot.subplots(figsize=(12,14))
sns.boxplot(x="Weight", y="Race", data=metadata[metadata.Gender == 'Male'])
# height and weight can be replaced by the mean of the same gender and race

w_means = metadata.groupby(["label", "Gender"])["Weight"].mean().unstack()
h_means = metadata.groupby(["label", "Gender"])["Height"].mean().unstack()

w_fh = w_means.loc["Human","Female"]
w_mh = w_means.loc["Human","Male"]
w_fn = w_means.loc["No-Human","Female"]
w_mn = w_means.loc["No-Human","Male"]

h_fh = h_means.loc["Human","Female"]
h_mh = h_means.loc["Human","Male"]
h_fn = h_means.loc["No-Human","Female"]
h_mn = h_means.loc["No-Human","Male"]

# Fill null values with means
metadata.loc[(metadata.label == "Human") & (metadata.Gender == "Female") & (metadata.Weight.isnull()), "Weight"] = w_fh
metadata.loc[(metadata.label == "Human") & (metadata.Gender == "Male") & (metadata.Weight.isnull()), "Weight"] = w_mh
metadata.loc[(metadata.label == "No-Human") & (metadata.Gender == "Female") & (metadata.Weight.isnull()), "Weight"] = w_fn
metadata.loc[(metadata.label == "No-Human") & (metadata.Gender == "Male") & (metadata.Weight.isnull()), "Weight"] = w_mn

metadata.loc[(metadata.label == "Human") & (metadata.Gender == "Female") & (metadata.Height.isnull()), "Height"] = h_fh
metadata.loc[(metadata.label == "Human") & (metadata.Gender == "Male") & (metadata.Height.isnull()), "Height"] = h_mh
metadata.loc[(metadata.label == "No-Human") & (metadata.Gender == "Female") & (metadata.Height.isnull()), "Height"] = h_fn
metadata.loc[(metadata.label == "No-Human") & (metadata.Gender == "Male") & (metadata.Height.isnull()), "Height"] = h_mn

# plot to see clearer differences
fig, (ax1,ax2) = pyplot.subplots(1,2, figsize=(12,6))
ax1.set_title("Weight")
ax2.set_title("Height")
w_means.plot(kind="bar", ax=ax1)
h_means.plot(kind="bar", ax=ax2)

metadata.isnull().sum()
metadata[metadata.Gender.isnull()]
metadata.drop(metadata[metadata.Gender.isnull()].index, axis=0, inplace=True)
metadata.isnull().sum()
sns.pairplot(x_vars=["Height"], y_vars=["Weight"], data=metadata, hue="label", height=10)
metadata[(metadata.Height > 400)]
metadata = metadata[(metadata.Weight < 450)]
for col in metadata.columns:
    if col in ("Eye color", "Hair color"):
        fig, ax = pyplot.subplots(figsize=(12,10))
        values = metadata.groupby([col, "label"]).count()['name'].unstack().sort_values(by="Human", ascending=False)
        values.plot(kind='barh', stacked=True, ax=ax)
        plt.show()
aux_eyes_colors = ["blue", "brown", "green"]
aux_hair_colors = ["Black", "Brown", "Blond", "Red", "No Hair"]
len_aux_eyes_colors = len(aux_eyes_colors)
len_aux_hair_colors = len(aux_hair_colors)

for ix in metadata[metadata["Eye color"].isnull()].index:
    metadata.at[ix, "Eye color"] = aux_eyes_colors[np.random.choice(len_aux_eyes_colors)]
    
for ix in metadata[metadata["Hair color"].isnull()].index:
    metadata.at[ix, "Hair color"] = aux_hair_colors[np.random.choice(len_aux_hair_colors)]

for col in metadata.columns:
    if col in ("Eye color", "Hair color"):
        fig, ax = pyplot.subplots(figsize=(12,10))
        values = metadata.groupby([col, "label"]).count()['name'].unstack().sort_values(by="Human", ascending=False)
        values.plot(kind='barh', stacked=True, ax=ax)
        plt.show()
metadata.isnull().sum()
metadata.drop(metadata[metadata.Publisher.isnull()].index, axis=0, inplace=True)
metadata.drop(metadata[metadata.Alignment.isnull()].index, axis=0, inplace=True)
metadata.isnull().sum()
metadata = metadata.drop(['Race'], axis=1)

high_card = ["Gender", "Alignment"]
low_card = ["Eye color", "Hair color", "Publisher"]

for hc in high_card:
    one_hot = pd.get_dummies(metadata[hc])
    metadata.drop(hc, axis=1, inplace=True)
    metadata = metadata.join(one_hot)

for lc in low_card:
    metadata[lc] = metadata[lc].astype('category').cat.codes

# transform label into 0 (Human) or 1 (No-Human)
metadata['label'] = metadata['label'].astype('category').cat.codes
# transform powers data into 0,1 binary features
cols = powers.select_dtypes(['bool']).columns
for col in cols:
    powers[col] = powers[col].astype(int)
metadata.head()
powers.head()
heroes = pd.merge(metadata, powers, how='inner', left_on = 'name', right_on = 'hero_names')

heroes.drop(["hero_names","name"], axis=1, inplace=True)

powers_cols = powers.columns.drop("hero_names")
metadata_cols = metadata.columns.drop("name")
heroes.shape
heroes.head()
# store dataframe
metadata.to_pickle("metadata.pkl")
powers.to_pickle("powers.pkl")
heroes.to_pickle("heroes.pkl")
# load back again
metadata = pd.read_pickle("metadata.pkl")
powers = pd.read_pickle("powers.pkl")
heroes = pd.read_pickle("heroes.pkl")
from sklearn import preprocessing

X = heroes.drop(["label"], axis=1).values
y = heroes["label"].values

X = preprocessing.scale(X)

print( "X - training data shape ", X.shape)
print( "y - label ", y.shape )
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
# Initialize a stratified split of our dataset for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = ["LogReg", "SVM", "RF", "XGB"]

for model in models:
    print( "Training ", model)
    if model == "LogReg":
        clf = LogisticRegression(random_state=0, solver='liblinear')
    elif model == "SVM":
        clf = svm.SVC(kernel='linear',C=1)
    elif model == 'RF':
        clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=1)
    else:
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        clf = XGBClassifier(n_estimators=50, max_depth=6)
    
    results = cross_val_score(clf, X, y, cv=5).mean()
    print( model, " CV accuracy score: {:.2f}%".format(results.mean()*100) )

# Random Forest optimization
clf_rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=1)

rf_params = {'max_features': [4, 7, 10, 13], 
             'min_samples_leaf': [1, 3, 5, 7], 
             'max_depth': [5,8,10,15], 
             "n_estimators": [50, 100] }

gcv = GridSearchCV(clf_rf, rf_params, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X, y)
gcv.best_estimator_, gcv.best_score_
# XGBoost optimization
clf_xgb = XGBClassifier(n_estimators=50, max_depth=8, random_state=1)

xgb_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 8, 10],
        'n_estimators' : [50,100]
        }

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
gcv_xgb = GridSearchCV(clf_xgb, xgb_params, n_jobs=-1, cv=skf, verbose=1)
gcv_xgb.fit(X, y)
gcv_xgb.best_estimator_, gcv_xgb.best_score_
importances = gcv_xgb.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importancies
features = dict()
count = 1
for col in heroes.drop("label",axis=1).columns:
    index = "f"+str(count)
    features[index] = col
    count+=1

num_to_plot = 20
feature_indices = [ind+1 for ind in indices[:num_to_plot]]
top_features = list()
# Print the feature ranking
print("Feature ranking:")
  
for f in range(num_to_plot):
    print("%d. %s %f " % (f + 1, 
            features["f"+str(feature_indices[f])], 
            importances[indices[f]]))
    top_features.append(features["f"+str(feature_indices[f])])
plt.figure(figsize=(15,5))
plt.title(u"Feature Importance")
bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
       color=([str(i/float(num_to_plot+1)) 
               for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f"+str(i)]) 
                  for i in feature_indices]);
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, n_iter=1000)
tsne_results = tsne.fit_transform(X)

df_tsne = pd.DataFrame(data=tsne_results, columns=["tsne1", "tsne2"])
df_tsne['label'] = y

sns.pairplot(x_vars=["tsne1"], y_vars=["tsne2"], data=df_tsne, hue="label", height=10)
from sklearn.decomposition import PCA
models = ["LogReg", "SVM", "RF", "XGB"]
reductions = [20, 30, 40, 50, 70, 100, 150]

for red in reductions:
    
    print( "Applying PCA on ", red, " components")
    pca = PCA(n_components=red)
    X_reduced = pca.fit_transform(X)

    for model in models:
        print( "Training ", model )
        if model == "LogReg":
            clf = LogisticRegression(random_state=0, solver='liblinear')
        elif model == "SVM":
            clf = svm.SVC(kernel='linear',C=1)
        elif model == 'RF':
            clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=1)
        else:
            clf = XGBClassifier(n_estimators=50, max_depth=6)

        results = cross_val_score(clf, X_reduced, y, cv=5).mean()
        print( model, " CV accuracy score: {:.2f}%".format(results.mean()*100) )
        
    print( "\n\n" )
# let's try to train xgboost with 50 PCA component

pca_50 = PCA(n_components=50)
X_reduced_p50 = pca_50.fit_transform(X)

# XGBoost optimization
clf_xgb = XGBClassifier(n_estimators=50, max_depth=8, random_state=1)

xgb_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 8, 10],
        'n_estimators' : [50]
        }

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
gcv_xgb = GridSearchCV(clf_xgb, xgb_params, n_jobs=-1, cv=skf, verbose=1)
gcv_xgb.fit(X_reduced_p50, y)
gcv_xgb.best_estimator_, gcv_xgb.best_score_