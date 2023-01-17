# This notebook predicts a hero is a human or not by using Random Forests and it has 78% accruacy. 
# The important features are:
# 1. Weight 0.057069 
# 2. Height 0.032676 
# 3. Weapons Master_False 0.030967 
# 4. num_powers 0.026448 
# 5. Weapons Master_True 0.026126 
# 6. Publisher_DC Comics 0.019808 
# 7. Super Strength_True 0.015057 
# 8. Marksmanship_True 0.014874 
# 9. Hair color_Brown 0.012977 
# 10. Accelerated Healing_True 0.012774 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
heros_info = pd.read_csv('../input/heroes_information.csv')
heros_info.head()
# Only want to predict human/non-human. transform race to binary
heros_info['Human'] = heros_info['Race'].apply(lambda i: 1 if i == 'Human' else 0)
heros_info.drop(['Unnamed: 0'], inplace=True, axis=1)
heros_info.head()
print(heros_info.columns[heros_info.isnull().any()])
heros_info['Weight'].fillna(heros_info['Weight'].median(), inplace=True)
heros_info['Publisher'].fillna(heros_info['Publisher'].mode()[0], inplace=True)
heros_info.head()
print(heros_info.columns[heros_info.isnull().any()])
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=-99.0, strategy='median', axis=0)
heros_info["Height"]=imp.fit_transform(heros_info[["Height"]])
heros_info["Weight"]=imp.fit_transform(heros_info[["Weight"]])
heros_power = pd.read_csv('./../input/super_hero_powers.csv')
heros_power.head()
power_cat_columns = heros_power.columns.drop("hero_names")
heros_power_dummies = pd.get_dummies(heros_power, columns=power_cat_columns)
heros_power_dummies.head()
info_cat_columns = ['Gender', 'Eye color', 'Hair color', 'Publisher', 'Skin color', 'Alignment']
heros_info_dummies = pd.get_dummies(heros_info, columns=info_cat_columns)
heros_info_dummies.head()
heros = pd.merge(heros_info_dummies, heros_power_dummies, left_on=['name'], right_on=['hero_names'], how='inner')
heros.head()
X_columns_drop = ['name', 'hero_names', 'Race', 'Human']
X, y = heros.drop(X_columns_drop, axis=1), heros['Human']
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

# Initialize a stratified split of our dataset for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the classifier with the default parameters 
rfc = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)

# Train it on the training set
results = cross_val_score(rfc, X, y, cv=skf)

# Evaluate the accuracy on the test set
print("CV accuracy score: {:.2f}%".format(results.mean()*100))
from matplotlib import pyplot as plt
import seaborn as sns
    
rfc.fit(X, y)
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest
features = dict()
count = 1
for factor in X.columns:
    index = "f"+str(count)
    features[index] = factor
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

X_important = X[top_features[0:17]]

# Train it on the training set
results = cross_val_score(rfc, X_important, y, cv=skf)

# Evaluate the accuracy on the test set
print("CV accuracy score: {:.2f}%".format(results.mean()*100))

# Accruracy doesn't change much after using only top 18 features
parameters = {'max_features': [4, 7, 10, 13], 'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5,10,15,20]}

gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X_important, y)
gcv.best_estimator_, gcv.best_score_
# Still not more than 80%. let's add one more feature #powers
heros_power_numeric=heros_power*1
heros_power_numeric.head()
heros_power_numeric.loc[:, 'num_powers'] = heros_power_numeric.iloc[:, 1:].sum(axis=1)
heros_power_numeric = heros_power_numeric[['hero_names', 'num_powers']]
heros_power_numeric.head()
heros_num_power = pd.merge(heros, heros_power_numeric, left_on=['name'], right_on=['hero_names'], how='inner')
heros_num_power.head()
X_columns_drop = ['name', 'hero_names_x', 'Race', 'Human', 'hero_names_y']
X_num_power = heros_num_power.drop(X_columns_drop, axis=1)
features = dict()
count = 1
for factor in X_num_power.columns:
    index = "f"+str(count)
    features[index] = factor
    count+=1
    
rfc.fit(X_num_power, y)
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest
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

# num_powers is an import factor
parameters = {'max_features': [4, 7, 10, 13], 'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5,10,15,20]}
X_important = X_num_power[top_features[0:13]]
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X_important, y)
gcv.best_estimator_, gcv.best_score_
# Still below 80%. perhaps join with the power details and trasnform the data into numeric 