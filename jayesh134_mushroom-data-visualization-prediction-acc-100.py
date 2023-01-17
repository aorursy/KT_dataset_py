# Set up



import numpy as np               # Data Cleaning Library              

import pandas as pd           



import matplotlib.pyplot as plt          # Data Visulation Library

import seaborn as sns

%matplotlib inline
# Data import and info



df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

df.info()     # Eagle eye view
# Indepth view

df.head()
# Mapping alphabets to words they indicate

df['class'] = df['class'].map({'e':'edible', 'p':'poisonous'})

df['cap-shape'] = df['cap-shape'].map({'b':'bell', 'c':'conical', 'x':'convex', 'f':'flat', 'k':'knobbed', 's':'sunken'})

df['cap-surface'] = df['cap-surface'].map({'f':'fibrous','g':'grooves','y':'scaly','s':'smooth'})

df['cap-color'] = df['cap-color'].map({'n':'brown','b':'buff','c':'cinnamon','g':'gray','r':'green','p':'pink','u':'purple','e':'red','w':'white','y':'yellow'})

df['bruises'] = df['bruises'].map({'t':'True', 'f':'False'})

df['odor'] = df['odor'].map({'a':'almond', 'l':'anise', 'c':'creosote', 'y':'fishy', 'f':'foul', 'm':'musty', 'n':'none', 'p':'pungent', 's':'spicy'})

df['gill-attachment'] = df['gill-attachment'].map({'a':'attached', 'd':'descending', 'f':'free', 'n':'notched'})

df['gill-spacing'] = df['gill-spacing'].map({'c':'close', 'w':'crowded', 'd':'distant'})

df['gill-size'] = df['gill-size'].map({'b':'broad', 'n':'narrow'})

df['gill-color'] = df['gill-color'].map({'k':'black', 'n':'brown', 'b':'buff', 'h':'chocolate', 'g':'gray', 'r':'green', 'o':'orange', 'p':'pink', 'u':'purple', 'e':'red', 'w':'white', 'y':'yellow'})

df['stalk-shape'] = df['stalk-shape'].map({'e':'enlarging', 't':'tapering'})

df['stalk-root'] = df['stalk-root'].map({'b':'bulbous', 'c':'club', 'u':'cup', 'e':'equal', 'z':'rhizomorphs', 'r':'rooted', '?':'missing'})

df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].map({'f':'fibrous', 'y':'scaly', 'k':'silky', 's':'smooth'})

df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].map({'f':'fibrous', 'y':'scaly', 'k':'silky', 's':'smooth'})

df['stalk-color-above-ring'] = df['stalk-color-above-ring'].map({'n':'brown', 'b':'buff', 'c':'cinnamon', 'g':'gray', 'o':'orange', 'p':'pink', 'e':'red', 'w':'white', 'y':'yellow'})

df['stalk-color-below-ring'] = df['stalk-color-below-ring'].map({'n':'brown', 'b':'buff', 'c':'cinnamon', 'g':'gray', 'o':'orange', 'p':'pink', 'e':'red', 'w':'white', 'y':'yellow'})

df['veil-type'] = df['veil-type'].map({'p':'partial', 'u':'universal'})

df['veil-color'] = df['veil-color'].map({'n':'brown', 'o':'orange', 'w':'white', 'y':'yellow'})

df['ring-number'] = df['ring-number'].map({'n':'none', 'o':'one', 't':'two'})

df['ring-type'] = df['ring-type'].map({'c':'cobwebby', 'e':'evanescent', 'f':'flaring', 'l':'large', 'n':'none', 'p':'pendant', 's':'sheathing', 'z':'zone'})

df['spore-print-color'] = df['spore-print-color'].map({'k':'black', 'n':'brown', 'b':'buff', 'h':'chocolate', 'r':'green', 'o':'orange', 'u':'purple', 'w':'white', 'y':'yellow'})

df['population'] = df['population'].map({'a':'abundant', 'c':'clustered', 'n':'numerous', 's':'scattered', 'v':'several', 'y':'solitary'})

df['habitat'] = df['habitat'].map({'g':'grasses', 'l':'leaves', 'm':'meadows', 'p':'paths', 'u':'urban', 'w':'waste', 'd':'woods'})
# Indepth view

df.head()
# Checking for null values

df.isna().sum()
# Check if class is balanced or unbalanced

df['class'].value_counts()       
# Visualizing class

sns.set_style('whitegrid')

sns.countplot(x='class', data=df)

sns.despine()
class Universal:

    

    def __init__(self, col_name):

    # Counting mushrooms for each type

        self.col_name = col_name                                                                 # Use for giving df index name

        self.Edible_mushroom = df[df['class']=='edible'][self.col_name].value_counts()           # Counting only edible for each type

        self.Total_mushroom = df[self.col_name].value_counts()                                   # Counting total for each type

        self.Index = self.Total_mushroom.index                                                  # List of all unique value for particular column mentioned

        self.perc_storage = dict()                                                    # Holds :- key : column name, value : [Edibility %, Poisonous %]

        self.value_processer_from_list()                                                # Calling out method for creating perc.

        self.df_converter()                                                             # Calling out method to create df

    

    def edible_poison_perc(self, value):

    # Defining function to calculate edibility & poisonous percentage and saving it in dict. 

        edible = (self.Edible_mushroom[value]/self.Total_mushroom[value])*100            # edible % = (edible count / total count)*100

        poison = 100 - edible                                                                # poison % = 100 - edible %

        self.perc_storage[value] = [edible, poison]                                          # storing value in dict.



    def value_processer_from_list(self):

    # Applying percentage count func. on all types of unique value for column  

        for value in self.Index:                                                             # Index hold list unique value for column

            if value in self.Edible_mushroom.index :

                self.edible_poison_perc(value)                                                        # calculating edible/poison %

            else :

                edible = 0

                poison = 100

                self.perc_storage[value] = [edible, poison]                                          # storing value in dict.

                              

    def df_converter(self):

    # Creates a dataframe for column mentioned with data being perc_storage and index being edibility/poisonous

        self.df = pd.DataFrame(data=self.perc_storage, index=['Edibility', 'Poisonous']).T.sort_values(by='Edibility', axis=0, ascending = False)

        #self.df['Edibility'] = self.df['Edibility'].apply(lambda x : '{:.4}'.format(x))

        #self.df['Poisonous'] = self.df['Poisonous'].apply(lambda x : '{:.4}'.format(x))
# Cap Color Attribute 

cap_color = Universal('cap-color')

cap_color.df
fig, axes = plt.subplots(ncols=2, nrows=2,figsize=(25, 12))

sns.set_style('whitegrid')



# Counting variety of mushrooms's cap color

fig_1 = sns.countplot(x='cap-color', data=df, order=df['cap-color'].value_counts().index, ax=axes[0, 0])

fig_1.set_xlabel('Cap Color', fontsize=20)

fig_1.set_ylabel('Amount', fontsize=20)



# Counting variety of mushrooms's cap color and if edible/poisonous

fig_2 = sns.countplot(x='cap-color', data=df, order=df['cap-color'].value_counts().index, hue='class', ax=axes[0 , 1], palette='Set1')

fig_2.set_xlabel('Cap Color', fontsize=20)

fig_2.set_ylabel('Amount', fontsize=20)



# Visualizing variety of mushrooms's cap color by it's edibility percentage

fig_3 = sns.barplot(x=cap_color.df.index, y=cap_color.df['Edibility'], ax=axes[1,0], color='green')

fig_3.set_xlabel('Cap Color', fontsize=20)

fig_3.set_ylabel('Edibility Percentage', fontsize=20)



# Visualizing variety of mushrooms's cap color by it's poisonous percentage

fig_4 = sns.barplot(x=cap_color.df.index, y=cap_color.df['Poisonous'], ax=axes[1,1], color='red')

fig_4.set_xlabel('Cap Color', fontsize=20)

fig_4.set_ylabel('Poisonous Percentage', fontsize=20)
# Cap Shape Attribute

cap_shape = Universal('cap-shape')

cap_shape.df
fig, axes = plt.subplots(ncols=2, nrows=2,figsize=(25, 12))

sns.set_style('whitegrid')



# Counting variety of mushrooms's cap shape

fig_1 = sns.countplot(x='cap-shape', data=df, order=df['cap-shape'].value_counts().index, ax=axes[0, 0])

fig_1.set_xlabel('Cap Shape', fontsize=20)

fig_1.set_ylabel('Amount', fontsize=20)



# Counting variety of mushrooms's cap shape and if edible/poisonous

fig_2 = sns.countplot(x='cap-shape', data=df, order=df['cap-shape'].value_counts().index, hue='class', ax=axes[0 , 1], palette='Set1')

fig_2.set_xlabel('Cap Shape', fontsize=20)

fig_2.set_ylabel('Amount', fontsize=20)



# Visualizing variety of mushrooms's cap shape by it's edibility percentage

fig_3 = sns.barplot(x=cap_shape.df.index, y=cap_shape.df['Edibility'], ax=axes[1,0], color='green')

fig_3.set_xlabel('Cap Shape', fontsize=20)

fig_3.set_ylabel('Edibility Percentage', fontsize=20)



# Visualizing variety of mushrooms's cap shape by it's poisonous percentage

fig_4 = sns.barplot(x=cap_shape.df.index, y=cap_shape.df['Poisonous'], ax=axes[1,1], color='red')

fig_4.set_xlabel('Cap Shape', fontsize=20)

fig_4.set_ylabel('Poisonous Percentage', fontsize=20)
# Gill Color Attribute

gill_color = Universal('gill-color')

gill_color.df
fig, axes = plt.subplots(ncols=2, nrows=2,figsize=(25, 12))

sns.set_style('whitegrid')



# Counting variety of mushrooms's gill color

fig_1 = sns.countplot(x='gill-color', data=df, order=df['gill-color'].value_counts().index, ax=axes[0, 0])

fig_1.set_xlabel('Gill Color', fontsize=20)

fig_1.set_ylabel('Amount', fontsize=20)



# Counting variety of mushrooms's gill color and if edible/poisonous

fig_2 = sns.countplot(x='gill-color', data=df, order=df['gill-color'].value_counts().index, hue='class', ax=axes[0 , 1], palette='Set1')

fig_2.set_xlabel('Gill Color', fontsize=20)

fig_2.set_ylabel('Amount', fontsize=20)



# Visualizing variety of mushrooms's gill color by it's edibility percentage

fig_3 = sns.barplot(x=gill_color.df.index, y=gill_color.df['Edibility'], ax=axes[1,0], color='green')

fig_3.set_xlabel('Gill Color', fontsize=20)

fig_3.set_ylabel('Edibility Percentage', fontsize=20)



# Visualizing variety of mushrooms's gill color by it's poisonous percentage

fig_4 = sns.barplot(x=gill_color.df.index, y=gill_color.df['Poisonous'], ax=axes[1,1], color='red')

fig_4.set_xlabel('Gill Color', fontsize=20)

fig_4.set_ylabel('Poisonous Percentage', fontsize=20)
# Determining edibility or poisonous chances for mushroom by the attribute

safe_harm_attribute = dict()

for col in df.columns:

    obj = Universal(col_name=col)

    

    safe = ((obj.df.sum()['Edibility'])/(obj.df.sum()['Edibility'] + obj.df.sum()['Poisonous']))*100

    harm = ((obj.df.sum()['Poisonous'])/(obj.df.sum()['Edibility'] + obj.df.sum()['Poisonous']))*100

    safe_harm_attribute[col] = [safe, harm]



df2 = pd.DataFrame(data=safe_harm_attribute,index=['safe','harm']).T

df2.sort_values(by='safe', ascending=False)



# Looks like population should be the main attribute for determining a safe mushroom.

# Also odor can help us to determine poisonous mushroom.

# Whereas there is 50% chance of mushroom being safe/harmful based on cap shape, habitat , veil type. 
# Imports



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
# Review

df.head()
# Label Encoding

le = LabelEncoder()

df3 = df.apply(le.fit_transform)

df3.head()
# scaling feature data 

ss = StandardScaler()

ss.fit(df3.drop('class', axis=1))

X = ss.transform(df3.drop('class', axis=1))
# Features and label

y = df3['class']



# Splitting data into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Logistic Regression

lr = LogisticRegression(max_iter=500)



# training & prediction

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)



# evaluation

print('Confusion Matrix :\n', confusion_matrix(y_test, lr_pred))

print()

print('Classification Report :\n', classification_report(y_test, lr_pred))
# K Nearest Neighbor using Grid Search for n_neighbors



# Grid Search and training

parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}

search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=5)

search.fit(X_train, y_train)

print(search.best_params_)

print()



# prediction

knn_pred = search.best_estimator_.predict(X_test)



# evaluation

print('Confusion Matrix :\n', confusion_matrix(y_test, knn_pred))

print()

print('Classification Report :\n', classification_report(y_test, knn_pred))
# Decision Tree

tree = DecisionTreeClassifier()



# training and prediction

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)



# evaluation

print('Confusion Matrix :\n', confusion_matrix(y_test, tree_pred))

print()

print('Classification Report :\n', classification_report(y_test, tree_pred))
# Random Forest along with Grid Search for n_estimators



# Grid search and training

parameters = {'n_estimators': [100, 200, 300]}

search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5)

search.fit(X_train, y_train)

print(search.best_params_)



# prediction

rfc_pred = search.best_estimator_.predict(X_test)



# evaluation

print('Confusion Matrix :\n', confusion_matrix(y_test, rfc_pred))

print()

print('Classification Report :\n', classification_report(y_test, rfc_pred))
# Support Vector Classifier

svc = SVC()



# training and prediction

svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)



# evaluation

print('Confusion Matrix :\n', confusion_matrix(y_test, svc_pred))

print()

print('Classification Report :\n', classification_report(y_test, svc_pred))
# Creating a Performance report for all ML Models



classifier_pred = {'Logistic Regression ':lr_pred, 'K Nearest Neighbors':knn_pred, 'Decision Tree Classifier':tree_pred,

                   'Random Forest Classifier':rfc_pred, 'Support Vector Classifier':svc_pred}



report = dict()



for key, value in classifier_pred.items():

    # calculating scores 

    accuracy = accuracy_score(y_test, value)

    precision = precision_score(y_test, value)

    recall = recall_score(y_test, value)

    f1 = f1_score(y_test, value)

    # entering scores in report

    report[key] = [accuracy, precision, recall, f1]



# report dataframe

report_df = pd.DataFrame(data=report, index=['Accuracy', 'Precision', 'Recall', 'F1 Score']).T

report_df.index.name = 'ML Model'

report_df