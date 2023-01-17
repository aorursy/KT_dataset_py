import pandas as pd  # Data manipulation and reading

import numpy as np # Algebra operations package

import seaborn as sns # Seaborn data visualization

from scipy.stats import chi2_contingency # Chi-square analysis for categorical variables

import matplotlib.pyplot as plt  # Data visualization - matplotlib
pd.set_option('display.max_columns', None)
# Loading the data

data_mushrooms = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

data_mushrooms.head(5)
data_mushrooms['cap-shape'] = data_mushrooms['cap-shape'].replace({ 'b':'bell','c':'conical','x':'convex', 'f':'flat', 'k':'knobbed', 's':'sunken' })

data_mushrooms['cap-surface'] =  data_mushrooms['cap-surface'].replace({'f':'fibrous','g':'grooves','y':'scaly','s':'smooth' })

data_mushrooms['cap-color'] = data_mushrooms['cap-color'].replace({'n':'brown', 'b':'buff','c':'cinnamon','g':'gray', 'r':'green', 'p':'pink','u':'purple','e':'red','w':'white','y':'yellow' })

data_mushrooms['bruises'] =data_mushrooms['bruises'].replace({'t':'bruises','f':'no' })

data_mushrooms['odor'] = data_mushrooms['odor'].replace({ 'a':'almond','l':'anise','c':'creosote','y':'fishy','f':'foul', 'm':'musty','n':'none','p':'pungent','s':'spicy' })

data_mushrooms['gill-attachment'] = data_mushrooms['gill-attachment'].replace({ 'a':'attached','d':'descending','f':'free','n':'notched' })

data_mushrooms['gill-spacing'] = data_mushrooms['gill-spacing'].replace({'c':'close','w':'crowded','d':'distant' })

data_mushrooms['gill-size'] = data_mushrooms['gill-size'].replace({ 'b':'broad','n':'narrow' })

data_mushrooms['gill-color'] = data_mushrooms['gill-color'].replace({ 'k':'black','n':'brown','b':'buff','h':'chocolate','g':'gray','r':'green','o':'orange','p':'pink','u':'purple','e':'red', 'w':'white','y':'yellow' })

data_mushrooms['stalk-shape'] = data_mushrooms['stalk-shape'].replace({ 'e':'enlarging','t':'tapering' })

data_mushrooms['stalk-root'] = data_mushrooms['stalk-root'].replace({ 'b':'bulbous','c':'club','u':'cup','e':'equal', 'z':'rhizomorphs','r':'rooted','?':'missing' })

data_mushrooms['stalk-surface-above-ring'] = data_mushrooms['stalk-surface-above-ring'].replace({ 'f':'fibrous','y':'scaly','k':'silky','s':'smooth' })

data_mushrooms['stalk-surface-below-ring'] = data_mushrooms['stalk-surface-below-ring'].replace({ 'f':'fibrous','y':'scaly','k':'silky','s':'smooth' })

data_mushrooms['stalk-color-above-ring'] = data_mushrooms['stalk-color-above-ring'].replace({ 'n': 'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange', 'p':'pink','e':'red','w':'white','y':'yellow'})

data_mushrooms['stalk-color-below-ring'] = data_mushrooms['stalk-color-below-ring'].replace({ 'n': 'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange', 'p':'pink','e':'red','w':'white','y':'yellow'})

data_mushrooms['veil-type'] = data_mushrooms['veil-type'].replace({ 'p':'partial','u':'universal' })

data_mushrooms['veil-color'] = data_mushrooms['veil-color'].replace({ 'n':'brown','o':'orange','w':'white','y':'yellow' })

data_mushrooms['ring-number'] =data_mushrooms['ring-number'].replace({ 'n':'none', 'o':'one','t':'two' })

data_mushrooms['ring-type'] = data_mushrooms['ring-type'].replace({'c':'cobwebby','e':'evanescent','f':'flaring','l':'large', 'n':'none','p':'pendant','s':'sheathing','z':'zone' })

data_mushrooms['spore-print-color'] = data_mushrooms['spore-print-color'].replace({'k':'black','n':'brown','b':'buff','h':'chocolate','r':'green', 'o':'orange','u':'purple','w':'white','y':'yellow'})

data_mushrooms['population'] = data_mushrooms['population'].replace({ 'a':'abundant','c':'clustered','n':'numerous', 's':'scattered','v':'several','y':'solitary' })

data_mushrooms['habitat'] = data_mushrooms['habitat'].replace({  'g':'grasses','l':'leaves','m':'meadows','p':'paths', 'u':'urban','w':'waste','d':'woods' })

data_mushrooms['class'] = data_mushrooms['class'].replace({ 'p':'Poisonous' , 'e':'Edible'})
# Data Transformation

data_mushrooms.head(5)
LABELS = ["Edible", "Poisonous"]

class_counts = pd.value_counts(data_mushrooms['class'], sort = True)

class_counts.plot(kind = 'bar', rot=0, color=['#3399ff','red']) #Added colors

plt.title('Edible / Poisonous count')

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("");
g = sns.catplot(x="spore-print-color", col="class", col_wrap=4,

            data=data_mushrooms,

            kind="count", height=3.5, aspect=.9)

g.set_xticklabels(rotation=45)

plt.show()
g = sns.catplot(x="odor", col="class", col_wrap=4,

            data=data_mushrooms,

            kind="count", height=3.5, aspect=.9)

g.set_xticklabels(rotation=45)

plt.show()
# Determining missing values 

data_mushrooms.isnull().values.any() # There is possibly a category of missing value that differs from none.
cols = list(data_mushrooms.columns) # List of columns



obs =[] # Keeping the observed matrix..

chi_square = [] # calculated p-value

corr = pd.DataFrame()

k = 0



for j in cols:

    for i in cols:

        obs.append(pd.crosstab(data_mushrooms[j],data_mushrooms[i]))

        aux = chi2_contingency(observed= pd.crosstab(data_mushrooms['class'],data_mushrooms[i])) # Matriz de contigência para cada par (n,m) de variáveis

        chi_square.append(aux[1])

    corr[j] = chi_square

    chi_square = []

    k = k + 1
corr.index = cols

corr = round(corr, 2)

sns.heatmap(corr, annot=True, cmap="YlGnBu")

plt.show()
data_mushrooms['veil-type'].value_counts()
data_mushrooms['stalk-root'].value_counts() # 30% missing values = ?
data_mushrooms.drop(['veil-type','stalk-root'], axis=1, inplace = True)
from sklearn.base import BaseEstimator #Packages needed for development

from sklearn.base import TransformerMixin



class LowFrequency():

    

    def __init__(self):

        pass



    def fit(self, X,y=None):

        return self

    

    def transform(self, X,y=None):

        

        X_ = X.copy()

        

        def transf_low_freq(df, col, K):



            SIZE = len(df)



            freq_col = df[col].value_counts()/SIZE # frequency calculation

            quantile = freq_col.quantile(q=K) #custom percentile



            less_freq_col = freq_col[freq_col <= quantile]

            df.loc[df[col].isin(less_freq_col.index.tolist()), col] = "others" # Categoria outros.

 

            return (df)



        list_cols = list(X_.columns)



        for item in list_cols:

            transf_low_freq(X_, item, K = 0.1)

        

        return(X_)
# Pipeline libraries

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split



# Template packages

from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import FeatureUnion



le = LabelEncoder()

X = data_mushrooms.drop('class', axis = 1)

list_cols = list(X.columns)



# Using featureUnion, using my custom function

union = FeatureUnion([("lfreq", LowFrequency())])



# Features and transformation

X = union.fit_transform(X)

X = pd.DataFrame(X, columns=list_cols)

X = pd.get_dummies(X, columns = X.columns)



# Class 

y = data_mushrooms['class'].values

y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020) # Hould-out
from sklearn.model_selection import GridSearchCV



pipe_rf = Pipeline( steps = [('rf',RandomForestClassifier(random_state=2020))]) 



param_grid_rf = {'rf__criterion': ["gini", "entropy"], 

                  'rf__n_estimators': [50, 100, 150],

                  'rf__max_depth': range(0,7),

                  'rf__max_features': [5, 7, 10, 15],

                  'rf__min_samples_split' : [0.005, 0.08]}
grid_rf = GridSearchCV(pipe_rf, 

                    param_grid=param_grid_rf, 

                    cv=10, # CV = 10

                    verbose = 2,

                    scoring='accuracy', 

                    n_jobs = -1)



grid_rf.fit(X_train, y_train)
# Final Model

grid_rf.best_estimator_
feat_importances = pd.Series(grid_rf.best_estimator_.named_steps["rf"].feature_importances_, index=X_train.columns)

feat_importances.nlargest(10).plot(kind='barh');
ypred = grid_rf.predict(X_test) # Using the test base
from sklearn.metrics import confusion_matrix

M_confusao = confusion_matrix(y_test, ypred)



# Confusion matrix plot

ax= plt.subplot()

sns.heatmap(M_confusao, annot=True, ax = ax, annot_kws={"size": 10},fmt=".0f");

ax.set_xlabel('Predict');ax.set_ylabel('Real'); 

ax.set_title('Confusion Matrix'); 

plt.grid(False)

ax.xaxis.set_ticklabels(['Edible', 'Poisonous']); ax.yaxis.set_ticklabels(['Edible', 'Poisonous']);
# Table of classification measures in both classes 0 = Edible, 1 = Poisonous

from sklearn.metrics import classification_report

print(classification_report(y_test, ypred, digits=5))
# Generated ROC AUC curve



import sklearn.metrics as metrics



y_pred_proba = grid_rf.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()