from IPython.display import Image
Image("/kaggle/input/ufc245/UFC 245.jpg")
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
df = pd.read_csv('/kaggle/input/ufcdata/data.csv')

b_age = df['B_age']  #  we replace B_age to put it among B features 
df.drop(['B_age'], axis = 1, inplace = True)
df.insert(76, "B_age", b_age)

df_fe = df.copy() #  We make a copy of the dataframe for the feature engineering part later

df.head(5)
print(df.shape)
len(df[df['Winner'] == 'Draw'])
last_fight = df.loc[0, ['date']]
print(last_fight)
limit_date = '2001-04-01'
df = df[(df['date'] > limit_date)]
print(df.shape)
print("Total NaN in dataframe :" , df.isna().sum().sum())
print("Total NaN in each column of the dataframe")
na = []
for index, col in enumerate(df):
    na.append((index, df[col].isna().sum())) 
na_sorted = na.copy()
na_sorted.sort(key = lambda x: x[1], reverse = True) 

for i in range(len(df.columns)):
    print(df.columns[na_sorted[i][0]],":", na_sorted[i][1], "NaN")
from sklearn.impute import SimpleImputer

imp_features = ['R_Weight_lbs', 'R_Height_cms', 'B_Height_cms', 'R_age', 'B_age', 'R_Reach_cms', 'B_Reach_cms']
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

for feature in imp_features:
    imp_feature = imp_median.fit_transform(df[feature].values.reshape(-1,1))
    df[feature] = imp_feature

imp_stance = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_R_stance = imp_stance.fit_transform(df['R_Stance'].values.reshape(-1,1))
imp_B_stance = imp_stance.fit_transform(df['B_Stance'].values.reshape(-1,1))
df['R_Stance'] = imp_R_stance
df['B_Stance'] = imp_B_stance
print('Number of features with NaN values :', len([x[1] for x in na if x[1] > 0]))
na_features = ['B_avg_BODY_att', 'R_avg_BODY_att']
df.dropna(subset = na_features, inplace = True)

df.drop(['Referee', 'location'], axis = 1, inplace = True)
print(df.shape)
print("Total NaN in dataframe :" , df.isna().sum().sum())
df.info()
list(df.select_dtypes(include=['object', 'bool']))
print(df['B_draw'].value_counts())
print(df['R_draw'].value_counts())
df.drop(['B_draw', 'R_draw'], axis=1, inplace=True)
df = df[df['Winner'] != 'Draw']
df = df[df['weight_class'] != 'Catch Weight']
plt.figure(figsize=(50, 40))
corr_matrix = df.corr(method = 'pearson').abs()
sns.heatmap(corr_matrix, annot=True)
sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                 .astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
print(sol[0:10])
#  i = index of the fighter's fight, 0 means the last fight, -1 means first fight
def select_fight_row(df, name, i): 
    df_temp = df[(df['R_fighter'] == name) | (df['B_fighter'] == name)]  # filter df on fighter's name
    df_temp.reset_index(drop=True, inplace=True) #  as we created a new temporary dataframe, we have to reset indexes
    idx = max(df_temp.index)  #  get the index of the oldest fight
    if i > idx:  #  if we are looking for a fight that didn't exist, we return nothing
        return 
    arr = df_temp.iloc[i,:].values
    return arr

select_fight_row(df, 'Amanda Nunes', 0) #  we get the last fight of Amanda Nunes
# get all active UFC fighters (according to the limit_date parameter)
def list_fighters(df, limit_date):
    df_temp = df[df['date'] > limit_date]
    set_R = set(df_temp['R_fighter'])
    set_B = set(df_temp['B_fighter'])
    fighters = list(set_R.union(set_B))
    return fighters
fighters = list_fighters(df, '2017-01-01')
print(len(fighters))
def build_df(df, fighters, i):      
    arr = [select_fight_row(df, fighters[f], i) for f in range(len(fighters)) if select_fight_row(df, fighters[f], i) is not None]
    cols = [col for col in df] 
    df_fights = pd.DataFrame(data=arr, columns=cols)
    df_fights.drop_duplicates(inplace=True)
    df_fights['title_bout'] = df_fights['title_bout'].replace({True: 1, False: 0})
    df_fights.drop(['R_fighter', 'B_fighter', 'date'], axis=1, inplace=True)
    return df_fights

df_train = build_df(df, fighters, 0)
df_test = build_df(df, fighters, 1)
df_train.head(5)
print(df_train.shape)
print(df_test.shape)
print(len(df_train[df_train['Winner'] == 'Blue']))
print(len(df_train[df_train['Winner'] == 'Red']))
print(len(df_test[df_test['Winner'] == 'Blue']))
print(len(df_test[df_test['Winner'] == 'Red']))
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer((OrdinalEncoder(), ['weight_class', 'B_Stance', 'R_Stance']), remainder='passthrough')

# If the winner is from the Red corner, Winner label will be encoded as 1, otherwise it will be 0 (Blue corner)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_train['Winner'])
y_test = label_encoder.transform(df_test['Winner'])

X_train, X_test = df_train.drop(['Winner'], axis=1), df_test.drop(['Winner'], axis=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Random Forest composed of 100 decision trees. We optimized parameters using cross-validation and GridSearch tool paired together
random_forest = RandomForestClassifier(n_estimators=100, 
                                       criterion='entropy', 
                                       max_depth=10, 
                                       min_samples_split=2,
                                       min_samples_leaf=1, 
                                       random_state=0)

model = Pipeline([('encoding', preprocessor), ('random_forest', random_forest)])
model.fit(X_train, y_train)

# We use cross-validation with 5-folds to have a more precise accuracy (reduce variation)
accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=5)
print('Accuracy mean : ', accuracies.mean())
print('Accuracy standard deviation : ', accuracies.std())

y_pred = model.predict(X_test)
print('Testing accuracy : ', accuracy_score(y_test, y_pred), '\n')

target_names = ["Blue","Red"]
print(classification_report(y_test, y_pred, labels=[0,1], target_names=target_names))
#from sklearn.model_selection import GridSearchCV
#parameters = [{'random_forest__n_estimators': [10, 50, 100, 500, 1000],
#               'random_forest__criterion': ['gini', 'entropy'],
#               'random_forest__max_depth': [5, 10, 50],
#               'random_forest__min_samples_split': [2, 3, 4],
#               'random_forest__min_samples_leaf': [1, 2, 3],
#              }]
#model = Pipeline([('encoding', preprocessor), ('random_forest', RandomForestClassifier())])

#grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=-1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_

#best_params = grid_search.best_params_
#print('Best accuracy : ', best_accuracy)
#print('Best parameters : ', best_params)
from sklearn.metrics import confusion_matrix

# The confusion matrix looks like the shape below:
# [TN FN
#  FP TP]
cm = confusion_matrix(y_test, y_pred) 
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax, fmt = "d")
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(['Blue', 'Red'])
ax.yaxis.set_ticklabels(['Blue', 'Red'])
feature_names = [col for col in X_train]
feature_importances = model['random_forest'].feature_importances_
indices = np.argsort(feature_importances)[::-1]
n = 30 # maximum feature importances displayed
idx = indices[0:n] 
std = np.std([tree.feature_importances_ for tree in model['random_forest'].estimators_], axis=0)

#for f in range(n):
#    print("%d. feature %s (%f)" % (f + 1, feature_names[idx[f]], feature_importances[idx[f]])) 

plt.figure(figsize=(30, 8))
plt.title("Feature importances")
plt.bar(range(n), feature_importances[idx], color="r", yerr=std[idx], align="center")
plt.xticks(range(n), [feature_names[id] for id in idx], rotation = 45) 
plt.xlim([-1, n]) 
plt.show()
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

tree_estimator = model['random_forest'].estimators_[10]
export_graphviz(tree_estimator, 
                out_file='tree.dot', 
                filled=True, 
                rounded=True)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
Image(filename = 'tree.png')
def predict(df, pipeline, blue_fighter, red_fighter, weightclass, rounds, title_bout=False): 
    
    #We build two dataframes, one for each figther 
    f1 = df[(df['R_fighter'] == blue_fighter) | (df['B_fighter'] == blue_fighter)].copy()
    f1.reset_index(drop=True, inplace=True)
    f1 = f1[:1]
    f2 = df[(df['R_fighter'] == red_fighter) | (df['B_fighter'] == red_fighter)].copy()
    f2.reset_index(drop=True, inplace=True)
    f2 = f2[:1]
    
    # if the fighter was red/blue corner on his last fight, we filter columns to only keep his statistics (and not the other fighter)
    # then we rename columns according to the color of  the corner in the parameters using re.sub()
    if (f1.loc[0, ['R_fighter']].values[0]) == blue_fighter:
        result1 = f1.filter(regex='^R', axis=1).copy() #here we keep the red corner stats
        result1.rename(columns = lambda x: re.sub('^R','B', x), inplace=True)  #we rename it with "B_" prefix because he's in the blue_corner
    else: 
        result1 = f1.filter(regex='^B', axis=1).copy()
    if (f2.loc[0, ['R_fighter']].values[0]) == red_fighter:
        result2 = f2.filter(regex='^R', axis=1).copy()
    else:
        result2 = f2.filter(regex='^B', axis=1).copy()
        result2.rename(columns = lambda x: re.sub('^B','R', x), inplace=True)
        
    fight = pd.concat([result1, result2], axis = 1) # we concatenate the red and blue fighter dataframes (in columns)
    fight.drop(['R_fighter','B_fighter'], axis = 1, inplace = True) # we remove fighter names
    fight.insert(0, 'title_bout', title_bout) # we add tittle_bout, weight class and number of rounds data to the dataframe
    fight.insert(1, 'weight_class', weightclass)
    fight.insert(2, 'no_of_rounds', rounds)
    fight['title_bout'] = fight['title_bout'].replace({True: 1, False: 0})
    
    pred = pipeline.predict(fight)
    proba = pipeline.predict_proba(fight)
    if (pred == 1.0): 
        print("The predicted winner is", red_fighter, 'with a probability of', round(proba[0][1] * 100, 2), "%")
    else:
        print("The predicted winner is", blue_fighter, 'with a probability of ', round(proba[0][0] * 100, 2), "%")
    return proba
predict(df, model, 'Kamaru Usman', 'Colby Covington', 'Welterweight', 5, True) 
predict(df, model, 'Max Holloway', 'Alexander Volkanovski', 'Featherweight', 5, True) 
predict(df, model, 'Amanda Nunes', 'Germaine de Randamie', "Women's Bantamweight", 5, True)
predict(df, model, 'Jose Aldo', 'Marlon Moraes', 'Bantamweight', 3, False)
predict(df, model, 'Urijah Faber', 'Petr Yan', 'Bantamweight', 3, False)