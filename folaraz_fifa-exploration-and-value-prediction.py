# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

warnings.simplefilter('ignore')

from sklearn.preprocessing import LabelEncoder

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, accuracy_score

import pandas as pd

import numpy as np

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn import cross_validation, metrics   

from sklearn.grid_search import GridSearchCV  

from sklearn import metrics

from pylab import rcParams

import numpy as np

import seaborn as sns

import folium

from collections import Counter, defaultdict

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

%matplotlib inline



# Any results you write to the current directory are saved as output.
player_attribute=pd.read_csv("../input/fifa-18-demo-player-dataset/PlayerAttributeData.csv", index_col=0)

player_personal=pd.read_csv("../input/fifa-18-demo-player-dataset/PlayerPersonalData.csv", index_col=0)
player_personal.head(1)
f, ax = plt.subplots(figsize=(10, 25))

countries = Counter(player_personal["Nationality"])

Nations_to_players = defaultdict(list)

for c in countries:

    Nations_to_players["countries"].append(c)

    Nations_to_players["number of players"].append(countries[c])

countries_to_num = pd.DataFrame(Nations_to_players)

countries_to_num = countries_to_num.sort_values("number of players", ascending=False)

sns.barplot(y="countries", x="number of players", data = countries_to_num, palette="GnBu_d")
f, ax = plt.subplots(figsize=(10, 25))

club = Counter(player_personal["Club"])

club_to_players = defaultdict(list)

for c in club:

    club_to_players["club"].append(c)

    club_to_players["number of players"].append(club[c])

club_to_num = pd.DataFrame(club_to_players)

club_to_num = club_to_num.sort_values("number of players", ascending=False)

sns.barplot(y="club", x="number of players", data= club_to_num.head(50), palette="GnBu_d")
#Convert the values in the column Value and Wage into a usable form i.e String.

def convert_to_int(x):

    remove = "K€M"

    for i in remove:

        x = x.replace(i,"")

    return x

player_personal["Wage"] = player_personal["Wage"].apply(lambda x: convert_to_int(x))

player_personal["Value"] = player_personal["Value"].apply(lambda x: convert_to_int(x))

player_personal[["Value","Wage"]] = player_personal[["Value","Wage"]].astype(float)



player_personal.head(1)
f, axes = plt.subplots(2, 2, figsize=(15, 13), sharex=True)

sns.despine(left=True)

sns.distplot(player_personal.Age, rug=True, rug_kws={"color": "g"},

            kde_kws={"color": "k", "lw": 3, "label": "KDE"},

            hist_kws={"histtype": "step", "linewidth": 3,

            "alpha": 1, "color": "g"},ax=axes[0, 0])

sns.distplot(player_personal.Value, rug=True, rug_kws={"color": "g"},

            kde_kws={"color": "k", "lw": 3, "label": "KDE"},

            hist_kws={"histtype": "step", "linewidth": 3,

            "alpha": 1, "color": "g"},ax=axes[0, 1])

sns.distplot(player_personal.Wage, rug=True, rug_kws={"color": "g"},

            kde_kws={"color": "k", "lw": 3, "label": "KDE"},

            hist_kws={"histtype": "step", "linewidth": 3,

            "alpha": 1, "color": "g"},ax=axes[1, 0])

sns.distplot(player_personal.Potential, rug=True, rug_kws={"color": "g"},

            kde_kws={"color": "k", "lw": 3, "label": "KDE"},

            hist_kws={"histtype": "step", "linewidth": 3,

            "alpha": 1, "color": "g"},ax=axes[1, 1])

plt.setp(axes, yticks=[])

plt.tight_layout()
corr_variables = player_personal[["Potential", "Value","Age", "Wage"]]

colormap = plt.cm.inferno

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of Potential, Value , Age and Wage of players', 

          y=1.05, size=15)

sns.heatmap(corr_variables.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
continent_countries = pd.read_csv("../input/world-countries-and-continents-details/countries and continents.csv")[["name","Continent"]]

continent_countries.dropna(inplace=True)
continent_countries["Nationality"] = continent_countries["name"]

continent_countries.drop("name",1, inplace=True)
#We merge it with other dataset.

cont_player_personal = pd.merge(player_personal, continent_countries,on = "Nationality", how='left')

cont_player_personal["countries"] = cont_player_personal["Nationality"]

cont_player_personal.drop("Nationality",1, inplace=True)

cont_player_personal = pd.merge(cont_player_personal, countries_to_num,on = "countries", how='left')
cont_player_personal.dropna(inplace=True)

cont_player_personal.head(1)
group_cont = cont_player_personal.groupby("Continent", as_index=False).agg({"Age": lambda x: x.sum()/len(x),

                                                             "number of players": lambda x: x.sum()/len(x),

                                                             "Wage": lambda x: x.sum()/len(x),

                                                             "Value": lambda x: x.sum()/len(x),

                                                             "Potential": lambda x: x.sum()/len(x)})
group_cont
group_cont = pd.melt(group_cont, id_vars=["Continent"]).sort_values(['variable','value'])
rcParams['figure.figsize'] = 15,10



sns.barplot(x='Continent', y='value', hue='variable', data=group_cont, palette="GnBu")

plt.xticks(rotation=90)

plt.title('Variation of Wages, Age, Value, Potential and Number of players in Different Contitents')
fifa_df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)
#Converted the value and wage column into integer

fifa_df["Wage"] = fifa_df["Wage"].apply(lambda x: convert_to_int(x))

fifa_df["Value"] = fifa_df["Value"].apply(lambda x: convert_to_int(x))

fifa_df[["Value","Wage"]] = fifa_df[["Value","Wage"]].astype(float)



fifa_df.tail(1)
fifa_df.drop(["Photo", "Flag","Club Logo","Name"],1,inplace=True)
fifa_df.drop("ID",1,inplace=True)
l_encode = LabelEncoder()

obj_feat = ["Club", "Nationality","Preferred Positions"]

for var in obj_feat:

    fifa_df[var] = l_encode.fit_transform(fifa_df[var].astype(str))

fifa_df.shape
fifa_df.dtypes
# Clean the Object values in the data frame, discovered there was "67 + 3" in the dataset

def clean_values(x):

    try:

        if len(x)>2:

            y = x[:2]

            return y

        else:

            return x

    except TypeError:

        return x

columns_to_clean = [col for col in fifa_df.columns if col not in ["Age","Nationality",

                                                                  "Overall","Potential",

                                                                 "Club","Value","Wage",

                                                                  "Special"]]

for col in columns_to_clean:

    fifa_df[col] = fifa_df[col].apply(lambda x : clean_values(x))
fifa_df = fifa_df.dropna(axis=1, how="any")
fifa_df.astype(int)

fifa_df.tail(4)
def modelfit(alg, dtrain, features, performCV=True, printFeatureImportance=True, cv_folds=10):

    #Fit the algorithm on the data

    alg.fit(dtrain[features],dtrain["Value"] )

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[features])



    #Perform cross-validation:

    cv_score = cross_validation.cross_val_score(alg, dtrain[features], dtrain["Value"], cv=cv_folds,

                                                scoring='neg_mean_squared_error')

    cv_score = np.sqrt(np.abs(cv_score))

    

    #Print model report:

    print ("\nModel Report")

    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain["Value"], dtrain_predictions)))

    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),

                                                                             np.std(cv_score),np.min(cv_score),

                                                                              np.max(cv_score)))

    if printFeatureImportance:

        feat_imp = pd.Series(alg.feature_importances_, features).sort_values(ascending=False)

        feat_imp.plot(kind='bar', title='Feature Importances')

        plt.ylabel('Feature Importance Score')
features = [i for i in fifa_df.columns if i != "Value"]

target = "Value"

gbm0 = GradientBoostingRegressor(random_state=7)

modelfit(gbm0, fifa_df, features)
estimators = [x for x in range(700,750,10)]

param_test1 = {'n_estimators':estimators}

gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500,

                                  min_samples_leaf=50,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 

                       param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=10)

gsearch1.fit(fifa_df[features],fifa_df["Value"])



gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
modelfit(gsearch1.best_estimator_, fifa_df, features)