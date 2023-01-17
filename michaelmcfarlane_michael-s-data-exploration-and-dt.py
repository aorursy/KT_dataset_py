import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
np.random.seed(0)
#save file path
main_file_path = '../input/NBA 2017-2018 Data.csv'
#read data into DataFrame
data = pd.read_csv(main_file_path)
original_data = pd.read_csv(main_file_path)
data.columns = ['TEAM', 'DATE', 'HOMEADV', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PlusMinus']
import re

data.WL.replace("(W)", 1, regex=True, inplace=True)
data.WL.replace("(L)", 0, regex=True, inplace=True)

data.HOMEADV.replace("(@)", 0, regex=True, inplace=True)
data.HOMEADV.replace("(vs)", 1, regex=True, inplace=True)

#data.DATE = pd.Series.to_timestamp(data['DATE'])
data.head()
data.isnull().sum()
data.shape
#summarize data
data.describe()
predictors = ['HOMEADV', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF']
X = data[predictors]
X.describe()
y = data.WL
y.describe()
plotvariables = ['WL', 'HOMEADV', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF']

plotdata = data[plotvariables]
corr = plotdata.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())
#box plot WL/Points
var = 'WL'
data1 = pd.concat([data['PTS'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="PTS", data=data)
predictors1 = ['WL', 'PTS', 'FGM', 'FGPercentage',
       '3PM', '3Percentage', 'DREB', 'REB', 'AST'
       ]
filtered_col = data[predictors1]
corr = filtered_col.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
import pandas as pd
from sklearn.model_selection import train_test_split
y1 = data.WL
X1 = X
X_train, X_test, y_train, y_test = train_test_split(X1, y1,test_size=0.2)
print ("\nX_train:\n")
print(X_train.head())
print (X_train.shape)
print ("\nX_test:\n")
print(X_test.head())
print (X_test.shape)
from sklearn.tree import DecisionTreeClassifier
# Define model
model = DecisionTreeClassifier()
# Fit model
model.fit(X_train, y_train)
print("Making predictions for the following 5 games:")
print(X_test.head())
print("The predictions are")
print(model.predict(X_test.head()))
no_warriors_data = data[data.TEAM != 'GSW']
warriorsdata = data.loc[data['TEAM'] == 'GSW']
predictors5 = ['HOMEADV', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF']

NW_y = no_warriors_data.WL
NW_X = no_warriors_data[predictors]
WAR_y = warriorsdata.WL
WAR_X = warriorsdata[predictors]
# Define model
model2 = DecisionTreeClassifier()
# Fit model
model2.fit(NW_X, NW_y)
features_test = warriorsdata[['HOMEADV', 'PTS', 'FGM', 'FGA', 'FGPercentage',
       '3PM', '3PA', '3Percentage', 'FTM', 'FTA', 'FTPercentage', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF']].values
target_test = warriorsdata['WL'].values
predictions =  model2.predict(features_test)
result=confusion_matrix(target_test, predictions)
result
Sensitivity= result[0][0]/(result[0][0]+ result[0][1])
Sensitivity
Specificity = result[1][1]/(result[1][0]+ result[1][1])
Specificity
Precision= result[0][0]/(result[0][0]+ result[1][0])
Precision
F1_Score= 2*result[0][0]/(2*result[0][0]+ result[1][0]+result[0][1])
F1_Score
r2_score(target_test, predictions, sample_weight=None)












