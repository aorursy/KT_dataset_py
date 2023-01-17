import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/nba-games/games.csv")
data.head()
data1 = data[["HOME_TEAM_ID","VISITOR_TEAM_ID","PTS_home","FG_PCT_home","FT_PCT_home","AST_home","REB_home","PTS_away","FG_PCT_away","FT_PCT_away","FG3_PCT_away","AST_away","REB_away","HOME_TEAM_WINS"]]
data1.head()
data1.isnull().sum()
data1.dropna(inplace=True)
data1.isnull().sum()
# Splitting the data set into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    data1[["HOME_TEAM_ID","VISITOR_TEAM_ID","PTS_home","FG_PCT_home","FT_PCT_home","AST_home","REB_home","PTS_away","FG_PCT_away",
           "FT_PCT_away","FG3_PCT_away","AST_away","REB_away"]],
    data1["HOME_TEAM_WINS"],
    test_size=0.3
)
print ("Тренировочный датасет : ", X_train.shape)
print ("Тестировочный датасет : ", X_test.shape)
# Importing the required libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings; warnings.simplefilter('ignore')

drugTree = DecisionTreeClassifier(criterion="gini")
model_1 = drugTree.fit(X_train, y_train)


pred_1 = model_1.predict(X_test)
print("Точность для алгоритма Decision Tree Model: %.2f" % (accuracy_score(y_test, pred_1) * 100)+"%")
logistic = LogisticRegression(C = 0.5, max_iter = 500)
model_2 = logistic.fit(X_train, y_train)


pred_2 = model_2.predict(X_test)
print("Точность для алгоритма Logistic Regression Model: %.2f" % (accuracy_score(y_test, pred_2) * 100)+"%")
R_forest = RandomForestClassifier(n_estimators = 200)
model_3 = R_forest.fit(X_train, y_train)

pred_3 = model_3.predict(X_test)
print("Точность для алгоритма Random Forest Model: %.2f" % (accuracy_score(y_test, pred_3) * 100)+"%")
clf5 = MLPClassifier()
model_4 = clf5.fit(X_train, y_train)

pred_4 = model_4.predict(X_test)
print("Точность для алгоритма ANN Model: %.2f" % (accuracy_score(y_test, pred_4) * 100)+"%")
list_pred = [pred_1, pred_2, pred_3, pred_4]
model_names = [ "Decision Tree" ,"Logistic Regression","Random Forest Classifier","ANN",]

for i, pred in enumerate(list_pred) :
    print ("Confusion Matrix для : ", model_names[i])
    print()
    print (pd.DataFrame(confusion_matrix(y_test, pred)))
    print ()
from sklearn.metrics import roc_auc_score, roc_curve
models = [model_1, model_2, model_3, model_4]

plt.rcParams['figure.figsize'] = [10,8]
plt.style.use("bmh")

color = ['red', 'blue', 'green', 'fuchsia', 'cyan','yellow','brown']
plt.title("ROC CURVE", fontsize = 15)
plt.xlabel("Особенность", fontsize = 15)
plt.ylabel("Чувствительность", fontsize = 15)
i = 1

for i, model in enumerate(models) :
    prob = model.predict_proba(X_test)
    prob_positive = prob[:,1]
    fpr, tpr, threshold = roc_curve(y_test, prob_positive)
    plt.plot(fpr, tpr, color = color[i])
    plt.gca().legend(model_names, loc = 'lower right', frameon = True)

plt.plot([0,1],[0,1], linestyle = '--', color = 'black')
plt.show()
data.head()
plt.figure(figsize = (14,7))
sns.heatmap(dataViz.corr(),cmap='coolwarm',annot=True)
data.groupby('SEASON')['PTS_home'].agg('sum').plot(kind='bar',title='Статистика очков домашней команды по сезонам')
plt.rcParams['figure.figsize'] = [12, 8]
sns.set(style = 'whitegrid')

sns.distplot(data['PTS_home'], bins = 90, color = 'mediumslateblue')
plt.ylabel("Распределение", fontsize = 15)
plt.xlabel("Очко", fontsize = 15)
plt.margins(x = 0)

print ("Максимальное очко домашней команды", data['PTS_home'].max())
print ("Минимальнре очко домашней команды", data['PTS_home'].min())
features = list(set(data1.columns) - set(['State', 'International plan', 'Voice mail plan',  'Area code',
                                      'Total day charge',   'Total eve charge',   'Total night charge',
                                        'Total intl charge', 'Churn', 'Churn', 'Churn', 'Churn', 'Churn']))

data1[features].hist(figsize=(20,12));
plt.figure(figsize = (20,10))
sns.countplot(x='PTS_home', hue='HOME_TEAM_WINS', data=data1);