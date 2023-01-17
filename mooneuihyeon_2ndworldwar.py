# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


#data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import xgboost as xgb
import plotly.graph_objects as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# This Python 3 environment comes with many helpful analytics libraries installed
#algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
import statsmodels.api as sm

#dataframe display settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 50)
battle = pd.read_csv('/kaggle/input/database-of-battles/battles.csv')
display(battle.tail())
print(battle.columns)
terrain = pd.read_csv('/kaggle/input/database-of-battles/terrain.csv')
display(terrain.tail())
print(terrain.columns)
weather = pd.read_csv('/kaggle/input/database-of-battles/weather.csv')
display(weather.tail())
print(weather.columns)
battle['war'].unique()
war_list = ['WORLD WAR II (ITALY 1943-1944)', 'WORLD WAR II (ITALY 1944)', 'WORLD WAR II (EUROPEAN THEATER)', 'WORLD WAR II', 'WORLD WAR II (EASTERN FRONT)', 'WORLD WAR II (OKINAWA)']

is_col = battle['war'] == 'WORLD WAR II (NORTH AFRICA 1942-1943)'
df1 = battle[is_col]

df2 = []

for col in war_list:
    is_col = battle['war'] == col
    df2 = battle[is_col]
    result = pd.concat([df1,df2])

display(result.tail(20))
#The revised dataset that contains all information needed

df = pd.merge(result, terrain , on="isqno")
df = pd.merge(df, weather, on = "isqno")
df.set_index('isqno', inplace = True)
df = df[['surpa', 'post1', 'wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'terra1', 'terra2', 'aeroa' , 'wina']]

display(df.tail())
import missingno as msno
#checking if there's any empty values on the chart.


msno.matrix(df)
plt.show()

print(df.isnull().sum())
# Based on what I searched for, Nan datas in 'wina' are supposed to be -1.(attacker loss)
df['wina'] = df['wina'].fillna(-1)
display(df)
df_combined = df[['surpa', 'post1', 'wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'terra1', 'terra2', 'aeroa', 'wina']]

display(df_combined)

msno.matrix(df_combined)
plt.show()
print(df_combined.isnull().sum())
df_combined.columns

def asdf(*args):
    bomb = []
    for a in df_combined.columns:
        if a != 'wx2':
            bomb.append(a)
        
    return bomb

column_list = asdf(df_combined.columns)
print(column_list)
df_combined['aeroa'] = df_combined['aeroa'].fillna(-1)
#전처리하기 전 aerial superiority 없는 경우(-1)로 처리

df_combined = df_combined[column_list].dropna()
display(df_combined) #926 datas left
print(df_combined.isnull().sum())
df_mask= df_combined['wina'] != 0
df_combined = df_combined[df_mask]
df_combined['wina'].unique()
#changing 'wina' value range from -1~,1 to 1,3
df_combined['wina'] = df_combined['wina'].apply(lambda x: x+1)

df_combined['wina'].unique()
# 0: attacker loss, 2: attacker win 
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x='wina',hue='surpa',data= df_combined ,palette='RdBu_r')
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x = 'wina', hue='terra1',data= df_combined ,palette='RdBu_r')
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x = 'wina', hue='terra2',data= df_combined ,palette='RdBu_r')
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='wx1',data= df_combined ,palette='RdBu_r')
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='wx3',data= df_combined ,palette='RdBu_r')
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='wx4',data= df_combined ,palette='RdBu_r')
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='wx5',data= df_combined ,palette='RdBu_r')
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x='wina', hue='post1',data= df_combined ,palette='RdBu_r')
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.countplot(x='wina',hue='aeroa',data= df_combined ,palette='RdBu_r')
def check_post1():
    global df_combined
    
    df_combined['post1'] = df_combined['post1'].map(lambda ca: ca[0])
    #dummy encoding
    post1_dummies = pd.get_dummies(df_combined['post1'], prefix='post1')
    df_combined = pd.concat([df_combined, post1_dummies], axis=1)
    df_combined.drop('post1', inplace=True, axis=1)
    return df_combined

df_combined = check_post1()
def check_wx1():
    global df_combined
    df_combined['wx1'] = df_combined['wx1'].map(lambda s: 1 if s == 'W' else 0)
    return df_combined

df_combined = check_wx1()
def check_wx3():
    global df_combined
    
    df_combined['wx3'] = df_combined['wx3'].map(lambda ca: ca[0])
    #dummy encoding
    wx3_dummies = pd.get_dummies(df_combined['wx3'], prefix='wx3')
    df_combined = pd.concat([df_combined, wx3_dummies], axis=1)
    df_combined.drop('wx3', inplace=True, axis=1)
    return df_combined

df_combined = check_wx3()
def check_wx4():
    global df_combined
    
    df_combined['wx4'] = df_combined['wx4'].map(lambda ca: ca[0])
    #dummy encoding
    wx4_dummies = pd.get_dummies(df_combined['wx4'], prefix='wx4')
    df_combined = pd.concat([df_combined, wx4_dummies], axis=1)
    df_combined.drop('wx4', inplace=True, axis=1)
    return df_combined

df_combined = check_wx4()
def check_wx5():
    global df_combined
    
    df_combined['wx5'] = df_combined['wx5'].map(lambda ca: ca[0])
    #dummy encoding
    wx5_dummies = pd.get_dummies(df_combined['wx5'], prefix='wx5')
    df_combined = pd.concat([df_combined, wx5_dummies], axis=1)
    df_combined.drop('wx5', inplace=True, axis=1)
    return df_combined

df_combined = check_wx5()
#creating function for creating categories based on the cabin of passengers
def check_terra1():
    global df_combined
    df_combined['terra1'] = df_combined['terra1'].map(lambda ca: ca[0])
    #dummy encoding
    terra1_dummies = pd.get_dummies(df_combined['terra1'], prefix='terra1')
    df_combined = pd.concat([df_combined, terra1_dummies], axis=1)
    df_combined.drop('terra1', inplace=True, axis=1)
    return df_combined

df_combined = check_terra1()
#creating function for creating categories based on the cabin of passengers
def check_terra2():
    global df_combined


    df_combined['terra2'] = df_combined['terra2'].map(lambda ca: ca[0])
    #dummy encoding
    terra2_dummies = pd.get_dummies(df_combined['terra2'], prefix='terra2')
    df_combined = pd.concat([df_combined, terra2_dummies], axis=1)
    df_combined.drop('terra2', inplace=True, axis=1)
    return df_combined

df_combined = check_terra2()
Test = df_combined.sample(frac = 0.3, random_state = 2)
Train = df_combined - Test

X_set = df_combined.copy().drop('wina', axis = 1)
y_set = df_combined.copy()['wina']
df_combined

from sklearn.model_selection import KFold

cv = KFold(5, shuffle=True, random_state=0)
cross_val_score(RandomForestClassifier(max_depth = 2), X_set, y_set, scoring="accuracy", cv=cv).mean()
cv = KFold(5, shuffle=True, random_state=0)
cross_val_score(LogisticRegression(), X_set, y_set, scoring="accuracy", cv=cv).mean()
cv = KFold(5, shuffle=True, random_state=0)
cross_val_score(KNeighborsClassifier(n_neighbors=3), X_set, y_set, scoring="accuracy", cv=cv).mean()
cv = KFold(5, shuffle=True, random_state=0)
cross_val_score(LinearSVC(), X_set, y_set, scoring="accuracy", cv=cv).mean()
cv = KFold(5, shuffle=True, random_state=0)
cross_val_score(DecisionTreeClassifier(), X_set, y_set, scoring="accuracy", cv=cv).mean()
X = X_set.copy()
y = y_set.copy()
feature_names = X.columns

tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0).fit(X, y)

import io
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz


def draw_decision_tree(model):
    dot_buf = io.StringIO()
    export_graphviz(model, out_file=dot_buf, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
    return Image(image)


def plot_decision_regions(X, y, model, title):
    resolution = 0.01
    markers = ('s', '^', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = mpl.colors.ListedColormap(colors)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = model.predict(
        np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)

    plt.contour(xx1, xx2, Z, cmap=mpl.colors.ListedColormap(['k']))
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=[cmap(idx)], marker=markers[idx], s=80, label=cl)

    plt.xlabel(data.feature_names[2])
    plt.ylabel(data.feature_names[3])
    plt.legend(loc='upper left')
    plt.title('War Classification')

    return Z
draw_decision_tree(tree1)

# 0: attacker loss, 2: attacker win 
with_aeroa_super = X['aeroa'] == -1
was = df_combined[with_aeroa_super]
display(was)
with_aeroa_super2 = X['aeroa'] == 0
was2 = df_combined[with_aeroa_super2]
display(was2)
was_final = pd.concat([was, was2])

was_final.index
#공자에게 제공권이 없던 사례 isqno
no_aeroa_super = X['aeroa'] == 1
Nas = df_combined[no_aeroa_super]

NotSpring = Nas['wx4_$'] == 0
NotSpring = Nas[NotSpring]
NotSpring.index
#공자에게 제공권 있는데, 여름/가을/겨울에 싸운 전투사례 isqno
no_aeroa_super = X['aeroa'] == 1
Nas = df_combined[no_aeroa_super]

Spring = Nas['wx4_$'] == 1
Spring = Nas[Spring]


Fd = Spring['post1_F'] == 0
Fd = Spring[Fd]
Fd.index
#공자에게 제공권 있는데, 계절은 봄이고, 방자의 방어형태가 '요새화된 방어'가 아니었던 전투사례 isqno