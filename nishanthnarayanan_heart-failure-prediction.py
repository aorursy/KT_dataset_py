# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head()
df.shape
df.columns
df.dtypes
df.DEATH_EVENT = df.DEATH_EVENT.map({0: 'Recovered', 1: 'Died'})
df.diabetes = df.diabetes.map({0: 'No', 1: 'Yes'})
df['DEATH_EVENT'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.catplot(x="DEATH_EVENT", kind='count', data=df, height=5, legend=True)
plt.show()
df.age.value_counts().sort_index()
df.age = pd.cut(df.age, bins=3, labels=["Middle Age", "Adulthood", "Older Adulthood"])
df.age.value_counts()
df.insert(0, 'Id', range(1, 1 + len(df)))
df
def show_donut_plot(col):
    
    rating_data = df.groupby(col)[['Id']].count().head(10)
    plt.figure(figsize = (12, 8))
    plt.pie(rating_data[['Id']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)

    # create a center circle for more aesthetics to make it better
    gap = plt.Circle((0, 0), 0.5, fc = 'white')
    fig = plt.gcf()
    fig.gca().add_artist(gap)
    
    plt.axis('equal')
    
    cols = []
    for index, row in rating_data.iterrows():
        cols.append(index)
    plt.legend(cols)
    
    plt.title('Donut Plot: Age categories involving Heart Failures', loc='center')
    plt.show()
show_donut_plot('age')
sns.relplot(x="serum_creatinine", y="serum_sodium", data=df, kind='scatter', hue='age', col='diabetes', height=6, alpha=0.6)
plt.show()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
event = le.fit_transform(df.DEATH_EVENT)
import plotly.express as px
fig = px.sunburst(df, path=['age', 'ejection_fraction'], values=event)
fig.show()
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=df.creatinine_phosphokinase,
    y=df.platelets,
    z=df.serum_creatinine,
    name = 'Heart Failure Prediction',
    mode='markers',
    marker=dict(
        size=10,
        color = df['platelets'],
        colorscale = 'Viridis',
    )
)])
fig.show()
def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
df.dtypes
df.age = le.fit_transform(df.age)
df.DEATH_EVENT = le.fit_transform(df.DEATH_EVENT)
df.diabetes = le.fit_transform(df.diabetes)
rf_model = RandomForestClassifier().fit(df.drop(["Id","DEATH_EVENT"], axis=1),df["DEATH_EVENT"])
plot_feature_importance(rf_model.feature_importances_,df.drop(["Id","DEATH_EVENT"], axis=1).columns,'RANDOM FOREST')
clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    'svm2': SVC(kernel='rbf'),
    'svm3': SVC(kernel='sigmoid'),
    'mlp1': MLPClassifier(),
    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}
accuracy_scores = dict()
train_x, test_x, train_y, test_y = train_test_split(df.drop(["Id","DEATH_EVENT"], axis=1), df["DEATH_EVENT"], test_size= 0.3)
for clf_name in clfs:
    
    clf = clfs[clf_name]
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    accuracy_scores[clf_name] = accuracy_score(y_pred, test_y)
    print(clf, '-' , accuracy_scores[clf_name])
accuracy_scores = sorted(accuracy_scores.items(), key = lambda kv:(kv[1], kv[0]), reverse= True)
accuracy_scores
villi = list(dict(accuracy_scores).keys())[0]
villi
confusion_matrix(clfs[villi].predict(test_x), test_y)
fig,ax=plt.subplots(figsize=(10,5))
sns.regplot(y=test_y,x=clfs[villi].predict(test_x),marker="*")
plt.show()