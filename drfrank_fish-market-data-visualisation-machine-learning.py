# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt

# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go


import warnings
warnings.filterwarnings("ignore")
fish=pd.read_csv("/kaggle/input/fish-market/Fish.csv")
df=fish.copy()
df.head()
df.info()
df.shape
df.columns
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df[df.duplicated() == True]
df.corr()
df_Species=df['Species'].value_counts().to_frame().reset_index().rename(columns={'index':'Species','Species':'Count'})

fig = go.Figure(go.Bar(
    y=df_Species['Species'],x=df_Species['Count'],orientation="h",
    marker={'color': df_Species['Count'], 
    'colorscale': 'sunsetdark'},  
    text=df_Species['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Fish Count',xaxis_title="Species",yaxis_title="Count",title_x=0.5)
fig.show()
df_Species=df['Species'].value_counts().to_frame().reset_index().rename(columns={'index':'Species','Species':'Count'})
df_Species
fig = go.Figure(go.Bar(
    x=df_Species['Species'],y=df_Species['Count'],
    marker={'color': df_Species['Count'], 
    'colorscale': 'Viridis'},  
    text=df_Species['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Fish Count',xaxis_title="Species",yaxis_title="Count",title_x=0.5)
fig.show()
df_Species=df['Species'].value_counts().to_frame().reset_index().rename(columns={'index':'Species','Species':'Count'})
df_Species

colors=['cyan','darkcyan','slateblue3','brown1','cadetblue1','coral2','salmon1']
fig = go.Figure([go.Pie(labels=df_Species['Species'], values=df_Species['Count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Fish Categories",title_x=0.5)
fig.show()
df_Species=df['Species'].value_counts().to_frame().reset_index().rename(columns={'index':'Species','Species':'Count'})
df_Species

fig = go.Figure(data=[go.Scatter(
    x=df_Species['Species'], y=df_Species['Count'],
    mode='markers',
    marker=dict(
        color=df_Species['Count'],
        size=df_Species['Count'],
        showscale=True
    ))])

fig.update_layout(title='Fish Categories',xaxis_title="Species",yaxis_title="Number Of Fish",title_x=0.5)
fig.show()
ax = sns.countplot(x="Species", data=df, palette="colorblind")
ax = sns.countplot(y="Species", data=df, palette="colorblind")
ax = sns.countplot(x="Species", data=df,
                   facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("dark", 7))
fig = go.Figure(data=[go.Histogram(x=df['Weight'],  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="Crimson",
                       xbins=dict(
                      start=0, #start range of bin
                      end=1800,  #end range of bin
                      size=50   #size of bin
                      ))])
fig.update_layout(title="Distribution Of Fish Weight in Gram ",xaxis_title="Weight",yaxis_title="Counts",title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Histogram(x=df['Height'],  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="LightSalmon",
                       xbins=dict(
                      start=0, #start range of bin
                      end=20,  #end range of bin
                      size=1   #size of bin
                      ))])
fig.update_layout(title="Distribution Of Fish Height in Cm ",xaxis_title="Height",yaxis_title="Counts",title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Histogram(x=df['Width'],  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="CadetBlue",
                       xbins=dict(
                      start=0, #start range of bin
                      end=10,  #end range of bin
                      size=1   #size of bin
                      ))])
fig.update_layout(title="Distribution Of Fish Width in Cm ",xaxis_title="Width",yaxis_title="Counts",title_x=0.5)
fig.show()
x=df['Weight']
from scipy.stats import norm
ax = sns.distplot(x, fit=norm, kde=False,color="darkviolet")
x=df['Height']
ax = sns.distplot(x, fit=norm, kde=False,color="Tan")
x=df['Width']
ax = sns.distplot(x, fit=norm, kde=False,color="DimGrey")
fig = px.scatter_matrix(df,
    dimensions=["Weight", "Length1", "Length2", "Length3",'Height','Width'],
    color="Species")
fig.update_layout(
    title='Fish Data set',
    width=800,
    height=800,
)
fig.update_traces(diagonal_visible=False)
fig.show()
g = sns.pairplot(df, kind='scatter', hue='Species');
sns.heatmap(df.corr(), annot=True, cmap='cubehelix');
fig = go.Figure()
fig.add_trace(go.Box(y=df['Width'],
                     marker_color="blue",
                     name="Width"))
fig.add_trace(go.Box(y=df['Height'],
                     marker_color="red",
                     name="Height"))
fig.add_trace(go.Box(y=df['Length1'],
                     marker_color="DarkSalmon",
                     boxpoints='outliers',
                     name="Length1"))
fig.add_trace(go.Box(y=df['Length2'],
                     marker_color="IndianRed",
                     boxpoints='outliers',
                     name="Length2"))
fig.add_trace(go.Box(y=df['Length3'],
                     marker_color="Tomato",
                     boxpoints='outliers',
                     name="Length3"))
fig.update_layout(title="Distribution",title_x=0.5)
fig.show()
fig = go.Figure(data=go.Violin(y=df['Weight'], box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                               x0='Weight '))

fig.update_layout(yaxis_zeroline=False,title="Distribution of Weight ",title_x=0.5)
fig.show()
ax = sns.boxplot(x="Species", y="Weight",
                 data=df, palette="Set3")
ax = sns.swarmplot(x="Species" ,y="Weight", data=df, color="DarkKhaki")
ax = sns.boxplot(x="Species", y="Weight", data=df)
ax = sns.swarmplot(x="Species" ,y="Weight", data=df, color=".25")
ax = sns.violinplot(x=df["Weight"])
ax = sns.violinplot(x="Species", y="Width", data=df, inner=None)
ax = sns.swarmplot(x="Species", y="Width", data=df,
                   color="Purple", edgecolor="gray")
df=fish.copy()
df.head()
df.info()
df.describe().T
df.drop(['Species'],axis=1,inplace=True)
df.head(2)
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:10]
np.sort(df_scores)[0:20]
threshold_value = np.sort(df_scores)[4]
threshold_value
Outlier_df= df[df_scores < threshold_value]
indexs=Outlier_df.index
Outlier_df
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
lbe.fit_transform(df["Species"])
df=fish.copy()
df["Target"]=lbe.fit_transform(df["Species"])
df.head()
df.info()
# Kick Outliers
for i in indexs:
    df.drop(i, axis = 0,inplace = True)
df.info()
df.drop(['Species'],axis=1,inplace=True)
df.info()
df.describe().T
y=df['Target']
y.head(3)
X=df.drop('Target',axis=1)
X.head(3)
X = (X - np.min(X)) / (np.max(X) - np.min(X)).values
X.head()
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=42)
print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)
loj_model
y_pred_loj = loj_model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test, y_pred_loj)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_loj)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
print("Training Accuracy :", loj_model.score(X_train, y_train))

print("Testing Accuracy :", loj_model.score(X_test, y_test))
print(classification_report(y_test, y_pred_loj))
cross_val_score(loj_model, X_test, y_test, cv = 10).mean()
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model
y_pred_nb = nb_model.predict(X_test)
accuracy_score(y_test, y_pred_nb)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_nb)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred_knn = knn_model.predict(X_test)
accuracy_score(y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))
knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(X_train, y_train)
    scoreList.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
print("Maximum KNN Score is {:.2f}%".format(acc))
knn = KNeighborsClassifier(9)
knn_tuned = knn.fit(X_train, y_train)
y_pred_knn_tuned = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred_knn_tuned)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_knn_tuned)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
from sklearn.svm import SVC
svc_model_linear = SVC(kernel = "linear").fit(X_train, y_train)
y_pred_svc = svc_model_linear.predict(X_test)
accuracy_score(y_test, y_pred_svc)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_svc)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
svc_model_rbf = SVC(kernel = "rbf").fit(X_train, y_train)
y_pred_svc_model_rbf = svc_model_rbf.predict(X_test)
accuracy_score(y_test, y_pred_svc_model_rbf)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_svc_model_rbf)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier().fit(X_train, y_train)
y_pred_mlpc = mlpc.predict(X_test)
accuracy_score(y_test,y_pred_mlpc)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_mlpc)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_score(y_test, y_pred_rf)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
Importance = pd.DataFrame({"Importance": rf_model.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Variable Significance Levels")
models = [
    knn_model,
    loj_model,
    svc_model_linear,
    svc_model_rbf,
    nb_model,
    mlpc,
    rf_model,
      
]

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("-"*28)
    print(names + ":" )
    print("Accuracy: {:.4%}".format(accuracy))
result = []

results = pd.DataFrame(columns= ["Models","Accuracy"])

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)    
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])
    results = results.append(result)
    
    
sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")
plt.xlabel('Accuracy %')
plt.title('Accuracy Ratios of Models');  