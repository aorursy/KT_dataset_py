import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



%matplotlib inline
pd.set_option('display.max_columns',40)
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import GridSearchCV
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff
cancer = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
cancer.head()
cancer.columns
print("Cancer dataset dimensions : {}".format(cancer.shape))

print()

print("Rows:",cancer.shape[0])

print()

print("Columns:",cancer.shape[1])
cancer.describe().T
cancer.drop(['Unnamed: 32','id'],1,inplace=True)
cancer.head()
cancer.isnull().any().any()
trace = go.Pie(labels = ['benign','malignant'], values = cancer['diagnosis'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['pink', 'purple'], 

               line=dict(color='#000000', width=1.5)))

           



layout= go.Layout(

        title={

        'text': "Distribution of diagnosis variable",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig = go.Figure(data = [trace], layout=layout)

fig.show()
cancer['diagnosis']= cancer['diagnosis'].map({'M':1,'B':0})
M = cancer[(cancer['diagnosis'] != 0)]

B = cancer[(cancer['diagnosis'] == 0)]
def plots(column, bin_size) :  

    temp1 = M[column]

    temp2 = B[column]

    

    hist_data = [temp1, temp2]

    

    group_labels = ['Malignant', 'Benign']

    colors = ['purple', 'pink']



    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = bin_size, curve_type='kde')

    

    fig['layout'].update(title = column)

    fig.show()
plots('radius_mean', .5)

plots('texture_mean', .5)

plots('perimeter_mean',5)

plots('area_mean',15)
plots('radius_se', .1)

plots('texture_se', .1)

plots('perimeter_se', .5)

plots('area_se', 5)
plots('radius_worst', .5)

plots('texture_worst', .5)

plots('perimeter_worst', 5)

plots('area_worst', 10)
plt.figure(figsize=(20,10))

sns.heatmap(cancer.corr(),annot=True)
sns.scatterplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=cancer)
cancer.columns
features = ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']
X =cancer.iloc[:,1:32].values

y =cancer['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=22,stratify=y)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test= scaler.transform(X_test)
model = SVC(kernel='linear')

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
cnf = confusion_matrix(y_test,y_pred)

sns.heatmap(cnf,annot=True,cmap='summer',fmt='g')
acc = accuracy_score(y_test,y_pred)

print("Accuracy:",acc)
print(classification_report(y_test,y_pred))
param_grid={'C':[0.1,1,10,100,1000],

            'gamma':[1,0.1,0.01,0.001,0.0001],

            'kernel':['rbf']}
grid= GridSearchCV(SVC(),param_grid,refit=True,verbose=4)

grid.fit(X_train,y_train)
grid.best_params_
grid.best_score_
g_pred = grid.predict(X_test)
g_cnf = confusion_matrix(y_test,g_pred)

sns.heatmap(g_cnf,annot=True,fmt='g',cmap='Blues')
g_acc = accuracy_score(y_test,g_pred)

print("Accuracy with GridSearch:",g_acc)
print(classification_report(y_test,g_pred))
coef= model.coef_

coeffs = np.squeeze(coef)

coeffs
coefs = pd.DataFrame({"Features":features,"Coefficients":coeffs})

feature_imp = coefs.sort_values(by='Coefficients',ascending=False)
feature_imp
plt.figure(figsize=(15,10))

sns.barplot(y='Features',x='Coefficients',data=feature_imp)