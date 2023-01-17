#this cell is valid when you're working on google colab and you want to upload the data to colab environment to use in your notebook

#uploading the data file from your Desktop

#from google.colab import files

#files.upload()
import pandas as pd

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

py.init_notebook_mode(connected=False)

import plotly.graph_objs as go

import plotly.figure_factory as ff

def configure_plotly_browser_state():

  import IPython

  display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

        <script>

          requirejs.config({

            paths: {

              base: '/static/base',

              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',

            },

          });

        </script>

        '''))
#Loading the dataset in Pandas dataframe

df_cancer = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df_cancer.head()
print(df_cancer.columns)

print()

print("Cancer dataset dimensions : {}".format(df_cancer.shape))

print()

print("Rows:",df_cancer.shape[0])

print()

print("Columns:",df_cancer.shape[1])
df_cancer = df_cancer.drop('Unnamed: 32',axis=1)
print(df_cancer.columns)

df_cancer.head()
df_cancer.describe().T
print(df_cancer.isnull().any().any())
configure_plotly_browser_state()

trace = go.Pie(labels = ['benign','malignant'], values = df_cancer['diagnosis'].value_counts(), 

               textfont=dict(size=10), opacity = 0.7,

               marker=dict(colors=['green', 'red'], 

               line=dict(color='#000000', width=1.0)))

           



layout= go.Layout(

        title={

        'text': "Distribution of dependent(diagnosis) variable",

        'y':0.8,

        'x':0.45,

        'xanchor': 'center',

        'yanchor': 'top'})



fig = go.Figure(data = [trace], layout=layout)

fig.show()
df_cancer['diagnosis']= df_cancer['diagnosis'].map({'M':1,'B':0})

df_cancer.head()
df_cancer['diagnosis'].value_counts()
mal = df_cancer[(df_cancer['diagnosis'] != 0)]

print(mal.shape)

ben = df_cancer[(df_cancer['diagnosis'] == 0)]

print(ben.shape)

def show_plots(column, bin_size) :  

    t1 = mal[column]

    t2 = ben[column]

    

    hist_data = [t1, t2]

    

    group_labels = ['Malignant', 'Benign']

    colors = ['red', 'green']



    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = bin_size, curve_type='kde')

    

    fig['layout'].update(title = column)

    fig.show()
configure_plotly_browser_state()

show_plots('radius_mean', .3)

show_plots('texture_mean', .3)

show_plots('perimeter_mean',3)

show_plots('area_mean',20)

configure_plotly_browser_state()

show_plots('radius_se', 0.1)

show_plots('texture_se', .1)

show_plots('perimeter_se', .5)

show_plots('area_se', 5)

configure_plotly_browser_state()

show_plots('radius_worst', .5)

show_plots('texture_worst', .5)

show_plots('perimeter_worst', 5)

show_plots('area_worst', 15)

plt.figure(figsize=(25,12))

sns.heatmap(df_cancer.corr(),annot=True)
sns.scatterplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=df_cancer)
features = ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']

len(features)
X =df_cancer[features].values

y =df_cancer['diagnosis']

print(X.shape)

print(y.shape)
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3,random_state=22,stratify=y)

print("Shape of train dataset:")

print(X_train.shape)

print(y_train.shape)

print("\n")

print("Shape of val dataset:")

print(X_val.shape)

print(y_val.shape)

print("\n")
from sklearn.ensemble import RandomForestClassifier



model1 = RandomForestClassifier(max_depth=1, random_state=0, verbose=0,n_estimators=50)

model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_val)
cnf1 = confusion_matrix(y_val,y_pred1)

sns.heatmap(cnf1,annot=True,cmap='summer',fmt='g')
acc1 = accuracy_score(y_val,y_pred1)

print("Accuracy: for baseline model is: %0.3f"%acc1)



print("RF train accuracy: %0.3f" % model1.score(X_train, y_train))

print("RF test accuracy: %0.3f" % model1.score(X_val, y_val))
print(classification_report(y_val,y_pred1))
coef1= model1.feature_importances_

print(coef1.shape)

print(len(features))

coefs1 = pd.DataFrame({"Features":features,"Coefficients":coef1})

feature_imp1 = coefs1.sort_values(by='Coefficients',ascending=False)

plt.figure(figsize=(15,10))

sns.barplot(y='Features',x='Coefficients',data=feature_imp1)
param_grid={'n_estimators':[50,100,150,200,250],

            'max_depth':[1,2,3,4],

            'min_samples_split':[2,3,5],

            'max_features':['auto','sqrt','log2']}
model2= GridSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=0,n_jobs=-1)

model2.fit(X_train,y_train)
print(model2.best_params_)

y_pred2 = model2.predict(X_val)
cnf2 = confusion_matrix(y_val,y_pred2)

sns.heatmap(cnf2,annot=True,fmt='g',cmap='Blues')
acc2 = accuracy_score(y_val,y_pred2)

print("Accuracy with GridSearch: %0.3f"%acc2)



print("RF train accuracy: %0.3f" % model2.score(X_train, y_train))

print("RF test accuracy: %0.3f" % model2.score(X_val, y_val))
print(classification_report(y_val,y_pred2))
coef2= model2.best_estimator_.feature_importances_

print(coef2.shape)

print(len(features))

coefs2 = pd.DataFrame({"Features":features,"Coefficients":coef2})

feature_imp2 = coefs2.sort_values(by='Coefficients',ascending=False)

plt.figure(figsize=(15,10))

sns.barplot(y='Features',x='Coefficients',data=feature_imp2)
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier

model3 = Pipeline([

  ('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=50))),

  ('classification', RandomForestClassifier())

])

model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_val)

cnf3 = confusion_matrix(y_val,y_pred3)

sns.heatmap(cnf3,annot=True,cmap='summer',fmt='g')

acc3 = accuracy_score(y_val,y_pred3)

print("Accuracy on Model3 is: %0.3f"%acc3)

print("RF train accuracy: %0.3f" % model3.score(X_train, y_train))

print("RF test accuracy: %0.3f" % model3.score(X_val, y_val))
print(classification_report(y_val,y_pred3))
#Feature Importance

f1 = model3.steps[0][1].get_support()

new_f = [features[i] for i,val in enumerate(f1) if val==True]

print(new_f)

coef3 = model3.steps[1][1].feature_importances_

print(coef3.shape)

print(len(new_f))

coefs3 = pd.DataFrame({"Features":new_f,"Coefficients":coef3})

feature_imp3 = coefs3.sort_values(by='Coefficients',ascending=False)

plt.figure(figsize=(15,10))

sns.barplot(y='Features',x='Coefficients',data=feature_imp3)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

model4 = XGBClassifier()

model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_val)

cnf4 = confusion_matrix(y_val,y_pred4)

sns.heatmap(cnf4,annot=True,cmap='summer',fmt='g')

acc4 = accuracy_score(y_val,y_pred4)

print("Accuracy on Model3 is: %0.3f"%acc4)

print("RF train accuracy: %0.3f" % model4.score(X_train, y_train))

print("RF test accuracy: %0.3f" % model4.score(X_val, y_val))
print(classification_report(y_val,y_pred4))
coef4= model4.feature_importances_

print(coef4.shape)

print(len(features))

coefs4 = pd.DataFrame({"Features":features,"Coefficients":coef4})

feature_imp4 = coefs4.sort_values(by='Coefficients',ascending=False)

plt.figure(figsize=(15,10))

sns.barplot(y='Features',x='Coefficients',data=feature_imp4)
from lightgbm import LGBMClassifier

model5 = LGBMClassifier()

model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_val)

cnf5 = confusion_matrix(y_val,y_pred5)

sns.heatmap(cnf5,annot=True,cmap='summer',fmt='g')

acc5 = accuracy_score(y_val,y_pred5)

print("Accuracy on Model5 is: %0.3f"%acc5)

print("Ligtgbm train accuracy: %0.3f" % model5.score(X_train, y_train))

print("LightGBM test accuracy: %0.3f" % model5.score(X_val, y_val))
print(classification_report(y_val,y_pred5))
coef5= model5.feature_importances_

print(coef5.shape)

print(len(features))

coefs5 = pd.DataFrame({"Features":features,"Coefficients":coef5})

feature_imp5 = coefs5.sort_values(by='Coefficients',ascending=False)

plt.figure(figsize=(15,10))

sns.barplot(y='Features',x='Coefficients',data=feature_imp5)