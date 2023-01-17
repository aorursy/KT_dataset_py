import shap 

import warnings  

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.tree import DecisionTreeClassifier

import os

import seaborn as sns

print(os.listdir("../input"))

from sklearn.metrics import classification_report

import itertools

plt.style.use('fivethirtyeight')
df=pd.read_csv('../input/diabetes.csv')
df.head()
df.tail()
df.isna().any() # No NAs
print(df.dtypes)
print(df.info())
df.head(10)
# Calculate the median value for BMI

median_bmi = df['BMI'].median()

# Substitute it in the BMI column of the

# dataset where values are 0

df['BMI'] = df['BMI'].replace(

    to_replace=0, value=median_bmi)



median_bloodp = df['BloodPressure'].median()

# Substitute it in the BloodP column of the

# dataset where values are 0

df['BloodPressure'] = df['BloodPressure'].replace(

    to_replace=0, value=median_bloodp)



# Calculate the median value for PlGlcConc

median_plglcconc = df['Glucose'].median()

# Substitute it in the PlGlcConc column of the

# dataset where values are 0

df['Glucose'] = df['Glucose'].replace(

    to_replace=0, value=median_plglcconc)



# Calculate the median value for SkinThick

median_skinthick = df['SkinThickness'].median()

# Substitute it in the SkinThick column of the

# dataset where values are 0

df['SkinThickness'] = df['SkinThickness'].replace(

    to_replace=0, value=median_skinthick)



# Calculate the median value for SkinThick

median_skinthick = df['Insulin'].median()

# Substitute it in the SkinThick column of the

# dataset where values are 0

df['Insulin'] = df['Insulin'].replace(

    to_replace=0, value=median_skinthick)
df['Outcome'].value_counts().plot.bar();
f,ax=plt.subplots(1,2,figsize=(18,8))

df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('target')

ax[0].set_ylabel('')

sns.countplot('Outcome',data=df,ax=ax[1])

ax[1].set_title('Outcome')

plt.show()
columns=df.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
df1=df[df['Outcome']==1]

columns=df.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df1[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()


sns.pairplot(df, hue = 'Outcome', vars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age'] )
sns.jointplot("Pregnancies", "Insulin", data=df, kind="reg")
sns.jointplot("BloodPressure", "Insulin", data=df, kind="reg")
def set_bmi(row):

    if row["BMI"] < 18.5:

        return "Under"

    elif row["BMI"] >= 18.5 and row["BMI"] <= 24.9:

        return "Healthy"

    elif row["BMI"] >= 25 and row["BMI"] <= 29.9:

        return "Over"

    elif row["BMI"] >= 30:

        return "Obese"
df = df.assign(BM_DESC=df.apply(set_bmi, axis=1))



df.head()
def set_insulin(row):

    if row["Insulin"] >= 16 and row["Insulin"] <= 166:

        return "Normal"

    else:

        return "Abnormal"
df = df.assign(INSULIN_DESC=df.apply(set_insulin, axis=1))



df.head()
sns.countplot(data=df, x = 'INSULIN_DESC', label='Count')



AB, NB = df['INSULIN_DESC'].value_counts()

print('Number of patients Having Abnormal Insulin Levels: ',AB)

print('Number of patients Having Normal Insulin Levels: ',NB)
sns.countplot(data=df, x = 'BM_DESC', label='Count')



UD,H,OV,OB = df['BM_DESC'].value_counts()

print('Number of patients Having Underweight BMI Index: ',UD)

print('Number of patients Having Healthy BMI Index: ',H)

print('Number of patients Having Overweigth BMI Index: ',OV)

print('Number of patients Having Obese BMI Index: ',OB)
g = sns.FacetGrid(df, col="INSULIN_DESC", row="Outcome", margin_titles=True)

g.map(plt.scatter,"Glucose", "BloodPressure",  edgecolor="w")

plt.subplots_adjust(top=1.1)
g = sns.FacetGrid(df, col="INSULIN_DESC", row="Outcome", margin_titles=True)

g.map(plt.scatter,"DiabetesPedigreeFunction", "BloodPressure",  edgecolor="w")

plt.subplots_adjust(top=1.1)
g = sns.FacetGrid(df, col="Outcome", row="INSULIN_DESC", margin_titles=True)

g.map(plt.hist, "Age", color="red")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Disease by INSULIN and Age');
sns.boxplot(x="Age", y="INSULIN_DESC", hue="Outcome", data=df);
sns.boxplot(x="Age", y="BM_DESC", hue="Outcome", data=df);
df1=df.drop(['Outcome'],axis=1)
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(df1.corr(), annot=True,cmap ='RdYlGn')
X=pd.get_dummies(df)

cols_drop=['Outcome','BM_DESC_Under']

X=X.drop(cols_drop,axis=1)



y = df['Outcome']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y, random_state = 1234)
min_train = X_train.min()

range_train = (X_train - min_train).max()

X_train = (X_train - min_train)/range_train
min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test = (X_test - min_test)/range_test
X_test.isna().sum()
from sklearn.ensemble import RandomForestClassifier

Model=RandomForestClassifier(max_depth=2)

Model.fit(X_train,y_train)

y_pred=Model.predict(X_test)

print(classification_report(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))

#Accuracy Score

print('accuracy is ',accuracy_score(y_pred,y_test))
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist(),top=13)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
features = [c for c in X.columns if c not in ['Outcome']]
from sklearn import tree

import graphviz

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features)

display(graphviz.Source(tree_graph))
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='Glucose')



# plot it

pdp.pdp_plot(pdp_goals, 'Glucose')

plt.show()
feature_to_plot = 'BMI'

pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature=feature_to_plot)



pdp.pdp_plot(pdp_dist, feature_to_plot)

plt.show()
# Build Random Forest model

rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)



pdp_dist = pdp.pdp_isolate(model=rf_model, dataset=val_X, model_features=features, feature=feature_to_plot)



pdp.pdp_plot(pdp_dist, feature_to_plot)

plt.show()
pdp_dist = pdp.pdp_isolate(model=rf_model, dataset=val_X, model_features=features, feature='Glucose')



pdp.pdp_plot(pdp_dist, 'Glucose')

plt.show()
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot

features_to_plot = ['Glucose', 'BMI']

inter1  =  pdp.pdp_interact(model=tree_model, dataset=val_X, model_features=features, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot,  plot_type='grid',

                                  plot_pdp=True)



plt.show()
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(rfc_model)



# Calculate Shap values

shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
shap.summary_plot(shap_values, X_test)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], X_train.iloc[:,1:10])