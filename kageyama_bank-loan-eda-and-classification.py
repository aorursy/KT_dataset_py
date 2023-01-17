!pip install bubbly
# for basic operations

import numpy as np

import pandas as pd



# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for advanced visualizations 

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)

from bubbly.bubbly import bubbleplot



# for providing path

import os

print(os.listdir('../input/'))



# for model explanation

from sklearn.tree import export_graphviz

from sklearn.metrics import roc_curve

from sklearn.metrics import auc



# for classification

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



#for purmutation importance

import eli5 

from eli5.sklearn import PermutationImportance



#for SHAP values

import shap 

from pdpbox import pdp, info_plots #for partial plots



# for warning ignore

import warnings

warnings.filterwarnings('ignore')
# reading the data

data = pd.read_csv('../input/UniversalBank.csv')



# getting the shape

data.shape
# reading the head of the data



data.head()
# describing the data



data.describe()
# drop 'ID' and 'ZIP Code'

data = data.drop(["ID","ZIP Code"],axis=1)
# check missing data

data.isnull().sum()
figure = bubbleplot(dataset = data, x_column = 'Experience', y_column = 'Income', 

    bubble_column = 'Personal Loan', time_column = 'Age', size_column = 'Mortgage', color_column = 'Personal Loan', 

    x_title = "Experience", y_title = "Income", title = 'Experience vs Income. vs Age vs Mortgage vs Personal Loan',

    x_logscale = False, scale_bubble = 3, height = 650)



py.iplot(figure, config={'scrollzoom': True})
# making a heat map

plt.rcParams['figure.figsize'] = (20, 15)

plt.style.use('ggplot')



sns.heatmap(data.corr(), annot = True, cmap = 'Wistia')

plt.title('Heatmap for the Dataset', fontsize = 20)

plt.show()
# checking the distribution of age



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (15, 5)

sns.distplot(data['Age'], color = 'cyan')

plt.title('Distribution of Age', fontsize = 20)

plt.show()
# plotting a donut chart for visualizing 'Personal Loan','Securities Account','CD Account','Online','CreditCard'



fig, ax = plt.subplots(1,5,figsize=(20,20))

columns = ['Personal Loan','Securities Account','CD Account','Online','CreditCard']



for i,column in enumerate(columns):

    plt.subplot(1,5,i+1)

    size = data[column].value_counts()

    colors = ['lightblue', 'lightgreen']

    labels = "No", "Yes"

    explode = [0, 0.01]



    my_circle = plt.Circle((0, 0), 0.7, color = 'white')



    plt.rcParams['figure.figsize'] = (20, 20)

    plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')

    plt.title('Distribution of {}'.format(column), fontsize = 15)

    p = plt.gcf()

    p.gca().add_artist(my_circle)

plt.legend()

plt.show()
# show relation of family with personal loan

  

plt.rcParams['figure.figsize'] = (12, 9)

dat = pd.crosstab(data['Personal Loan'], data['Family']) 

dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar', 

                                                 stacked = False, 

                                                 color = plt.cm.rainbow(np.linspace(0, 1, 4)))

plt.title('Relation of Family with Personal Loan', fontsize = 20, fontweight = 30)

plt.show()

# show relation of education with personal loan

  

plt.rcParams['figure.figsize'] = (12, 9)

dat = pd.crosstab(data['Personal Loan'], data['Education']) 

dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar', 

                                                 stacked = False, 

                                                 color = plt.cm.rainbow(np.linspace(0, 1, 4)))

plt.title('Relation of Education with Personal Loan', fontsize = 20, fontweight = 30)

plt.show()

# show relation of income with personal loan



plt.rcParams['figure.figsize'] = (12, 9)

sns.boxplot(data['Personal Loan'], data['Income'], palette = 'viridis')

plt.title('Relation of Income with Personal Loan', fontsize = 20)

plt.show()
# show relation of CCAvg with personal loan



plt.rcParams['figure.figsize'] = (12, 9)

sns.violinplot(data['Personal Loan'], data['CCAvg'], palette = 'colorblind')

plt.title('Relation of CCAvg with Target', fontsize = 20, fontweight = 30)

plt.show()
# show relation of mortgage with personal loan



plt.rcParams['figure.figsize'] = (12, 9)

sns.violinplot(data['Personal Loan'], data['Mortgage'], palette = 'colorblind')

plt.title('Relation of Mortgage with Target', fontsize = 20, fontweight = 30)

plt.show()
# Give meaning to category data 



data['Securities Account'][data['Securities Account'] == 0] = 'No'

data['Securities Account'][data['Securities Account'] == 1] = 'Yes'



data['CD Account'][data['CD Account'] == 0] = 'No'

data['CD Account'][data['CD Account'] == 1] = 'Yes'



data['Online'][data['Online'] == 0] = 'No'

data['Online'][data['Online'] == 1] = 'Yes'



data['CreditCard'][data['CreditCard'] == 0] = 'No'

data['CreditCard'][data['CreditCard'] == 1] = 'Yes'
data['Securities Account'] = data['Securities Account'].astype('object')

data['CD Account'] = data['CD Account'].astype('object')

data['Online'] = data['Online'].astype('object')

data['CreditCard'] = data['CreditCard'].astype('object')



# drop age (Because the correlation with experience is high.)

data = data.drop(["Age"],axis=1)
# taking the labels out from the data



y = data['Personal Loan']

data = data.drop('Personal Loan', axis = 1)



print("Shape of y:", y.shape)
# One hot encoding

data = pd.get_dummies(data, drop_first=True)
# check data

data.head()
# Split the data

x = data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



# getting the shapes

print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
# MODELLING

# Random Forest Classifier



model = RandomForestClassifier(n_estimators = 50, max_depth = 5)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

y_pred_quant = model.predict_proba(x_test)[:, 1]

y_pred = model.predict(x_test)



# evaluating the model

print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))



# cofusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')



# classification report

cr = classification_report(y_test, y_pred)

print(cr)
# show tree map



estimator = model.estimators_[1]

feature_names = [i for i in x_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no loan'

y_train_str[y_train_str == '1'] = 'loan'

y_train_str = y_train_str.values





export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=50'])



from IPython.display import Image

Image(filename = 'tree.png')
# show ROC curve



fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])



plt.rcParams['figure.figsize'] = (15, 5)

plt.title('ROC curve for classifier', fontweight = 30)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
# check the auc score



auc = auc(fpr, tpr)

print("AUC Score :", auc)
# let's check the importance of each attributes



perm = PermutationImportance(model, random_state = 0).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())
# plotting the partial dependence plot for Income



base_features = data.columns.values.tolist()



feat_name = 'Income'

pdp_dist = pdp.pdp_isolate(model=model, dataset=x_test, model_features = base_features, feature = feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()

# let's plot the partial dependence plot for Education



base_features = data.columns.values.tolist()



feat_name = 'Education'

pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()

# let's plot the partial dependence plot for Family



base_features = data.columns.values.tolist()



feat_name = 'Family'

pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
# let's see the shap values



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(x_test)



shap.summary_plot(shap_values[1], x_test, plot_type="bar")
shap.summary_plot(shap_values[1], x_test)
# define function to make force plot

def show_forceplot(model, data):

  explainer = shap.TreeExplainer(model)

  shap_values = explainer.shap_values(data)

  shap.initjs()

  return shap.force_plot(explainer.expected_value[1], shap_values[1], data)
tmp = x_test.iloc[1,:].astype(float)

show_forceplot(model, tmp)
tmp = x_test.iloc[:, 2].astype(float)

show_forceplot(model, tmp)
tmp = x_test.iloc[:,3].astype(float)

show_forceplot(model, tmp)
shap_values = explainer.shap_values(x_train.iloc[:50])

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], x_test.iloc[:50])