#visualization

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import matplotlib.pyplot as plt # plotting

%matplotlib inline



#data handling

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import copy



# Import statements required for Plotly 

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



#data mining and machine learning

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE, ADASYN

import xgboost
df = pd.read_csv('../input/HR-Employees.csv')

nRow, nCol = df.shape

print('There are {} rows and {} columns'.format(nRow, nCol))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1]] # and nunique[col] < 50 For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title('{}'.format(columnNames[i]))

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print('No correlation plots shown: The number of non-NaN or constant columns ({}) is less than 2'.format(df.shape[1]))

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title('Correlation Matrix for Numerical Variables')

    plt.show()
display(df.isnull().any())
df.info()
df.describe()
df_dropped = df.drop(columns=['EmployeeNumber', 'StandardHours', 'EmployeeCount', 'Over18', 'PerformanceRating'])

df.nunique() > 1
plotPerColumnDistribution(df_dropped, df_dropped.size, 8)
df_attrition = df_dropped[lambda df: df_dropped.Attrition == 'Yes'].copy()

df_attrition.head()
plotPerColumnDistribution(df_attrition,df_attrition.size, 10)
df_attrition_sales = df_attrition[lambda df: df_attrition.Department == 'Sales'].copy()
df_attrition_rd = df_attrition[lambda df: df_attrition.Department == 'Research & Development'].copy()
plotPerColumnDistribution(df_attrition_sales,df_attrition_sales.size, 10)
plotPerColumnDistribution(df_attrition_rd,df_attrition_rd.size, 10)
df_attrition_rd_lvl1 = df_attrition_rd[df_attrition_rd.JobLevel==1].copy()

df_attrition_rd_lvl1.head()
plotPerColumnDistribution(df_attrition_rd_lvl1 ,df_attrition_rd_lvl1 .size, 8)
df_attrition_rd_lvl1.StockOptionLevel.hist()
numerical_df = df_dropped[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',

       'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',

       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',

       'PercentSalaryHike', 'RelationshipSatisfaction',

       'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',

       'YearsInCurrentRole', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']].copy()
seniority = numerical_df[["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",

                        "YearsWithCurrManager", "TotalWorkingYears","MonthlyIncome"]].copy()

pca = PCA(n_components=1)

numerical_df['Seniority'] = pca.fit_transform(seniority)

numerical_df = numerical_df.drop(["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",

                        "YearsWithCurrManager", "TotalWorkingYears","MonthlyIncome", 'DailyRate', 'HourlyRate', 'MonthlyRate'], axis=1)
categorical = []

for col, value in df_dropped.iteritems():

    if value.dtype == 'object':

        categorical.append(col)

cat_df = df_dropped[categorical].copy()

cat_df = pd.get_dummies(cat_df)

cat_df = cat_df.drop(['BusinessTravel_Non-Travel','Department_Human Resources','EducationField_Other'

                      ,'Gender_Male','JobRole_Laboratory Technician','MaritalStatus_Single',

                     'OverTime_No','Attrition_No'], axis=1)
df_final = pd.concat([numerical_df,cat_df],axis=1)

attr = df_final['Attrition_Yes']

df_final = df_final.drop(['Attrition_Yes'], axis=1)
# Import the train_test_split method

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



# Split data into train and test sets as well as for validation and testing

train, test, target_train, target_val = train_test_split(df_final, 

                                                         attr, test_size=.3,random_state=0);





oversampler=ADASYN(random_state=0)

smote_train, smote_target = oversampler.fit_sample(train,target_train)
seed = 0   # We set our random seed to zero for reproducibility

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 800,

    'warm_start': True, 

    'max_features': 0.3,

    'max_depth': 9,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}



rf = RandomForestClassifier(**rf_params)

rf.fit(smote_train, smote_target)

rf_predictions = rf.predict(test)

ranF_accuracy = accuracy_score(target_val, rf_predictions)

print('Our Random Forest model has an accuracy of {}'.format(round(ranF_accuracy,3)))
# Scatter plot 

trace = go.Scatter(

    y = rf.feature_importances_,

    x = df_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = rf.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = df_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance - Not Accounting Ordinal Data',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
# Gradient Boosting Parameters

gb_params ={

    'n_estimators': 500,

    'max_features': 0.9,

    'learning_rate' : 0.2,

    'max_depth': 11,

    'min_samples_leaf': 2,

    'subsample': 1,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}



gb = GradientBoostingClassifier(**gb_params)

# Fit the model to our SMOTEd train and target

gb.fit(smote_train, smote_target)

# Get our predictions

gb_predictions = gb.predict(test)
gb_accuracy = accuracy_score(target_val, gb_predictions)

print('Our Gradient Boosting model has an accuracy of {}'.format(round(gb_accuracy,3)))
# Scatter plot 

trace = go.Scatter(

    y = gb.feature_importances_,

    x = df_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = gb.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = df_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Model Feature Importance - Not Accounting Ordinal Data',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter')
from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re



decision_tree = tree.DecisionTreeClassifier(max_depth = 4)

decision_tree.fit(train, target_train) #change the training and target data



# Predicting results for test dataset

y_pred = decision_tree.predict(test) #change the test data



# Export our trained model as a .dot file

with open("tree_uncleaned.dot", 'w') as f: #change the name of the output

     f = tree.export_graphviz(decision_tree,

                              out_file=f,

                              max_depth = 4,

                              impurity = False,

                              feature_names = df_final.columns.values, #change the df

                              class_names = ['Attrition_No', "Attrition_Yes"],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree_uncleaned.dot','-o','tree_uncleaned.png']) #change the name of the output



# Annotating chart with PIL

img = Image.open("tree_uncleaned.png")#change the name of the output

draw = ImageDraw.Draw(img)#change the name of the output

img.save('tree_uncleaned_out.png')#change the name of the output

PImage("tree_uncleaned_out.png")#change the name of the output
interval_df = df_dropped[["Age", "DailyRate",

                           "DistanceFromHome","HourlyRate", "MonthlyIncome",

                           "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike",

                           "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",

                           "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]].copy()

interval_df.describe()
scaled_num = interval_df.copy()
plotCorrelationMatrix(scaled_num, 10)

scaled_num.head()
seniority = scaled_num[["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",

                        "YearsWithCurrManager", "TotalWorkingYears","MonthlyIncome", 'Age']].copy()

seniority.corr(method='pearson').round(4)
#PCA FOR MY NOTES



pca = PCA(n_components=1)

scaled_num['Seniority'] = pca.fit_transform(seniority)

scaled_num = scaled_num.drop(["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",

                        "YearsWithCurrManager", "TotalWorkingYears","MonthlyIncome", 

                              'DailyRate', 'HourlyRate', 'MonthlyRate', 'Age'], axis=1)



scaler = MinMaxScaler()

scaled_num = pd.DataFrame(scaler.fit_transform(scaled_num), columns=scaled_num.columns)
scaled_num.head()
cat_nominal = df_dropped[["Attrition", "BusinessTravel", "Department", "EducationField", "Gender",

                          "JobRole","OverTime"]].copy()

cat_nominal = pd.get_dummies(cat_nominal)

cat_nominal = cat_nominal.drop(["Attrition_Yes","BusinessTravel_Travel_Frequently", "Department_Human Resources",

                                "EducationField_Human Resources", "Gender_Male", "JobRole_Human Resources",

                               "OverTime_No"], axis=1)
cat_ordinal = df_dropped[["Education","EnvironmentSatisfaction","JobInvolvement",

                          "JobLevel","JobSatisfaction","RelationshipSatisfaction",

                          "StockOptionLevel","WorkLifeBalance"]].copy()

cat_ordinal.corr(method='spearman')
cat_ordinal = cat_ordinal.applymap(str)

cat_ordinal = pd.get_dummies(cat_ordinal)

cat_ordinal = cat_ordinal.drop(["Education_5", "EnvironmentSatisfaction_4", 

                               "JobInvolvement_4", "JobSatisfaction_4", "RelationshipSatisfaction_4", 

                              "StockOptionLevel_3", "WorkLifeBalance_4"], axis=1)

cat_ordinal.info()
cleaned_final = pd.concat([scaled_num, cat_nominal, cat_ordinal],axis=1)

attr_final = cleaned_final['Attrition_No']

cleaned_final = cleaned_final.drop(['Attrition_No'], axis=1)
# Import the train_test_split method

from sklearn.model_selection import train_test_split

#from sklearn.model_selection import StratifiedShuffleSplit



# Split data into train and test sets as well as for validation and testing

train_r2, test_r2, target_train_r2, target_val_r2 = train_test_split(cleaned_final, 

                                                         attr_final, test_size= 0.3,random_state=0);





oversampler=ADASYN(random_state=0)

smote_train_r2, smote_target_r2 = oversampler.fit_sample(train_r2,target_train_r2)
# Random Forest Model

seed = 0   # We set our random seed to zero for reproducibility

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 800,

    'warm_start': True, 

    'max_features': 0.3,

    'max_depth': 9,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}



rf_r2 = RandomForestClassifier(**rf_params)

rf_r2.fit(smote_train_r2, smote_target_r2)

rf_predictions_r2 = rf_r2.predict(test_r2)

ranF_accuracy = accuracy_score(target_val_r2, rf_predictions_r2)

print('Our Random Forest model has an accuracy of {}'.format(round(ranF_accuracy,3)))
# Scatter plot 

trace = go.Scatter(

    y = rf_r2.feature_importances_,

    x = cleaned_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = rf_r2.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = cleaned_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
# Gradient Boosting Parameters

gb_params ={

    'n_estimators': 500,

    'max_features': 0.9,

    'learning_rate' : 0.2,

    'max_depth': 11,

    'min_samples_leaf': 2,

    'subsample': 1,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}



gb = GradientBoostingClassifier(**gb_params)

# Fit the model to our SMOTEd train and target

gb.fit(smote_train_r2, smote_target_r2)

# Get our predictions

gb_predictions_r2 = gb.predict(test_r2)



gb_accuracy = accuracy_score(target_val_r2, gb_predictions_r2)

print('Our Gradient Boosting model has an accuracy of {}'.format(round(gb_accuracy,3)))
# Scatter plot 

trace = go.Scatter(

    y = gb.feature_importances_,

    x = cleaned_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = gb.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = cleaned_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Model Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter')
from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re



decision_tree = tree.DecisionTreeClassifier(max_depth = 4)

decision_tree.fit(train_r2, target_train_r2) #change the training and target data



# Predicting results for test dataset

y_pred = decision_tree.predict(test_r2) #change the test data



# Export our trained model as a .dot file

with open("tree_cleaned.dot", 'w') as f: #change the name of the output

     f = tree.export_graphviz(decision_tree,

                              out_file=f,

                              max_depth = 4,

                              impurity = False,

                              feature_names = cleaned_final.columns.values, #change the df

                              class_names = ['Attrition_No', "Attrition_Yes"],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree_cleaned.dot','-o','tree_cleaned.png']) #change the name of the output



# Annotating chart with PIL

img = Image.open("tree_cleaned.png")#change the name of the output

draw = ImageDraw.Draw(img)#change the name of the output

img.save('tree_cleaned_out.png')#change the name of the output

PImage("tree_cleaned_out.png")#change the name of the output
clf = MultinomialNB()

clf.fit(train_r2, target_train_r2)

clf_predictions = clf.predict(test_r2)

clf_accuracy = accuracy_score(target_val_r2, clf_predictions)

print('Our Classifying Model has an accuracy of {}'.format(round(clf_accuracy,5)))
from pandas_ml import ConfusionMatrix

cm = ConfusionMatrix(target_val_r2, clf_predictions)



cmap = sns.cubehelix_palette(start=0, light=.75, as_cmap=True)

ax = cm.plot(backend='seaborn', cmap=cmap)

ax.set_title("Confusion Matrix - Predicted Attrition")
cm.print_stats()
reduction_df = scaled_num.copy()

red_cat = df_dropped[["Attrition", "Gender","OverTime"]].copy()

red_ord = df_dropped[["EnvironmentSatisfaction",

                          "JobLevel","JobSatisfaction",

                          "StockOptionLevel"]].copy()

red_cat = pd.get_dummies(red_cat)

red_cat = red_cat.drop(['Attrition_Yes', 'Gender_Male','OverTime_No'], axis=1)

red_ord = red_ord.applymap(str)

red_ord = pd.get_dummies(red_ord)

red_ord = red_ord.drop(['EnvironmentSatisfaction_4','EnvironmentSatisfaction_3',

                        'EnvironmentSatisfaction_2','JobLevel_5','JobLevel_4','JobLevel_3',

                        'JobSatisfaction_3', 'JobSatisfaction_2',

                        'JobSatisfaction_4', 'StockOptionLevel_3', 'StockOptionLevel_2'], axis=1)
red_final = pd.concat([reduction_df, red_cat, red_ord], axis=1)

attrition_red = red_final['Attrition_No']

red_final = red_final.drop(['Attrition_No'], axis=1)

red_final.head()
train_red, test_red, target_train_red, target_val_red = train_test_split(red_final, 

                                                         attrition_red, test_size= 0.4,random_state=0);





oversampler=ADASYN(random_state=0)

smote_train_red, smote_target_red = oversampler.fit_sample(train_red,target_train_red)
seed = 0   # We set our random seed to zero for reproducibility

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 800,

    'warm_start': True, 

    'max_features': 0.3,

    'max_depth': 9,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}



rf_red = RandomForestClassifier(**rf_params)

rf_red.fit(smote_train_red, smote_target_red)

rf_predictions_red = rf_red.predict(test_red)

ranF_accuracy = accuracy_score(target_val_red, rf_predictions_red)

print('Our Random Forest model has an accuracy of {}'.format(round(ranF_accuracy,3)))
# Scatter plot 

trace = go.Scatter(

    y = rf_red.feature_importances_,

    x = red_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = rf_red.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = red_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
# Gradient Boosting Parameters

gb_params ={

    'n_estimators': 500,

    'max_features': 0.9,

    'learning_rate' : 0.2,

    'max_depth': 11,

    'min_samples_leaf': 2,

    'subsample': 1,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}



gb = GradientBoostingClassifier(**gb_params)

# Fit the model to our SMOTEd train and target

gb.fit(smote_train_red, smote_target_red)

# Get our predictions

gb_predictions_red = gb.predict(test_red)



gb_accuracy = accuracy_score(target_val_red, gb_predictions_red)

print('Our Gradient Boosting model has an accuracy of {}'.format(round(gb_accuracy,3)))
# Scatter plot 

trace = go.Scatter(

    y = gb.feature_importances_,

    x = red_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = gb.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = red_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Model Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter')
from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re



decision_tree = tree.DecisionTreeClassifier(max_depth = 4)

decision_tree.fit(train_red, target_train_red) #change the training and target data



# Predicting results for test dataset

y_pred = decision_tree.predict(test_red) #change the test data



# Export our trained model as a .dot file

with open("tree_reduced.dot", 'w') as f: #change the name of the output

     f = tree.export_graphviz(decision_tree,

                              out_file=f,

                              max_depth = 4,

                              impurity = False,

                              feature_names = red_final.columns.values, #change the df

                              class_names = ['Attrition', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree_reduced.dot','-o','tree_reduced.png']) #change the name of the output



# Annotating chart with PIL

img = Image.open("tree_reduced.png")#change the name of the output

draw = ImageDraw.Draw(img)#change the name of the output

img.save('tree_reduced_out.png')#change the name of the output

PImage("tree_reduced_out.png")#change the name of the output
classficiation = pd.concat([cat_nominal,cat_ordinal], axis=1)

attr_class = classficiation['Attrition_No']

classficiation = classficiation.drop(['Attrition_No'], axis=1)

classficiation.head()
train_nb, test_nb, target_train_nb, target_val_nb = train_test_split(classficiation, 

                                                         attr_class, test_size= 0.45,random_state=0);



oversampler=ADASYN(random_state=0)

smote_train_nb, smote_target_nb = oversampler.fit_sample(train_nb,target_train_nb)
clf = MultinomialNB()

clf.fit(smote_train_nb, smote_target_nb)

clf_predictions = clf.predict(test_nb)

clf_accuracy = accuracy_score(target_val_nb, clf_predictions)

print('Our Classifying Model has an accuracy of {}'.format(round(clf_accuracy,5)))



from pandas_ml import ConfusionMatrix

cm = ConfusionMatrix(target_val_nb, clf_predictions)

print(cm)

cm.print_stats()
# Analysis of job related categories

job_df = df[['JobInvolvement','JobLevel','JobRole','JobSatisfaction', 'Attrition']].copy()

job_df.describe()
# Encoding 'JobRole' to numerical labels for further analysis

le = preprocessing.LabelEncoder()

job_df['JobRole'] = le.fit_transform(job_df['JobRole'])

job_df['Attrition'] = le.fit_transform(job_df['Attrition'])
# Plotting the KDEplots

f, axes = plt.subplots(1, 3, figsize=(10, 4), 

                       sharex=False, sharey=False)



# Defining our colormap scheme

s = np.linspace(0, 1, 10)

cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)



# Generate and plot

y = job_df['JobInvolvement'].values

x = job_df['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0])

axes[0].set( title = 'Job Involvement vs Job Satisfaction')



cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)

# Generate and plot

y = job_df['JobLevel'].values

x = job_df['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[1])

axes[1].set( title = 'Job Level vs Job Satisfaction')



cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)

# Generate and plot

y = job_df['JobRole'].values

x = job_df['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[2])

axes[2].set( title = 'Job Role vs Job Satisfaction')





f.tight_layout()