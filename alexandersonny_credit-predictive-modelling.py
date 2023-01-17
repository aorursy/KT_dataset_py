import pandas as pd

import numpy as np

import seaborn as sns

import datetime as dt



import warnings

warnings.filterwarnings('ignore')



from collections import Counter



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.offline as pyo

import plotly.tools as tls

import plotly.graph_objs as go

pyo.init_notebook_mode(connected= True)



pd.options.display.max_columns = 100
df_ucicreditcard = pd.read_csv('../input/UCI_Credit_Card.csv')
df_ucicreditcard.info()
df_ucicreditcard.head()
df_ucicreditcard.drop(['ID'], axis=1, inplace = True)

df_ucicreditcard.rename(columns = {'default.payment.next.month': 'default_payment_next_month', 'PAY_0':'PAY_1'}, inplace = True)
df_ucicreditcard.shape
df_ucicreditcard.describe().T
df_ucicreditcard.isnull().sum()
df_ucicreditcard_numerical_columns = df_ucicreditcard.columns

df_ucicreditcard_numerical_columns = ['LIMIT_BAL', 'AGE', 'PAY_1', 

             'PAY_2', 'PAY_3', 'PAY_4', 

             'PAY_5', 'PAY_6', 'BILL_AMT1', 

             'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 

             'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 

             'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 

             'PAY_AMT5', 'PAY_AMT6']

df_ucicreditcard_numerical_columns

print("Sex : ", df_ucicreditcard.SEX.unique())

print("Education : ", df_ucicreditcard.EDUCATION.unique())

print("Marriage : ", df_ucicreditcard.MARRIAGE.unique())
df_ucicreditcard.default_payment_next_month.value_counts()
plt.title('Default Next Month Payment')

ax = sns.countplot(x = df_ucicreditcard.default_payment_next_month ,palette="Set2")

sns.set(font_scale= 1.0)

ax.set_ylim(top = 30000)

ax.set_xticklabels(['No Default','Default'])

ax.set_xlabel('Next Month Default Payment')

ax.set_ylabel('Frequency')

fig = plt.gcf()

fig.set_size_inches(10,5)



plt.show()
#First plot

trace0 = go.Bar(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['SEX'].value_counts().index.values,

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['SEX'].value_counts().values,

    name = 'No Default'

)



#First plot 2

trace1 = go.Bar(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['SEX'].value_counts().index.values,

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['SEX'].value_counts().values,

    name = 'Default'

)



#Second plot

trace2 = go.Box(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['SEX'],

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['LIMIT_BAL'],

    name = trace0.name,

    boxmean = True

)



#Second plot 2

trace3 = go.Box(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['SEX'],

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['LIMIT_BAL'],

    name = trace1.name,

    boxmean = True

)



data = [trace0, trace1, trace2,trace3]





fig = tls.make_subplots(rows= 1, cols= 2, 

                        subplot_titles= ('Sex Count', 'Credit Amount by Sex'))



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 2)



fig['layout'].update(height= 400, width= 800, title= 'Sex Distribution', boxmode= 'group')

pyo.iplot(fig, filename= 'sex-subplot')
default_ratio = pd.pivot_table(df_ucicreditcard,

                               columns= 'default_payment_next_month',

                               values= 'AGE',

                               index= 'SEX',

                               aggfunc= 'count')



default_ratio.reset_index(inplace= True)

default_ratio.columns = ['Sex', 'No Default', 'Default']



total = default_ratio['No Default'] + default_ratio['Default']

default_ratio['Total'] = total



default_ratio['No Default ratio'] = default_ratio['No Default'] / total

default_ratio['Default ratio'] = default_ratio['Default'] / total

default_ratio
#First plot

trace0 = go.Bar(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['EDUCATION'].value_counts().index.values,

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['EDUCATION'].value_counts().values,

    name = 'No Default'

)



#First plot 2

trace1 = go.Bar(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['EDUCATION'].value_counts().index.values,

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['EDUCATION'].value_counts().values,

    name = 'Default'

)



#Second plot

trace2 = go.Box(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['EDUCATION'],

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['LIMIT_BAL'],

    name = trace0.name,

    boxmean = True

)



#Second plot 2

trace3 = go.Box(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['EDUCATION'],

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['LIMIT_BAL'],

    name = trace1.name,

    boxmean = True

)



data = [trace0, trace1, trace2,trace3]





fig = tls.make_subplots(rows= 1, cols= 2, 

                        subplot_titles= ('Education Distribution', 'Credit Amount by Education'))



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 2)



fig['layout'].update(height= 400, width= 1000, title= 'Education Distribution', boxmode= 'group')

pyo.iplot(fig, filename= 'education-subplot')
#First plot

trace0 = go.Bar(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['MARRIAGE'].value_counts().index.values,

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['MARRIAGE'].value_counts().values,

    name = 'No Default'

)



#First plot 2

trace1 = go.Bar(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['MARRIAGE'].value_counts().index.values,

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['MARRIAGE'].value_counts().values,

    name = 'Default'

)



#Second plot

trace2 = go.Box(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['MARRIAGE'],

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 0]['LIMIT_BAL'],

    name = trace0.name,

    boxmean = True

)



#Second plot 2

trace3 = go.Box(

    x = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['MARRIAGE'],

    y = df_ucicreditcard[df_ucicreditcard['default_payment_next_month'] == 1]['LIMIT_BAL'],

    name = trace1.name,

    boxmean = True

)



data = [trace0, trace1, trace2,trace3]





fig = tls.make_subplots(rows= 1, cols= 2, 

                        subplot_titles= ('Marriage Distribution', 'Credit Amount by Marital status'))



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 2)



fig['layout'].update(height= 400, width= 1000, title= 'Marriage Distribution', boxmode= 'group')

pyo.iplot(fig, filename= 'marriage-subplot')
def plot_histogram(data, cols, bins = 10, hist = True, norm_hist=False):

    for col in cols:

        fig = plt.figure(figsize = (7,3))

        sns.set_style('whitegrid')

        sns.distplot(a=data[col].dropna(), hist = hist)

        plt.title('Histogram of ' + col, fontweight='bold', size=11)

        plt.xlabel(col, size=10)

        plt.ylabel('Frequency Density', size=10)

        plt.show



plot_histogram(df_ucicreditcard, df_ucicreditcard_numerical_columns)
df_good = df_ucicreditcard.loc[df_ucicreditcard['default_payment_next_month'] == 0]['AGE'].values.tolist()

df_bad = df_ucicreditcard.loc[df_ucicreditcard['default_payment_next_month'] == 1]['AGE'].values.tolist()

df_age = df_ucicreditcard['AGE'].values.tolist()



#First plot

trace0 = go.Histogram(

    x= df_good,

    histnorm= 'probability',

    name= 'No Default', 

    marker= dict(

                    color= 'rgb(61,145,64)'

                ) 

)

#Second plot

trace1 = go.Histogram(

    x= df_bad,

    histnorm= 'probability',

    name= 'Default',

    marker= dict(

                    color= 'rgb(179,27,27)'

                ) 

    

)

#Third plot

trace2 = go.Histogram(

    x= df_age,

    histnorm= 'probability',

    name= 'Overall Age',

    marker= dict(

                    color= 'rgb(0,72,186)'

                ) 

)



#Creating the grid

fig = tls.make_subplots(rows= 2, cols= 2, specs= [[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('No Default', 'Default', 'Overall Age Distribution'))



#setting the figs

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)



fig['layout'].update(showlegend= True, title= 'Age Distribution', bargap=0.05)

pyo.iplot(fig, filename= 'Age Distribution subplot')
df_ucicreditcard_numerical_columns_with_y = ['LIMIT_BAL', 'AGE', 'PAY_1', 

             'PAY_2', 'PAY_3', 'PAY_4', 

             'PAY_5', 'PAY_6', 'BILL_AMT1', 

             'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 

             'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 

             'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 

             'PAY_AMT5', 'PAY_AMT6', 'default_payment_next_month']

df_ucicreditcard_num =  df_ucicreditcard[df_ucicreditcard_numerical_columns_with_y]



plt.figure(figsize=(50,30))

sns.set(font_scale=2.0)

sns.heatmap(df_ucicreditcard_num.corr(),annot=True)
df_ucicreditcard_numerical_columns_bill_amt = ['BILL_AMT1', 

             'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 

             'BILL_AMT5', 'BILL_AMT6', 'default_payment_next_month']

df_ucicreditcard_num_bill =  df_ucicreditcard[df_ucicreditcard_numerical_columns_bill_amt]



plt.figure(figsize=(50,30))

sns.set(font_scale=2.0)

sns.heatmap(df_ucicreditcard_num_bill.corr(),annot=True)
#Drop the unnecessary features for BILL_AMT



df_ucicreditcard = df_ucicreditcard.drop(['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 

             'BILL_AMT5', 'BILL_AMT6'], axis=1)
def detect_outliers(df,n,features):

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers



# These are the numerical features present in the dataset

Outliers_to_drop = detect_outliers(df_ucicreditcard,2,['LIMIT_BAL',

                                                 'PAY_1',

                                                 'PAY_2',

                                                 'PAY_3',

                                                 'PAY_4',

                                                 'PAY_5',

                                                 'PAY_6',

                                                 'BILL_AMT1',

                                                 'PAY_AMT1',

                                                 'PAY_AMT2',

                                                 'PAY_AMT3',

                                                 'PAY_AMT4',

                                                 'PAY_AMT5',

                                                 'PAY_AMT6'])
df_ucicreditcard.loc[Outliers_to_drop]
edu_replacement = (df_ucicreditcard.EDUCATION == 5) | (df_ucicreditcard.EDUCATION == 6) | (df_ucicreditcard.EDUCATION == 0)

df_ucicreditcard.loc[edu_replacement, 'EDUCATION'] = 4

df_ucicreditcard.EDUCATION.value_counts()
df_ucicreditcard.loc[df_ucicreditcard.MARRIAGE == 0, 'MARRIAGE'] = 3

df_ucicreditcard.MARRIAGE.value_counts()
#StandardScaller is being used to normalize the features

from sklearn.preprocessing import StandardScaler



standardized_features = ['LIMIT_BAL', 'AGE', 'PAY_1',  'PAY_2',  'PAY_3',  'PAY_4', 'PAY_5', 'PAY_6',

                         'BILL_AMT1', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']



numeric_features = df_ucicreditcard[standardized_features]

sc = StandardScaler()

standardized = pd.DataFrame(sc.fit_transform(numeric_features))

standardized.columns = ['LIMIT_BAL',

                         'AGE',

                         'PAY_1',

                         'PAY_2',

                         'PAY_3',

                         'PAY_4',

                         'PAY_5',

                         'PAY_6',

                         'BILL_AMT1',

                         'PAY_AMT1',

                         'PAY_AMT2',

                         'PAY_AMT3',

                         'PAY_AMT4',

                         'PAY_AMT5',

                         'PAY_AMT6']



df_ucicreditcard_stdized = df_ucicreditcard.copy()

df_ucicreditcard_stdized[standardized_features] = standardized

df_ucicreditcard_stdized.head()
df_ucicreditcard_stdized.head()
df_ucicreditcard_stdized = pd.get_dummies(df_ucicreditcard_stdized, columns = ["SEX"], prefix= "SEX")

df_ucicreditcard_stdized = pd.get_dummies(df_ucicreditcard_stdized, columns = ["EDUCATION"], prefix= "EDUCATION")

df_ucicreditcard_stdized = pd.get_dummies(df_ucicreditcard_stdized, columns = ["MARRIAGE"], prefix= "MARRIAGE")
df_ucicreditcard_stdized.sample(10)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, make_scorer, roc_curve

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
# Prepare the features to predict Y

features = ['LIMIT_BAL', 

            'AGE', 

            'PAY_1', 

            'PAY_2',

            'PAY_3', 

            'PAY_4', 

            'PAY_5', 

            'PAY_6', 

            'BILL_AMT1', 

            'PAY_AMT1',

            'PAY_AMT2', 

            'PAY_AMT3', 

            'PAY_AMT4', 

            'PAY_AMT5', 

            'PAY_AMT6', 

            'SEX_1', 

            'SEX_2', 

            'EDUCATION_1', 'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'MARRIAGE_1', 'MARRIAGE_2', 'MARRIAGE_3']



X = df_ucicreditcard_stdized[features].copy()

X.columns
# Prepare the target variable as Y

y = df_ucicreditcard_stdized['default_payment_next_month'].copy()

y.sample(5)
from sklearn.model_selection import train_test_split

# Split train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.30, 

                                                    shuffle = True, 

                                                    random_state = 5)



print("Number of X_train dataset: ", X_train.shape)

print("Number of y_train dataset: ", y_train.shape)

print("Number of X_test dataset: ", X_test.shape)

print("Number of y_test dataset: ", y_test.shape)
print('Before Over-sampling, the shape of train_X: {}'.format(X_train.shape))

print('Before Over-sampling, the shape of train_y: {} \n'.format(y_train.shape))



print("Before Over-sampling, counts of label '1': {}".format(sum(y_train == 1)))

print("Before Over-sampling, counts of label '0': {}".format(sum(y_train == 0)))
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state= 2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)



print("Number of X_train_res dataset: ", X_train_res.shape)

print("Number of y_train_res dataset: ", y_train_res.shape)
print('After Over-sampling, the shape of train_X_res: {}'.format(X_train_res.shape))

print('After Over-sampling, the shape of train_y_res: {} \n'.format(y_train_res.shape))



print("After Over-sampling, counts of label '1': {}".format(sum(y_train_res == 1)))

print("After Over-sampling, counts of label '0': {}".format(sum(y_train_res == 0)))
# Create Decision Tree model before optimization 

clf_DT_before_opt = DecisionTreeClassifier(random_state= 0)



# Apply the model

clf_DT_before_opt.fit(X_train, y_train)



# Predicted value

y_pred_DT_before_opt = clf_DT_before_opt.predict(X_test)
TP = np.sum(np.logical_and(y_pred_DT_before_opt == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_DT_before_opt == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_DT_before_opt == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_DT_before_opt == 0, y_test == 1))



#Overall Classification Report

print(classification_report(y_test, y_pred_DT_before_opt))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test,y_pred_DT_before_opt).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_DT_before_opt).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_DT_before_opt).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_DT_before_opt).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_DT_before_opt).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
# define the parameters grid

param_grid = {'max_depth': np.arange(3, 10),

              'criterion' : ['gini','entropy'],

              'max_leaf_nodes': [5,10,20,100],

              'min_samples_split': [2, 5, 10, 20]}



# create the grid

grid_DT_opt = GridSearchCV(DecisionTreeClassifier(), param_grid, cv= 5, scoring= 'recall')



#training

grid_DT_opt.fit(X_train, y_train)

#let's see the best estimator

print(grid_DT_opt.best_estimator_)

#with its score

print(np.abs(grid_DT_opt.best_score_))
clf_DT = DecisionTreeClassifier(class_weight= None, criterion= 'entropy', max_depth= 3,

            max_features= None, max_leaf_nodes=20,

            min_impurity_decrease= 0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf= 0.0, presort= False, random_state= None,

            splitter='best')



clf_DT.fit(X_train, y_train)



y_pred_DT = clf_DT.predict(X_test)



TP = np.sum(np.logical_and(y_pred_DT == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_DT == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_DT == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_DT == 0, y_test == 1))



#Overall Classification Report

print(classification_report(y_test, y_pred_DT))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_DT).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_DT).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_DT).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_DT).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_DT).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
# define the parameters grid

param_grid = {'max_depth': np.arange(3, 10),

              'criterion' : ['gini','entropy'],

              'max_leaf_nodes': [5,10,20,100],

              'min_samples_split': [2, 5, 10, 20]}



# create the grid

grid_DT_res = GridSearchCV(DecisionTreeClassifier(), param_grid, cv= 5, scoring= 'recall')



#training

grid_DT_res.fit(X_train_res, y_train_res)

#let's see the best estimator

print(grid_DT_res.best_estimator_)

#with its score

print(np.abs(grid_DT_res.best_score_))
clf_DT_res = DecisionTreeClassifier(class_weight= None, criterion= 'gini', max_depth= 9,

            max_features= None, max_leaf_nodes=100,

            min_impurity_decrease= 0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf= 0.0, presort= False, random_state= None,

            splitter='best')



clf_DT_fit_res = clf_DT_res.fit(X_train_res, y_train_res)



y_pred_DT_res = clf_DT_res.predict(X_test)



TP = np.sum(np.logical_and(y_pred_DT_res == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_DT_res == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_DT_res == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_DT_res == 0, y_test == 1))



#Overall Classification Report

print(classification_report(y_test, y_pred_DT_res))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_DT_res).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_DT_res).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_DT_res).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_DT_res).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_DT_res).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))

cm = pd.crosstab(y_test.values, y_pred_DT_res, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.set(font_scale=1.0)

sns.heatmap(cm, 

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues", fmt='g')

plt.title('Confusion Matrix', fontsize= 10)

plt.show()
#Predicting proba

y_pred_prob_DT = clf_DT_fit_res.predict_proba(X_test)[:,1]





sns.set(font_scale=1.0)

# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_DT)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve', fontsize= 10)

plt.show()
features_DT = pd.DataFrame()

features_DT['feature'] = X_train.columns

features_DT['importance'] = clf_DT_fit_res.feature_importances_

features_DT.sort_values(by=['importance'], ascending=True, inplace=True)

features_DT.set_index('feature', inplace=True)



features_DT.plot(kind='barh', figsize=(20, 20))
from sklearn.linear_model import LogisticRegression
clf_LR_before_opt = LogisticRegression()



clf_LR_before_opt.fit(X_train, y_train)



y_pred_LR_before_opt = clf_LR_before_opt.predict(X_test)



TP = np.sum(np.logical_and(y_pred_LR_before_opt == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_LR_before_opt == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_LR_before_opt == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_LR_before_opt == 0, y_test == 1))



#Overall Classification Report

print(classification_report(y_test, y_pred_LR_before_opt))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_LR_before_opt).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_LR_before_opt).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_LR_before_opt).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_LR_before_opt).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_LR_before_opt).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
# define the parameters grid

tol = [0.01, 0.001, 0.0001]

max_iter = [100, 150, 200]

param_grid = {'tol': tol,

              'max_iter' : max_iter}



# create the grid

grid_LR = GridSearchCV(LogisticRegression(), param_grid, cv= 5, scoring= 'recall')



#training

grid_LR.fit(X_train, y_train)

#let's see the best estimator

print(grid_LR.best_estimator_)

#with its score

print(np.abs(grid_LR.best_score_))
clf_LR_tuned = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='warn', n_jobs=None, penalty='l2',

                   random_state=None, solver='warn', tol=0.001, verbose=0,

                   warm_start=False)



clf_LR_fit = clf_LR_tuned.fit(X_train, y_train)



y_pred_LR = clf_LR_tuned.predict(X_test)



TP = np.sum(np.logical_and(y_pred_LR == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_LR == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_LR == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_LR == 0, y_test == 1))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_LR).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_LR).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_LR).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_LR).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_LR).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
# define the parameters grid

tol = [0.01, 0.001, 0.0001]

max_iter = [100, 150, 200]

param_grid = {'tol': tol,

              'max_iter' : max_iter}



# create the grid

grid_LR_res = GridSearchCV(LogisticRegression(), param_grid, cv= 5, scoring= 'recall')



#training

grid_LR_res.fit(X_train_res, y_train_res)

#let's see the best estimator

print(grid_LR_res.best_estimator_)

#with its score

print(np.abs(grid_LR_res.best_score_))
clf_LR_res_tuned = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='warn', n_jobs=None, penalty='l2',

                   random_state=None, solver='warn', tol=0.01, verbose=0,

                   warm_start=False)



clf_LR_fit_res = clf_LR_res_tuned.fit(X_train_res, y_train_res)



y_pred_LR_res = clf_LR_res_tuned.predict(X_test)



TP = np.sum(np.logical_and(y_pred_LR_res == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_LR_res == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_LR_res == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_LR_res == 0, y_test == 1))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_LR_res).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_LR_res).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_LR_res).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_LR_res).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_LR_res).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
cm_LR = pd.crosstab(y_test.values, y_pred_LR_res, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm_LR, 

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues", fmt='g')

plt.title('Confusion Matrix', fontsize=14)

plt.show()
#Predicting proba

y_pred_proba_LR = clf_LR_fit_res.predict_proba(X_test)[:,1]



# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_LR)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
clf_RF = RandomForestClassifier(random_state = 0)

clf_RF_fit = clf_RF.fit(X_train_res, y_train_res)



y_pred_RF = clf_RF_fit.predict(X_test)
TP = np.sum(np.logical_and(y_pred_RF == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_RF == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_RF == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_RF == 0, y_test == 1))



#Overall Classification Report

print(classification_report(y_test, y_pred_RF))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_RF).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_RF).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_RF).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_RF).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_RF).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
#Setting the Hyper Parameters

param_grid = {'max_depth': [3, 5, 7, 10],

              'n_estimators':[3, 5, 10, 25, 50, 100],

              'max_features': ['auto', 'sqrt', 4, 7, 15, 20], 

              'bootstrap': [True, False],

              'criterion': ['gini', 'entropy']

             }



grid_RF_res = GridSearchCV(RandomForestClassifier(), param_grid= param_grid, cv=5, scoring= 'recall', verbose= 4)

grid_RF_res.fit(X_train_res, y_train_res)



#let's see the best estimator

print(grid_RF_res.best_estimator_)

#with its score

print(np.abs(grid_RF_res.best_score_))
clf_RF_res_tuned = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=10, max_features=20, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=25,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)



clf_RF_fit_res_tuned = clf_RF_res_tuned.fit(X_train_res, y_train_res)



y_pred_RF_res_tuned = clf_RF_res_tuned.predict(X_test)



TP = np.sum(np.logical_and(y_pred_RF_res_tuned == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_RF_res_tuned == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_RF_res_tuned == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_RF_res_tuned == 0, y_test == 1))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_RF_res_tuned).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_RF_res_tuned).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_RF_res_tuned).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_RF_res_tuned).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_RF_res_tuned).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
cm_LR = pd.crosstab(y_test.values, y_pred_RF_res_tuned, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm_LR, 

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues", fmt='g')

plt.title('Confusion Matrix', fontsize=14)

plt.show()
#Predicting proba

y_pred_prob = clf_RF_fit_res_tuned.predict_proba(X_test)[:,1]



# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
features_RF = pd.DataFrame()

features_RF['feature'] = X_train.columns

features_RF['importance'] = clf_RF_fit_res_tuned.feature_importances_

features_RF.sort_values(by=['importance'], ascending=True, inplace=True)

features_RF.set_index('feature', inplace=True)



features_RF.plot(kind='barh', figsize=(20, 20))
clf_SVC_before_opt = SVC()



clf_SVC_before_opt.fit(X_train_res, y_train_res)



y_pred_SVC_before_opt = clf_SVC_before_opt.predict(X_test)
TP = np.sum(np.logical_and(y_pred_SVC_before_opt == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_SVC_before_opt == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_SVC_before_opt == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_SVC_before_opt == 0, y_test == 1))



#Overall Classification Report

print(classification_report(y_test, y_pred_SVC_before_opt))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_SVC_before_opt).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_SVC_before_opt).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_SVC_before_opt).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_SVC_before_opt).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_SVC_before_opt).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
# define the parameters grid

C= [0.123,0.124, 0.125, 0.126, 0.127]

kernel = ['linear','rbf','poly']

gamma = [0, 0.0000000000001, 0.000000000001, 0.00000000001]



random_grid_svm = {'C': C,

                   'kernel': kernel,

                   'gamma': gamma}

# create the grid

grid_LR = GridSearchCV(SVC(), param_grid= random_grid_svm, cv=5, scoring= 'recall', verbose= 4)



#training

grid_LR.fit(X_train_res, y_train_res)

#let's see the best estimator

print(grid_LR.best_estimator_)

#with its score

print(np.abs(grid_LR.best_score_))
clf_SVC_res_tuned = SVC(C=0.123, cache_size=200, class_weight=None, coef0=0.0,

                        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1,

                        probability=True, random_state=None, shrinking=True, tol=0.001,

                        verbose=False)



clf_SVC_fit_res_tuned = clf_SVC_res_tuned.fit(X_train_res, y_train_res)



y_pred_SVC_res_tuned = clf_SVC_res_tuned.predict(X_test)



TP = np.sum(np.logical_and(y_pred_SVC_res_tuned == 1, y_test == 1))

TN = np.sum(np.logical_and(y_pred_SVC_res_tuned == 0, y_test == 0))

FP = np.sum(np.logical_and(y_pred_SVC_res_tuned == 1, y_test == 0))

FN = np.sum(np.logical_and(y_pred_SVC_res_tuned == 0, y_test == 1))



# Accuracy Score

print('Accuracy score is ', accuracy_score(y_test, y_pred_SVC_res_tuned).round(2))



# Precision Score

print('Precision score is ', precision_score(y_test, y_pred_SVC_res_tuned).round(2))



# Recall Score

print('Recall_score is ', recall_score(y_test, y_pred_SVC_res_tuned).round(2))



# F1 Score

print('F1 score is ', f1_score(y_test, y_pred_SVC_res_tuned).round(2))



# ROC_AUC

print('ROC AUC is ', roc_auc_score(y_test, y_pred_SVC_res_tuned).round(2))



# Specificity Ratio

print('Specificity ratio: {}'.format(round(TN/(TN+FP),2)))



# False Negative Ratio

print('False Negative ratio: {}'.format(round(FN/(FN+TP),2)))
cm_svm = pd.crosstab(y_test.values, y_pred_SVC_res_tuned, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm_svm, 

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues", fmt='g')

plt.title('Confusion Matrix', fontsize=14)

plt.show()
#Predicting proba

y_SVC_pred_prob = clf_SVC_fit_res_tuned.predict_proba(X_test)[:,1]



# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_SVC_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()