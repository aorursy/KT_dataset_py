import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

%matplotlib inline

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier
#setting up for customized printing

from IPython.display import Markdown, display

from IPython.display import HTML

def printmd(string, color=None):

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))

    

#function to display dataframes side by side    

from IPython.display import display_html

def display_side_by_side(args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline;margin-left:50px !important;margin-right: 40px !important"'),raw=True)
def distplot(figRows,figCols,xSize, ySize, data, features, colors):

    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    

    features = np.array(features).reshape(figRows, figCols)

    colors = np.array(colors).reshape(figRows, figCols)

    

    for row in range(figRows):

        for col in range(figCols):

            if (figRows == 1 and figCols == 1) :

                axesplt = axes

            elif (figRows == 1 and figCols > 1) :

                axesplt = axes[col]

            elif (figRows > 1 and figCols == 1) :

                axesplt = axes[row]

            else:

                axesplt = axes[row][col]

            plot = sns.distplot(data[features[row][col]], color=colors[row][col], ax=axesplt, kde=True, hist_kws={"edgecolor":"k"})

            plot.set_xlabel(features[row][col],fontsize=20)
def boxplot(figRows,figCols,xSize, ySize, data,features, colors=None, palette=None, hue=None, orient='h', rotation=30):

    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    

    features = np.array(features).reshape(figRows, figCols)

    if(palette is None):

        colors = np.array(colors).reshape(figRows, figCols)

    

    for row in range(figRows):

        for col in range(figCols):

            if (figRows == 1 and figCols == 1) :

                axesplt = axes

            elif (figRows == 1 and figCols > 1) :

                axesplt = axes[col]

            elif (figRows > 1 and figCols == 1) :

                axesplt = axes[row]

            else:

                axesplt = axes[row][col]

            if(palette is None):

                plot = sns.boxplot(features[row][col], data= data, color=colors[row][col], ax=axesplt, orient=orient, hue=hue)

            else:

                plot = sns.boxplot(features[row][col], data= data, ax=axesplt, orient=orient, hue=hue)

            plot.set_ylabel('',fontsize=20)

            plot.set_xticklabels(rotation=rotation, labels=[features[row][col]], fontweight='demibold',fontsize='large')
def boxplot_all(xSize, ySize, data, palette):

    f, axes = plt.subplots(1, 1, figsize=(xSize, ySize))

    plot = sns.boxplot(x='variable',y='value', data= pd.melt(data), palette='Set1', ax=axes, orient='v')

    plot.set_xlabel('',fontsize=20)

    plot.set_xticklabels(rotation=60, labels=plot.get_xticklabels(),fontweight='demibold',fontsize='x-large')
def countplot(figRows,figCols,xSize, ySize, data, features, colors=None,palette=None,hue=None, orient=None, rotation=90):

    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    

    features = np.array(features).reshape(figRows, figCols)

    if(colors is not None):

        colors = np.array(colors).reshape(figRows, figCols)

    if(palette is not None):

        palette = np.array(palette).reshape(figRows, figCols)

    

    for row in range(figRows):

        for col in range(figCols):

            if (figRows == 1 and figCols == 1) :

                axesplt = axes

            elif (figRows == 1 and figCols > 1) :

                axesplt = axes[col]

            elif (figRows > 1 and figCols == 1) :

                axesplt = axes[row]

            else:

                axesplt = axes[row][col]

                

            if(colors is None):

                plot = sns.countplot(features[row][col], data=data, palette=palette[row][col], ax=axesplt, orient=orient, hue=hue)

            elif(palette is None):

                plot = sns.countplot(features[row][col], data=data, color=colors[row][col], ax=axesplt, orient=orient, hue=hue)

            plot.set_title(features[row][col],fontsize=20)

            plot.set_xlabel(None)

            plot.set_xticklabels(rotation=rotation, labels=plot.get_xticklabels(),fontweight='demibold',fontsize='large')

            
def heatmap(xSize, ySize, data, palette = 'YlGnBu', fmt='.2f', lineColor='white', lineWidths=0.3, square = True,upper=False, rotation=60):

    f, axes = plt.subplots(1, 1, figsize=(xSize, ySize))

    if(not upper):        

        cor_mat = data.corr()

        hmapData = cor_mat

    else:

        cor_mat_abs = data.corr().abs()

        upperHalf = cor_mat_abs.where(np.triu(np.ones(cor_mat_abs.shape), k=1).astype(np.bool))

        hmapData = upperHalf        

    sns.heatmap(hmapData,cmap=palette, annot=True, fmt=fmt, ax=axes, linecolor=lineColor, linewidths=lineWidths, square=square)

    plt.xticks(rotation=rotation)
def catdist(cols, data):

    dfs = []

    for col in cols:

        colData = pd.DataFrame(data[col].value_counts(), columns=[col])

        colData['%'] = round((colData[col]/colData[col].sum())*100,2)

        dfs.append(colData)

    display_side_by_side(dfs)
def scatterplot(xSize, ySize, data,x,y, palette, hue=None,size=None, sizes=(40,200),alpha=1):

    f, axes = plt.subplots(1, 1, figsize=(xSize, ySize))

    if(size is None):

        plot = sns.scatterplot(x,y, data= data, palette=palette, ax=axes, hue=hue, alpha=alpha)

    else:

        plot = sns.scatterplot(x,y, data= data, palette=palette, ax=axes, hue=hue, size=size, sizes=sizes, alpha=alpha, legend='full')

    plot.set_ylabel(y,fontsize=20)

    plot.set_xlabel(x,fontsize=20)
def scatter_box_plot(xSize, ySize, x,y,data, palette, hue=None, orient='h', rotation=30):

    f, axes = plt.subplots(1, 2, figsize=(xSize, ySize))

    splot = sns.scatterplot(x=x,y=y, data= data, palette=palette, ax=axes[0], hue=hue)

    splot.set_ylabel(y,fontsize=20)

    splot.set_xlabel(x,fontsize=20)

    bplot = sns.boxplot(x=x,y=y, data= data, palette=palette, ax=axes[1], orient=orient, hue=hue)

    bplot.set_ylabel('',fontsize=20)

    bplot.set_xlabel(x,fontsize=20)

    #bplot.set_xticklabels(rotation=rotation, labels=[features[row][col]], fontweight='demibold',fontsize='medium')
def point_bar_plot(row, col, data, hue, figRow, figCol, palette='rocket', fontsize='large', fontweight='demibold'):

    sns.set(style="whitegrid")

    f, axes = plt.subplots(2, 1, figsize=(figRow, figCol))

    pplot=sns.pointplot(row,col, data=data, ax=axes[0], linestyles=['--'])

    pplot.set_xlabel(None)

    pplot.set_xticklabels(labels=pplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)    

    bplot=sns.barplot(row,col, data=data, hue=hue, ax=axes[1],palette=palette)

    bplot.set_xlabel(row,fontsize=20)

    bplot.set_xticklabels(labels=bplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)
titanic = pd.read_csv('../input/titanic/train.csv')

titanic_test = pd.read_csv('../input/titanic/test.csv')

passengerid = titanic_test['PassengerId']

titanic.head()
print('The total number of rows :', titanic.shape[0])

print('The total number of columns :', titanic.shape[1])
#continuous - Age, Fare

#cat - Survived, Pclass, Sex, Embarked

titanic.info()
display(titanic.isna().sum())

print('======================================')

printmd('**CONCLUSION**: As seen from the data above, we conclude there are **"Missing"** values in the data', color="red")



display(titanic_test.isna().sum())

print('======================================')

printmd('**CONCLUSION**: As seen from the data above, we conclude there are **"Missing"** values in the data', color="red")
titanic.describe().transpose()
titanic.drop('PassengerId', axis=1, inplace=True)

titanic.drop('Ticket', axis=1, inplace=True)

titanic.drop('Cabin', axis=1, inplace=True)

titanic.drop('Name', axis=1, inplace=True)



titanic_test.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1, inplace=True)
from sklearn.impute import SimpleImputer



impute = SimpleImputer(missing_values=np.nan, strategy='median')

transformed = impute.fit_transform(titanic[['Age']])

titanic.Age = transformed



transformed = impute.fit_transform(titanic_test[['Age']])

titanic_test.Age = transformed
display(titanic.Age.isna().sum())

display(titanic_test.Age.isna().sum())
titanic[['Age']].describe().transpose()
catdist(['Embarked'], titanic)

catdist(['Embarked'], titanic_test)
titanic.Embarked.fillna('Q', inplace=True)
catdist(['Embarked'], titanic)
distplot(1,2,15,7, titanic, ['Age','Fare'], ['red', 'blue'])
distplot(1,1,15,7, titanic, ['SibSp'], ['green'])
boxplot(2, 1, 20, 8, orient='h', data=titanic, features=['Age','Fare',], colors=['green','brown'])
titanic[['Age','Fare']].skew()
catdist(['Survived','Pclass', 'Sex','Embarked', 'Parch', 'SibSp'], titanic)
countplot(2,3,20,15,data=titanic,features=['Survived','Pclass', 'Sex','Embarked', 'Parch', 'SibSp'], palette=['Set1', 'Dark2', 'Paired', 'viridis','afmhot','tab10'], rotation=0)
f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.boxplot('Survived', 'Age', 'Sex', data=titanic, orient='v', palette='Set1_r', ax=axes[0])

sns.violinplot('Survived', 'Age', 'Sex',data=titanic, orient='v', palette='Set3', ax=axes[0])

sns.swarmplot('Survived', 'Age', 'Sex',data=titanic, orient='v', palette='Set1', ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.boxplot('Survived', 'Age', 'Pclass', data=titanic, orient='v', palette='Set1_r', ax=axes[0])

sns.violinplot('Survived', 'Age', 'Pclass',data=titanic, orient='v', palette='Set3', ax=axes[0])

sns.swarmplot('Survived', 'Age', 'Pclass',data=titanic, orient='v', palette='Set1', ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.boxplot('Survived', 'Age', 'Embarked', data=titanic, orient='v', palette='Set1_r', ax=axes[0])

sns.violinplot('Survived', 'Age', 'Embarked',data=titanic, orient='v', palette='Set3', ax=axes[0])

sns.swarmplot('Survived', 'Age', 'Embarked',data=titanic, orient='v', palette='Set1', ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.boxplot('Survived', 'Fare', 'Sex', data=titanic, orient='v', palette='Set1_r', ax=axes[0])

sns.violinplot('Survived', 'Fare', 'Sex',data=titanic, orient='v', palette='Set3', ax=axes[0])

sns.swarmplot('Survived', 'Fare', 'Sex',data=titanic, orient='v', palette='Set1', ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.boxplot('Survived', 'Fare', 'Pclass', data=titanic, orient='v', palette='Set1_r', ax=axes[0])

sns.violinplot('Survived', 'Fare', 'Pclass',data=titanic, orient='v', palette='Set3', ax=axes[0])

sns.swarmplot('Survived', 'Fare', 'Pclass',data=titanic, orient='v', palette='Set1', ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.boxplot('Survived', 'Fare', 'Embarked', data=titanic, orient='v', palette='Set1_r', ax=axes[0])

sns.violinplot('Survived', 'Fare', 'Embarked',data=titanic, orient='v', palette='Set3', ax=axes[0])

sns.swarmplot('Survived', 'Fare', 'Embarked',data=titanic, orient='v', palette='Set1', ax=axes[1])
scatterplot(20,8,data=titanic, x='Age',y='Fare',hue='Survived',palette='Set1')
scatterplot(20,8,data=titanic, x='Age',y='Fare', hue='Sex',palette='Dark2')
scatterplot(20,8,data=titanic, x='Age',y='Fare', hue='Embarked',size='Embarked',palette='Set1', sizes=(200,50),alpha=0.7)
scatterplot(20,8,data=titanic, x='Age',y='Fare', hue='Pclass', size='Pclass', palette='tab10', sizes=(210,70),alpha=0.8)
scatterplot(20,8,data=titanic, x='Age',y='Fare', hue='SibSp',size='SibSp', palette='Set1', alpha=0.8)
scatterplot(20,8,data=titanic, x='Age',y='Fare', hue='Parch',size='Parch', palette='Set1', alpha=0.8)
point_bar_plot('Pclass','Age', data=titanic, hue='Survived', figRow=20, figCol=8, palette='Dark2')

point_bar_plot('Sex','Age', data=titanic, hue='Survived', figRow=20, figCol=8, palette='Paired')

point_bar_plot('Embarked','Age', data=titanic, hue='Survived', figRow=20, figCol=8, palette='CMRmap')

point_bar_plot('SibSp','Age', data=titanic, hue='Survived', figRow=20, figCol=8, palette='tab10_r')

point_bar_plot('Parch','Age', data=titanic, hue='Survived', figRow=20, figCol=8, palette='summer')
point_bar_plot('Pclass','Fare', data=titanic, hue='Survived', figRow=20, figCol=8, palette='Dark2')

point_bar_plot('Sex','Fare', data=titanic, hue='Survived', figRow=20, figCol=8, palette='Paired')

point_bar_plot('Embarked','Fare', data=titanic, hue='Survived', figRow=20, figCol=8, palette='CMRmap')

point_bar_plot('SibSp','Fare', data=titanic, hue='Survived', figRow=20, figCol=8, palette='tab10_r')

point_bar_plot('Parch','Fare', data=titanic, hue='Survived', figRow=20, figCol=8, palette='summer')
countplot(1,1,15,6,data=titanic,features=['Pclass'], hue='Survived', palette=['tab20b_r'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Pclass'], hue='Sex', palette=['afmhot'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Pclass'], hue='Embarked', palette=['viridis'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Pclass'], hue='SibSp', palette=['tab10_r'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Pclass'], hue='Parch', palette=['Dark2_r'], rotation=0)
countplot(1,1,15,6,data=titanic,features=['Sex'], hue='Survived', palette=['tab20b_r'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Sex'], hue='Embarked', palette=['viridis'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Sex'], hue='SibSp', palette=['tab10_r'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Sex'], hue='Parch', palette=['Dark2_r'], rotation=0)
countplot(1,1,15,6,data=titanic,features=['Embarked'], hue='Survived', palette=['tab20b_r'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Embarked'], hue='SibSp', palette=['tab10_r'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['Embarked'], hue='Parch', palette=['Dark2_r'], rotation=0)
countplot(1,1,15,6,data=titanic,features=['SibSp'], hue='Survived', palette=['tab20b_r'], rotation=0)

countplot(1,1,15,6,data=titanic,features=['SibSp'], hue='Parch', palette=['tab10_r'], rotation=0)
countplot(1,1,15,6,data=titanic,features=['Parch'], hue='Survived', palette=['tab20b_r'], rotation=0)
from sklearn.preprocessing import LabelEncoder   # import label encoder



def lencode(col, data):

    labelencoder = LabelEncoder()

    data[col] = labelencoder.fit_transform(data[col]) # returns label encoded variable(s)

    return data
display(titanic.head(3))

display(titanic_test.head(3))
titanic = lencode('Sex', titanic)

titanic = lencode('Embarked', titanic)



titanic_test = lencode('Sex', titanic_test)

titanic_test = lencode('Embarked', titanic_test)
heatmap(10,8, data=titanic)
titanic= pd.get_dummies(titanic, prefix=['Pclass', 'Embarked'], columns=['Pclass', 'Embarked'])

titanic.head()
titanic_test= pd.get_dummies(titanic_test, prefix=['Pclass', 'Embarked'], columns=['Pclass', 'Embarked'])

titanic_test.head()
titanic.head()
from sklearn.preprocessing import StandardScaler

std_scale = StandardScaler()

titanic[['Age', 'Fare']] = std_scale.fit_transform(titanic[['Age', 'Fare']])

titanic_test[['Age', 'Fare']] = std_scale.fit_transform(titanic_test[['Age', 'Fare']])
from sklearn.preprocessing import FunctionTransformer  

log_transformer = FunctionTransformer(np.log1p)

titanic[['Fare']] = log_transformer.fit_transform(titanic[['Fare']])

titanic_test[['Fare']] = log_transformer.fit_transform(titanic_test[['Fare']])
titanic['Fare'].skew()
distplot(1,1, 8,5, 

         features=['Fare'], data=titanic,

         colors=['indigo'])

X = titanic.loc[:, titanic.columns != 'Survived']

y = titanic['Survived']
#Balance the target class using SMOTE 
#from imblearn.over_sampling import SMOTE

#oversample = SMOTE()

#X, y = oversample.fit_resample(X, y)
printmd('**As "Personal Loan" attribute is imbalanced, STRATIFYING the same to maintain the same percentage of distribution**',color='brown')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20, random_state=1)



printmd('**Training and Testing Set Distribution**', color='brown')



print(f'Training set has {X_train.shape[0]} rows and {X_train.shape[1]} columns')

print(f'Testing set has {X_test.shape[0]} rows and {X_test.shape[1]} columns')



printmd('**Original Set Survived Value Distribution**', color='brown')



print("Original Survived '1' Values    : {0} ({1:0.2f}%)".format(len(titanic.loc[titanic['Survived'] == 1]), (len(titanic.loc[titanic['Survived'] == 1])/len(titanic.index)) * 100))

print("Original Survived '0' Values   : {0} ({1:0.2f}%)".format(len(titanic.loc[titanic['Survived'] == 0]), (len(titanic.loc[titanic['Survived'] == 0])/len(titanic.index)) * 100))



printmd('**Training Set Survived Value Distribution**', color='brown')



print("Training Survived '1' Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Survived '0' Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))



printmd('**Testing Set Survived Value Distribution**', color='brown')

print("Test Survived '1' Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Survived '0' Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))
def Modelling_Prediction_Scores(model ,X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    #predict on train and test

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)



    #predict the probabilities on train and test

    y_train_pred_proba = model.predict_proba(X_train) 

    y_test_pred_proba = model.predict_proba(X_test)



    #get Accuracy Score for train and test

    accuracy_train = metrics.accuracy_score(y_train, y_train_pred)

    accuracy_test = metrics.accuracy_score(y_test, y_test_pred)

    accdf = pd.DataFrame([[accuracy_train, accuracy_test, ]], columns=['Training', 'Testing'], index=['Accuracy'])    



    #get Precision Score on train and test

    precision_train = metrics.precision_score(y_train, y_train_pred)

    precision_test = metrics.precision_score(y_test, y_test_pred)

    precdf = pd.DataFrame([[precision_train, precision_test, ]], columns=['Training', 'Testing'], index=['Precision'])



    #get Recall Score on train and test

    recall_train = metrics.recall_score(y_train, y_train_pred)

    recall_test = metrics.recall_score(y_test, y_test_pred)

    recdf = pd.DataFrame([[recall_train, recall_test, ]], columns=['Training', 'Testing'], index=['Recall'])



    #get F1-Score on train and test

    f1_score_train = metrics.f1_score(y_train, y_train_pred)

    f1_score_test = metrics.f1_score(y_test, y_test_pred)

    f1sdf = pd.DataFrame([[f1_score_train, f1_score_test, ]], columns=['Training', 'Testing'], index=['F1 Score'])



    #get Area Under the Curve (AUC) for ROC Curve on train and test

    roc_auc_score_train = metrics.roc_auc_score(y_train, y_train_pred)

    roc_auc_score_test = metrics.roc_auc_score(y_test, y_test_pred)

    rocaucsdf = pd.DataFrame([[roc_auc_score_train, roc_auc_score_test, ]], columns=['Training', 'Testing'], index=['ROC AUC Score'])



    #get Area Under the Curve (AUC) for Precision-Recall Curve on train and test

    precision_train, recall_train, thresholds_train = metrics.precision_recall_curve(y_train, y_train_pred_proba[:,1])

    precision_recall_auc_score_train = metrics.auc(recall_train, precision_train)

    precision_test, recall_test, thresholds_test = metrics.precision_recall_curve(y_test,y_test_pred_proba[:,1])

    precision_recall_auc_score_test = metrics.auc(recall_test, precision_test)

    precrecaucsdf = pd.DataFrame([[precision_recall_auc_score_train, precision_recall_auc_score_test]], columns=['Training', 'Testing'], index=['Precision Recall AUC Score'])



    #calculate the confusion matrix 

    #print('tn, fp, fn, tp')

    confusion_matrix_test = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])



    #display confusion matrix in a heatmap

    f, axes = plt.subplots(1, 2, figsize=(20, 8))

    hmap = sns.heatmap(confusion_matrix_test, cmap='YlGnBu', annot=True, fmt=".0f", ax=axes[0], )

    hmap.set_xlabel('Predicted', fontsize=15)

    hmap.set_ylabel('Actual', fontsize=15)



    #plotting the ROC Curve and Precision-Recall Curve

    fpr, tpr, threshold = metrics.roc_curve(y_test,y_test_pred_proba[:,1])

    plt.plot(fpr, tpr, marker='.', label='ROC Curve')

    plt.plot(recall_test, precision_test, marker='.', label='Precision Recall Curve')

    plt.axes(axes[1])

    plt.title(type(model).__name__, fontsize=15)

    # axis labels

    plt.xlabel('ROC Curve - False Positive Rate \n Precision Recall Curve - Recall', fontsize=15)    

    plt.ylabel('ROC Curve - True Positive Rate \n Precision Recall Curve - Precision', fontsize=15)

    # show the legend

    plt.legend()

    # show the plot

    plt.show()



    #concatenating all the scores and displaying as single dataframe

    consolidatedDF= pd.concat([accdf, precdf,recdf,f1sdf, rocaucsdf, precrecaucsdf])



    printmd('**Confusion Matrix**', color='brown')

    display_side_by_side([confusion_matrix_test, consolidatedDF])

    

    return confusion_matrix_test, consolidatedDF
from sklearn.model_selection import GridSearchCV



def find_best_model_gridsearch(model, parameters, X_train, y_train):

    clf = GridSearchCV(model, parameters, scoring='accuracy')

    clf.fit(X_train, y_train)             

    print(clf.best_score_)

    print(clf.best_params_)

    print(clf.best_estimator_)

    return clf
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform



def find_best_model_randomsearch(model, parameters, X_train, y_train):

    clf = RandomizedSearchCV(model, parameters, scoring='neg_mean_absolute_error', n_jobs=-1, n_iter=50, random_state=10, cv=5)

    clf.fit(X_train, y_train)             

    print(clf.best_score_)

    print(clf.best_params_)

    print(clf.best_estimator_)

    return clf
logRegModel = LogisticRegression(max_iter=200)

params = dict(C=uniform(loc=0, scale=4),penalty=['l2', 'l1'], class_weight=['balanced', None], solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])

clf = find_best_model_randomsearch(logRegModel, params, X_train, y_train)
logRegModel = clf.best_estimator_

cmLR, dfLR = Modelling_Prediction_Scores(logRegModel, X_train, X_test, y_train, y_test)
nbModel = GaussianNB()

cmNB, dfNB = Modelling_Prediction_Scores(nbModel, X_train, X_test, y_train, y_test)
def Optimal_k_Plot(model, X_train, X_test, y_train, y_test):

    # creating odd list of K for KNN

    myList = list(range(3,20))



    # subsetting just the odd ones

    klist = list(filter(lambda x: x % 2 != 0, myList))

    # empty list that will hold accuracy scores

    scores = []



    # perform accuracy metrics for values from 3,5....19

    for k in klist:        

        model.n_neighbors = k

        model.fit(X_train, y_train)

        # predict the response

        y_test_pred = model.predict(X_test)        

        test_score= metrics.accuracy_score(y_test, y_test_pred)

        scores.append(test_score)



    # determining best k

    optimal_k = klist[scores.index(max(scores))]

    print("The optimal number of neighbors is %d" % optimal_k)



    import matplotlib.pyplot as plt

    # plot misclassification error vs k

    plt.plot(klist, scores)

    plt.xlabel('Number of Neighbors K')

    plt.ylabel('Score')

    plt.show()

knnModel = KNeighborsClassifier(n_jobs=-1, weights='uniform')

params = dict(n_neighbors=range(2, 20, 1), algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])

clf = find_best_model_randomsearch(knnModel, params, X_train, y_train)
#knnModel = KNeighborsClassifier(n_jobs=-1)

knnModel = clf.best_estimator_

Optimal_k_Plot(knnModel, X_train, X_test, y_train, y_test)
#knnModel = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

knnModel.n_neighbors = 15

cmKNN, dfKNN = Modelling_Prediction_Scores(knnModel, X_train, X_test, y_train, y_test)
from xgboost import XGBClassifier as XGB



xgb = XGB(n_jobs=-1, random_state=10)



cmXGB, dfXGB = Modelling_Prediction_Scores(xgb, X_train, X_test, y_train, y_test)
params = dict(booster=('gbtree', 'gblinear'),max_depth=range(2,10,1), learning_rate=np.arange(0.01, 0.5, 0.01), n_estimators=range(100, 300, 25), gamma=np.arange(0.1, 1, 0.1), importance_type=('gain', 'weight', 'cover'))



clf = find_best_model_randomsearch(xgb, params, X_train, y_train)
xgb = clf.best_estimator_

cmXGB, dfXGB = Modelling_Prediction_Scores(xgb, X_train, X_test, y_train, y_test)
def model_show_feature_importance(model, X_train, feature_importance=False):

    f, axes = plt.subplots(1, 1, figsize=(20, 10))

    

    if (not feature_importance):

        coef = pd.DataFrame(model.coef_.ravel())

    elif (feature_importance):

        coef = pd.DataFrame(model.feature_importances_)

    

    coef["feat"] = X_train.columns

    bplot = sns.barplot(coef["feat"],coef[0],palette="Set1",linewidth=2,edgecolor="k", ax=axes)    

    bplot.set_facecolor("white")

    bplot.axhline(0,color="k",linewidth=2)

    bplot.set_ylabel("coefficients/weights", fontdict=dict(fontsize=20))

    bplot.set_xlabel("features", fontdict=dict(fontsize=20))

    bplot.set_title('FEATURE IMPORTANCES')

    bplot.set_xticklabels(rotation=60, labels=bplot.get_xticklabels(),fontweight='demibold',fontsize='x-large')
model_show_feature_importance(xgb, X_train, feature_importance=True)
titanic_test.Fare.fillna(titanic_test.Fare.median(), inplace=True)

titanic_test.isna().sum()
final_predictions = xgb.predict(titanic_test)

final_predictions
submission = pd.DataFrame({'PassengerId':passengerid,'Survived':final_predictions})



#Visualize the first 5 rows

submission.head()
filename = 'Titanic_Predictions_6.csv'

submission.to_csv(filename,index=False)