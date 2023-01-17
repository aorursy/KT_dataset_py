import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import pandas_profiling 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_color_codes()

sns.set(style="darkgrid")

%matplotlib inline

from scipy.stats import zscore

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

	

from sklearn.utils import resample



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
def distplot(figRows,figCols,xSize, ySize, features, colors):

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

            plot = sns.distplot(bank[features[row][col]], color=colors[row][col], ax=axesplt, kde=True, hist_kws={"edgecolor":"k"})

            plot.set_xlabel(features[row][col],fontsize=20)
def boxplot(figRows,figCols,xSize, ySize, features, colors, hue=None, orient='h'):

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

            plot = sns.boxplot(features[row][col], data= bank, color=colors[row][col], ax=axesplt, orient=orient, hue=hue)

            plot.set_xlabel(features[row][col],fontsize=20)
def countplot(figRows,figCols,xSize, ySize, features, colors=None,palette=None,hue=None, orient=None, rotation=90):

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

                plot = sns.countplot(features[row][col], data=bank, palette=palette[row][col], ax=axesplt, orient=orient, hue=hue)

            elif(palette is None):

                plot = sns.countplot(features[row][col], data=bank, color=colors[row][col], ax=axesplt, orient=orient, hue=hue)

            plot.set_title(features[row][col],fontsize=20)

            plot.set_xlabel(None)

            plot.set_xticklabels(rotation=rotation, labels=plot.get_xticklabels(),fontweight='demibold',fontsize='large')

            
def catdist(cols):

    dfs = []

    for col in cols:

        colData = pd.DataFrame(bank[col].value_counts(), columns=[col])

        colData['%'] = round((colData[col]/colData[col].sum())*100,2)

        dfs.append(colData)

    display_side_by_side(dfs)
bank = pd.read_csv("../input/portuguese-banking-institution/bank-full.csv")

bank.head()
print('The total number of rows :', bank.shape[0])

print('The total number of columns :', bank.shape[1])
bank.info()

print('===========================================')
print(bank.isna().sum())

print('===================')

print(bank.isnull().sum())

print('===================')

printmd('**CONCLUSION**: As seen from the data above, we conclude there are **"NO Missing"** values in the data', color="blue")
display(bank.describe().transpose())

print('==============================')

printmd('Total negative values under **balance**', color="brown")

display(bank[bank['balance'] < 0].shape[0])

print('==============================')

display(bank[bank['pdays'] == -1]['pdays'].value_counts())

print('==============================')

display(bank[bank['previous'] == 0]['previous'].value_counts())
pandas_profiling.ProfileReport(bank)
distplot(3,2, 15,15, 

         ['age', 'balance','day', 'duration', 'campaign', 'pdays'], 

         ['olive', 'indigo', 'blue', 'teal', 'brown', 'red'])



distplot(1,1, 7, 5, 

         ['previous'], 

         ['green'])
pd.DataFrame.from_dict(dict(

    {

        'age':bank.age.skew(), 

        'balance': bank.balance.skew(), 

        'day': bank.day.skew(),

        'duration': bank.duration.skew(),

        'campaign': bank.campaign.skew(),

        'pdays': bank.pdays.skew(),

        'previous': bank.previous.skew(),        

    }), orient='index', columns=['Skewness'])
boxplot(7,1, 20,31, 

         ['age', 'balance','day', 'duration', 'campaign', 'pdays', 'previous'], 

         ['olive', 'indigo', 'blue', 'teal', 'brown', 'red', 'cyan'])
def catdist(cols):

    dfs = []

    for col in cols:

        colData = pd.DataFrame(bank[col].value_counts(), columns=[col])

        colData['%'] = round((colData[col]/colData[col].sum())*100,2)

        dfs.append(colData)

    display_side_by_side(dfs)

        
catdist(['Target', 'default', 'housing', 'loan', 'marital', 'contact', 'education', 'poutcome', 'job', 'month'])
countplot(2,4, 20,16, 

         ['Target', 'marital','education', 'default', 'housing', 'loan', 'contact', 'poutcome', ], 

         ['olive', 'indigo', 'blue', 'teal', 'brown', 'red', 'cyan','darkgreen'], rotation=30)



countplot(1,2, 20,5, 

         ['job', 'month'], 

         palette=['Set1_r', 'Set1'], rotation=60)
countplot(2,4, 20,15, 

         ['Target', 'marital','education', 'default', 'housing', 'loan', 'contact', 'poutcome'], 

         palette=['winter', 'Accent', 'Paired', 'Spectral', 'bone', 'cool', 'PuRd_r','inferno'], hue='Target', rotation=30)



countplot(1,2, 20,5, 

         ['job', 'month'], 

         palette=['viridis', 'Dark2_r'], hue='Target', rotation=60)
def point_bar_plot(row, col, target, figRow, figCol, palette='rocket', fontsize='large', fontweight='demibold'):

    sns.set(style="whitegrid")

    f, axes = plt.subplots(2, 1, figsize=(figRow, figCol))

    pplot=sns.pointplot(row,col, data=bank, ax=axes[0], linestyles=['--'])

    pplot.set_xlabel(None)

    pplot.set_xticklabels(labels=pplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)    

    bplot=sns.barplot(row,col, data=bank, hue=target, ax=axes[1],palette=palette)

    bplot.set_xlabel(row,fontsize=20)

    bplot.set_xticklabels(labels=bplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)
def point_box_bar_plot(row, col, target, figRow, figCol, palette='rocket', fontsize='large', fontweight='demibold'):

    sns.set(style="whitegrid")

    f, axes = plt.subplots(3, 1, figsize=(figRow, figCol))

    pplot=sns.pointplot(row,col, data=bank, ax=axes[0], linestyles=['--'])

    pplot.set_xlabel(None)

    pplot.set_xticklabels(labels=pplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)

    bxplot=sns.boxplot(row,col, data=bank, hue=target, ax=axes[1],palette='viridis')

    bxplot.set_xlabel(None)

    bxplot.set_xticklabels(labels=bxplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)

    bplot=sns.barplot(row,col, data=bank, hue=target, ax=axes[2],palette=palette)

    bplot.set_xlabel(row,fontsize=20)

    bplot.set_xticklabels(labels=bplot.get_xticklabels(),fontweight=fontweight,fontsize=fontsize)
point_box_bar_plot('job', 'age', 'Target', 20, 15)
point_bar_plot('job', 'balance', 'Target', 20, 10, palette='winter')
point_bar_plot('marital', 'balance', 'Target', 12, 8, palette='summer')
point_bar_plot('education', 'balance', 'Target', 12, 8, palette='tab20b_r')
point_bar_plot('default', 'balance', 'Target', 8, 8, palette='afmhot')
point_bar_plot('housing', 'balance', 'Target', 8, 8, palette='autumn')
point_bar_plot('loan', 'balance', 'Target', 8, 8, palette='binary')
point_bar_plot('Target', 'duration', 'Target', 10, 8, palette='cool')
point_bar_plot('contact', 'campaign', 'Target', 10, 8, palette='copper')
point_bar_plot('month', 'campaign', 'Target', 20, 8, palette='Greens_r')
point_bar_plot('job', 'campaign', 'Target', 20, 8, palette='YlGnBu')
point_bar_plot('poutcome', 'previous', 'Target', 10, 8, palette='Paired_r')
sns.pairplot(bank, hue='Target', diag_kind = 'kde', palette='rocket')
bank.corr()
f, axes = plt.subplots(1, 1, figsize=(12, 6))

sns.heatmap(bank.corr().abs(), cmap='YlGnBu', annot=True, fmt=".2f", ax=axes, linecolor='white', linewidths=0.3, square=True)
bank.drop(bank[bank.job == 'unknown'].index, axis=0, inplace=True)

bank.shape
bank.drop(bank[bank.education == 'unknown'].index, axis=0, inplace=True)

bank.shape
catdist(['Target', 'default', 'housing', 'loan', 'marital', 'contact', 'education', 'poutcome', 'job', 'month'])
bank.drop(['duration','default', 'day'], axis=1, inplace=True)

bank.shape
def remove_outliers(col, data):

    outlier_col = col + "_outliers"

    data[outlier_col] = data[col]

    data[outlier_col]= zscore(data[outlier_col])



    condition = (data[outlier_col]>3) | (data[outlier_col]<-3)

    print(data[condition].shape)

    data.drop(data[condition].index, axis = 0, inplace = True)

    data.drop(outlier_col, axis=1, inplace=True)
remove_outliers('balance', bank)
f, axes = plt.subplots(1, 1, figsize=(20, 5))

sns.boxplot(bank['balance'], ax =axes)
remove_outliers('pdays', bank)
f, axes = plt.subplots(1, 1, figsize=(20, 5))

sns.boxplot(bank['pdays'], ax =axes)
remove_outliers('previous', bank)
f, axes = plt.subplots(1, 1, figsize=(20, 5))

sns.boxplot(bank['previous'], ax =axes)
bank_enc= pd.get_dummies(bank, prefix=['job','marital','education','contact','month','poutcome'], columns=['job','marital','education','contact','month','poutcome'])
bank_enc.info()
bank_enc.drop(['job_student','marital_divorced', 'education_primary', 'contact_telephone', 'month_dec', 'poutcome_success'], 

                  axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder   # import label encoder



def lencode(col):

    labelencoder = LabelEncoder()

    bank_enc[col] = labelencoder.fit_transform(bank_enc[col]) # returns label encoded variable(s)

    return bank_enc
bank_enc = lencode('housing')

bank_enc = lencode('loan')

bank_enc = lencode('Target')
from sklearn.preprocessing import StandardScaler

std_scale = StandardScaler()

bank_enc[['age', 'balance', 'campaign','pdays','previous']] = std_scale.fit_transform(bank_enc[['age', 'balance', 'campaign','pdays','previous']])
distplot(1,2, 12,5, 

         ['balance', 'campaign'], 

         ['indigo', 'blue'])



distplot(1,2, 12, 5, 

         ['previous', 'pdays'], 

         ['green', 'red'])
bank_enc[['balance', 'campaign','pdays','previous']].skew()
def lognew(a):    

    a = np.array(a)

    x = np.min(a)

    a = a + 1 - x

    return np.log1p(a)
from sklearn.preprocessing import FunctionTransformer  

log_transformer = FunctionTransformer(lognew)

bank_enc[['balance', 'campaign','pdays','previous']] = log_transformer.fit_transform(bank_enc[['balance', 'campaign','pdays','previous']])
bank_enc[['balance', 'campaign','pdays','previous']].skew()
distplot(1,2, 12,5, 

         ['balance', 'campaign'], 

         ['indigo', 'blue'])



distplot(1,2, 12, 5, 

         ['previous', 'pdays'], 

         ['green', 'red'])
def fit_n_score(model, X_train, X_test, y_train, y_test):  # take the model, and data as inputs    

    model.fit(X_train, y_train)   # fit the model with the train data

    

    iterables = [[type(model).__name__], ['Training', 'Testing']]

    

    multiIndex = pd.MultiIndex.from_product(iterables, names=['Algorithm', 'DataSet'])



    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    

    #get Precision Score on train and test

    accuracy_train = round(metrics.accuracy_score(y_train, y_train_pred),3)

    accuracy_test = round(metrics.accuracy_score(y_test, y_test_pred),3)

    accdf = pd.DataFrame([[accuracy_train],[accuracy_test]], index=multiIndex, columns=['Accuracy'])    



    #get Precision Score on train and test

    precision_train = round(metrics.precision_score(y_train, y_train_pred),3)

    precision_test = round(metrics.precision_score(y_test, y_test_pred),3)

    precdf = pd.DataFrame([[precision_train],[precision_test]], index=multiIndex, columns=['Precision'])



    #get Recall Score on train and test

    recall_train = round(metrics.recall_score(y_train, y_train_pred),3)

    recall_test = round(metrics.recall_score(y_test, y_test_pred),3)

    recdf = pd.DataFrame([[recall_train],[recall_test]], index=multiIndex, columns=['Recall'])    



    #get F1-Score on train and test

    f1_score_train = round(metrics.f1_score(y_train, y_train_pred),3)

    f1_score_test = round(metrics.f1_score(y_test, y_test_pred),3)

    f1sdf = pd.DataFrame([[f1_score_train],[f1_score_test]], index=multiIndex, columns=['F1 Score'])   

    

    consolidatedDF= pd.concat([accdf, precdf,recdf, f1sdf], axis=1)

    

    confusion_matrix_test = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])   

    

    display_side_by_side([consolidatedDF, confusion_matrix_test])

    

    return consolidatedDF, confusion_matrix_test

    
# function for model fitting, prediction and calculating different scores

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
X = bank_enc.loc[:, bank_enc.columns != 'Target']

y = bank_enc['Target']
printmd('**As "Personal Loan" attribute is imbalanced, STRATIFYING the same to maintain the same percentage of distribution**', color='brown')

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size =.30, random_state=1)



printmd('**Training and Testing Set Distribution**', color='brown')



print(f'Training set has {X_train.shape[0]} rows and {X_train.shape[1]} columns')

print(f'Testing set has {X_test.shape[0]} rows and {X_test.shape[1]} columns')



printmd('**Original Set Target Value Distribution**', color='brown')



print("Original Personal Loan '1' Values    : {0} ({1:0.2f}%)".format(len(bank_enc.loc[bank_enc['Target'] == 1]), (len(bank_enc.loc[bank_enc['Target'] == 1])/len(bank_enc.index)) * 100))

print("Original Personal Loan '0' Values   : {0} ({1:0.2f}%)".format(len(bank_enc.loc[bank_enc['Target'] == 0]), (len(bank_enc.loc[bank_enc['Target'] == 0])/len(bank_enc.index)) * 100))



printmd('**Training Set Target Value Distribution**', color='brown')



print("Training Personal Loan '1' Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Personal Loan '0' Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))



printmd('**Testing Set Target Value Distribution**', color='brown')

print("Test Personal Loan '1' Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Personal Loan '0' Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))

# We do upsampling only from the train dataset to preserve the sanctity of the test data

y_train_0 = y_train[y_train == 0]



extra_samples = y_train[y_train == 1].sample(10000,replace = True, random_state=1).index # Generate duplicate samples

y_train = y_train.append(y_train.loc[extra_samples])  # use the index of the duplicate samples to append to the y_train



extra_samples = X_train.loc[extra_samples]   # use the same index to generate duplicate rows in X_train

X_train = X_train.append(extra_samples)  # append these duplicate rows to X_train
printmd('**Training Set Target Value Distribution**', color='brown')



print("Training Personal Loan '1' Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Personal Loan '0' Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))



printmd('**Testing Set Target Value Distribution**', color='brown')

print("Test Personal Loan '1' Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Personal Loan '0' Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))
#combine them back for resampling

train_data = pd.concat([X_train, y_train], axis=1)

train_data.shape



# separate minority and majority classes

negative = train_data[train_data.Target==0]

positive = train_data[train_data.Target==1]



df_majority_downsampled = resample(negative,

 replace=False, # sample without replacement

 n_samples=20000, # match number in minority class

 random_state=1) # reproducible results

# combine minority and downsampled majority

downsampled = pd.concat([positive, df_majority_downsampled])

# check new class counts

downsampled.Target.value_counts()

X_train = downsampled.loc[:, downsampled.columns != 'Target']

y_train = downsampled['Target']
printmd('**Training Set Target Value Distribution**', color='brown')



print("Training Personal Loan '1' Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Personal Loan '0' Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))



printmd('**Testing Set Target Value Distribution**', color='brown')

print("Test Personal Loan '1' Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Personal Loan '0' Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))
logRegModel = LogisticRegression(n_jobs=-1)

cmLR, dfLR = Modelling_Prediction_Scores(logRegModel, X_train, X_test, y_train, y_test)
gnb = GaussianNB()

cmNB, dfNB = Modelling_Prediction_Scores(gnb, X_train, X_test, y_train, y_test)
#plot the f1-scores for different values of k for a model and see which is optimal

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

        y_test_pred = knn.predict(X_test)        

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
knn = KNeighborsClassifier(n_jobs=-1)

Optimal_k_Plot(knn, X_train, X_test, y_train, y_test)
knn = KNeighborsClassifier(n_neighbors=19, n_jobs=-1)

cmKNN, dfKNN = Modelling_Prediction_Scores(knn, X_train, X_test, y_train, y_test)
lr = LogisticRegression(C=0.1) 

lrDf, lrCm = fit_n_score(lr, X_train, X_test, y_train, y_test)



knn = KNeighborsClassifier(n_neighbors=3)

knnDf, knnCm = fit_n_score(knn, X_train, X_test, y_train, y_test)



nb = GaussianNB()

nbDf, nbCm = fit_n_score(nb, X_train, X_test, y_train, y_test)



svm = SVC(gamma='auto')

svmDf, svmCm = fit_n_score(svm, X_train, X_test, y_train, y_test)



result1 = pd.concat([lrDf,knnDf,nbDf,svmDf])
lr = LogisticRegression(C=0.1, class_weight='balanced')  

lrDf, lrCm = fit_n_score(lr, X_train, X_test, y_train, y_test)



knn = KNeighborsClassifier(n_neighbors=3)

knnDf, knnCm = fit_n_score(knn, X_train, X_test, y_train, y_test)



svm = SVC(gamma='auto', C=0.1)

svmDf, svmCm = fit_n_score(svm, X_train, X_test, y_train, y_test)



result2 = pd.concat([lrDf,knnDf,svmDf])
from sklearn.model_selection import GridSearchCV



def find_best_model(model, parameters):

    clf = GridSearchCV(model, parameters, scoring='accuracy')

    clf.fit(X_train, y_train)             

    print(clf.best_score_)

    print(clf.best_params_)

    print(clf.best_estimator_)

    return clf
dTree= DecisionTreeClassifier()

dTreeDf, dTreeCm = fit_n_score(dTree, X_train, X_test, y_train, y_test)

result3  = dTreeDf
parameters = {'criterion':('gini', 'entropy'), 'max_depth':[1, 10], 'max_features':(None,'auto')}

clf = find_best_model(dTree, parameters)
dTree2 = clf.best_estimator_

dTreeDf, dTreeCm = fit_n_score(dTree2, X_train, X_test, y_train, y_test)

dTreeDf

result4 = dTreeDf
bagging = BaggingClassifier()

baggingDf, baggingCm = fit_n_score(bagging, X_train, X_test, y_train, y_test)

result5  = baggingDf
bagging = BaggingClassifier(dTree, max_samples=0.1)

baggingDf, baggingCm = fit_n_score(bagging, X_train, X_test, y_train, y_test)

result5  = baggingDf
parameters = {'n_estimators': [5,50]}

clf = find_best_model(bagging, parameters)
bagging2 = clf.best_estimator_

baggingDf, baggingCm = fit_n_score(bagging2, X_train, X_test, y_train, y_test)

result6  = baggingDf
rf = RandomForestClassifier()

rfDf, rfgCm = fit_n_score(rf, X_train, X_test, y_train, y_test)

result7  = rfDf
rf2 = RandomForestClassifier(n_estimators=150, max_depth=15)

rfDf, rfCm = fit_n_score(rf2, X_train, X_test, y_train, y_test)

result8  = rfDf
ab = AdaBoostClassifier()

abDf, abCm = fit_n_score(ab, X_train, X_test, y_train, y_test)

result9  = abDf
dTree3= DecisionTreeClassifier(max_depth=5)
ab = AdaBoostClassifier(dTree3)

abDf, abCm = fit_n_score(ab, X_train, X_test, y_train, y_test)

result9  = abDf
gb = GradientBoostingClassifier()

gbDf, gbCm = fit_n_score(gb, X_train, X_test, y_train, y_test)

result10 = gbDf
gb = GradientBoostingClassifier(learning_rate=0.2, max_depth=7)

gbDf, gbCm = fit_n_score(gb, X_train, X_test, y_train, y_test)

result11 = gbDf
display_side_by_side([result8, result11])