import numpy as np

import pandas as pd



from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.impute import SimpleImputer

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import os # accessing directory structure



from scipy.stats import randint as sp_randint



import seaborn as sns

import matplotlib.pyplot as plt



import h2o

from h2o.automl import H2OAutoML



from scipy import stats



from random import seed

from random import randint



import pickle



from subprocess import check_output



import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # suppress "future warning" messages
print(os.listdir('../input'))
def preProc_df(df,cutoff_catg_conv,cutoff_num,cutoff_excess,cutoff_sparse):

    """    

    Returns

    -------

    DataFrame with the following fields for each variable:

        Basic Statistics : Mean, Median, Min, Max, Quantiles, Count

        % Missing and # of Missing values

        # of distinct values

        % Zero and # of Zero values

        Binary Indicator indicating high cardinality or low based on cutoff value provided

        Binary indicator indicating singular value for predictor

    Parameters

    ----------

    df : Any Pandas DataFrame

    cutoff : int, optional (default = 10)

          Threshold of distinct values to classify a feature as having

          high or low ordinality.  



    """

    #Adding 1 & 99th quantiles

    stats = df.describe().T.reset_index()

    p1 = df.quantile(0.01)

    p99 = df.quantile(0.99)

    stats1 = pd.concat([p1,p99],axis=1)

    stats1.columns = ['1%','99%']

    stats2 = stats.merge(stats1,left_on='index',right_index=True)

    stats2 = stats2.drop('count',axis=1)

    stats2.rename(columns={'index':'feature'},inplace=True)



    #Number of unique values per column

    distinct = []

    for col in df.columns:

        n_unique = df[col].nunique()

        distinct.append([col,n_unique])

    #distinct



    # add count, nzero, pct zero, pct missing, combine distinct and dtypes,

    length = len(df)

    count = df.count()

    nzero = df.apply(lambda x: (x == 0).sum())

    nmiss = length-count

    pctzero = nzero/length*100

    pctmiss = nmiss/length*100

    sparse=pd.Series()

    for i in df.columns:

        sparse[i]=1 if df[i].astype(bool).sum() < cutoff_sparse else 0

    base = pd.concat([count,nmiss,nzero,pctzero,pctmiss,df.dtypes,sparse],axis=1)

    base.columns = ['count','Missing','Zero','%Zero','%Missing','Dtype','Sparse']

    distinct = pd.DataFrame(distinct,columns=['Var','Distinct'])

    merge1 = base.merge(distinct,left_index=True,right_on='Var')

    merged = merge1.merge(stats2,left_on='Var',right_on='feature',how='left', suffixes=('','_1'))

    merged = merged.drop('feature',axis=1)

    merged = merged[['Var','count','Distinct','Dtype','Sparse','Missing','%Missing','Zero','%Zero','mean','std','min','1%','25%','50%','75%','99%','max']]

    merged['Low_Cardinality'] = (merged['Distinct'] <= cutoff_catg_conv)*1

    merged['Medium_Cardinality'] = ((merged['Distinct'] > cutoff_catg_conv) & (merged['Distinct'] < cutoff_num)) *1

    merged['High_Cardinality'] = (merged['Distinct'] >= cutoff_num)*1

    merged['Excess_Cardinality'] = (merged['Distinct'] >= cutoff_excess)*1

    merged['String'] = (merged['Dtype'] == 'object')*1

    merged['Float'] = (merged['Dtype'] == 'float64')*1

    merged['All_Missing'] = (merged['Missing'] == len(df))*1

    merged['Some_Missing'] = (merged['Missing'] > 0)*1

    merged['Singular_Value'] = (merged['Distinct'] <= 1)*1

    merged['Sparse'] = (merged['Sparse'] == 1)*1

    merged['Outlier']= ((merged['max']>=1.5*(merged['75%'] - merged['25%'])) | (merged['min']<=1.5*(merged['75%'] - merged['25%'])))*1

    

    #merged['Sparse']=[1 if df[merged['Var']].astype(bool).sum()<cutoff_sparse else 0] #Number of non zero elements in each column

#    merged['D_type']=df.columns.dtypes

    return merged
def pred_histo(df,var):

    g = sns.FacetGrid(df, col="class", margin_titles=True)

    g.map(sns.distplot, var, color="steelblue", kde=True)
def pred_box(df,var):

    g = sns.FacetGrid(df, col="class", margin_titles=True)

    g.map(sns.boxplot, var, color="steelblue", order=['d ' 'h ' 's ' 'o '])
def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.5, 0.8), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
def unique(df,var):

    unique = pd.unique(df[[var]].values.ravel('K'))

    print(var)

    print(unique)

    print("");
# generate random integer values

from random import seed

from random import randint

# seed random number generator

seed(57720)  # run random generator once without a seed value here and swap in this random seed for the seed value to use in future steps



# generate some integers

for _ in range(1):

    seed_value = randint(0, 100000)

    print(seed_value)

    print('CheckPoint - Last random number was: 80683')
df1 = pd.read_csv('../input/testing.csv', delimiter=',')

nRow, nCol = df1.shape

classes = df1['class'].unique()

print(f'There are {nRow} rows and {nCol} columns in the testing.csv dataset')

print('')

print(df1.head(5))

print('')

print(f'Classes: {classes}')

print('')

print(df1.groupby(['class']).describe())
df2 = pd.read_csv('../input/training.csv', delimiter=',')

nRow, nCol = df2.shape

classes = df2['class'].unique()

print(f'There are {nRow} rows and {nCol} columns in the training.csv dataset')

print('')

print(df2.head(5))

print('')

print(f'Classes: {classes}')

print('')

print(df2.groupby(['class']).describe())
df3 = df2.append(df1, ignore_index=True)

nRow, nCol = df3.shape

classes = df3['class'].unique()

print(f'There are {nRow} rows and {nCol} columns in the combined dataset')

print('')

print(df3.head(5))

print('')

print(f'Classes: {classes}')

print('')

print(df3.groupby(['class']).describe())
df3.info()
# Labels are the target values we want to predict

labels = np.array(df3['class'])



# Remove the target from the features

df = df3.drop('class', axis = 1)
train_features, test_features, train_labels, test_labels = model_selection.train_test_split(df, labels, test_size = 0.20, random_state = seed_value)



print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)
print('Columns that are categorical:', train_features.columns[train_features.select_dtypes(include=['object']).any()].tolist())
print('Columns with missing values:', train_features.columns[train_features.isna().any()].tolist())
print('distinct target values:')

unique(df=df3, var='class')
df_tgt = pd.concat([train_features, df3['class']], join='inner', axis=1, ignore_index=False)

stats = df_tgt['class'].value_counts()

print('Frequency for target:')

print(stats)
stats = pd.DataFrame(preProc_df(df=train_features, cutoff_catg_conv=20, cutoff_num=1000, cutoff_excess=22500, cutoff_sparse=0))

stats.to_csv('stats_train_df.csv', index = False) # save to csv
df_num = df3.drop('class',axis=1)



n=len(df_num.columns)



for i in range(n):

    col = df_num.columns[i]

    pred_histo(df=df3,var=col)
df_num = df3.drop('class',axis=1)



n=len(df_num.columns)



for i in range(n):

    col = df_num.columns[i]

    pred_box(df=df3,var=col)
df3.columns
df_tmp = df3.loc[:, ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9']]

plotScatterMatrix(df_tmp, 20, 10)
df_tmp = df3.loc[:, ['pred_minus_obs_H_b1', 'pred_minus_obs_H_b2', 'pred_minus_obs_H_b3',

       'pred_minus_obs_H_b4', 'pred_minus_obs_H_b5', 'pred_minus_obs_H_b6',

       'pred_minus_obs_H_b7', 'pred_minus_obs_H_b8', 'pred_minus_obs_H_b9']]

plotScatterMatrix(df_tmp, 20, 10)
df_tmp = df3.loc[:, ['pred_minus_obs_S_b1', 'pred_minus_obs_S_b2', 'pred_minus_obs_S_b3',

       'pred_minus_obs_S_b4', 'pred_minus_obs_S_b5', 'pred_minus_obs_S_b6',

       'pred_minus_obs_S_b7', 'pred_minus_obs_S_b8', 'pred_minus_obs_S_b9']]

plotScatterMatrix(df_tmp, 20, 10)
nzv_tmp = stats.query("Sparse == 1")

nzv_list = nzv_tmp['Var'].tolist()

print('Variables with near zero variance:')

print(nzv_list)



# with open('nzv_list.txt', 'w') as filehandle:  

#     filehandle.writelines("%s\n" % place for place in nzv_list)



# remove nzv list (none - not needed)

# train_features = train_features.drop(nzv_list, axis=1)
# find correlation

corr_df = df3.corr(method='pearson') # choosing pearson since data appear normally distributed



c1 = corr_df.abs().unstack()

c1 = pd.DataFrame(c1.sort_values(ascending = False))



c1.to_csv('stats_corr_pairs.csv', index = True)



print(c1.head())

print(c1.tail())
h2o.init()
df = h2o.H2OFrame(df3)



y = "class"

x = ['pred_minus_obs_H_b1', 'pred_minus_obs_H_b2', 'pred_minus_obs_H_b3']



splits = df.split_frame(ratios = [0.8], seed = value )

train = splits[0]

test = splits[1]
y_train = train.as_data_frame()

y_train = y_train.loc[:, ['class']]

y_train.squeeze()

# C = np.delete(y_train, 1, 1)  # delete second column of C

# abc = np.delete(y_train)

# y_train = y_train.reset_index().values



y_test = test.as_data_frame()

y_test = y_test.loc[:, ['class']]

y_test.squeeze()

# y_test = y_test.reset_index().values



x_train = train.as_data_frame()

x_train = x_train.drop(["class"], axis = 1)

x_train = x_train.reset_index().values



x_test = test.as_data_frame()

x_test = x_test.drop(["class"], axis = 1)

x_test = x_test.reset_index().values
print(y_train.shape)

# print(abc.shape)

# print(y_train.dtype.names)
print('y_train shape is:')

print(y_train.shape)

print('')

print('x_train shape is:')

print(x_train.shape)

print('')

print('y_test shape is:')

print(y_test.shape)

print('')

print('x_test shape is:')

print(x_test.shape)

print('')
fs = ReliefF(n_neighbors=100, n_features_to_keep=5)

# X_train = fs.fit_transform(x_train, y_train)

X_test_subset = fs.transform(x_test)

# print(x_test.shape, X_test_subset.shape)
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

# from sklearn.cross_validation import train_test_split

from ReliefF import ReliefF



digits = load_digits(2)

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
# print(digits.data)

# print(digits.target)

# print(X_train)

# print(X_train.shape)

# print(X_test)

# print(X_test.shape)

# print(y_train)

# print(y_train.shape)

# print(y_test)

# print(y_test.shape)

print('y_train shape is:')

print(y_train.shape)

print('')

print('X_train shape is:')

print(X_train.shape)

print('')

print('y_test shape is:')

print(y_test.shape)

print('')

print('X_test shape is:')

print(X_test.shape)

print('')
fs = ReliefF(n_neighbors=100, n_features_to_keep=5)

X_train = fs.fit_transform(X_train, y_train)

X_test_subset = fs.transform(X_test)

print(X_test.shape, X_test_subset.shape)
aml = H2OAutoML(max_runtime_secs = 300, seed = value)

aml.train(y = y,

         training_frame = train)
lb = aml.leaderboard

lb
from h2o.estimators.glm import H2OGeneralizedLinearEstimator



glm = H2OGeneralizedLinearEstimator(family = 'multinomial')

glm.train(x = x, y = y, training_frame = train, validation_frame = test)



# print the auc for the validation data

# glm.auc(valid = True)
preds = aml.predict(test)
# Labels are the target values we want to predict

labels = np.array(df3['class'])



# Remove the target from the features

df = df3.drop('class', axis = 1)



# # Save feature names for later use

feature_list = np.array(df.columns)



# # Convert to numpy array

df = np.array(df)



train_features, test_features, train_labels, test_labels = model_selection.train_test_split(df, labels, test_size = 0.20, random_state = seed_value)



from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy



rf = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=7, random_state=seed_value)

# Define Boruta feature selection method

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=seed_value)
feat_selector.fit(train_features, train_labels)
# Check selected features

print(feat_selector.support_)

# Select the chosen features from our dataframe.

selected = train_features[:, feat_selector.support_]

print ("")

print ("Selected Feature Matrix Shape")

print (selected.shape)


    
selected_features = [feature_list[i] for i, x in enumerate(feat_selector.support_) if x]

print(selected_features)
drop_list = []

drop_list.append('b1')

drop_list.append('b2')

drop_list.append('b3')

drop_list.append('b4')

drop_list.append('b5')

drop_list.append('b6')

drop_list.append('b7')

drop_list.append('b8')

drop_list.append('b9')

drop_list.append('pred_minus_obs_H_b1')

drop_list.append('pred_minus_obs_H_b2')

drop_list.append('pred_minus_obs_H_b3')

drop_list.append('pred_minus_obs_H_b4')

drop_list.append('pred_minus_obs_H_b5')

drop_list.append('pred_minus_obs_H_b6')

drop_list.append('pred_minus_obs_H_b7')

drop_list.append('pred_minus_obs_H_b8')

drop_list.append('pred_minus_obs_H_b9')

drop_list
df4 = df3.drop(drop_list)



# Labels are the target values we want to predict

labels = np.array(df4['class'])



# Remove the target from the features

df = df4.drop('class', axis = 1)



# # Save feature names for later use

feature_list = np.array(df.columns)



# # Convert to numpy array

df = np.array(df)
train_features, test_features, train_labels, test_labels = model_selection.train_test_split(df, labels, test_size = 0.20, random_state = seed_value)

# import pandas as pd

# import sklearn



# from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression



# training=pd.read_csv('../input/training.csv')

# testing=pd.read_csv('../input/testing.csv')



# X_train=training.drop(['class'],axis=1)

# y_train=training['class']

# clf=LogisticRegression(random_state=42)



# clf.fit(X_train,y_train)

# X_test=testing.drop(['class'],axis=1)

# y_test=testing['class']



# y_pred=clf.predict(X_test)

# print("Classification Report:")

# print(sklearn.metrics.classification_report(y_test,y_pred))

# print("Confusion Matrix")

# print(sklearn.metrics.confusion_matrix(y_test,y_pred))

# print("Accuracy Score: ",sklearn.metrics.accuracy_score(y_test,y_pred))
import pandas as pd

import sklearn



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# training=pd.read_csv('../input/training.csv')

# test=pd.read_csv('../input/testing.csv')



train = train.as_data_frame() # convert h20 training frame to pandas frame

test = test.as_data_frame() # convert h20 testing frame to pandas frame



X_train=train.drop(['class'],axis=1)

y_train=train['class']

clf=LogisticRegression(random_state=42)



clf.fit(X_train,y_train)

X_test=test.drop(['class'],axis=1)

y_test=test['class']



y_pred=clf.predict(X_test)

print("Classification Report:")

print(sklearn.metrics.classification_report(y_test,y_pred))

print("Confusion Matrix")

print(sklearn.metrics.confusion_matrix(y_test,y_pred))

print("Accuracy Score: ",sklearn.metrics.accuracy_score(y_test,y_pred))

# print("AUC: ", roc_auc_score(y_test,y_pred))
# importance = clf.feature_importances_
plotPerColumnDistribution(train, 10, 5)
plotCorrelationMatrix(train, 8)
plotScatterMatrix(df1, 20, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file

df2 = pd.read_csv('../input/training.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'training.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
print(df2['class'].unique())

print(df2.groupby(['class']).describe())
plotPerColumnDistribution(df2, 10, 5)
sns.boxplot(x="b4", y="class", data=df2)
sns.jointplot(x="b1", y="b2", data=df2)
# corr_train = df2.drop(['class'], axis=1).corr(method='pearson').reset_index()

corr_train = df2.select_dtypes([np.number]).corr(method='pearson')

corr_train = corr_train.reset_index()

print(corr_train.head())

print('The shape of corr_train is:', corr_train.shape)

print(corr_train.info())

print(corr_train)

corr_train.to_csv('corr_train.csv', index = True)

# hi_corr = corr_train.abs().unstack()

# hi_corr = pd.DataFrame(hi_corr.sort_values(ascending = False))

# print(hi_corr)
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

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

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

#     filename = df.dataframeName

    filename = [x for x in globals() if globals()[x] is df][0]

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

#     plt.title(f'Correlation Matrix for %s' % df, fontsize=15)

    plt.show()
plotScatterMatrix(df2, 20, 10)
plotCorrelationMatrix(df2, 8)