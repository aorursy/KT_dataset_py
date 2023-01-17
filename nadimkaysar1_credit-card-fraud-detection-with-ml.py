import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.shape
df.head()
df.info()
df.dtypes.sort_values().to_frame('feature_type').groupby(by = 'feature_type').size().to_frame('count').reset_index()
df_dtypes = pd.merge(df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),

         df.dtypes.to_frame('feature_type').reset_index(),

         on = 'index',

         how = 'inner')
df_dtypes.sort_values(['missing_value', 'feature_type'])
df.describe().round()
def find_constant_features(dataFrame):

    const_features = []

    for column in list(dataFrame.columns):

        if dataFrame[column].unique().size < 2:

            const_features.append(column)

    return const_features
const_features = find_constant_features(df)
const_features
df.drop_duplicates(inplace= True)
df.shape
def duplicate_columns(frame):

    groups = frame.columns.to_series().groupby(frame.dtypes).groups

    dups = []



    for t, v in groups.items():



        cs = frame[v].columns

        vs = frame[v]

        lcs = len(cs)



        for i in range(lcs):

            ia = vs.iloc[:,i].values

            for j in range(i+1, lcs):

                ja = vs.iloc[:,j].values

                if np.array_equal(ia, ja):

                    dups.append(cs[i])

                    break

    return dups
duplicate_cols = duplicate_columns(df)
duplicate_cols

df.shape

df.columns
sns.countplot('Class', data=df)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=11)


sns.catplot(x="Class",y="Amount",kind="bar",data=df);
amount = df['Amount'].values

time= df['Time'].values
sns.distplot(amount,bins=20,color='r')
sns.distplot(time,bins=50,color='r')
df.describe().T
from sklearn.utils import shuffle, class_weight

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

cf.go_offline()



corr = df.corr(method = 'spearman')

layout = cf.Layout(height=600,width=600)

corr.abs().iplot(kind = 'heatmap', layout=layout.to_plotly_json(), colorscale = 'RdBu')
new_corr = corr.abs()

new_corr.loc[:,:] = np.tril(new_corr, k=-1) # below main lower triangle of an array

new_corr = new_corr.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)
new_corr[new_corr.correlation > 0.3]
corr_with_target = df.corrwith(df.Class).sort_values(ascending = False).abs().to_frame('correlation_with_target').reset_index().head(20)

unique_values = df.nunique().to_frame('unique_values').reset_index()

corr_with_unique = pd.merge(corr_with_target, unique_values, on = 'index', how = 'inner')
corr_with_unique
df_major=df[df.Class==0]
df_minor=df[df.Class==1]
df_major.shape
from sklearn.utils import resample
df_minor_upsmapled = resample(df_minor, replace = True, n_samples = 283253, random_state = 2018)
df_minor_upsmapled.shape
final_data=pd.concat([df_minor_upsmapled,df_major])

final_data.shape
X = final_data.drop('Class', axis = 1)

Y = final_data.Class
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=0)
mms = StandardScaler()

mms.fit(xtrain)

xtrain_scaled = mms.transform(xtrain)

xtest_scaled = mms.transform(xtest)
def evaluate_model(ytest, ypred, ypred_proba = None):

    if ypred_proba is not None:

        print('ROC-AUC score of the model: {}'.format(roc_auc_score(ytest, ypred_proba[:, 1])))

    print('Accuracy of the model: {}\n'.format(accuracy_score(ytest, ypred)))

    print('Classification report: \n{}\n'.format(classification_report(ytest, ypred)))

    print('Confusion matrix: \n{}\n'.format(confusion_matrix(ytest, ypred)))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
logisticRegr = LogisticRegression()
logisticRegr.fit(xtrain_scaled, ytrain)
lr_pred = logisticRegr.predict(xtest_scaled)
evaluate_model(ytest, lr_pred)
def random_forest(xtrain, xtest, ytrain):

    rf_params = {

        'n_estimators': 126, 

        'max_depth': 14

    }



    rf = RandomForestClassifier(**rf_params)

    rf.fit(xtrain, ytrain)

    rfpred = rf.predict(xtest)

    rfpred_proba = rf.predict_proba(xtest)

    

    return rfpred, rfpred_proba, rf
rfpred, rfpred_proba, rf = random_forest(xtrain_scaled, xtest_scaled, ytrain)
from sklearn.metrics import recall_score, roc_auc_score, f1_score

from sklearn.metrics import accuracy_score, roc_auc_score,                             classification_report, confusion_matrix
evaluate_model(ytest, rfpred, rfpred_proba)