import numpy as np

import pandas as pd



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from collections import Counter

import matplotlib.pyplot as plt

#import seaborn as sns

%matplotlib inline

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()



import warnings

warnings.filterwarnings('ignore')
import collections



from IPython.display import display



class Exploration(object):

    """

        How to run:

                    fa = Exploration(df).DescribeAnalysis(True)

    """



    def __init__(self, dataframe):



        self.dataframe = dataframe



    def _getFirstNumericalColNameIndex(self,tem_df):

        for col in tem_df.columns:

            col_type = tem_df[col].dtype

            if (col_type != 'object') and (col_type != 'datetime64') and (col_type != 'bool'):



                first_numerical_col_name = col

#                     print(tem_df.columns.get_loc(col))

                break

        return tem_df.columns.get_loc(col)





    def DescribeAnalysis(self, dis=False):



        """

            Input:

                    DataFrame

            Output:

                    Dataframe:

                        Features: Name of Features

                        Dtype: type of data

                        Nunique: number of unique value

                        freq1: most frequent value

                        freq1_val: number of occurance of the most frequent value

                        freq2: second most frequent value

                        freq2_val: number of occurance of the second most frequent value

                        freq3: 3rd most frequent value, if available

                        freq3_val: number of occurance of the thrid most frequent value, if available

                        describe stats: the following ones are the stat offer by our best friend .describe methods.

        """





        # get input column dataframe name

        cols = self.dataframe.columns



        # set name of output dataframe

        stat_cols= ['Dtype', 'Nunique', 'nduplicate', 'freq1', 'freq1_val', 'freq2', 'freq2_val',

             'freq3', 'freq3_val'] + self.dataframe[cols[self._getFirstNumericalColNameIndex(self.dataframe)]].describe().index.tolist()[1:]

        stat_cols = ['Features']+stat_cols



        feature_stat = pd.DataFrame(columns=stat_cols)

        i = 0



        for col in cols:

            col_type = self.dataframe[col].dtype

            if (col_type != 'object') and (col_type != 'datetime64[ns]') and (col_type != 'bool'):

                stat_vals = []



                # get stat value

                stat_vals.append(col)

                # Data type

                stat_vals.append(self.dataframe[col].dtype)

                # number of unique

                stat_vals.append(self.dataframe[col].nunique())

                # nduplicate

                stat_vals.append(self.dataframe.shape[0]-self.dataframe[col].nunique())

                # 'freq1' & freq1_val'

                stat_vals.append(self.dataframe[col].value_counts().index[0])

                stat_vals.append(self.dataframe[col].value_counts().iloc[0])

                #'freq2', 'freq2_val'

                try:

                    stat_vals.append(self.dataframe[col].value_counts().index[1])

                    stat_vals.append(self.dataframe[col].value_counts().iloc[1])

                except Exception:

                    stat_vals.append(np.nan)

                    stat_vals.append(np.nan)

                # 'freq3', 'freq3_val'

                if len(self.dataframe[col].value_counts())>2:

                    stat_vals.append(self.dataframe[col].value_counts().index[2])

                    stat_vals.append(self.dataframe[col].value_counts().iloc[2])

                else:

                    stat_vals.append(np.nan)

                    stat_vals.append(np.nan)



                stat_vals += self.dataframe[col].describe().tolist()[1:]

                feature_stat.loc[i] = stat_vals

                i += 1



        # dipslay dataframe

        if dis:

            # display(nunique_duplicate)

            display(feature_stat)



        return feature_stat

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display



import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls



py.init_notebook_mode(connected=True)







class PlotingMissingValues(object):

    """

    input:

    df: dataframe

    verbosity: show the resul of finding missing values

    output:

        list of tuple of missitng value and key: [(key,value)]



    Methods:

        FindingMissingValue(Ture shows the result)

        PlotMissingValue: show the plot of miasing value

    """



    def __init__(self, df):

        self.df =  df



    def FindingMissingValue(self, verbosity =True):

        df_size = self.df.shape[0]



        if self.df.isnull().values.any():

            if  verbosity:

                print(self.df.isnull().sum())



            value, key = [], []

            for col in self.df.columns:

                value.append(round(self.df[col].isnull().sum()/df_size *100,2))

                key.append(col)

                missingValue = list(zip(key,value))



            return missingValue



        else:

            print('There is no NAN in dataframe')

            return None





    def __getListofTupleValues(self,listOfTuple):

        """

         input:

             [(key,value)]

         output:

             list of keys & list of values

        """

        key = [listOfTuple[i][0] for i in range(len(listOfTuple))]

        value = [listOfTuple[i][1] for i in range(len(listOfTuple))]

        return key,value



    def __Plot(self,key, value):

        data_array = value

        hist_data = np.histogram(data_array)

        x = value

        y = key

        """

            x: list of value

            y : list of key

        """

        data = [go.Bar(

                x=x,

                y=y,

                text=x,

                orientation = 'h',

                textposition = 'auto',

                marker=dict(

                    color='rgb(227,67,45)',

                    line=dict(

                        color='rgb(8,48,107)',

                        width=2.5),

                ),

                opacity=0.6

            )]

        layout = go.Layout(

            title='Missing Values',

            autosize=False,

            width=600,

            height=1200,

            margin=go.Margin(

                l=200,

                r=0,

                b=100,

                t=100,

                pad=4

            )

        )



        fig = go.Figure(data=data, layout=layout)

        py.iplot(fig, filename='bar-direct-labels')



    def PlotMissingValue(self):

        finding_missing_values= self.FindingMissingValue(False)

        if finding_missing_values==None:

            pass

        else:

            key, value = self.__getListofTupleValues(finding_missing_values)

            self.__Plot(key, value)

def wordFrequencyPlot(df,colname):

    temp = df[colname].value_counts()

    trace = go.Bar(

        y=temp.index[::-1],

        x=(temp / temp.sum() * 100)[::-1],

        orientation = 'h',

        marker=dict(

            color='blue',

        ),

    )



    layout = go.Layout(

        title = colname + " Top  (%)",

        xaxis=dict(

            title=' count',

            tickfont=dict(size=14,)),

        yaxis=dict(

#             title='',

            titlefont=dict(size=16),

            tickfont=dict(

                size=14)),

        margin=dict(

        l=200,

    ),



    )

    data = [trace]

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
from scipy.stats import skew

from scipy.stats import kurtosis

def plotBarCat(df,feature,target,nbins=2):

    

    

    

    x0 = df[df[target]==0][feature]

    x1 = df[df[target]==1][feature]



    trace1 = go.Histogram(

        x=x0,

        nbinsx =nbins, 

        opacity=0.75

    )

    trace2 = go.Histogram(

        x=x1,

        nbinsx = nbins, 

        opacity=0.75

    )



    data = [trace1, trace2]

    layout = go.Layout(barmode='overlay',

                      title=feature,

                       yaxis=dict(title='Count'

        ))

    fig = go.Figure(data=data, layout=layout)



    py.iplot(fig, filename='overlaid histogram')

    

    def DescribeFloatSkewKurt(df,target):

        """

            A fundamental task in many statistical analyses is to characterize

            the location and variability of a data set. A further

            characterization of the data includes skewness and kurtosis.

            Skewness is a measure of symmetry, or more precisely, the lack

            of symmetry. A distribution, or data set, is symmetric if it

            looks the same to the left and right of the center point.

            Kurtosis is a measure of whether the data are heavy-tailed

            or light-tailed relative to a normal distribution. That is,

            data sets with high kurtosis tend to have heavy tails, or

            outliers. Data sets with low kurtosis tend to have light

            tails, or lack of outliers. A uniform distribution would

            be the extreme case

        """

        print('-*-'*25)

        print("{0} mean : ".format(feature), np.mean(df[feature]))

        print("{0} var  : ".format(feature), np.var(df[feature]))

        print("{0} skew : ".format(feature), skew(df[feature]))

        print("{0} kurt : ".format(feature), kurtosis(df[feature]))

        print('-*-'*25)

    

    DescribeFloatSkewKurt(df,feature)

def piePlot(sizes,labels,title):





    fig1, ax1 = plt.subplots(figsize=(8,8))

    plt.rcParams.update({'font.size': 22})

    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

            shadow=True, startangle=90)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



    plt.title(title)

    plt.show()

    

def piePlotLables(df,col,target,v_target):

    select_df = df[[col, target]]

    

    col_isFraud =  select_df[select_df[target]==v_target].groupby(col).count().reset_index(drop=False)

    

    labels = col_isFraud[col].tolist()

    sizes =  col_isFraud[target].tolist()

    

    return sizes, labels 
path = '../input/'

train_identity    = pd.read_csv(path+'train_identity.csv')

train_transaction = pd.read_csv(path+'train_transaction.csv')



test_identity     = pd.read_csv(path+'test_identity.csv')

test_transaction  = pd.read_csv(path+'test_transaction.csv')
print('Shape:')

print('train_identity.shape {} train_transaction.shape {}'.format(train_identity.shape, train_transaction.shape))

print('test_identity.shape {} test_transaction.shape {}'.format(test_identity.shape, test_transaction.shape))
train_identity.head().T
train_identity.describe(include=['O']).T
train_identity_col_cat = train_identity.describe(include=['O']).columns
feature_info_train_identity  = Exploration(train_identity).DescribeAnalysis(True)
PlotingMissingValues(train_identity).PlotMissingValue()
wordFrequencyPlot(train_identity,train_identity_col_cat[0])
# wordFrequencyPlot(train_identity,train_identity_col_cat[1])
# wordFrequencyPlot(train_identity,train_identity_col_cat[2])
# wordFrequencyPlot(train_identity,train_identity_col_cat[3])
# wordFrequencyPlot(train_identity,train_identity_col_cat[4])
# wordFrequencyPlot(train_identity,train_identity_col_cat[5])
# wordFrequencyPlot(train_identity,train_identity_col_cat[6])
# wordFrequencyPlot(train_identity,train_identity_col_cat[7])
# wordFrequencyPlot(train_identity,train_identity_col_cat[8])
wordFrequencyPlot(train_identity,train_identity_col_cat[15])
# wordFrequencyPlot(train_identity,train_identity_col_cat[16])
col_remain = train_transaction.columns[17:]

ff=set()

for col in col_remain:

    ff.add(col[0])

    

print('Columns: ', ff)





col_remain = train_transaction.columns[17:]

C_col = []

D_col = []

M_col = []

V_col = []



rest_col = []



for col in col_remain:

    if col[0] == 'C':

        C_col.append(col)

    elif col[0] == 'D':

        D_col.append(col)

    elif col[0] == 'M':

        M_col.append(col)

    elif col[0] =='V':

        V_col.append(col)

    else:

        rest_col.append(col)

        

print('len C: {}, D: {}, M: {}, V: {}'.format(len(C_col),len(D_col),len(M_col),len(V_col)))
train_transaction_16=train_transaction[train_transaction.columns[0:17]]

train_transaction_16.head().T
feature_info_train_transaction_16  = Exploration(train_transaction_16).DescribeAnalysis(True)
# train_transaction_16['log_TransactionAmt'] = train_transaction_16['TransactionAmt'].apply(np.log)

# plotBarCat(train_transaction_16,'log_TransactionAmt','isFraud',40)
feature = 'TransactionAmt'

plt.boxplot(train_transaction_16[feature].dropna(),vert=False)

plt.title(feature)
feature = 'TransactionAmt'

plt.boxplot(train_transaction_16[train_transaction_16[feature]<300][feature].dropna(),vert=False)

plt.title(feature)
# plotBarCat(train_transaction_16,'isFraud','isFraud')
print('  {:.4f}% of Transactions that are fraud in train '.format(train_transaction['isFraud'].mean() * 100))
feature = 'TransactionDT'

plt.boxplot(train_transaction_16[feature].dropna(),vert=False)

plt.title(feature)
feature = 'card1'

plt.boxplot(train_transaction_16[feature].dropna(),vert=False)

plt.title(feature)
feature = 'card2'

plt.boxplot(train_transaction_16[feature].dropna(),vert=False)

plt.title(feature)
feature = 'card5'

plt.boxplot(train_transaction_16[feature].dropna(),vert=False)

plt.title(feature)
train_transaction_16.describe(include=['O']).T
# wordFrequencyPlot(train_transaction_16,'ProductCD')
col = 'ProductCD'

target = 'isFraud'    

sizes, labels = piePlotLables(train_transaction_16,col,target,0)

piePlot(sizes,labels,'Not Fraud(0)')
col = 'ProductCD'

target = 'isFraud'    

sizes, labels = piePlotLables(train_transaction_16,col,target,1)

piePlot(sizes,labels,'Fraud(1)')
# wordFrequencyPlot(train_transaction_16,'card4')
wordFrequencyPlot(train_transaction_16,'card6')
wordFrequencyPlot(train_transaction_16,'P_emaildomain')
# wordFrequencyPlot(train_transaction_16,'R_emaildomain')
# PlotingMissingValues(train_transaction_16).PlotMissingValue()
train_transaction[C_col].head()
C_col_info = Exploration(train_transaction[C_col]).DescribeAnalysis(True)
PlotingMissingValues(train_transaction[C_col]).PlotMissingValue()
train_transaction[D_col].head()
D_col_info = Exploration(train_transaction[D_col]).DescribeAnalysis(True)
# PlotingMissingValues(train_transaction[D_col]).PlotMissingValue()
train_transaction[M_col].head()
train_transaction[M_col].describe(include=['O'])
wordFrequencyPlot(train_transaction[M_col],'M4')
col = 'M4'

target = 'isFraud'    

sizes, labels = piePlotLables(train_transaction,col,target,0)

piePlot(sizes,labels,'NOT Fraud(0)')
col = 'M4'

target = 'isFraud'    

sizes, labels = piePlotLables(train_transaction,col,target,1)

piePlot(sizes,labels,'Fraud(1)')
from sklearn import base

from sklearn.model_selection import KFold



class KFoldMeanEncoder(base.BaseEstimator, base.TransformerMixin):



    def __init__(self, colnames,targetName,n_fold=5,verbosity=True,discardOriginal_col=False):



        self.colnames = colnames

        self.targetName = targetName

        self.n_fold = n_fold

        self.verbosity = verbosity

        self.discardOriginal_col = discardOriginal_col



    def fit(self, X, y=None):

        return self





    def transform(self,X):



        assert(type(self.targetName) == str)

        assert(type(self.colnames) == list)



        mean_of_target = X[self.targetName].mean()

        kf = KFold(n_splits = self.n_fold, shuffle = False)



        for col in self.colnames:



            col_mean_name = col + '_' + 'KfoldMeanEnc'

            X[col_mean_name] = np.nan



            for tr_ind, val_ind in kf.split(X):

                X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]

                X.loc[X.index[val_ind], col_mean_name] = X_val[col].map(X_tr.groupby(col)[self.targetName].mean())



            X[col_mean_name].fillna(mean_of_target, inplace = True)



            if self.verbosity:

                #print correlation

                encoded_feature = X[col_mean_name].values

                print('Correlation between the new feature, {} and, {} is {:.4f}.'.format(col_mean_name,

                                                                                      self.targetName,

                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))

        if self.discardOriginal_col:

            X = X.drop(self.targetName, axis=1)



        return X
KFME= KFoldMeanEncoder(['ProductCD','card4','card6','P_emaildomain','R_emaildomain',

                       'M1','M2','M3','M4','M5','M6','M7','M8','M9'],'isFraud')

train_transaction_KFME = KFME.fit_transform(train_transaction)