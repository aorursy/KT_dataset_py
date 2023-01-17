import os
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import numpy as np
import pandas as pd
import pylab 
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
%matplotlib inline
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import skew, norm, probplot, boxcox
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import f_classif, chi2, SelectKBest, SelectFromModel
from boruta import BorutaPy
from rfpimp import *

from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

#from sklearn.base import BaseEstimator, TransformerMixin, clone, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from itertools import combinations
train = pd.read_csv('../input/train.csv') 

test = pd.read_csv('../input/test.csv') 
Test_ID = test.PassengerId
test.insert(loc=1, column='Survived', value=-1)

data = pd.concat([train, test], ignore_index=True)
def rstr(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str
details = rstr(data.loc[: ,'Survived' : 'Embarked'], 'Survived')
details.sort_values(by='corr Survived', ascending=False)
print('Data is not balanced! Has {:2.2%} survives'.format(train.Survived.describe()[1]))
display(data.loc[: ,'Pclass' : 'Embarked'].describe().transpose())
print('Survived: [1] Survived; [0] Died; [-1] Test Data set:\n',data.Survived.value_counts())
def charts(feature, df):
    print('\n _____________________ Plots of', feature, 'per Survived and Dead: ____________________')
    # Pie of all Data
    fig = plt.figure(figsize=(20,5))
    f1 = fig.add_subplot(131)
    cnt = df[feature].value_counts()
    g = plt.pie(cnt, labels=cnt.index, autopct='%1.1f%%', shadow=True, startangle=90)
    
    # Count Plot By Survived and Dead
    f = fig.add_subplot(132)
    g = sns.countplot(x=feature, hue='Survived', hue_order=[1,0], data=df, ax=f)

    # Percent stacked Plot
    survived = df[df['Survived']==1][feature].value_counts()
    dead = df[df['Survived']==0][feature].value_counts()
    df2 = pd.DataFrame([survived,dead])
    df2.index = ['Survived','Dead']
    df2 = df2.T
    df2 = df2.fillna(0)
    df2['Total'] = df2.Survived + df2.Dead
    df2.Survived = df2.Survived/df2.Total
    df2.Dead = df2.Dead/df2.Total
    df2.drop(['Total'], axis=1, inplace=True)
    f = fig.add_subplot(133)
    df2.plot(kind='bar', stacked=True, ax=f)
    del df2, g, f, cnt, dead, fig
same_ticket = data.Ticket.value_counts()
data['qtd_same_ticket'] = data.Ticket.apply(lambda x: same_ticket[x])
del same_ticket
charts('qtd_same_ticket', data[data.Survived>=0])
data[(data.qtd_same_ticket==11)]
data['qtd_same_ticket_bin'] = data.qtd_same_ticket.apply(lambda x: 3 if (x>2 and x<5) else (5 if x>4 else x))
charts('qtd_same_ticket_bin', data[data.Survived>=0])
print('Percent. survived from unique ticket: {:3.2%}'.\
      format(data.Survived[(data.qtd_same_ticket==1) & (data.Survived>=0)].sum()/
             data.Survived[(data.qtd_same_ticket==1) & (data.Survived>=0)].count()))
print('Percent. survived from same tickets: {:3.2%}'.\
      format(data.Survived[(data.qtd_same_ticket>1) & (data.Survived>=0)].sum()/
             data.Survived[(data.qtd_same_ticket>1) & (data.Survived>=0)].count()))

data['same_tckt'] = data.qtd_same_ticket.apply(lambda x: 1 if (x> 1) else 0)
charts('same_tckt', data[data.Survived>=0])
data.Ticket.str.findall('[A-z]').apply(lambda x: ''.join(map(str, x))).value_counts().head(7)
data['distinction_in_tikect'] =\
   (data.Ticket.str.findall('[A-z]').apply(lambda x: ''.join(map(str, x)).strip('[]')))

data.distinction_in_tikect = data.distinction_in_tikect.\
  apply(lambda y: 'Without' if y=='' else y if (y in ['PC', 'CA', 'A', 'SOTONOQ', 'STONO', 'WC', 'SCPARIS']) else 'Others')

charts('distinction_in_tikect', data[(data.Survived>=0)])
data.distinction_in_tikect = data.distinction_in_tikect.\
  apply(lambda y: 'Others' if (y in ['Without', 'Others', 'CA']) else\
        'Low' if (y in ['A', 'SOTONOQ', 'WC']) else\
        'High' if (y in ['STONO', 'SCPARIS']) else y)

charts('distinction_in_tikect', data[(data.Survived>=0)])
# Fill null with median of most likely type passenger
data.loc[data.Fare.isnull(), 'Fare'] = data.Fare[(data.Pclass==3) & (data.qtd_same_ticket==1) & (data.Age>60)].median()

fig = plt.figure(figsize=(20,5))
f = fig.add_subplot(121)
g = sns.distplot(data[(data.Survived>=0)].Fare)
f = fig.add_subplot(122)
g = sns.boxplot(y='Fare', x='Survived', data=data[data.Survived>=0], notch = True)
data['passenger_fare'] = data.Fare / data.qtd_same_ticket

fig = plt.figure(figsize=(20,6))
a = fig.add_subplot(141)
g = sns.distplot(data[(data.Survived>=0)].passenger_fare)
a = fig.add_subplot(142)
g = sns.boxplot(y='passenger_fare', x='Survived', data=data[data.Survived>=0], notch = True)
a = fig.add_subplot(143)
g = pd.qcut(data.Fare[(data.Survived==0)], q=[.0, .25, .50, .75, 1.00]).value_counts().plot(kind='bar', ax=a, title='Died')
a = fig.add_subplot(144)
g = pd.qcut(data.Fare[(data.Survived>0)], q=[.0, .25, .50, .75, 1.00]).value_counts().plot(kind='bar', ax=a, title='Survived')
plt.tight_layout(); plt.show()
print('Passengers with higets passenger fare:')
display(data[data.passenger_fare>120])
print('\nSurivived of passenger fare more than 50:\n',
    pd.pivot_table(data.loc[data.passenger_fare>50, ['Pclass', 'Survived']], aggfunc=np.count_nonzero, 
                       columns=['Survived'] , index=['Pclass']))
charts('Pclass', data[(data.Survived>=0)])
charts('SibSp', data[(data.Survived>=0)])
data['SibSp_bin'] = data.SibSp.apply(lambda x: 6 if x > 2 else x)
charts('SibSp_bin', data[(data.Survived>=0)])
charts('Parch', data[data.Survived>=0])
data['Parch_bin'] = data.Parch.apply(lambda x: x if x< 3 else 4)
charts('Parch_bin', data[(data.Survived>=0)])
data['family'] = data.SibSp + data.Parch + 1
charts('family', data[data.Survived>=0])
charts('Pclass', data[(data.family>4) & (data.Survived>=0)])
data['non_relatives'] = data.qtd_same_ticket - data.family
charts('non_relatives', data[data.Survived>=0])
charts('Sex', data[(data.Survived>=0)])
display(data[data.Embarked.isnull()])
data.loc[data.Embarked=='NA', 'Embarked'] = data[(data.Cabin.str.match('B2')>0) & (data.Pclass==1)].Embarked.mode()[0]
charts('Embarked', data[(data.Survived>=0)])
def Personal_Titles(df):
    df['Personal_Titles'] = df.Name.str.findall('Mrs\.|Mr\.|Miss\.|Maste[r]|Dr\.|Lady\.|Countess\.|'
                                                +'Sir\.|Rev\.|Don\.|Major\.|Col\.|Jonkheer\.|'
                                                + 'Capt\.|Ms\.|Mme\.|Mlle\.').apply(lambda x: ''.join(map(str, x)).strip('[]'))

    df.Personal_Titles[df.Personal_Titles=='Mrs.'] = 'Mrs'
    df.Personal_Titles[df.Personal_Titles=='Mr.'] = 'Mr'
    df.Personal_Titles[df.Personal_Titles=='Miss.'] = 'Miss'
    df.Personal_Titles[df.Personal_Titles==''] = df[df.Personal_Titles==''].Sex.apply(lambda x: 'Mr' if (x=='male') else 'Mrs')
    df.Personal_Titles[df.Personal_Titles=='Mme.'] = 'Mrs' 
    df.Personal_Titles[df.Personal_Titles=='Ms.'] = 'Mrs'
    df.Personal_Titles[df.Personal_Titles=='Lady.'] = 'Royalty'
    df.Personal_Titles[df.Personal_Titles=='Mlle.'] = 'Miss'
    df.Personal_Titles[(df.Personal_Titles=='Miss.') & (df.Age>-1) & (df.Age<15)] = 'Kid' 
    df.Personal_Titles[df.Personal_Titles=='Master'] = 'Kid'
    df.Personal_Titles[df.Personal_Titles=='Don.'] = 'Royalty'
    df.Personal_Titles[df.Personal_Titles=='Jonkheer.'] = 'Royalty'
    df.Personal_Titles[df.Personal_Titles=='Capt.'] = 'Technical'
    df.Personal_Titles[df.Personal_Titles=='Rev.'] = 'Technical'
    df.Personal_Titles[df.Personal_Titles=='Sir.'] = 'Royalty'
    df.Personal_Titles[df.Personal_Titles=='Countess.'] = 'Royalty'
    df.Personal_Titles[df.Personal_Titles=='Major.'] = 'Technical'
    df.Personal_Titles[df.Personal_Titles=='Col.'] = 'Technical'
    df.Personal_Titles[df.Personal_Titles=='Dr.'] = 'Technical'

Personal_Titles(data)
display(pd.pivot_table(data[['Personal_Titles', 'Survived']], aggfunc=np.count_nonzero, 
                       columns=['Survived'] , index=['Personal_Titles']).T)

charts('Personal_Titles', data[(data.Survived>=0)])
data['distinction_in_name'] =\
   ((data.Name.str.findall('\(').apply(lambda x: ''.join(map(str, x)).strip('[]'))=='(')
    | (data.Name.str.findall(r'"[A-z"]*"').apply(lambda x: ''.join(map(str, x)).strip('""'))!=''))

data.distinction_in_name = data.distinction_in_name.apply(lambda x: 1 if x else 0)

charts('distinction_in_name', data[(data.Survived>=0)])
print('Total of differents surnames aboard:',
      ((data.Name.str.findall(r'[A-z]*\,').apply(lambda x: ''.join(map(str, x)).strip(','))).value_counts()>1).shape[0])
print('More then one persons aboard with smae surnames:',
      ((data.Name.str.findall(r'[A-z]*\,').apply(lambda x: ''.join(map(str, x)).strip(','))).value_counts()>1).sum())

surnames = (data.Name.str.findall(r'[A-z]*\,').apply(lambda x: ''.join(map(str, x)).strip(','))).value_counts()

data['surname'] = (data.Name.str.findall(r'[A-z]*\,').\
 apply(lambda x: ''.join(map(str, x)).strip(','))).apply(lambda x: x if surnames.get_value(x)>1 else 'Alone')

test_surnames = set(data.surname[data.Survived>=0].unique().tolist())
print('Surnames with more than one member aboard that happens only in the test data set:', 
      240-len(test_surnames))

train_surnames = set(data.surname[data.Survived<0].unique().tolist())
print('Surnames with more than one member aboard that happens only in the train data set:', 
      240-len(train_surnames))

both_surnames = test_surnames.intersection(train_surnames)

data.surname = data.surname.apply(lambda x : x if test_surnames.issuperset(set([x])) else 'Exclude')

del surnames, both_surnames, test_surnames, train_surnames
CabinByTicket = data.loc[~data.Cabin.isnull(), ['Ticket', 'Cabin']].groupby(by='Ticket').agg(min)
before = data.Cabin.isnull().sum()
data.loc[data.Cabin.isnull(), 'Cabin'] = data.loc[data.Cabin.isnull(), 'Ticket'].\
   apply(lambda x: CabinByTicket[CabinByTicket.index==x].min())
print('Cabin nulls reduced:', (before - data.Cabin.isnull().sum()))
del CabinByTicket, before
data.Cabin[data.Cabin.isnull()] = 'N999'
data['Cabin_Letter'] = data.Cabin.str.findall('[^a-z]\d\d*')
data['Cabin_Number'] = data.apply(lambda x: 0 if len(str(x.Cabin))== 1 else np.int(np.int(x.Cabin_Letter[0][1:])/10), axis=1)
data.Cabin_Letter = data.apply(lambda x: x.Cabin if len(str(x.Cabin))== 1 else x.Cabin_Letter[0][0], axis=1)

display(data[['Fare', 'Cabin_Letter']].groupby(['Cabin_Letter']).agg([np.median, np.mean, np.count_nonzero, np.max, np.min]))
display(data[data.Cabin=='T'])
display(data.Cabin_Letter[data.passenger_fare==35.5].value_counts())

data.Cabin_Letter[data.Cabin_Letter=='T'] = 'C'
data.loc[(data.passenger_fare<6.237) & (data.passenger_fare>=0.0) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
  data[(data.passenger_fare<6.237) & (data.passenger_fare>=0.0) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare<6.237) & (data.passenger_fare>=0.0) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Number'] =\
  data[(data.passenger_fare<6.237) & (data.passenger_fare>=0.0) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare<7.225) & (data.passenger_fare>=6.237) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
  data[(data.passenger_fare<7.225) & (data.passenger_fare>=6.237) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare<7.225) & (data.passenger_fare>=6.237) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Number'] =\
  data[(data.passenger_fare<7.225) & (data.passenger_fare>=6.237) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare<7.65) & (data.passenger_fare>=7.225) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
  data[(data.passenger_fare<7.65) & (data.passenger_fare>=7.225) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare<7.65) & (data.passenger_fare>=7.225) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Number'] =\
  data[(data.passenger_fare<7.65) & (data.passenger_fare>=7.225) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Number.min()

data.loc[(data.passenger_fare<7.75) & (data.passenger_fare>=7.65) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
  data[(data.passenger_fare<7.75) & (data.passenger_fare>=7.65) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare<7.75) & (data.passenger_fare>=7.65) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Number'] =\
  data[(data.passenger_fare<7.75) & (data.passenger_fare>=7.65) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Number.min()

data.loc[(data.passenger_fare<8.0) & (data.passenger_fare>=7.75) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
  data[(data.passenger_fare<8.0) & (data.passenger_fare>=7.75) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare<8.0) & (data.passenger_fare>=7.75) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Number'] =\
  data[(data.passenger_fare<8.0) & (data.passenger_fare>=7.75) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Number.min()

data.loc[(data.passenger_fare>=8.0) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
  data[(data.passenger_fare>=8.0) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>=8.0) & (data.Pclass==3) & (data.Cabin=='N999'), 'Cabin_Number'] =\
  data[(data.passenger_fare>=8.0) & (data.Pclass==3) & (data.Cabin!='N999')].Cabin_Number.mode()[0]
data.loc[(data.passenger_fare>=0) & (data.passenger_fare<8.59) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>=0) & (data.passenger_fare<8.59) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>=0) & (data.passenger_fare<8.59) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>=0) & (data.passenger_fare<8.59) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>=8.59) & (data.passenger_fare<10.5) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>=8.59) & (data.passenger_fare<10.5) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>=8.59) & (data.passenger_fare<10.5) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>=8.59) & (data.passenger_fare<10.5) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>=10.5) & (data.passenger_fare<10.501) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>=10.5) & (data.passenger_fare<10.501) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>=10.5) & (data.passenger_fare<10.501) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>=10.5) & (data.passenger_fare<10.501) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>=10.501) & (data.passenger_fare<12.5) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>=10.501) & (data.passenger_fare<12.5) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>=10.501) & (data.passenger_fare<12.5) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>=10.501) & (data.passenger_fare<12.5) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>=12.5) & (data.passenger_fare<13.) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>=12.5) & (data.passenger_fare<13.) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>=12.5) & (data.passenger_fare<13.) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>=12.5) & (data.passenger_fare<13.) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>=13.) & (data.passenger_fare<13.1) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>=13.) & (data.passenger_fare<13.1) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>=13.) & (data.passenger_fare<13.1) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>=13.) & (data.passenger_fare<13.1) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>=13.1) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>=13.1) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>=13.1) & (data.Pclass==2) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>=13.1) & (data.Pclass==2) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare==0) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare==0) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare==0) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare==0) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>0) & (data.passenger_fare<=19.69) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>0) & (data.passenger_fare<=19.69) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>0) & (data.passenger_fare<=19.69) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>0) & (data.passenger_fare<=19.69) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>19.69) & (data.passenger_fare<=23.374) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>19.69) & (data.passenger_fare<=23.374) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>19.69) & (data.passenger_fare<=23.374) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>19.69) & (data.passenger_fare<=23.374) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>23.374) & (data.passenger_fare<=25.25) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>23.374) & (data.passenger_fare<=25.25) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>23.374) & (data.passenger_fare<=25.25) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>23.374) & (data.passenger_fare<=25.25) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>25.69) & (data.passenger_fare<=25.929) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>25.69) & (data.passenger_fare<=25.929) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>25.69) & (data.passenger_fare<=25.929) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>25.69) & (data.passenger_fare<=25.929) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>25.99) & (data.passenger_fare<=26.) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>25.99) & (data.passenger_fare<=26.) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>25.99) & (data.passenger_fare<=26.) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>25.99) & (data.passenger_fare<=26.) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>26.549) & (data.passenger_fare<=26.55) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>26.549) & (data.passenger_fare<=26.55) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>26.549) & (data.passenger_fare<=26.55) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>26.549) & (data.passenger_fare<=26.55) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>27.4) & (data.passenger_fare<=27.5) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>27.4) & (data.passenger_fare<=27.5) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>27.4) & (data.passenger_fare<=27.5) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>27.4) & (data.passenger_fare<=27.5) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>27.7207) & (data.passenger_fare<=27.7208) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>27.7207) & (data.passenger_fare<=27.7208) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>27.7207) & (data.passenger_fare<=27.7208) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>27.7207) & (data.passenger_fare<=27.7208) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>29.69) & (data.passenger_fare<=29.7) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>29.69) & (data.passenger_fare<=29.7) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>29.69) & (data.passenger_fare<=29.7) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>29.69) & (data.passenger_fare<=29.7) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>30.49) & (data.passenger_fare<=30.5) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>30.49) & (data.passenger_fare<=30.5) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>30.49) & (data.passenger_fare<=30.5) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>30.49) & (data.passenger_fare<=30.5) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>30.6) & (data.passenger_fare<=30.7) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>30.6) & (data.passenger_fare<=30.7) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>30.6) & (data.passenger_fare<=30.7) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>30.6) & (data.passenger_fare<=30.7) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>31.67) & (data.passenger_fare<=31.684) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>31.67) & (data.passenger_fare<=31.684) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>31.67) & (data.passenger_fare<=31.684) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>31.67) & (data.passenger_fare<=31.684) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>39.599) & (data.passenger_fare<=39.6) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>39.599) & (data.passenger_fare<=39.6) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>39.599) & (data.passenger_fare<=39.6) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>39.599) & (data.passenger_fare<=39.6) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>41) & (data.passenger_fare<=41.2) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>41) & (data.passenger_fare<=41.2) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>41) & (data.passenger_fare<=41.2) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>41) & (data.passenger_fare<=41.2) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>45.49) & (data.passenger_fare<=45.51) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>45.49) & (data.passenger_fare<=45.51) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>45.49) & (data.passenger_fare<=45.51) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>45.49) & (data.passenger_fare<=45.51) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>49.5) & (data.passenger_fare<=49.51) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>49.5) & (data.passenger_fare<=49.51) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>49.5) & (data.passenger_fare<=49.51) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>49.5) & (data.passenger_fare<=49.51) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]

data.loc[(data.passenger_fare>65) & (data.passenger_fare<=70) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Letter'] =\
    data[(data.passenger_fare>65) & (data.passenger_fare<=70) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Letter.mode()[0]
data.loc[(data.passenger_fare>65) & (data.passenger_fare<=70) & (data.Pclass==1) & (data.Cabin=='N999'), 'Cabin_Number'] =\
    data[(data.passenger_fare>65) & (data.passenger_fare<=70) & (data.Pclass==1) & (data.Cabin!='N999')].Cabin_Number.mode()[0]
charts('Cabin_Letter', data[(data.Survived>=0)])
display(data[data.Name.str.findall('Bourke').apply(lambda x: ''.join(map(str, x)).strip('[]'))=='Bourke'])
family_w_age = data.Ticket[(data.Parch>0) & (data.SibSp>0) & (data.Age==-1)].unique().tolist()
data['sons'] = data.apply(lambda x : \
                          1 if ((x.Ticket in (['2661', '2668', 'A/5. 851', '4133'])) & (x.SibSp>0)) else 0, axis=1)

data.sons += data.apply(lambda x : \
                        1 if ((x.Ticket in (['CA. 2343'])) & (x.SibSp>1)) else 0, axis=1)


data.sons += data.apply(lambda x : \
                        1 if ((x.Ticket in (['W./C. 6607'])) & (x.Personal_Titles not in (['Mr', 'Mrs']))) else 0, axis=1)

data.sons += data.apply(lambda x: 1 if ((x.Parch>0) & (x.Age>=0) & (x.Age<20)) else 0, axis=1)
data.sons.loc[data.PassengerId==594] = 1 # Sun with diferente pattern (family with two tickets)
data.sons.loc[data.PassengerId==1252] = 1 # Case of 'CA. 2343' and last rule
data.sons.loc[data.PassengerId==1084] = 1 # Case of 'A/5. 851' and last rule
data.sons.loc[data.PassengerId==1231] = 1 # Case of 'A/5. 851' and last rule

charts('sons', data[(data.Survived>=0)])
data['parents'] = data.apply(lambda x : \
                              1 if ((x.Ticket in (['2661', '2668', 'A/5. 851', '4133'])) & (x.SibSp==0)) else 0, axis=1)

data.parents += data.apply(lambda x : \
                              1 if ((x.Ticket in (['CA. 2343'])) & (x.SibSp==1)) else 0, axis=1)

data.parents += data.apply(lambda x : 1 if ((x.Ticket in (['W./C. 6607'])) & (x.Personal_Titles in (['Mr', 'Mrs']))) \
                                else 0, axis=1)

# Identify parents and care nulls ages
data.parents += data.apply(lambda x: 1 if ((x.Parch>0) & (x.SibSp>0) & (x.Age>19) & (x.Age<=45) ) else 0, axis=1)
charts('parents', data[(data.Survived>=0)])
data['parent_alone'] = data.apply(lambda x: 1 if ((x.Parch>0) & (x.SibSp==0) & (x.Age>19) & (x.Age<=45) ) else 0, axis=1)
charts('parent_alone', data[(data.Survived>=0)])
t_p_alone = data.Ticket[data.parent_alone==1].tolist()

data[data.Ticket.isin(t_p_alone)].sort_values('Ticket')[96:]

data.parent_alone.loc[data.PassengerId==141] = 1

data.parent_alone.loc[data.PassengerId==541] = 0
data.sons.loc[data.PassengerId==541] = 1

data.parent_alone.loc[data.PassengerId==1078] = 0
data.sons.loc[data.PassengerId==1078] = 1

data.parent_alone.loc[data.PassengerId==98] = 0
data.sons.loc[data.PassengerId==98] = 1

data.parent_alone.loc[data.PassengerId==680] = 0
data.sons.loc[data.PassengerId==680] = 1

data.parent_alone.loc[data.PassengerId==915] = 0
data.sons.loc[data.PassengerId==915] = 1

data.parent_alone.loc[data.PassengerId==333] = 0
data.sons.loc[data.PassengerId==333] = 1

data.parent_alone.loc[data.PassengerId==119] = 0
data.sons[data.PassengerId==119] = 1

data.parent_alone.loc[data.PassengerId==319] = 0
data.sons.loc[data.PassengerId==319] = 1

data.parent_alone.loc[data.PassengerId==103] = 0
data.sons.loc[data.PassengerId==103] = 1

data.parents.loc[data.PassengerId==154] = 0
data.sons.loc[data.PassengerId==1084] = 1

data.parents.loc[data.PassengerId==581] = 0
data.sons.loc[data.PassengerId==581] = 1

data.parent_alone.loc[data.PassengerId==881] = 0
data.sons.loc[data.PassengerId==881] = 1

data.parent_alone.loc[data.PassengerId==1294] = 0
data.sons.loc[data.PassengerId==1294] = 1

data.parent_alone.loc[data.PassengerId==378] = 0
data.sons.loc[data.PassengerId==378] = 1

data.parent_alone.loc[data.PassengerId==167] = 1
data.parent_alone.loc[data.PassengerId==357] = 0
data.sons.loc[data.PassengerId==357] = 1

data.parent_alone.loc[data.PassengerId==918] = 0
data.sons.loc[data.PassengerId==918] = 1

data.parent_alone.loc[data.PassengerId==1042] = 0
data.sons.loc[data.PassengerId==1042] = 1

data.parent_alone.loc[data.PassengerId==540] = 0
data.sons.loc[data.PassengerId==540] = 1

data.parents += data.parent_alone 
charts('parents', data[(data.Survived>=0)])
data['grandparents'] = data.apply(lambda x: 1 if ((x.Parch>0) & (x.SibSp>0) & (x.Age>19) & (x.Age>45) ) else 0, axis=1)
charts('grandparents', data[(data.Survived>=0)])
data['grandparent_alone'] = data.apply(lambda x: 1 if ((x.Parch>0) & (x.SibSp==0) & (x.Age>45) ) else 0, axis=1)
charts('grandparent_alone', data[(data.Survived>=0)])
data.parents += data.grandparent_alone + data.grandparents
charts('parents', data[(data.Survived>=0)])
data['relatives'] = data.apply(lambda x: 1 if ((x.SibSp>0) & (x.Parch==0)) else 0, axis=1)
charts('relatives', data[(data.Survived>=0)])
data['companions'] = data.apply(lambda x: 1 if ((x.SibSp==0) & (x.Parch==0) & (x.same_tckt==1)) else 0, axis=1)
charts('companions', data[(data.Survived>=0)])
data['alone'] = data.apply(lambda x: 1 if ((x.SibSp==0) & (x.Parch==0) & (x.same_tckt==0)) else 0, axis=1)
charts('alone', data[(data.Survived>=0)])
fig = plt.figure(figsize=(20, 10))
fig1 = fig.add_subplot(221)
g = sns.distplot(data.Age.fillna(0), fit=norm, label='Nulls as Zero')
g = sns.distplot(data[~data.Age.isnull()].Age, fit=norm, label='Withou Nulls')
plt.legend(loc='upper right')
print('Survived without Age:')
display(data[data.Age.isnull()].Survived.value_counts())
fig2 = fig.add_subplot(222)
g = sns.scatterplot(data = data[(~data.Age.isnull())], y='Age', x='SibSp',  hue='Survived')
print('Mean and median ages by siblings:')
data.loc[data.Age.isnull(), 'Age'] = -1
display(data.loc[(data.Age>=0), ['SibSp', 'Age']].groupby('SibSp').agg([np.mean, np.median]).T)

print('\nMedian ages by Personal_Titles:')
Ages = { 'Age' : {'median'}}
display(data[data.Age>=0][['Age', 'Personal_Titles', 'parents', 'grandparents', 'sons', 'relatives', 'companions', 'alone']].\
        groupby('Personal_Titles').agg(Ages).T)

print('\nMedian ages by Personal Titles and Family Relationships:')
display(pd.pivot_table(data[data.Age>=0][['Age', 'Personal_Titles', 'parents', 'grandparents', 
                                          'sons', 'relatives', 'companions','alone']],
                       aggfunc=np.median, 
                       index=['parents', 'grandparents', 'sons', 'relatives', 'companions', 'alone'] , 
                       columns=['Personal_Titles']))

print('\nNulls ages by Personal Titles and Family Relationships:')
display(data[data.Age<0][['Personal_Titles', 'parents', 'grandparents', 'sons', 'relatives', 'companions', 'alone']].\
        groupby('Personal_Titles').agg([sum]))
data['Without_Age'] = data.Age.apply(lambda x: 0 if x>0 else 1)

data.Age.loc[(data.Age<0) & (data.companions==1) & (data.Personal_Titles=='Miss')] = \
   data.Age[(data.Age>=0) & (data.companions==1) & (data.Personal_Titles=='Miss')].median()

data.Age.loc[(data.Age<0) & (data.companions==1) & (data.Personal_Titles=='Mr')] = \
   data.Age[(data.Age>=0) & (data.companions==1) & (data.Personal_Titles=='Mr')].median()

data.Age.loc[(data.Age<0) & (data.companions==1) & (data.Personal_Titles=='Mrs')] = \
   data.Age[(data.Age>=0) & (data.companions==1) & (data.Personal_Titles=='Mrs')].median()

data.Age.loc[(data.Age<0) & (data.alone==1) & (data.Personal_Titles=='Kid')] = \
   data.Age[(data.Age>=0) & (data.alone==1) & (data.Personal_Titles=='Kid')].median()

data.Age.loc[(data.Age<0) & (data.alone==1) & (data.Personal_Titles=='Technical')] = \
   data.Age[(data.Age>=0) & (data.alone==1) & (data.Personal_Titles=='Technical')].median()

data.Age.loc[(data.Age<0) & (data.alone==1) & (data.Personal_Titles=='Miss')] = \
   data.Age[(data.Age>=0) & (data.alone==1) & (data.Personal_Titles=='Miss')].median()

data.Age.loc[(data.Age<0) & (data.alone==1) & (data.Personal_Titles=='Mr')] = \
   data.Age[(data.Age>=0) & (data.alone==1) & (data.Personal_Titles=='Mr')].median()

data.Age.loc[(data.Age<0) & (data.alone==1) & (data.Personal_Titles=='Mrs')] = \
   data.Age[(data.Age>=0) & (data.alone==1) & (data.Personal_Titles=='Mrs')].median()

data.Age.loc[(data.Age<0) & (data.parents==1) & (data.Personal_Titles=='Mr')] = \
   data.Age[(data.Age>=0) & (data.parents==1) & (data.Personal_Titles=='Mr')].median()

data.Age.loc[(data.Age<0) & (data.parents==1) & (data.Personal_Titles=='Mrs')] = \
   data.Age[(data.Age>=0) & (data.parents==1) & (data.Personal_Titles=='Mrs')].median()

data.Age.loc[(data.Age<0) & (data.sons==1) & (data.Personal_Titles=='Kid')] = \
   data.Age[(data.Age>=0) & (data.Personal_Titles=='Kid')].median()
data.Age.loc[(data.Age.isnull()) & (data.sons==1) & (data.Personal_Titles=='Kid')] = \
   data.Age[(data.Age>=0) & (data.Personal_Titles=='Kid')].median()

data.Age.loc[(data.Age<0) & (data.sons==1) & (data.Personal_Titles=='Miss')] = \
   data.Age[(data.Age>=0) & (data.sons==1) & (data.Personal_Titles=='Miss')].median()

data.Age.loc[(data.Age<0) & (data.sons==1) & (data.Personal_Titles=='Mr')] = \
   data.Age[(data.Age>=0) & (data.sons==1) & (data.Personal_Titles=='Mr')].median()

data.Age.loc[(data.Age<0) & (data.sons==1) & (data.Personal_Titles=='Mrs')] = \
   data.Age[(data.Age>=0) & (data.sons==1) & (data.Personal_Titles=='Mrs')].median()

data.Age.loc[(data.Age<0) & (data.relatives==1) & (data.Personal_Titles=='Miss')] = \
   data.Age[(data.Age>=0) & (data.relatives==1) & (data.Personal_Titles=='Miss')].median()

data.Age.loc[(data.Age<0) & (data.relatives==1) & (data.Personal_Titles=='Mr')] = \
   data.Age[(data.Age>=0) & (data.sons==1) & (data.Personal_Titles=='Mr')].median()

data.Age.loc[(data.Age<0) & (data.relatives==1) & (data.Personal_Titles=='Mrs')] = \
   data.Age[(data.Age>=0) & (data.relatives==1) & (data.Personal_Titles=='Mrs')].median()

print('Age correlation with survived:',data.corr()['Survived'].Age)
g = sns.distplot(data.Age, fit=norm, label='With nulls filled')
plt.legend(loc='upper right')
plt.show()
def binningAge(df):
    # Binning Age based on custom ranges
    bin_ranges = [0, 1.7, 8, 15, 18, 25, 55, 65, 100] 
    bin_names = [0, 1, 2, 3, 4, 5, 6, 7]
    df['Age_bin_custom_range'] = pd.cut(np.array(df.Age), bins=bin_ranges)
    df['Age_bin_custom_label'] = pd.cut(np.array(df.Age), bins=bin_ranges, labels=bin_names)
    return df

data = binningAge(data)
display(data[['Age', 'Age_bin_custom_range', 'Age_bin_custom_label']].sample(5))
display(pd.pivot_table(data[['Age_bin_custom_range', 'Survived']], aggfunc=np.count_nonzero, 
                       index=['Survived'] , columns=['Age_bin_custom_range']))
charts('Age_bin_custom_label', data[(data.Survived>=0)])
data['genre'] = data.Sex.apply(lambda x: 1 if x=='male' else 0)
data.drop(['Name', 'Cabin', 'Ticket', 'Sex', 'same_tckt', 'qtd_same_ticket', 'parent_alone', 'grandparents', 
           'grandparent_alone', 'Age_bin_custom_range'], axis=1, inplace=True) # , 'Age', 'Parch', 'SibSp',
data = pd.get_dummies(data, columns = ['Cabin_Letter', 'Personal_Titles', 'Embarked', 'distinction_in_tikect'])

data = pd.get_dummies(data, columns = ['surname']) # 'Age_bin_custom_label'
data.drop(['surname_Exclude'], axis=1, inplace=True)
corr = data.loc[:, 'Survived':].corr()
top_corr_cols = corr[abs(corr.Survived)>=0.06].Survived.sort_values(ascending=False).keys()
top_corr = corr.loc[top_corr_cols, top_corr_cols]
dropSelf = np.zeros_like(top_corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
plt.figure(figsize=(15, 15))
sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)
sns.set(font_scale=0.8)
plt.show()
display(corr[(abs(corr.Survived)>=0.05) & (abs(corr.Survived)<0.06)].Survived.sort_values(ascending=False).keys())
del corr, dropSelf, top_corr
def VIF(predict, dt, y):
    scale = StandardScaler(with_std=False)
    df = pd.DataFrame(scale.fit_transform(dt.loc[dt[predict]>=0, cols]), columns= cols)
    features = "+".join(cols)
    df2 = pd.concat([y, df], axis=1)

    # get y and X dataframes based on this regression:
    y, X = dmatrices(predict + ' ~' + features, data = df2, return_type='dataframe')

    #Step 2: Calculate VIF Factors
    # For each X, calculate VIF and save in dataframe
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns

    #Step 3: Inspect VIF Factors
    display(vif.sort_values('VIF Factor'))
    return vif

#Step 1: Remove the higest correlations and run a multiple regression
cols = [ 'family',
         'non_relatives',
         'surname_Alone',
         'surname_Baclini',
         'surname_Carter',
         'surname_Richards',
         'surname_Harper', 'surname_Beckwith', 'surname_Goldenberg',
         'surname_Moor', 'surname_Chambers', 'surname_Hamalainen',
         'surname_Dick', 'surname_Taylor', 'surname_Doling', 'surname_Gordon',
         'surname_Beane', 'surname_Hippach', 'surname_Bishop',
         'surname_Mellinger', 'surname_Yarred', 
         'Pclass',
         'Age',
         'SibSp',
         'Parch',
         #'Fare',
         'qtd_same_ticket_bin',
         'passenger_fare',
         #'SibSp_bin',
         #'Parch_bin',
         'distinction_in_name',
         'Cabin_Number',
         'sons',
         'parents',
         'relatives',
         'companions',
         'alone',
         'Without_Age',
         'Age_bin_custom_label',
         'genre',
         'Cabin_Letter_A',
         'Cabin_Letter_B',
         'Cabin_Letter_C',
         'Cabin_Letter_D',
         'Cabin_Letter_E',
         'Cabin_Letter_F',
         'Cabin_Letter_G',
         'Personal_Titles_Kid',
         'Personal_Titles_Miss',
         #'Personal_Titles_Mr',
         #'Personal_Titles_Mrs',
         'Personal_Titles_Royalty',
         'Personal_Titles_Technical',
         'Embarked_C',
         'Embarked_Q',
         'Embarked_S',
         'distinction_in_tikect_High',
         'distinction_in_tikect_Low',
         'distinction_in_tikect_Others',
         'distinction_in_tikect_PC'
]

data.Age_bin_custom_label = data.Age_bin_custom_label.astype(np.uint8)
y_train = data.Survived[data.Survived>=0]
vif = VIF('Survived', data, y_train)
# Remove one feature with VIF on Inf from the same category and run a multiple regression
cols.remove('alone')
vif = VIF('Survived', data, y_train)
# Remove one feature with VIF on Inf from the same category and run a multiple regression
cols.remove('Cabin_Letter_A')

vif = VIF('Survived', data, y_train)
cols.remove('distinction_in_tikect_High')

vif = VIF('Survived', data, y_train)
cols.remove('family')

vif = VIF('Survived', data, y_train)
scale = StandardScaler()
df = pd.DataFrame(scale.fit_transform(data.loc[data.Survived>=0, cols]), columns = cols)

rf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=101)
scores = []
for i in range(df.shape[1]):
     score = cross_val_score(rf, df.iloc[:, i:i+1], y_train, scoring="accuracy", cv=10)
     scores.append((round(np.mean(score), 3), cols[i]))
MBR = pd.DataFrame(sorted(scores, reverse=True), columns=['Score', 'Feature'])
g = MBR.iloc[:15, :].plot(x='Feature', kind='barh', figsize=(20,10), fontsize=12, grid=True)
plt.show()
MBR = MBR.iloc[:15, 1]
cols = pd.Index(cols)

skb = SelectKBest(score_func=f_classif, k=10)
skb.fit(df, y_train)

select_features_kbest = skb.get_support()
feature_f_clas = cols[select_features_kbest]
feature_f_clas_scores = [(item, score) for item, score in zip(cols, skb.scores_)]
print('Total features slected by f_classif Statistical Methods',len(feature_f_clas))
fig = plt.figure(figsize=(20,7))
f1 = fig.add_subplot(121)
g = pd.DataFrame(sorted(feature_f_clas_scores, key=lambda x: -x[1])[:len(feature_f_clas)], columns=['Feature','F-Calss Score']).\
plot(x='Feature', kind='barh', title= 'F Class Score', fontsize=18, ax=f1, grid=True)

scale = MinMaxScaler()
df2 = scale.fit_transform(data.loc[data.Survived>=0, cols])
skb = SelectKBest(score_func=chi2, k=10)
skb.fit(df2, y_train)
select_features_kbest = skb.get_support()
feature_chi2 = cols[select_features_kbest]
feature_chi2_scores = [(item, score) for item, score in zip(cols, skb.scores_)]
print('Total features slected by chi2 Statistical Methods',len(feature_chi2))
f2 = fig.add_subplot(122)
g = pd.DataFrame(sorted(feature_chi2_scores, key=lambda x: -x[1])[:len(feature_chi2)], columns=['Feature','Chi2 Score']).\
plot(x='Feature', kind='barh',  title= 'Chi2 Score', fontsize=18, ax=f2, grid=True)

SMcols = set(feature_f_clas).union(set(feature_chi2))
print("Extra features select by f_class:\n", set(feature_f_clas).difference(set(feature_chi2)), '\n')
print("Extra features select by chi2:\n", set(feature_chi2).difference(set(feature_f_clas)), '\n')
print("Intersection features select by f_class and chi2:\n",set(feature_f_clas).intersection(set(feature_chi2)), '\n')
print('Total number of features selected:', len(SMcols))
print(SMcols)

plt.tight_layout(); plt.show()
logit_model=sm.Logit(y_train,df)
result=logit_model.fit(method='bfgs', maxiter=2000)
print(result.summary2())
pv_cols = cols.values

def backwardElimination(x, Y, sl, columns):
    numVars = x.shape[1]
    for i in range(0, numVars):
        regressor = sm.Logit(Y, x).fit(method='bfgs', maxiter=2000, disp=False)
        maxVar = max(regressor.pvalues) #.astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor.pvalues[j].astype(float) == maxVar):
                    columns = np.delete(columns, j)
                    x = x.loc[:, columns]
                    
    print(regressor.summary2())
    print('\nSelect {:d} features from {:d} by best p-values.'.format(len(columns), len(pv_cols)))
    print('The max p-value from the features selecte is {:.3f}.'.format(maxVar))
    
    # odds ratios and 95% CI
    conf = np.exp(regressor.conf_int())
    conf['Odds Ratios'] = np.exp(regressor.params)
    conf.columns = ['2.5%', '97.5%', 'Odds Ratios']
    display(conf)
    
    return columns, regressor

SL = 0.1
df2 = scale.fit_transform(data.loc[data.Survived>=0, pv_cols])
df2 = pd.DataFrame(df2, columns = pv_cols)

pv_cols, Logit = backwardElimination(df2, y_train, SL, pv_cols)
pred = Logit.predict(df2[pv_cols])
train = data[data.Survived>=0]
train['proba'] = pred
train['Survived'] = y_train
y_pred = pred.apply(lambda x: 1 if x > 0.5 else 0)
print('Accurancy: {0:2.2%}'.format(accuracy_score(y_true=y_train, y_pred=y_pred)))

def plot_proba(continous, predict, discret, data):
    grouped = pd.pivot_table(data, values=[predict], index=[continous, discret], aggfunc=np.mean)
    colors = 'rbgyrbgy'
    for col in data[discret].unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        plt.plot(plt_data.index.get_level_values(0), plt_data[predict], color=colors[int(col)])
    plt.xlabel(continous)
    plt.ylabel("Probabilities")
    plt.legend(np.sort(data[discret].unique()), loc='upper left', title=discret)
    plt.title("Probabilities with " + continous + " and " + discret)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(231)
plot_proba('non_relatives', 'Survived', 'Pclass', train)
ax = fig.add_subplot(232)
plot_proba('non_relatives', 'Survived', 'genre', train)
ax = fig.add_subplot(233)
plot_proba('non_relatives', 'Survived', 'qtd_same_ticket_bin', train)
ax = fig.add_subplot(234)
plot_proba('qtd_same_ticket_bin', 'Survived', 'distinction_in_name', train)
ax = fig.add_subplot(235)
plot_proba('qtd_same_ticket_bin', 'Survived', 'Embarked_S', train)
ax = fig.add_subplot(235)
plot_proba('qtd_same_ticket_bin', 'Survived', 'Embarked_S', train)
ax = fig.add_subplot(236)
plot_proba('qtd_same_ticket_bin', 'Survived', 'parents', train)
plt.show()
class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=101):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = list(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, list(p))
                scores.append(score)
                subsets.append(list(p))
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
            
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X.iloc[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train.iloc[:, indices], y_train)
        y_pred = self.estimator.predict(X_test.iloc[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
knn = KNeighborsClassifier(n_neighbors=3)
sbs = SBS(knn, k_features=1)
df2 = df.drop(['surname_Harper', 'surname_Beckwith', 'surname_Goldenberg',
                'surname_Moor', 'surname_Chambers', 'surname_Hamalainen',
                'surname_Dick', 'surname_Taylor', 'surname_Doling', 'surname_Gordon',
                'surname_Beane', 'surname_Hippach', 'surname_Bishop',
                'surname_Mellinger', 'surname_Yarred'], axis = 1)

sbs.fit(df2, y_train)

print('Best Score:',max(sbs.scores_))

k_feat = [len(k) for k in sbs.subsets_]
fig = plt.figure(figsize=(10,5))
plt.plot(k_feat, sbs.scores_, marker='o')
#plt.ylim([0.7, max(sbs.scores_)+0.01])
plt.xlim([1, len(sbs.subsets_)])
plt.xticks(np.arange(1, len(sbs.subsets_)+1))
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid(b=1)
plt.show()

print('First best accuracy with:\n',list(df.columns[sbs.subsets_[np.argmax(sbs.scores_)]]))

SBS = list(df.columns[list(sbs.subsets_[max(np.arange(0, len(sbs.scores_))[(sbs.scores_==max(sbs.scores_))])])])

print('\nBest accuracy with {0:2d} features:\n{1:}'.format(len(SBS), SBS))
from sklearn.feature_selection import RFE

lr = LogisticRegression()
rfe = RFE(estimator=lr,  step=1)
rfe.fit(df, y_train)

FRFE = cols[rfe.ranking_==1]
print('\nFeatures selected:\n',FRFE)
print('\n Total Features selected:',len(FRFE))
rfc = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=101)
rfc.fit(df, y_train)

feature_importances = [(feature, score) for feature, score in zip(cols, rfc.feature_importances_)]

MDI = cols[rfc.feature_importances_>0.010]
print('Total features slected by Random Forest:',len(MDI))

g = pd.DataFrame(sorted(feature_importances, key=lambda x: -x[1])[:len(MDI)], columns=['Feature','Importance']).\
plot(x='Feature', kind='barh', figsize=(20,7), fontsize=18, grid=True)
plt.show()
X_train, X_test, y, y_test = train_test_split(df, y_train , test_size=0.20,  random_state=101)

# Add column of random numbers
X_train['random'] = np.random.random(size=len(X_train))
X_test['random'] = np.random.random(size=len(X_test))

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, oob_score=True, random_state=101)
rf.fit(X_train, y)

imp = importances(rf, X_test, y_test, n_samples=-1) # permutation
MDA = imp[imp!=0].dropna().index
if 'random' in MDA:
   MDA =  MDA.drop('random')
print('%d features are selected.' % len(MDA))
plot_importances(imp[imp!=0].dropna(), figsize=(20,7))
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X = df.values
y = y_train.values.ravel()

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
#rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=5, n_jobs=-1, oob_score=True, random_state=101)
rf = ExtraTreesClassifier(n_estimators=100, max_depth=4, n_jobs=-1, oob_score=True, bootstrap=True, random_state=101)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=101)

# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)

shadow = cols[feat_selector.support_]
# check selected features - first 5 features are selected
print('Features selected:',shadow)

# call transform() on X to filter it down to selected features
print('Data transformaded has %d features' % feat_selector.n_features_) #feat_selector.transform(X).shape[1])
print('Check the selector ranking:')
display(pd.concat([pd.DataFrame(cols, columns=['Columns']), 
           pd.DataFrame(feat_selector.ranking_, columns=['Rank'])], axis=1).sort_values(by=['Rank']))
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# split data into train and test sets
X_train, X_test, y, y_test = train_test_split(df, y_train, test_size=0.30, random_state=101)

# fit model on all training data
model = XGBClassifier(importance_type='gain', scale_pos_weight=((len(y)-y.sum())/y.sum()))
model.fit(X_train, y)
fig=plt.figure(figsize=(20,5))
ax = fig.add_subplot(121)
g = plot_importance(model, height=0.5, ax=ax)

# Using each unique importance as a threshold
thresholds = np.sort(np.unique(model.feature_importances_)) #np.sort(model.feature_importances_[model.feature_importances_>0])
best = 0
colsbest = 31
my_model = model
threshold = 0

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier(importance_type='gain', scale_pos_weight=((len(y)-y.sum())/y.sum()))
    selection_model.fit(select_X_train, y)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh={:1.3f}, n={:d}, Accuracy: {:2.2%}".format(thresh, select_X_train.shape[1], accuracy))
    if (best <= accuracy):
        best = accuracy
        colsbest = select_X_train.shape[1]
        my_model = selection_model
        threshold = thresh
        
ax = fig.add_subplot(122)
g = plot_importance(my_model,height=0.5, ax=ax, 
                    title='The best accuracy: {:2.2%} with {:d} features'.\
                    format(best, colsbest))

feature_importances = [(score, feature) for score, feature in zip(model.feature_importances_, cols)]
XGBest = pd.DataFrame(sorted(sorted(feature_importances, reverse=True)[:colsbest]), columns=['Score', 'Feature'])
g = XGBest.plot(x='Feature', kind='barh', figsize=(20,7), fontsize=14, grid= True,
     title='Original feature importance from selected features')
plt.tight_layout(); plt.show()
XGBestCols = XGBest.iloc[:, 1].tolist()
X_train, X_test, y, y_test = train_test_split(df, y_train , test_size=0.20,  random_state=101)

# Add column of random numbers
X_train['random'] = np.random.random(size=len(X_train))
X_test['random'] = np.random.random(size=len(X_test))

svm = SGDClassifier(penalty='elasticnet', class_weight='balanced', n_jobs = - 1, random_state=101)
svm.fit(X_train, y)

imp = importances(svm, X_test, y_test, n_samples=-1) # permutation
RM = imp[imp!=0].dropna().index
if 'random' in RM:
    RM = RM.drop('random')
    
print('%d features are selected.' % len(RM))
plot_importances(imp[imp!=0].dropna(), figsize=(20,7))
bcols = set(pv_cols).union(set(FRFE)).union(set(MDI)).union(set(MDA)).union(set(MBR)).union(set(SMcols)).union(set(RM)).\
        union(set(XGBestCols)).union(set(SBS))
print("Extra features select by RFE:", set(FRFE).difference(set(pv_cols).union(set(MDI)).union(set(MDA)).union(set(MBR)).union(set(RM)).\
                                                            union(set(SMcols)).union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by pv_cols:", set(pv_cols).difference(set(FRFE).union(set(MDI)).union(set(MDA)).union(set(MBR)).union(set(SMcols)).\
                                              union(set(RM)).union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by Statistical Methods:", set(SMcols).difference(set(pv_cols).union(set(FRFE)).union(set(MDI)).\
                                                         union(set(MDA)).union(set(MBR)).union(set(RM)).\
                                                        union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by MDI:", set(MDI).difference(set(pv_cols).union(set(FRFE)).union(set(MDA)).union(set(MBR)).\
                                          union(set(SMcols)).union(set(RM)).union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by MDA:", set(MDA).difference(set(pv_cols).union(set(FRFE)).union(set(MDI)).union(set(MBR)).\
                                          union(set(SMcols)).union(set(RM)).union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by MBR:", set(MBR).difference(set(pv_cols).union(set(FRFE)).union(set(MDI)).union(set(MDA)).\
                                          union(set(SMcols)).union(set(RM)).union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by RM:", set(RM).difference(set(pv_cols).union(set(FRFE)).union(set(MDI)).union(set(MDA)).\
                                          union(set(SMcols)).union(set(MBR)).union(set(XGBestCols)).union(set(SBS))), '\n')
print("Extra features select by XGBestCols:", set(XGBestCols).difference(set(pv_cols).union(set(FRFE)).union(set(MDI)).union(set(MDA)).\
                                          union(set(SMcols)).union(set(MBR)).union(set(RM)).union(set(SBS))), '\n')
print("Extra features select by SBS:", set(SBS).difference(set(pv_cols).union(set(FRFE)).union(set(MDI)).union(set(MDA)).\
                                          union(set(SMcols)).union(set(MBR)).union(set(RM)).union(set(XGBestCols))), '\n')
print("Intersection features:",set(MDI).intersection(set(SMcols)).intersection(set(FRFE)).intersection(set(pv_cols)).\
                                  intersection(set(RM)).intersection(set(MDA)).intersection(set(MBR)).\
                                  intersection(set(XGBestCols)).intersection(set(SBS)), '\n')
print('Total number of features selected:', len(bcols))
print(bcols)
print('\n{0:2d} features removed if use the union of selections:\n{1:}'.format(len(cols.difference(bcols)), cols.difference(bcols)))
pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
res = pf.fit_transform(data[['Pclass', 'passenger_fare']])

display(pd.DataFrame(pf.powers_, columns=['Pclass', 'passenger_fare']))
del res 

# We can contact the new res with data, but we need treat the items without interactions and power, 
# or if is few features it can generate and incorporate to data manually.
data['Pclass^2'] = data.Pclass**2
data['Plcass_X_p_fare'] = data.Pclass * data.passenger_fare
data['p_fare^2'] = data.passenger_fare**2

cols = cols.insert(33, 'Pclass^2')
cols = cols.insert(34, 'Plcass_X_p_fare')
cols = cols.insert(35, 'p_fare^2')

bcols.add('Pclass^2')
bcols.add('Plcass_X_p_fare')
bcols.add('p_fare^2')

scale = StandardScaler(with_std=False)
df = pd.DataFrame(scale.fit_transform(data.loc[data.Survived>=0, cols]), columns= cols)
data.Pclass = data.Pclass.astype('category')
data.genre = data.genre.astype('category')
data.distinction_in_tikect_Low = data.distinction_in_tikect_Low.astype('category')
data.distinction_in_tikect_PC = data.distinction_in_tikect_PC.astype('category')
data.distinction_in_tikect_Others = data.distinction_in_tikect_Others.astype('category')
data.Cabin_Letter_B = data.Cabin_Letter_B.astype('category')
data.Cabin_Letter_C = data.Cabin_Letter_C.astype('category')
data.Cabin_Letter_D = data.Cabin_Letter_D.astype('category')
data.Cabin_Letter_E = data.Cabin_Letter_E.astype('category')
data.Cabin_Letter_F = data.Cabin_Letter_F.astype('category')
data.Cabin_Letter_G = data.Cabin_Letter_G.astype('category')
data.Embarked_C = data.Embarked_C.astype('category')
data.Embarked_S = data.Embarked_S.astype('category')
data.Embarked_Q = data.Embarked_Q.astype('category')
data.Personal_Titles_Royalty = data.Personal_Titles_Royalty.astype('category')
data.Personal_Titles_Technical = data.Personal_Titles_Technical.astype('category')
data.Personal_Titles_Kid = data.Personal_Titles_Kid.astype('category')
data.Personal_Titles_Mrs = data.Personal_Titles_Mrs.astype('category')
data.Personal_Titles_Mr = data.Personal_Titles_Mr.astype('category')
data.Personal_Titles_Miss = data.Personal_Titles_Miss.astype('category')
data.Without_Age = data.Without_Age.astype('category')
data.distinction_in_name = data.distinction_in_name.astype('category')
data.parents = data.parents.astype('category')
data.relatives = data.relatives.astype('category')
data.sons = data.sons.astype('category')
data.companions = data.companions.astype('category')
data.surname_Alone = data.surname_Alone.astype('category')
data.surname_Baclini = data.surname_Baclini.astype('category')
data.surname_Carter = data.surname_Carter.astype('category')
data.surname_Richards = data.surname_Richards.astype('category')
data.surname_Harper = data.surname_Harper.astype('category')
data.surname_Beckwith = data.surname_Beckwith.astype('category')
data.surname_Goldenberg = data.surname_Goldenberg.astype('category')
data.surname_Moor = data.surname_Moor.astype('category')
data.surname_Chambers = data.surname_Chambers.astype('category')
data.surname_Hamalainen = data.surname_Hamalainen.astype('category')
data.surname_Dick = data.surname_Dick.astype('category')
data.surname_Taylor = data.surname_Taylor.astype('category')
data.surname_Doling = data.surname_Doling.astype('category')
data.surname_Gordon = data.surname_Gordon.astype('category')
data.surname_Beane = data.surname_Beane.astype('category')
data.surname_Hippach = data.surname_Hippach.astype('category')
data.surname_Bishop = data.surname_Bishop.astype('category')
data.surname_Mellinger = data.surname_Mellinger.astype('category')
data.surname_Yarred = data.surname_Yarred.astype('category')
numeric_features = list(data.loc[:, cols].dtypes[data.dtypes != "category"].index)

# non_relative is skwed and have negatives values, so we need adding 6 as a shift parameter.
data['non_relatives_shift'] = data.non_relatives + 6
numeric_features.remove('non_relatives')
numeric_features.append('non_relatives_shift')

skewed_features = data[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

#compute skewness
skewness = pd.DataFrame({'Skew' :skewed_features})   

# Get only higest skewed features
skewness = skewness[abs(skewness) > 0.7]
skewness = skewness.dropna()
print ("There are {} higest skewed numerical features to box cox transform".format(skewness.shape[0]))

l_opt = {}

#df = pd.DataFrame()    
for feat in skewness.index:
    #df[feat] = boxcox1p(data[feat], l_opt[feat])
    #data[feat] = boxcox1p(data[feat], l_opt[feat])
    data[feat], l_opt[feat] = boxcox((data[feat]+1))

#skewed_features2 = df.apply(lambda x : skew (x.dropna())).sort_values(ascending=False)
skewed_features2 = data[skewness.index].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

#compute skewness
skewness2 = pd.DataFrame({'New Skew' :skewed_features2})   
display(pd.concat([skewness, skewness2], axis=1).sort_values(by=['Skew'], ascending=False))
def QQ_plot(data, measure):
    fig = plt.figure(figsize=(12,4))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.distplot(data, fit=norm)
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('Frequency')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.\
                   format(data.skew(), data.kurt()), loc='center')

    plt.tight_layout()
    plt.show()
    
for feat in skewness.index:
    QQ_plot(data[feat], ('Boxcox1p of {}'.format(feat)))
pca_all = PCA(random_state=101, whiten=True).fit(df)

my_color=y_train.astype('category').cat.codes

# Store results of PCA in a data frame
result=pd.DataFrame(pca_all.transform(df), columns=['PCA%i' % i for i in range(df.shape[1])], index=df.index)

# Plot initialisation
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=my_color, cmap="Set2_r", s=60)
 
# make simple, bare axis lines through space:
xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
 
# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on the Titanic data set")
plt.show()

X_train , X_test, y, y_test = train_test_split(df , y_train, test_size=0.3, random_state=0)

lr = LogisticRegression(class_weight='balanced', random_state=101)
lr = lr.fit(X_train, y)
print('LR Training Accuracy: {:2.2%}'.format(accuracy_score(y, lr.predict(X_train))))
y_pred = lr.predict(X_test)
print('LR Accuracy: {:2.2%}'.format(accuracy_score(y_test, y_pred)))

print('_' * 40)
print('\nApply PCA:\n')
AccPca = pd.DataFrame(columns=['Components', 'Var_ratio', 'Train_Acc', 'Test_Acc'])

for componets in np.arange(1, df.shape[1]):
    variance_ratio = sum(pca_all.explained_variance_ratio_[:componets])*100
    pca = PCA(n_components=componets, random_state=101, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    Components = X_train_pca.shape[1]
    lr = LogisticRegression(class_weight='balanced', random_state=101)
    lr = lr.fit(X_train_pca, y)
    Training_Accuracy = accuracy_score(y, lr.predict(X_train_pca))
    X_test_pca = pca.transform(X_test)
    y_pred = lr.predict(X_test_pca)
    Test_Accuracy = accuracy_score(y_test, y_pred)
    AccPca = AccPca.append(pd.DataFrame([(Components, variance_ratio, Training_Accuracy, Test_Accuracy)],
                                        columns=['Components', 'Var_ratio', 'Train_Acc', 'Test_Acc']))#], axis=0)

AccPca.set_index('Components', inplace=True)
display(AccPca.sort_values(by='Test_Acc', ascending=False))
X_train , X_test, y, y_test = train_test_split(df , y_train, test_size=0.3, random_state=0)

lr = LogisticRegression(class_weight='balanced', random_state=101)
lr = lr.fit(X_train, y)
print('LR Training Accuracy: {:2.2%}'.format(accuracy_score(y, lr.predict(X_train))))
y_pred = lr.predict(X_test)
print('LR Accuracy: {:2.2%}'.format(accuracy_score(y_test, y_pred)))
print('_' * 40)
print('\nApply LDA:\n')
lda = LDA(store_covariance=True)
X_train_lda = lda.fit_transform(X_train, y)
#X_train_lda = pd.DataFrame(X_train_lda)

print('Number of features after LDA:',X_train_lda.shape[1])
lr = LogisticRegression(class_weight='balanced', random_state=101)
lr = lr.fit(X_train_lda, y)
print('LR Training Accuracy With LDA: {:2.2%}'.format(accuracy_score(y, lr.predict(X_train_lda))))
X_test_lda = lda.transform(X_test)
y_pred = lr.predict(X_test_lda)
print('LR Test Accuracy With LDA: {:2.2%}'.format(accuracy_score(y_test, y_pred)))

fig = plt.figure(figsize=(20,5))
fig.add_subplot(121)
plt.scatter(X_train_lda[y==0, 0], np.zeros((len(X_train_lda[y==0, 0]),1)), color='red', alpha=0.1)
plt.scatter(X_train_lda[y==1, 0], np.zeros((len(X_train_lda[y==1, 0]),1)), color='blue', alpha=0.1)
plt.title('LDA on Training Data Set')
plt.xlabel('LDA')
fig.add_subplot(122)
plt.scatter(X_test_lda[y_test==0, 0], np.zeros((len(X_test_lda[y_test==0, 0]),1)), color='red', alpha=0.1)
plt.scatter(X_test_lda[y_test==1, 0], np.zeros((len(X_test_lda[y_test==1, 0]),1)), color='blue', alpha=0.1)
plt.title('LDA on Test Data Set')
plt.xlabel('LDA')

plt.show()
X_train , X_test, y, y_test = train_test_split(df , y_train, test_size=0.3, random_state=0)

X_train = X_train.append(pd.DataFrame(-np.ones((20,len(cols)))/10, columns = X_train.columns), ignore_index=True)
y = y.append(pd.Series(-np.ones((20))), ignore_index=True)

lr = LogisticRegression(class_weight='balanced', random_state=101)
lr = lr.fit(X_train, y)

print('Artficial training %d observations' % X_train.Age[y==-1].count())
print('LR Training Accuracy: {:2.2%}'.format(accuracy_score(y, lr.predict(X_train))))
y_pred = lr.predict(X_test)
print('LR Accuracy: {:2.2%}'.format(accuracy_score(y_test, y_pred)))

print('_' * 40)
print('\nApply LDA:\n')
lda = LDA(store_covariance=True)
X_train_lda = lda.fit_transform(X_train, y)

print('Number of features after LDA:',X_train_lda.shape[1])
print('Number test observations predit as -1:', len(X_test_lda[y_test==-1, :]))
lr = LogisticRegression(class_weight='balanced', random_state=101)
lr = lr.fit(X_train_lda, y)
print('LR Training Accuracy With LDA: {:2.2%}'.format(accuracy_score(y, lr.predict(X_train_lda))))
X_test_lda = lda.transform(X_test)
y_pred = lr.predict(X_test_lda)
print('LR Test Accuracy With LDA: {:2.2%}'.format(accuracy_score(y_test, y_pred)))

fig = plt.figure(figsize=(20,5))
fig.add_subplot(121)
plt.scatter(x=X_train_lda[y==0, 0], y=X_train_lda[y==0, 1], color='red', alpha=0.1)
plt.scatter(x=X_train_lda[y==1, 0], y=X_train_lda[y==1, 1], color='blue', alpha=0.1)
plt.title('LDA on Training Data Set')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')

fig.add_subplot(122)
plt.scatter(x=X_test_lda[y_test==0, 0], y=X_test_lda[y_test==0, 1], color='red', alpha=0.1)
plt.scatter(x=X_test_lda[y_test==1, 0], y=X_test_lda[y_test==1, 1], color='blue', alpha=0.1)
plt.title('LDA on Test Data Set')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')

plt.show()
n_components = 3
kernel = 'linear' 
degree = 3
gamma = 1/df.shape[0]

kpca = KernelPCA(n_components = n_components, degree = degree, random_state = 101, #gamma = gamma,
                kernel = kernel, eigen_solver='arpack')
X_kpca = kpca.fit_transform(df)

# Plot first two KPCA components
fig = plt.figure(figsize=(20,6))
ax  = fig.add_subplot(121)
plt.scatter(x = X_kpca[y_train==0, 0], y = X_kpca[y_train==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(x = X_kpca[y_train==1, 0], y = X_kpca[y_train==1, 1], color='blue', marker='o', alpha=0.5)
ax.set_xlabel("KPCA_0")
ax.set_ylabel("KPCA_1")
ax.set_title("Plot of first 2 KPCA Components on the Titanic data set")

my_color=y_train.astype('category').cat.codes

# Store results of PCA in a data frame
result=pd.DataFrame(X_kpca, columns=['KPCA%i' % i for i in range(n_components)], index=df.index)

# Plot initialisation
ax = fig.add_subplot(122, projection='3d')
ax.scatter(result['KPCA0'], result['KPCA1'], result['KPCA2'], c=my_color, cmap="Set2_r", s=60)
 
# make simple, bare axis lines through space:
xAxisLine = ((min(result['KPCA0']), max(result['KPCA0'])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(result['KPCA1']), max(result['KPCA1'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(result['KPCA2']), max(result['KPCA2'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
 
# label the axes
ax.set_xlabel("KPCA_0")
ax.set_ylabel("KPCA_1")
ax.set_zlabel("KPCA_2")
ax.set_title("KPCA of 3 Components on the Titanic data set")
plt.tight_layout(); plt.show()

X_train , X_test, y, y_test = train_test_split(df , y_train, test_size=0.3, random_state=0)

lr = LogisticRegression(class_weight='balanced', random_state=101)
lr = lr.fit(X_train, y)
print('\nLogistic Regression over data without transformation:\n' + '_' * 53 + '\n')
print('LR Training Accuracy: {:2.2%}'.format(accuracy_score(y, lr.predict(X_train))))
y_pred = lr.predict(X_test)
print('LR Test Accuracy: {:2.2%}'.format(accuracy_score(y_test, y_pred)))


print('\nApply KPCA:\n' + '_' * 53)
kpca = KernelPCA(kernel = kernel, random_state = 101, degree = degree, eigen_solver='arpack', n_components = 23)
X_train_kpca = kpca.fit_transform(X_train)
print('Number of features after KPCA:', X_train_kpca.shape[1])
lr = LogisticRegression(class_weight='balanced', random_state=101)
lr = lr.fit(X_train_kpca, y)
print('LR Training Accuracy: {:2.2%}'.format(accuracy_score(y, lr.predict(X_train_kpca))))
X_test_kpca = kpca.transform(X_test)
y_pred = lr.predict(X_test_kpca)
print('LR Test Accuracy: {:2.2%}'.format(accuracy_score(y_test, y_pred)))
class select_fetaures(object): # BaseEstimator, TransformerMixin, 
    def __init__(self, select_cols):
        self.select_cols_ = select_cols
    
    def fit(self, X, Y ):
        print('Recive {0:2d} features...'.format(X.shape[1]))
        return self

    def transform(self, X):
        print('Select {0:2d} features'.format(X.loc[:, self.select_cols_].shape[1]))
        return X.loc[:, self.select_cols_]    

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        df = self.transform(X)
        return df 
        #X.loc[:, self.select_cols_]    

    def __getitem__(self, x):
        return self.X[x], self.Y[x]
        
data.Pclass = data.Pclass.astype(np.uint8)
data.genre = data.genre.astype(np.uint8)
data.distinction_in_tikect_Low = data.distinction_in_tikect_Low.astype(np.uint8)
data.distinction_in_tikect_PC = data.distinction_in_tikect_PC.astype(np.uint8)
data.distinction_in_tikect_Others = data.distinction_in_tikect_Others.astype(np.uint8)
data.Cabin_Letter_B = data.Cabin_Letter_B.astype(np.uint8)
data.Cabin_Letter_C = data.Cabin_Letter_C.astype(np.uint8)
data.Cabin_Letter_D = data.Cabin_Letter_D.astype(np.uint8)
data.Cabin_Letter_E = data.Cabin_Letter_E.astype(np.uint8)
data.Cabin_Letter_F = data.Cabin_Letter_F.astype(np.uint8)
data.Cabin_Letter_G = data.Cabin_Letter_G.astype(np.uint8)
data.Embarked_C = data.Embarked_C.astype(np.uint8)
data.Embarked_S = data.Embarked_S.astype(np.uint8)
data.Embarked_Q = data.Embarked_Q.astype(np.uint8)
data.Personal_Titles_Royalty = data.Personal_Titles_Royalty.astype(np.uint8)
data.Personal_Titles_Technical = data.Personal_Titles_Technical.astype(np.uint8)
data.Personal_Titles_Kid = data.Personal_Titles_Kid.astype(np.uint8)
data.Personal_Titles_Mrs = data.Personal_Titles_Mrs.astype(np.uint8)
data.Personal_Titles_Mr = data.Personal_Titles_Mr.astype(np.uint8)
data.Personal_Titles_Miss = data.Personal_Titles_Miss.astype(np.uint8)
data.Without_Age = data.Without_Age.astype(np.uint8)
data.distinction_in_name = data.distinction_in_name.astype(np.uint8)
data.parents = data.parents.astype(np.uint8)
data.relatives = data.relatives.astype(np.uint8)
data.sons = data.sons.astype(np.uint8)
data.companions = data.companions.astype(np.uint8)
data.surname_Alone = data.surname_Alone.astype(np.uint8)
data.surname_Baclini = data.surname_Baclini.astype(np.uint8)
data.surname_Carter = data.surname_Carter.astype(np.uint8)
data.surname_Richards = data.surname_Richards.astype(np.uint8)
data.surname_Harper = data.surname_Harper.astype(np.uint8)
data.surname_Beckwith = data.surname_Beckwith.astype(np.uint8)
data.surname_Goldenberg = data.surname_Goldenberg.astype(np.uint8)
data.surname_Moor = data.surname_Moor.astype(np.uint8)
data.surname_Chambers = data.surname_Chambers.astype(np.uint8)
data.surname_Hamalainen = data.surname_Hamalainen.astype(np.uint8)
data.surname_Dick = data.surname_Dick.astype(np.uint8)
data.surname_Taylor = data.surname_Taylor.astype(np.uint8)
data.surname_Doling = data.surname_Doling.astype(np.uint8)
data.surname_Gordon = data.surname_Gordon.astype(np.uint8)
data.surname_Beane = data.surname_Beane.astype(np.uint8)
data.surname_Hippach = data.surname_Hippach.astype(np.uint8)
data.surname_Bishop = data.surname_Bishop.astype(np.uint8)
data.surname_Mellinger = data.surname_Mellinger.astype(np.uint8)
data.surname_Yarred = data.surname_Yarred.astype(np.uint8)

Test_ID = data.PassengerId[data.Survived<0]
y_train = data.Survived[data.Survived>=0]

scale = StandardScaler()
train =  pd.DataFrame(scale.fit_transform(data.loc[data.Survived>=0, cols]), columns = cols)
test = pd.DataFrame(scale.transform(data.loc[data.Survived<0, cols]), columns = cols)
def get_results(model, name, results=None, data=train, reasume=False):

    modelo = model.fit(data, y_train)
    print('Mean Best Accuracy: {:2.2%}'.format(gs.best_score_))
    print(gs.best_params_,'\n')
    best = gs.best_estimator_
    param_grid = best
    y_pred = model.predict(data)
    display_model_performance_metrics(true_labels=y_train, predicted_labels=y_pred)

    print('\n\n              ROC AUC Score: {:2.2%}'.format(roc_auc_score(y_true=y_train, y_score=y_pred)))
    if hasattr(param_grid, 'predict_proba'):
            prob = model.predict_proba(data)
            score_roc = prob[:, prob.shape[1]-1] 
            prob = True
    elif hasattr(param_grid, 'decision_function'):
            score_roc = model.decision_function(data)
            prob = False
    else:
            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")
    fpr, tpr, thresholds = roc_curve(y_true=y_train, y_score=score_roc)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot(fpr, tpr, 'b', label='AUC = {:2.2%}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()

    r1 = pd.DataFrame([(prob, gs.best_score_, np.round(accuracy_score(y_train, y_pred), 4), 
                         roc_auc_score(y_true=y_train, y_score=y_pred), roc_auc)], index = [name],
                         columns = ['Prob', 'CV Accuracy', 'Acc All', 'ROC AUC Score', 'ROC Area'])
    if reasume:
        results = r1
    elif (name in results.index):        
        results.loc[[name], :] = r1
    else: 
        results = results.append(r1)
        
    return results, modelo

"""
Created on Mon Jul 31 20:05:23 2017

@author: DIP
@Copyright: Dipanjan Sarkar
"""

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp


def get_metrics(true_labels, predicted_labels):
    
    print('Accuracy:  {:2.2%} '.format(metrics.accuracy_score(true_labels, predicted_labels)))
    print('Precision: {:2.2%} '.format(metrics.precision_score(true_labels, predicted_labels, average='weighted')))
    print('Recall:    {:2.2%} '.format(metrics.recall_score(true_labels, predicted_labels, average='weighted')))
    print('F1 Score:  {:2.2%} '.format(metrics.f1_score(true_labels, predicted_labels, average='weighted')))
                        

def train_predict_model(classifier,  train_features, train_labels,  test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    return predictions    


def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]):
    
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, 
                                  labels=classes)
    cm_frame = pd.DataFrame(data=cm, 
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes], labels=level_labels), 
                            index=pd.MultiIndex(levels=[['Actual:'], classes], labels=level_labels)) 
    print(cm_frame) 
    
def display_classification_report(true_labels, predicted_labels, classes=[1,0]):

    report = metrics.classification_report(y_true=true_labels, y_pred=predicted_labels, labels=classes) 
    print(report)
    
    
    
def display_model_performance_metrics(true_labels, predicted_labels, classes=[1,0]):
    print('Model Performance metrics:')
    print('-'*30)
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    print('\nModel Classification report:')
    print('-'*30)
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, classes=classes)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, classes=classes)


def plot_model_decision_surface(clf, train_features, train_labels, plot_step=0.02, cmap=plt.cm.RdYlBu,
                                markers=None, alphas=None, colors=None):
    
    if train_features.shape[1] != 2:
        raise ValueError("X_train should have exactly 2 columnns!")
    
    x_min, x_max = train_features[:, 0].min() - plot_step, train_features[:, 0].max() + plot_step
    y_min, y_max = train_features[:, 1].min() - plot_step, train_features[:, 1].max() + plot_step
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    clf_est = clone(clf)
    clf_est.fit(train_features,train_labels)
    if hasattr(clf_est, 'predict_proba'):
        Z = clf_est.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    else:
        Z = clf_est.predict(np.c_[xx.ravel(), yy.ravel()])    
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(train_labels)
    n_classes = len(le.classes_)
    plot_colors = ''.join(colors) if colors else [None] * n_classes
    label_names = le.classes_
    markers = markers if markers else [None] * n_classes
    alphas = alphas if alphas else [None] * n_classes
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_enc == i)
        plt.scatter(train_features[idx, 0], train_features[idx, 1], c=color,
                    label=label_names[i], cmap=cmap, edgecolors='black', 
                    marker=markers[i], alpha=alphas[i])
    plt.legend()
    plt.show()


def plot_model_roc_curve(clf, features, true_labels, label_encoder=None, class_names=None):
    
    ## Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if hasattr(clf, 'classes_'):
        class_labels = clf.classes_
    elif label_encoder:
        class_labels = label_encoder.classes_
    elif class_names:
        class_labels = class_names
    else:
        raise ValueError('Unable to derive prediction classes, please specify class_names!')
    n_classes = len(class_labels)
    y_test = label_binarize(true_labels, classes=class_labels)
    if n_classes == 2:
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(features)
            y_score = prob[:, prob.shape[1]-1] 
        elif hasattr(clf, 'decision_function'):
            prob = clf.decision_function(features)
            y_score = prob[:, prob.shape[1]-1]
        else:
            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")
        
        fpr, tpr, _ = roc_curve(y_test, y_score)      
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = {0:2.2%})'
                                 ''.format(roc_auc),
                 linewidth=2.5)
        
    elif n_classes > 2:
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(features)
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(features)
        else:
            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        ## Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        ## Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        ## Plot ROC curves
        plt.figure(figsize=(6, 4))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:2.2%})'
                       ''.format(roc_auc["micro"]), linewidth=3)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:2.2%})'
                       ''.format(roc_auc["macro"]), linewidth=3)

        for i, label in enumerate(class_labels):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:2.2%})'
                                           ''.format(label, roc_auc[i]), 
                     linewidth=2, linestyle=':')
    else:
        raise ValueError('Number of classes should be atleast 2 or more')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', LogisticRegression(random_state=101))])  

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [25, 22, 31, 54]
whiten = [True, False]
C =  [0.008, 0.007, 0.009, 0.01]#, 0.1, 1.0, 10.0, 100.0, 1000.0]
tol = [0.001, 0.003, 0.002, 0.005] # [1e-06, 5e-07, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01]

param_grid =\
    [{'clf__C': C
     ,'clf__solver': ['liblinear', 'saga'] 
     ,'clf__penalty': ['l1', 'l2']
     ,'clf__tol' : tol 
     ,'clf__class_weight': ['balanced']
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
},
    {'clf__C': C
     ,'clf__max_iter': [3, 9, 2, 7, 4]
     ,'clf__solver': ['newton-cg', 'sag', 'lbfgs']
     ,'clf__penalty': ['l2']
     ,'clf__tol' : tol 
     ,'clf__class_weight': ['balanced'] 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
}]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(shadow))),
        #('scl', StandardScaler()),
        ('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  


results, lr = get_results(main_pip, 'LogisticRegression', reasume=True)
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', SGDClassifier(random_state=101))])

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [30, 22, 21, 50]
whiten = [True, False]
alpha = [4e-03, 5e-03, 6e-03, 1e-03]
tol = [1e-08, 1e-07, 5e-09]

param_grid =\
    [{'clf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
     ,'clf__tol': tol
     ,'clf__alpha': alpha
     ,'clf__penalty': ['l2', 'l1']
     ,'clf__class_weight' : ['balanced'] 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     },
    {'clf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
     ,'clf__tol': tol
     ,'clf__alpha': alpha
     ,'clf__penalty': ['elasticnet']
     ,'clf__l1_ratio' : [0.3, 0.5, 0.1]
     ,'clf__class_weight' : ['balanced'] 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }]


gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(FRFE))),
        #('scl', StandardScaler()),
        ('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  

results, svm = get_results(main_pip, 'SGDClassifier', results)
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', LinearSVC(random_state=101))])

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [25, 22, 31, 54]
whiten = [True, False]
C =  [0.5, 0.3, 0.05, 0.1] #, 1.0, 10.0, 100.0, 1000.0]
tol = [1e-06, 3e-06, 5e-07]
max_iter = [9, 15, 7]

param_grid =\
    [{'clf__loss': ['hinge']
     ,'clf__tol': tol
     ,'clf__C': C
     ,'clf__penalty': ['l2']
     ,'clf__class_weight' : ['balanced'] 
     ,'clf__max_iter' : max_iter
     ,'clf__dual' : [True]
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }
    ,{'clf__loss': ['squared_hinge']
     ,'clf__tol': tol
     ,'clf__C': C
     ,'clf__penalty': ['l2', 'l1']
     ,'clf__class_weight' : ['balanced'] 
     ,'clf__max_iter' : max_iter
     ,'clf__dual' : [False]
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(FRFE))),
        #('scl', StandardScaler()),
        ('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  

results, lsvc = get_results(main_pip, 'LinearSVC', results)
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', GaussianProcessClassifier(1.0 * RBF(1.0), random_state=101))
])

# n_restarts_optimizer=5
# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [25, 22, 31, 54]
whiten = [True, False]
max_iter_predict = [5, 10, 15, 20]

param_grid =\
    [{'clf__max_iter_predict':  max_iter_predict
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(bcols))),
        #('scl', StandardScaler()),
        ('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  

results, gpc = get_results(main_pip, 'GaussianProcessClassifier', results)
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', RandomForestClassifier(random_state=101))])

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [25, 22, 31, 54]
whiten = [True, False]
param_grid =\
    [{'clf__n_estimators' : [500, 3000]
      ,'clf__criterion': ['gini', 'entropy']
      ,'clf__min_samples_split': [4, 3, 5]
      #,'clf__min_impurity_split': [0.05, 0.03, 0.07]
      #,'clf__max_depth': [5, 10]
      ,'clf__min_impurity_decrease': [0.0003]
      #,'clf__min_samples_leaf': [1,2,3,4]
      ,'clf__class_weight': ['balanced']
      #,'clf__bootstrap': [True, False]
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }]

sele = bcols
gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(sele))),
        #('scl', StandardScaler()),
        ('gs', gs)
 ])  

results, rfc = get_results(main_pip, 'RandomForestClassifier', results)
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', AdaBoostClassifier(random_state=101))])
# , max_iter_predict=500, n_restarts_optimizer=5

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [25, 22, 31, 54]
whiten = [True, False]

param_grid =\
    [{'clf__learning_rate': [3e-03, 15e-02, 5e-02]
     ,'clf__n_estimators': [300, 350, 400, 500] # np.arange(96,115)
     ,'clf__algorithm' : ['SAMME', 'SAMME.R']
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(FRFE))),
        #('scl', StandardScaler()),
        ('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  

results, AdaB = get_results(main_pip, 'AdaBoostClassifier', results)
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', KNeighborsClassifier())])

#max_iter_predict=500, n_restarts_optimizer=5
# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [25, 22, 31, 54]
whiten = [True, False]
param_grid =\
    [{'clf__n_neighbors': [3, 7, 8, 9] #
     ,'clf__weights': ['uniform', 'distance'] 
     ,'clf__algorithm' : ['ball_tree', 'kd_tree'] # ['auto', 'ball_tree', 'kd_tree', 'brute']
     ,'clf__leaf_size': [12, 15, 16, 20]
     ,'clf__p': [1, 2] 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     }]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(FRFE))),
        #('scl', StandardScaler()),
        ('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  

results, KNNC = get_results(main_pip, 'KNeighborsClassifier', results)
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', MLPClassifier(random_state=101))])

# a list of dictionaries to specify the parameters that we'd want to tune
n_components= [25, 22, 31, 54]
whiten = [True, False]
param_grid =\
    [{#'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
      'clf__solver': ['adam'] # , 'lbfgs', 'sgd'
     ,'clf__tol': [5e-04] #, 3e-04, 7e-04]
     #,'clf__max_iter': [200, 1000]
     ,'clf__alpha': [1e-06] #, 1e-07, 1e-08] 
     ,'clf__learning_rate_init': [3e-04]
     ,'clf__hidden_layer_sizes': [(512, 256, 128, 64, )]#, (1024, 512, 256, 128, 64, )]
     ,'clf__batch_size': [64]
     ,'clf__epsilon': [1e-08] 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
     },
     {'clf__solver': ['sgd'] 
     ,'clf__tol': [5e-04]
     ,'clf__learning_rate_init': [3e-04]
     ,'clf__learning_rate': ['constant', 'adaptive']
     ,'clf__alpha': [1e-06] #, 1e-07, 1e-08] #, 1e-03, 1e-02, 1e-01]
     ,'clf__hidden_layer_sizes': [(512, 256, 128, 64, )]#, (1024, 512, 256, 128, 64, )]
     ,'clf__batch_size': [64]
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
    },
     {'clf__solver': ['sgd'] 
     ,'clf__tol': [5e-04]
     ,'clf__learning_rate_init': [3e-04]
     ,'clf__learning_rate': ['invscaling']
     ,'clf__power_t' : [ 0.25, 0.5]
     ,'clf__alpha': [1e-06]
     ,'clf__hidden_layer_sizes': [(256, 128, 64, 32, )]
     ,'clf__batch_size': [64]
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
    }]
    
gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(cols))),
        #('scl', StandardScaler()),
        ('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  

results, mlpc = get_results(main_pip, 'MLPClassifier', results)
clf = Pipeline([
        #('pca', PCA(random_state = 101)),
        ('clf', GradientBoostingClassifier(random_state=101))])  

# a list of dictionaries to specify the parameters that we'd want to tune
#cv=None, dual=False,  scoring=None, refit=True,  multi_class='ovr'
n_components= [25, 22, 31, 54]
whiten = [True, False]
learning_rate =  [1e-02] #, 5e-03, 2e-02]
n_estimators= [140, 150, 160, 145]
max_depth = [2, 3, 5]

param_grid =\
    [{'clf__learning_rate': learning_rate
     ,'clf__max_depth': max_depth
     ,'clf__n_estimators' : n_estimators 
     #,'pca__n_components' : n_components
     #,'pca__whiten' : whiten
}]

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(FRFE))),
        ('scl', StandardScaler()),
        ('lda', LDA(store_covariance=True)),
        ('gs', gs)
 ])  

results, GBC = get_results(main_pip, 'GradientBoostingClassifier', results)
def categorical_change_back(df):
    categorical_features = list(df.dtypes[df.dtypes == "category"].index)
    for feat in categorical_features:
        if len(df[feat].unique())==2:
            df[feat] = df[feat].astype(bool)
        else:
            df[feat] = df[feat].astype(int)
    return df

trainXGB = data.loc[data.Survived>=0, cols].copy()
trainXGB = categorical_change_back(trainXGB)
testXGB = data.loc[data.Survived<0, cols].copy()
testXGB = categorical_change_back(testXGB)
# a list of dictionaries to specify the parameters that we'd want to tune
scale = ((len(y_train)-y_train.sum())/y_train.sum())
param_grid = \
    [{
     'clf__learning_rate': [0.1, 0.09, 0.03, 0.01, 0.001],
     'clf__n_estimators': [200, 3000]
}]

clf = Pipeline([
        ('clf', XGBClassifier(learning_rate =0.1, n_estimators=2000, max_depth=3, min_child_weight=2, gamma=0.0, 
                              subsample=0.9, colsample_bytree=0.7, objective= 'binary:logistic', importance_type='gain', 
                              reg_alpha = 0.9, n_jobs=4, scale_pos_weight=scale, seed=101, random_state=101))])  

gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=4)
main_pip = Pipeline([
        ('sel', select_fetaures(select_cols=list(pv_cols))),
        ('gs', gs)
 ])  

results, xgbF = get_results(main_pip, 'XGBClassifier Final', results, data = trainXGB)
display(results.sort_values(by='ROC Area', ascending=False))
n_folds = 10

def cvscore(model):
    kf = KFold(n_folds, shuffle=True, random_state=101).get_n_splits(train.values)
    score= cross_val_score(estimator=model, X=train.values, y=y_train, scoring="accuracy", verbose=1, n_jobs=3, cv = kf)
    return(score)
models = ( xgbF, rfc, GBC, AdaB, mlpc, gpc, lr, KNNC )

trained_models = []
for model in models:
    #model.fit(train, targets) models all ready fited
    trained_models.append(model)

predictions = []
for i, model in enumerate(trained_models):
    if i < 1:
         predictions.append(model.predict_proba(testXGB)[:, 1])
    else:
        predictions.append(model.predict_proba(test)[:, 1])

# Preper Submission File of Probabilities Classifier
predictions_df = pd.DataFrame(predictions).T

ensemble = predictions_df.mean(axis=1).map(lambda s: 1 if s >= 0.5 else 0)
submit = pd.DataFrame()
submit['PassengerId'] = Test_ID.values
submit['Survived'] = ensemble

# ----------------------------- Create File to Submit --------------------------------
submit.to_csv('Titanic_Probabilities_submission.csv', index = False)
print('Sample of Probabilities Submit:')
display(submit.head())