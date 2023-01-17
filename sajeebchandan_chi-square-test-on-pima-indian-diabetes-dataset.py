import pandas

import numpy

import matplotlib

from matplotlib import pyplot

from sklearn import preprocessing

%matplotlib inline

pandas.set_option('display.max_rows', 500)

pandas.set_option('display.max_columns', 500)

pandas.set_option('display.width', 1000)

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

from scipy.stats import chi2_contingency

from scipy.stats import chi2

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2 as sklearn_chi2

from sklearn.preprocessing.imputation import Imputer
data_frame= pandas.read_table("../input/pima-data.csv", sep=',')
data_frame.shape
data_frame.head()
for col in data_frame.columns:

    data_frame[col] = data_frame[col].map(lambda x: numpy.nan if isinstance(x, str) and '\t?' in x else(x[1:] if isinstance(x, str) and '\t' in x and x.find('\t')==0 else (x[:-1] if isinstance(x, str) and '\t' in x and x.find('\t')>0 else x)))
print((data_frame[data_frame.columns] == 0).sum())
data_frame[data_frame.columns] = data_frame[data_frame.columns].replace(0, numpy.NaN)
print(data_frame.isnull().sum())
data_frame.fillna(value=data_frame.mean(), inplace=True)
labelencoder=preprocessing.LabelEncoder()

data_frame.diabetes=pandas.Series(data=labelencoder.fit_transform(data_frame.diabetes), index=data_frame.index)
diabetes_col=data_frame.diabetes
data_frame.head()
data_frame.drop(labels=['diabetes'], axis=1, inplace=True)

data_frame=pandas.concat([data_frame, diabetes_col], axis=1)
data_frame.head()
iteration_val=len(data_frame.columns)-1

selecte_feature_index=[]
for i in range(0,iteration_val,1):

    print('Feature Name : {0}'.format(data_frame.columns[i]))

    table= data_frame[data_frame.iloc[:,[i,iteration_val]].columns].values

    chi_squared_stat, p_value, dof, expected=chi2_contingency(table)

    print(chi_squared_stat)

    probability=0.95

    critical=chi2.ppf(probability, dof)

    if abs(chi_squared_stat)>=critical:

        print('Dependent : Reject Hypothesis 0 (null Hypothesis)')

    elif abs(chi_squared_stat)<critical:

        print('Inependent : Fail to Reject Hypothesis 0 (null Hypothesis)')

        selecte_feature_index.append(i)

    

    alpha = 1.0 - probability

    print('Significance {0}, {1}'.format(alpha, p_value))

    if p_value<=alpha:

        print("Dependent : Reject Hypothesis 0 (null Hypothesis)")

    else:

        print('Inependent : Fail to Reject Hypothesis 0 (null Hypothesis)')

        selecte_feature_index.append(i)

    print('================================================================')
array=data_frame.values
array
X = data_frame.iloc[:,0:-1]

y = data_frame.iloc[:,-1:]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=sklearn_chi2, k='all')

fit = bestfeatures.fit(X,y)
fit.scores_
dfscores = pandas.DataFrame(fit.scores_)

dfcolumns = pandas.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pandas.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']

print(featureScores.nlargest(8,'Score'))