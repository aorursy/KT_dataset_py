# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import pandas_profiling

import scipy.stats as stats



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.rc("font", size=14)

plt.rcParams['axes.grid'] = True

plt.figure(figsize=(6,3))

plt.gray()



from matplotlib.backends.backend_pdf import PdfPages



import statsmodels.formula.api as sm



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics



from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





titanic_train = pd.read_csv("../input/titanic/train.csv")

titanic_train.head()
titanic_train = titanic_train.drop(['Name', 'Ticket'],axis = 1)

titanic_train
titanic_train['Survived'] = titanic_train.Survived.astype(str)
titanic_train
titanic_test = pd.read_csv("../input/titanic/test.csv")

titanic_test
titanic_train.describe()
titanic_train.info()
titanic_test.describe()
titanic_test.info()
titanic_train.columns
len(titanic_train.columns)
titanic_train.dtypes
titanic_train.isnull()
titanic_train.shape
# Plotting correlation between all important features

corr = titanic_train.corr()

plt.figure(figsize=(15, 10))

sns.heatmap(corr, annot=True)

plt.plot()
titanic_train['Sex'] = pd.factorize(titanic_train.Sex)[0]

titanic_train
titanic_train['Embarked'] = pd.factorize(titanic_train.Embarked)[0]

titanic_train
numeric_var_names=[key for key in dict(titanic_train.dtypes) if dict(titanic_train.dtypes)[key] in ['float64', 'int64', 'float32', 'int32','uint8']]

cat_var_names=[key for key in dict(titanic_train.dtypes) if dict(titanic_train.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)
numeric_var_names
cat_var_names
titanic_train_num = titanic_train[numeric_var_names]

titanic_train_num
titanic_train_cat = titanic_train[cat_var_names]

titanic_train_cat
# Use a general function that returns multiple values

def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_summary = titanic_train_num.apply(lambda x: var_summary(x)).T
num_summary
#def outlier_capping(x):

#    x = x.clip_upper(x.quantile(0.99))

#    x = x.clip_lower(x.quantile(0.01))

#    return x
#titanic_train_num=titanic_train_num.apply(lambda x: outlier_capping(x))

#titanic_train_num
def Missing_imputation(x):

    x = x.fillna(x.median())

    return x
titanic_train_num = titanic_train_num.apply(lambda x: Missing_imputation(x))

titanic_train_num
def Cat_Missing_imputation(x):

    x = x.fillna(x.mode())

    return x
titanic_train_cat = titanic_train_cat.apply(lambda x:  Cat_Missing_imputation(x))

titanic_train_cat
titanic_train_new = pd.concat([titanic_train_num, titanic_train_cat], axis=1)

titanic_train_new
pandas_profiling.ProfileReport(titanic_train_new)
titanic_train_new.head()
titanic_train_new.describe().T.head(10)
titanic_train_num.Fare.hist()
titanic_train_num.Sex.hist()
titanic_train_num.Embarked.hist()
titanic_train_num.Age.hist()
titanic_train_num.Pclass.hist()
# An utility function to create dummy variable

def create_dummies( titanic_train_new, colname ):

    col_dummies = pd.get_dummies(titanic_train_new[colname], prefix=colname, drop_first=True)

    #col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)

    titanic_train_new = pd.concat([titanic_train_new, col_dummies], axis=1)

    titanic_train_new.drop( colname, axis = 1, inplace = True )

    return titanic_train_new
cat_var_names
#for c_feature in categorical_features

titanic_train_cat_new = titanic_train_cat

for c_feature in cat_var_names:

    titanic_train_cat_new[c_feature] = titanic_train_cat_new[c_feature].astype('category')

    titanic_train_cat_new = create_dummies(titanic_train_cat_new , c_feature )
titanic_train_cat_new
titanic_train_new = pd.concat([titanic_train_num, titanic_train_cat_new], axis=1)

titanic_train_new
titanic_train_new.columns
bp = PdfPages('WOE Plots.pdf')



for num_variable in titanic_train_new.columns.difference(['Survived_1']):

    binned = pd.cut(titanic_train_new[num_variable], bins=10, labels=list(range(1,11)))

    #binned = binned.dropna()

    odds = titanic_train_new.groupby(binned)['Survived_1'].sum() / (titanic_train_new.groupby(binned)['Survived_1'].count()-titanic_train_new.groupby(binned)['Survived_1'].sum())

    log_odds = np.log(odds)

    fig,axes = plt.subplots(figsize=(10,4))

    sns.barplot(x=log_odds.index,y=log_odds)

    plt.ylabel('Log Odds Ratio')

    plt.title(str('Logit Plot for identifying if the bucketing is required or not for variable ') + str(num_variable))

    bp.savefig(fig)



bp.close()
titanic_train_new.columns
pandas_profiling.ProfileReport(titanic_train_new)
import statsmodels.formula.api as sm
logreg_model = sm.logit('Survived_1~Pclass',data = titanic_train_new).fit()
p = logreg_model.predict(titanic_train_new)

p
metrics.roc_auc_score(titanic_train_new['Survived_1'],p)
2*metrics.roc_auc_score(titanic_train_new['Survived_1'],p)-1
titanic_train_new1 = titanic_train_new.loc[:,["Age",

"Cabin_A14",

"Cabin_A16",

"Cabin_A19",

"Cabin_A20",

"Cabin_A23",

"Cabin_A24",

"Cabin_A26",

"Cabin_A31",

"Cabin_A32",

"Cabin_A34",

"Cabin_A36",

"Cabin_A5",

"Cabin_A6",

"Cabin_A7",

"Cabin_B101",

"Cabin_B102",

"Cabin_B18",

"Cabin_B19",

"Cabin_B20",

"Cabin_B22",

"Cabin_B28",

"Cabin_B3",

"Cabin_B30",

"Cabin_B35",

"Cabin_B37",

"Cabin_B38",

"Cabin_B39",

"Cabin_B4",

"Cabin_B41",

"Cabin_B42",

"Cabin_B49",

"Cabin_B5",

"Cabin_B50",

"Cabin_B51_B53_B55",

"Cabin_B57_B59_B63_B66",

"Cabin_B58_B60",

"Cabin_B69",

"Cabin_B71",

"Cabin_B73",

"Cabin_B77",

"Cabin_B78",

"Cabin_B79",

"Cabin_B80",

"Cabin_B82_B84",

"Cabin_B86",

"Cabin_B94",

"Cabin_B96_B98",

"Cabin_C101",

"Cabin_C103",

"Cabin_C104",

"Cabin_C106",

"Cabin_C110",

"Cabin_C111",

"Cabin_C118",

"Cabin_C123",

"Cabin_C124",

"Cabin_C125",

"Cabin_C126",

"Cabin_C128",

"Cabin_C148",

"Cabin_C2",

"Cabin_C22_C26",

"Cabin_C23_C25_C27",

"Cabin_C30",

"Cabin_C32",

"Cabin_C45",

"Cabin_C46",

"Cabin_C47",

"Cabin_C49",

"Cabin_C50",

"Cabin_C52",

"Cabin_C54",

"Cabin_C62_C64",

"Cabin_C65",

"Cabin_C68",

"Cabin_C7",

"Cabin_C70",

"Cabin_C78",

"Cabin_C82",

"Cabin_C83",

"Cabin_C85",

"Cabin_C86",

"Cabin_C87",

"Cabin_C90",

"Cabin_C91",

"Cabin_C92",

"Cabin_C93",

"Cabin_C95",

"Cabin_C99",

"Cabin_D",

"Cabin_D10_D12",

"Cabin_D11",

"Cabin_D15",

"Cabin_D17",

"Cabin_D19",

"Cabin_D20",

"Cabin_D21",

"Cabin_D26",

"Cabin_D28",

"Cabin_D30",

"Cabin_D33",

"Cabin_D35",

"Cabin_D36",

"Cabin_D37",

"Cabin_D45",

"Cabin_D46",

"Cabin_D47",

"Cabin_D48",

"Cabin_D49",

"Cabin_D50",

"Cabin_D56",

"Cabin_D6",

"Cabin_D7",

"Cabin_D9",

"Cabin_E10",

"Cabin_E101",

"Cabin_E12",

"Cabin_E121",

"Cabin_E17",

"Cabin_E24",

"Cabin_E25",

"Cabin_E31",

"Cabin_E33",

"Cabin_E34",

"Cabin_E36",

"Cabin_E38",

"Cabin_E40",

"Cabin_E44",

"Cabin_E46",

"Cabin_E49",

"Cabin_E50",

"Cabin_E58",

"Cabin_E63",

"Cabin_E67",

"Cabin_E68",

"Cabin_E77",

"Cabin_E8",

"Cabin_F2",

"Cabin_F33",

"Cabin_F38",

"Cabin_F4",

"Cabin_F_E69",

"Cabin_F_G63",

"Cabin_F_G73",

"Cabin_G6",

"Cabin_T",

"Embarked",

"Fare",

"Parch",

"PassengerId",

"Pclass",

"Sex",

"SibSp",

"Survived_1"]]

titanic_train_new1
somersd_df = pd.DataFrame()

for num_variable in titanic_train_new1.columns.difference(['Survived_1']):

    logreg_model = sm.logit(formula = str('Survived_1 ~ ')+str(num_variable), data=titanic_train_new1)

    result = logreg_model.fit()

    #result = logit.fit(method='bfgs')

    y1_score = pd.DataFrame(result.predict())

    y1_score.columns = ['Score']

    somers_d = 2*metrics.roc_auc_score(titanic_train_new1['Survived_1'],y1_score) - 1

    temp = pd.DataFrame([num_variable,somers_d]).T

    temp.columns = ['Variable Name', 'SomersD']

    somersd_df = pd.concat([somersd_df, temp], axis=0)

somersd_df
somersd_df.sort_values('SomersD', ascending=False, inplace=True)
somersd_df
titanic_train_new2 = titanic_train_new.loc[:,["Age",

"Cabin_A14",

"Cabin_A16",

"Cabin_A19",

"Cabin_A20",

"Cabin_A23",

"Cabin_A24",

"Cabin_A26",

"Cabin_A31",

"Cabin_A32",

"Cabin_A34",

"Cabin_A36",

"Cabin_A5",

"Cabin_A6",

"Cabin_A7",

"Cabin_B101",

"Cabin_B102",

"Cabin_B18",

"Cabin_B19",

"Cabin_B20",

"Cabin_B22",

"Cabin_B28",

"Cabin_B3",

"Cabin_B30",

"Cabin_B35",

"Cabin_B37",

"Cabin_B38",

"Cabin_B39",

"Cabin_B4",

"Cabin_B41",

"Cabin_B42",

"Cabin_B49",

"Cabin_B5",

"Cabin_B50",

"Cabin_B51_B53_B55",

"Cabin_B57_B59_B63_B66",

"Cabin_B58_B60",

"Cabin_B69",

"Cabin_B71",

"Cabin_B73",

"Cabin_B77",

"Cabin_B78",

"Cabin_B79",

"Cabin_B80",

"Cabin_B82_B84",

"Cabin_B86",

"Cabin_B94",

"Cabin_B96_B98",

"Cabin_C101",

"Cabin_C103",

"Cabin_C104",

"Cabin_C106",

"Cabin_C110",

"Cabin_C111",

"Cabin_C118",

"Cabin_C123",

"Cabin_C124",

"Cabin_C125",

"Cabin_C126",

"Cabin_C128",

"Cabin_C148",

"Cabin_C2",

"Cabin_C22_C26",

"Cabin_C23_C25_C27",

"Cabin_C30",

"Cabin_C32",

"Cabin_C45",

"Cabin_C46",

"Cabin_C47",

"Cabin_C49",

"Cabin_C50",

"Cabin_C52",

"Cabin_C54",

"Cabin_C62_C64",

"Cabin_C65",

"Cabin_C68",

"Cabin_C7",

"Cabin_C70",

"Cabin_C78",

"Cabin_C82",

"Cabin_C83",

"Cabin_C85",

"Cabin_C86",

"Cabin_C87",

"Cabin_C90",

"Cabin_C91",

"Cabin_C92",

"Cabin_C93",

"Cabin_C95",

"Cabin_C99",

"Cabin_D",

"Cabin_D10_D12",

"Cabin_D11",

"Cabin_D15",

"Cabin_D17",

"Cabin_D19",

"Cabin_D20",

"Cabin_D21",

"Cabin_D26",

"Cabin_D28",

"Cabin_D30",

"Cabin_D33",

"Cabin_D35",

"Cabin_D36",

"Cabin_D37",

"Cabin_D45",

"Cabin_D46",

"Cabin_D47",

"Cabin_D48",

"Cabin_D49",

"Cabin_D50",

"Cabin_D56",

"Cabin_D6",

"Cabin_D7",

"Cabin_D9",

"Cabin_E10",

"Cabin_E101",

"Cabin_E12",

"Cabin_E121",

"Cabin_E17",

"Cabin_E24",

"Cabin_E25",

"Cabin_E31",

"Cabin_E33",

"Cabin_E34",

"Cabin_E36",

"Cabin_E38",

"Cabin_E40",

"Cabin_E44",

"Cabin_E46",

"Cabin_E49",

"Cabin_E50",

"Cabin_E58",

"Cabin_E63",

"Cabin_E67",

"Cabin_E68",

"Cabin_E77",

"Cabin_E8",

"Cabin_F2",

"Cabin_F33",

"Cabin_F38",

"Cabin_F4",

"Cabin_F_E69",

"Cabin_F_G63",

"Cabin_F_G73",

"Cabin_G6",

"Cabin_T",

"Embarked",

"Fare",

"Parch",

"PassengerId",

"Pclass",

"Sex",

"SibSp",

"Survived_1"]]

titanic_train_new2
from sklearn import datasets

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



X = titanic_train_new2[titanic_train_new2.columns.difference(['Survived_1'])]

logreg = LogisticRegression()

rfe = RFE(logreg, 15)

rfe = rfe.fit(X, titanic_train_new2[['Survived_1']] )



print(rfe.support_)

print(rfe.ranking_)
X.columns
# summarize the selection of the attributes

import itertools

feature_map = [(i, v) for i, v in itertools.zip_longest(X.columns, rfe.get_support())]



feature_map



#Alternative of capturing the important variables

RFE_features=X.columns[rfe.get_support()]



selected_features_from_rfe = X[RFE_features]
RFE_features
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
X = titanic_train_new2[titanic_train_new2.columns.difference(['Survived_1'])]

X_new = SelectKBest(f_classif, k=15).fit(X, titanic_train_new2[['Survived_1']] )
X_new.get_support()
X_new.scores_
# summarize the selection of the attributes

import itertools

feature_map = [(i, v) for i, v in itertools.zip_longest(X.columns, X_new.get_support())]



feature_map



#Alternative of capturing the important variables

KBest_features=X.columns[X_new.get_support()]



selected_features_from_KBest = X[KBest_features]
KBest_features
X = pd.concat([titanic_train_new2[titanic_train_new2.columns.difference(['Survived_1'])],titanic_train_new2['Survived_1']], axis=1)

features = "+".join(titanic_train_new2.columns.difference(['Survived_1']))

X.head()
features
a,b = dmatrices(formula_like='Survived_1 ~ '+ 'Age+Cabin_A14+Cabin_A16+Cabin_A19+Cabin_A20+Cabin_A23+Cabin_A24+Cabin_A26+Cabin_A31+Cabin_A32+Cabin_A34+Cabin_A36+Cabin_A5+Cabin_A6+Cabin_A7+Cabin_B101+Cabin_B102+Cabin_B18+Cabin_B19+Cabin_B20+Cabin_B22+Cabin_B28+Cabin_B3+Cabin_B30+Cabin_B35+Cabin_B37+Cabin_B38+Cabin_B39+Cabin_B4+Cabin_B41+Cabin_B42+Cabin_B49+Cabin_B5+Cabin_B50+Cabin_B51_B53_B55+Cabin_B57_B59_B63_B66+Cabin_B58_B60+Cabin_B69+Cabin_B71+Cabin_B73+Cabin_B77+Cabin_B78+Cabin_B79+Cabin_B80+Cabin_B82_B84+Cabin_B86+Cabin_B94+Cabin_B96_B98+Cabin_C101+Cabin_C103+Cabin_C104+Cabin_C106+Cabin_C110+Cabin_C111+Cabin_C118+Cabin_C123+Cabin_C124+Cabin_C125+Cabin_C126+Cabin_C128+Cabin_C148+Cabin_C2+Cabin_C22_C26+Cabin_C23_C25_C27+Cabin_C30+Cabin_C32+Cabin_C45+Cabin_C46+Cabin_C47+Cabin_C49+Cabin_C50+Cabin_C52+Cabin_C54+Cabin_C62_C64+Cabin_C65+Cabin_C68+Cabin_C7+Cabin_C70+Cabin_C78+Cabin_C82+Cabin_C83+Cabin_C85+Cabin_C86+Cabin_C87+Cabin_C90+Cabin_C91+Cabin_C92+Cabin_C93+Cabin_C95+Cabin_C99+Cabin_D+Cabin_D10_D12+Cabin_D11+Cabin_D15+Cabin_D17+Cabin_D19+Cabin_D20+Cabin_D21+Cabin_D26+Cabin_D28+Cabin_D30+Cabin_D33+Cabin_D35+Cabin_D36+Cabin_D37+Cabin_D45+Cabin_D46+Cabin_D47+Cabin_D48+Cabin_D49+Cabin_D50+Cabin_D56+Cabin_D6+Cabin_D7+Cabin_D9+Cabin_E10+Cabin_E101+Cabin_E12+Cabin_E121+Cabin_E17+Cabin_E24+Cabin_E25+Cabin_E31+Cabin_E33+Cabin_E34+Cabin_E36+Cabin_E38+Cabin_E40+Cabin_E44+Cabin_E46+Cabin_E49+Cabin_E50+Cabin_E58+Cabin_E63+Cabin_E67+Cabin_E68+Cabin_E77+Cabin_E8+Cabin_F2+Cabin_F33+Cabin_F38+Cabin_F4+Cabin_F_E69+Cabin_F_G63+Cabin_F_G73+Cabin_G6+Cabin_T+Embarked+Parch+PassengerId+Sex+SibSp', data = titanic_train_new2, return_type='dataframe')



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]

vif["features"] = b.columns



print(vif)
vif.to_csv("vif.csv")
#for logistic regression using statsmodels

train1, test1 = train_test_split(titanic_train_new2, test_size=0.3, random_state=0)
train1
import statsmodels.formula.api as sm

import sklearn.metrics as metrics
logreg = sm.logit(formula='Survived_1 ~ Age+Sex+Cabin_A5+Cabin_A6+Cabin_A7+Cabin_B3+Cabin_B4+Cabin_B5+Embarked+PassengerId+SibSp', data = train1)

result = logreg.fit()
print(result.summary())
train_gini = 2*metrics.roc_auc_score(train1['Survived_1'], result.predict(train1)) - 1

print("The Gini Index for the model built on the Train Data is : ", train_gini)



test_gini = 2*metrics.roc_auc_score(test1['Survived_1'], result.predict(test1)) - 1

print("The Gini Index for the model built on the Test Data is : ", test_gini)



train_auc = metrics.roc_auc_score(train1['Survived_1'], result.predict(train1))

test_auc = metrics.roc_auc_score(test1['Survived_1'], result.predict(test1))



print("The AUC for the model built on the Train Data is : ", train_auc)

print("The AUC for the model built on the Test Data is : ", test_auc)

                                 
## Intuition behind ROC curve - predicted probability as a tool for separating the '1's and '0's

train_predicted_prob = pd.DataFrame(result.predict(train1))

train_predicted_prob.columns = ['prob']

train_actual = train1['Survived_1']

# making a DataFrame with actual and prob columns

train_predict = pd.concat([train_actual, train_predicted_prob], axis=1)

train_predict.columns = ['actual','prob']

train_predict.head()
## Intuition behind ROC curve - predicted probability as a tool for separating the '1's and '0's

test_predicted_prob = pd.DataFrame(result.predict(test1))

test_predicted_prob.columns = ['prob']

test_actual = test1['Survived_1']

# making a DataFrame with actual and prob columns

test_predict = pd.concat([test_actual, test_predicted_prob], axis=1)

test_predict.columns = ['actual','prob']

test_predict.head()
## Intuition behind ROC curve - confusion matrix for each different cut-off shows trade off in sensitivity and specificity

roc_like_df = pd.DataFrame()

train_temp = train_predict.copy()



for cut_off in np.linspace(0,1,50):

    train_temp['cut_off'] = cut_off

    train_temp['predicted'] = train_temp['prob'].apply(lambda x: 0.0 if x < cut_off else 1.0)

    train_temp['tp'] = train_temp.apply(lambda x: 1.0 if x['actual']==1.0 and x['predicted']==1 else 0.0, axis=1)

    train_temp['fp'] = train_temp.apply(lambda x: 1.0 if x['actual']==0.0 and x['predicted']==1 else 0.0, axis=1)

    train_temp['tn'] = train_temp.apply(lambda x: 1.0 if x['actual']==0.0 and x['predicted']==0 else 0.0, axis=1)

    train_temp['fn'] = train_temp.apply(lambda x: 1.0 if x['actual']==1.0 and x['predicted']==0 else 0.0, axis=1)

    sensitivity = train_temp['tp'].sum() / (train_temp['tp'].sum() + train_temp['fn'].sum())

    specificity = train_temp['tn'].sum() / (train_temp['tn'].sum() + train_temp['fp'].sum())

    accuracy = (train_temp['tp'].sum()  + train_temp['tn'].sum() ) / (train_temp['tp'].sum() + train_temp['fn'].sum() + train_temp['tn'].sum() + train_temp['fp'].sum())

    roc_like_table = pd.DataFrame([cut_off, sensitivity, specificity, accuracy]).T

    roc_like_table.columns = ['cutoff', 'sensitivity', 'specificity', 'accuracy']

    roc_like_df = pd.concat([roc_like_df, roc_like_table], axis=0)
roc_like_df
## Finding ideal cut-off for checking if this remains same in OOS validation

roc_like_df['total'] = roc_like_df['sensitivity'] + roc_like_df['specificity']
roc_like_df.head()
#Cut-off based on highest sum(sensitivity+specicity)   - common way of identifying cut-off

roc1=roc_like_df[roc_like_df['total']==roc_like_df['total'].max()]

roc1
#Cut-off based on highest accuracy   - some teams use this as methodology to decide the cut-off

roc2=roc_like_df[roc_like_df['accuracy']==roc_like_df['accuracy'].max()]

roc2
#Cut-off based on highest sensitivity

roc3=roc_like_df[roc_like_df['sensitivity']==roc_like_df['sensitivity'].max()]

roc3
#Choosen Best Cut-off is 0.367347 based on highest (sensitivity+specicity)



test_predict['predicted'] = test_predict['prob'].apply(lambda x: 1 if x > 0.367347 else 0)

train_predict['predicted'] = train_predict['prob'].apply(lambda x: 1 if x > 0.367347 else 0)
test_predict.head()
train_predict.head()
sns.heatmap(pd.crosstab(train_predict['actual'], train_predict['predicted']), annot=True, fmt='.0f')

plt.title('Train Data Confusion Matrix')

plt.show()

sns.heatmap(pd.crosstab(test_predict['actual'], test_predict['predicted']), annot=True, fmt='.0f')

plt.title('Test Data Confusion Matrix')

plt.show()
print("The overall accuracy score for the Train Data is : ", metrics.accuracy_score(train_predict.actual, train_predict.predicted))

print("The overall accuracy score for the Test Data  is : ", metrics.accuracy_score(test_predict.actual, test_predict.predicted))
print(metrics.classification_report(train_predict.actual, train_predict.predicted))
print(metrics.classification_report(test_predict.actual, test_predict.predicted))
train_predict['Deciles']=pd.qcut(train_predict['prob'],10, labels=False)



train_predict.head()
test_predict['Deciles']=pd.qcut(test_predict['prob'],10, labels=False)



test_predict.head()
# Decile Analysis for train data



no_1s = train_predict[['Deciles','actual']].groupby(train_predict.Deciles).sum().sort_index(ascending=False)['actual']

no_total = train_predict[['Deciles','actual']].groupby(train_predict.Deciles).count().sort_index(ascending=False)['actual']

max_prob = train_predict[['Deciles','prob']].groupby(train_predict.Deciles).max().sort_index(ascending=False)['prob']

min_prob = train_predict[['Deciles','prob']].groupby(train_predict.Deciles).min().sort_index(ascending=False)['prob']
Decile_analysis_train1 = pd.concat([max_prob, min_prob, no_1s, no_total-no_1s, no_total], axis=1)



Decile_analysis_train1.columns = ['max_prob','min_prob','#1','#0','total']
Decile_analysis_train1
# Decile Analysis for train data



no_1s = test_predict[['Deciles','actual']].groupby(test_predict.Deciles).sum().sort_index(ascending=False)['actual']

no_total = test_predict[['Deciles','actual']].groupby(test_predict.Deciles).count().sort_index(ascending=False)['actual']

max_prob = test_predict[['Deciles','prob']].groupby(test_predict.Deciles).max().sort_index(ascending=False)['prob']

min_prob = test_predict[['Deciles','prob']].groupby(test_predict.Deciles).min().sort_index(ascending=False)['prob']



Decile_analysis_test1 = pd.concat([max_prob, min_prob, no_1s, no_total-no_1s, no_total], axis=1)



Decile_analysis_test1.columns = ['max_prob','min_prob','#1','#0','total']
Decile_analysis_test1