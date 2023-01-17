# conventional way to import pandas

import pandas as pd
#read file.

df = pd.read_csv('../input/anonymized_full_release_competition_dataset20181128.csv')

#replace spaces with underscores for all columns 

df.columns = df.columns.str.replace(' ', '_')

df.head()
df.shape
#locate a value in a column as Nan https://stackoverflow.com/questions/45416684/python-pandas-replace-multiple-columns-zero-to-nan?rq=1

import numpy as np

df.loc[df['MCAS'] == -999.0,'MCAS'] = np.nan
# create the 'genderFemale' dummy variable using the 'map' method

df['genderFemale'] = df.InferredGender.map({'Female':1, 'Male':0})

# Removing unused columns

list_drop = ['InferredGender']

df.drop(list_drop, axis=1, inplace=True)
df.shape
df.head()
# create dummy variables for multiple categories; this drops nominal columns and creates dummy variables

dfDummy=pd.get_dummies(df, columns=['MiddleSchoolId'], drop_first=True)

dfDummy.shape
#use observations only with no missing in isSTEM

stud=df.dropna(subset=['isSTEM'], how='any')

stud.shape
#Find column names with missing data because sklearn does not allow missing data

stud.columns[stud.isnull().any()]
# impute MCAS missing values with median

MCAS_median = np.nanmedian(stud['MCAS'])

new_MCAS = np.where(stud['MCAS'].isnull(), MCAS_median, stud['MCAS'])

stud['MCAS'] = new_MCAS

stud.head()
feature_colsCor= [

 'isSTEM',

 'RES_BORED',

 'RES_CONCENTRATING',

 'RES_CONFUSED',

 'RES_FRUSTRATED',

 'RES_OFFTASK',

 'RES_GAMING']

corrAll = stud[feature_colsCor]
from scipy.stats import pearsonr

import pandas as pd



def calculate_pvalues(df):

    df = df.dropna()._get_numeric_data()

    dfcols = pd.DataFrame(columns=df.columns)

    pvalues = dfcols.transpose().join(dfcols, how='outer')

    for r in df.columns:

        for c in df.columns:

            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)

    return pvalues



rho = corrAll.corr()

rho = rho.round(4)

pval = calculate_pvalues(corrAll) 

# create three masks

r1 = rho.applymap(lambda x: '{}*'.format(x))

r2 = rho.applymap(lambda x: '{}**'.format(x))

# apply them where appropriate

rho = rho.mask(pval< 0.05,r1)

rho = rho.mask(pval< 0.01,r2)

rho
#reference: http://seaborn.pydata.org/tutorial/distributions.html

import seaborn as sns

sns.pairplot(corrAll); #corrAll = stud[feature_colsCor]
corrAll.describe()
corrAll.skew()
corrAll.kurt()
# list(stud) to copy column names

feature_cols = [

 'RES_BORED',

 'RES_CONCENTRATING',

 'RES_CONFUSED',

 'RES_FRUSTRATED',

 'RES_OFFTASK',

 'RES_GAMING']

X = stud[feature_cols]

y = stud.isSTEM
#Find column names with missing data because sklearn does not allow missing data

X.columns[X.isnull().any()]
#Compute linear regression standardized coefficient (beta) with Python

#https://stackoverflow.com/questions/33913868/compute-linear-regression-standardized-coefficient-beta-with-python

import statsmodels.api as sm

from scipy.stats.mstats import zscore
#logistic regression: negative pseudo r-squared; negative concentrating; unique: off-task positive

logit = sm.Logit(y, X).fit()

logit.summary()
#odds ratio with conf. intervals; odds ratio as effect sizes; results hard to explain with relative importance

params = logit.params

conf = logit.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
#negative pseudo R-square, cannot do y2 = zscore(y)

Xz = zscore(X)

logitXz = sm.Logit(y, Xz).fit()

logitXz.summary()
#odds ratio with conf. intervals; odds ratio as effect sizes; results hard to explain with relative importance

params = logitXz.params

conf = logitXz.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit1 = sm.Logit(y, stud.RES_BORED).fit()

logit1.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit1.params

conf = logit1.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit2 = sm.Logit(y, stud.RES_CONCENTRATING).fit() # significant but negative, not good because Engcon is positive in meaning.

logit2.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit2.params

conf = logit2.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit3 = sm.Logit(y, stud.RES_CONFUSED).fit()

logit3.summary()
#odds ratio with conf. intervals

params = logit3.params

conf = logit3.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit4 = sm.Logit(y, stud.RES_FRUSTRATED).fit()

logit4.summary()
#odds ratio with conf. intervals

params = logit4.params

conf = logit4.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit5 = sm.Logit(y, stud.RES_OFFTASK).fit()

logit5.summary()
#odds ratio with conf. intervals

params = logit5.params

conf = logit5.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit6 = sm.Logit(y, stud.RES_GAMING).fit()

logit6.summary()
#odds ratio with conf. intervals

params = logit6.params

conf = logit6.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit1z = sm.Logit(y, zscore(stud.RES_BORED)).fit()

logit1z.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit1z.params

conf = logit1z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit2z = sm.Logit(y, zscore(stud.RES_CONCENTRATING)).fit()

logit2z.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit2z.params

conf = logit2z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit3z = sm.Logit(y, zscore(stud.RES_CONFUSED)).fit()

logit3z.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit3z.params

conf = logit3z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit4z = sm.Logit(y, zscore(stud.RES_FRUSTRATED)).fit()

logit4z.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit4z.params

conf = logit4z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit5z = sm.Logit(y, zscore(stud.RES_OFFTASK)).fit()

logit5z.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit5z.params

conf = logit5z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit6z = sm.Logit(y, zscore(stud.RES_GAMING)).fit()

logit6z.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit6z.params

conf = logit6z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
sm.OLS(y, X).fit().summary() # large R-squared, but positive bored, confused, and frustrated. 
# as effect size measures; ok coefficient: negative bored and gaming

Xz = zscore(X)

yz = zscore(y)

orzxy = sm.OLS(yz, Xz).fit()

orzxy.summary()
orzx = sm.OLS(y, Xz).fit() #same Adj. R-squared and coefficient patterns as both y and x zscore;

orzx.summary()
sm.OLS(y, stud.RES_BORED).fit().summary()
sm.OLS(y, stud.RES_CONCENTRATING).fit().summary()
sm.OLS(y, stud.RES_CONFUSED).fit().summary()
sm.OLS(y, stud.RES_FRUSTRATED).fit().summary()
sm.OLS(y, stud.RES_OFFTASK).fit().summary()
sm.OLS(y, stud.RES_GAMING).fit().summary()
sm.OLS(zscore(y), zscore(stud.RES_BORED)).fit().summary()
sm.OLS(zscore(y), zscore(stud.RES_CONCENTRATING)).fit().summary()
sm.OLS(zscore(y), zscore(stud.RES_CONFUSED)).fit().summary()
sm.OLS(zscore(y), zscore(stud.RES_FRUSTRATED)).fit().summary()
sm.OLS(zscore(y), zscore(stud.RES_OFFTASK)).fit().summary()
sm.OLS(zscore(y), zscore(stud.RES_GAMING)).fit().summary()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)



model = LogisticRegression()

model.fit(X_train, y_train)



y_predict = model.predict(X_test)



from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



y_predict_probabilities = model.predict_proba(X_test)[:,1]



fpr, tpr, _ = roc_curve(y_test, y_predict_probabilities)

roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='darkorange',

         lw=2, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

val_y = my_model.predict(train_X)



from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



y_predict_probabilities2 = model.predict_proba(train_X)[:,1]



fpr2, tpr2, _ = roc_curve(val_y, y_predict_probabilities2)

roc_auc2 = auc(fpr2, tpr2)



plt.figure()

plt.plot(fpr2, tpr2, color='darkorange',

         lw=2, label='ROC curve (area = %0.3f)' % roc_auc2)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()



import numpy as np
female=stud[stud.genderFemale == 1]

male=stud[stud.genderFemale == 0]
female.shape
male.shape
corrF = female[feature_colsCor]

rho = corrF.corr()

rho = rho.round(4)

pval = calculate_pvalues(corrF) 

# create three masks

r1 = rho.applymap(lambda x: '{}*'.format(x))

r2 = rho.applymap(lambda x: '{}**'.format(x))

# apply them where appropriate

rho = rho.mask(pval< 0.05,r1)

rho = rho.mask(pval< 0.01,r2)

rho
feature_colsF = [

 'RES_BORED',

 'RES_CONCENTRATING',

 'RES_CONFUSED',

 'RES_FRUSTRATED',

 'RES_OFFTASK',

 'RES_GAMING']

XF = female[feature_colsF]

yF = female.isSTEM
#Compute linear regression standardized coefficient (beta) with Python

#https://stackoverflow.com/questions/33913868/compute-linear-regression-standardized-coefficient-beta-with-python

import statsmodels.api as sm

from scipy.stats.mstats import zscore



#logistic regression result is ok.

logitF = sm.Logit(yF, XF).fit()

logitF.summary()
#odds ratio with conf. intervals; results hard to explain with relative importance

params = logitF.params

conf = logitF.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF1 = sm.Logit(yF, female.RES_BORED).fit()

logitF1.summary()

#odds ratio with conf. intervals

params = logitF1.params

conf = logitF1.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF2 = sm.Logit(yF, female.RES_CONCENTRATING).fit()

logitF2.summary()
#odds ratio with conf. intervals

params = logitF2.params

conf = logitF2.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF3 = sm.Logit(yF, female.RES_CONFUSED).fit()

logitF3.summary()
#odds ratio with conf. intervals

params = logitF3.params

conf = logitF3.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF4 = sm.Logit(yF, female.RES_FRUSTRATED).fit()

logitF4.summary()
#odds ratio with conf. intervals

params = logitF4.params

conf = logitF4.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF5 = sm.Logit(yF, female.RES_OFFTASK).fit()

logitF5.summary()
#odds ratio with conf. intervals

params = logitF5.params

conf = logitF5.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF6 = sm.Logit(yF, female.RES_GAMING).fit()

logitF6.summary()
#odds ratio with conf. intervals

params = logitF6.params

conf = logitF6.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
sm.OLS(yF, XF).fit().summary()
sm.OLS(zscore(yF), zscore(XF)).fit().summary() 
sm.OLS(zscore(yF), zscore(female.RES_BORED)).fit().summary()
sm.OLS(zscore(yF), zscore(female.RES_CONCENTRATING)).fit().summary()
sm.OLS(zscore(yF), zscore(female.RES_CONFUSED)).fit().summary()
sm.OLS(zscore(yF), zscore(female.RES_FRUSTRATED)).fit().summary()
sm.OLS(zscore(yF), zscore(female.RES_OFFTASK)).fit().summary()
sm.OLS(zscore(yF), zscore(female.RES_GAMING)).fit().summary()
# Female_ROC and AUC: Logit regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



XF_train, XF_test, yF_train, yF_test = train_test_split(XF, yF, random_state=1)



modelF = LogisticRegression()

modelF.fit(XF_train, yF_train)



y_predictF = model.predict(XF_test)



from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



y_predict_probabilitiesF = modelF.predict_proba(XF_test)[:,1]



fprF, tprF, _ = roc_curve(yF_test, y_predict_probabilitiesF)

roc_aucF = auc(fprF, tprF)



plt.figure()

plt.plot(fprF, tprF, color='darkorange',

         lw=2, label='ROC curve (area = %0.3f)' % roc_aucF)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()

#Female_ROC and AUC: Random Forest



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_XF, val_XF, train_yF, val_yF = train_test_split(XF, yF, random_state=1)

my_modelF = RandomForestClassifier(random_state=0).fit(train_XF, train_yF)

val_yF = my_modelF.predict(train_XF)



from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



y_predict_probabilities2F = model.predict_proba(train_XF)[:,1]



fpr2F, tpr2F, _ = roc_curve(val_yF, y_predict_probabilities2F)

roc_auc2F = auc(fpr2F, tpr2F)



plt.figure()

plt.plot(fpr2F, tpr2F, color='darkorange',

         lw=2, label='ROC curve (area = %0.3f)' % roc_auc2F)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()





corrM = male[feature_colsCor]

rho = corrM.corr()

rho = rho.round(4)

pval = calculate_pvalues(corrM) 

# create three masks

r1 = rho.applymap(lambda x: '{}*'.format(x))

r2 = rho.applymap(lambda x: '{}**'.format(x))

# apply them where appropriate

rho = rho.mask(pval< 0.05,r1)

rho = rho.mask(pval< 0.01,r2)

rho
feature_colsM = [

 'RES_BORED',

 'RES_CONCENTRATING',

 'RES_CONFUSED',

 'RES_FRUSTRATED',

 'RES_OFFTASK',

 'RES_GAMING']

XM = male[feature_colsM]

yM = male.isSTEM
#Compute linear regression standardized coefficient (beta) with Python

#https://stackoverflow.com/questions/33913868/compute-linear-regression-standardized-coefficient-beta-with-python

import statsmodels.api as sm

from scipy.stats.mstats import zscore



#logistic regression result: bad: negative concentrating

logitM = sm.Logit(yM, XM).fit()

logitM.summary()
#odds ratio with conf. intervals; results hard to explain with relative importance

params = logitM.params

conf = logitM.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM1 = sm.Logit(yM, male.RES_BORED).fit()

logitM1.summary()
#odds ratio with conf. intervals

params = logitM1.params

conf = logitM1.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM2 = sm.Logit(yM, male.RES_CONCENTRATING).fit()

logitM2.summary()
#odds ratio with conf. intervals

params = logitM2.params

conf = logitM2.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM3 = sm.Logit(yM, male.RES_CONFUSED).fit()

logitM3.summary()
#odds ratio with conf. intervals

params = logitM3.params

conf = logitM3.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM4 = sm.Logit(yM, male.RES_FRUSTRATED).fit()

logitM4.summary()
#odds ratio with conf. intervals

params = logitM4.params

conf = logitM4.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM5 = sm.Logit(yM, male.RES_OFFTASK).fit()

logitM5.summary()
#odds ratio with conf. intervals

params = logitM5.params

conf = logitM5.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM6 = sm.Logit(yM, male.RES_GAMING).fit()

logitM6.summary()
#odds ratio with conf. intervals

params = logitM6.params

conf = logitM6.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
sm.OLS(yM, XM).fit().summary() #bad: postive bored, confused.
sm.OLS(zscore(yM), zscore(XM)).fit().summary() # as effect sizes; good: positive concentrating, negative furstrating and gaming
sm.OLS(zscore(yM), zscore(male.RES_BORED)).fit().summary()
sm.OLS(zscore(yM), zscore(male.RES_CONCENTRATING)).fit().summary()
sm.OLS(zscore(yM), zscore(male.RES_CONFUSED)).fit().summary()
sm.OLS(zscore(yM), zscore(male.RES_FRUSTRATED)).fit().summary()
sm.OLS(zscore(yM), zscore(male.RES_OFFTASK)).fit().summary()
sm.OLS(zscore(yM), zscore(male.RES_GAMING)).fit().summary()
# Male_ROC and AUC: Logit regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



XM_train, XM_test, yM_train, yM_test = train_test_split(XM, yM, random_state=1)



modelM = LogisticRegression()

modelM.fit(XM_train, yM_train)



y_predictM = model.predict(XM_test)



from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



y_predict_probabilitiesM = modelM.predict_proba(XM_test)[:,1]



fprM, tprM, _ = roc_curve(yM_test, y_predict_probabilitiesM)

roc_aucM = auc(fprM, tprM)



plt.figure()

plt.plot(fprM, tprM, color='darkorange',

         lw=2, label='ROC curve (area = %0.3f)' % roc_aucM)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()

#Male_ROC and AUC: Random Forest



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_XM, val_XM, train_yM, val_yM = train_test_split(XM, yM, random_state=1)

my_modelM = RandomForestClassifier(random_state=0).fit(train_XM, train_yM)

val_yM = my_modelM.predict(train_XM)



from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



y_predict_probabilities2M = model.predict_proba(train_XM)[:,1]



fpr2M, tpr2M, _ = roc_curve(val_yM, y_predict_probabilities2M)

roc_auc2M = auc(fpr2M, tpr2M)



plt.figure()

plt.plot(fpr2M, tpr2M, color='darkorange',

         lw=2, label='ROC curve (area = %0.3f)' % roc_auc2M)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()

#Writing a function to calculate the VIF values; https://statinfer.com/204-1-9-issue-of-multicollinearity-in-python/

import statsmodels.formula.api as sm

def vif_cal(input_data, dependent_col):

    x_vars=input_data.drop([dependent_col], axis=1)

    xvar_names=x_vars.columns

    for i in range(0,xvar_names.shape[0]):

        y=x_vars[xvar_names[i]] 

        x=x_vars[xvar_names.drop(xvar_names[i])]

        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  

        vif=round(1/(1-rsq),3)

        print (xvar_names[i], " VIF = " , vif)#Calculating VIF values using that function
vif_cal(input_data=corrAll, dependent_col="isSTEM") #all student data
vif_cal(input_data=corrF, dependent_col="isSTEM") #female data
vif_cal(input_data=corrM, dependent_col="isSTEM") #male data