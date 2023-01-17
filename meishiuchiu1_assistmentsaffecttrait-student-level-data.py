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

dfStem=dfDummy.dropna(subset=['isSTEM'], how='any')

dfStem.shape
#locate columns with the data type of object

dfStem.loc[:, dfStem.dtypes == 'object'].head()
# use means to transform df to student level data

stud = dfStem.groupby('studentId').mean()

stud.shape # from 84 to 78 columns due to delete object columns and 'isSTEM'
stud.head()
#Find column names with missing data because sklearn does not allow missing data

stud.columns[stud.isnull().any()]
# impute MCAS missing values with median

MCAS_median = np.nanmedian(stud['MCAS'])

new_MCAS = np.where(stud['MCAS'].isnull(), MCAS_median, stud['MCAS'])

stud['MCAS'] = new_MCAS

stud.head()
feature_colsCor= [

  'isSTEM',

 'AveResBored',

 'AveResEngcon',

 'AveResConf',

 'AveResFrust',

 'AveResOfftask',

 'AveResGaming']

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

 'AveResBored',

 'AveResEngcon',

 'AveResConf',

 'AveResFrust',

 'AveResOfftask',

 'AveResGaming']

X = stud[feature_cols]

y = stud.isSTEM
#Find column names with missing data because sklearn does not allow missing data

X.columns[X.isnull().any()]
#Compute linear regression standardized coefficient (beta) with Python

#https://stackoverflow.com/questions/33913868/compute-linear-regression-standardized-coefficient-beta-with-python

import statsmodels.api as sm

from scipy.stats.mstats import zscore
#logistic regression result is ok.

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
logit1 = sm.Logit(y, stud.AveResBored).fit()

logit1.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit1.params

conf = logit1.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit2 = sm.Logit(y, stud.AveResEngcon).fit() # significant but negative, not good because Engcon is positive in meaning.

logit2.summary()
#odds ratio with conf. intervals; OR hard to explain

params = logit2.params

conf = logit2.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit3 = sm.Logit(y, stud.AveResConf).fit()

logit3.summary()
#odds ratio with conf. intervals

params = logit3.params

conf = logit3.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit4 = sm.Logit(y, stud.AveResFrust).fit()

logit4.summary()
#odds ratio with conf. intervals

params = logit4.params

conf = logit4.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit5 = sm.Logit(y, stud.AveResOfftask).fit()

logit5.summary()
#odds ratio with conf. intervals

params = logit5.params

conf = logit5.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit6 = sm.Logit(y, stud.AveResGaming).fit()

logit6.summary()
#odds ratio with conf. intervals

params = logit6.params

conf = logit6.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit1z = sm.Logit(y, zscore(stud.AveResBored)).fit()

logit1z.summary()
#odds ratio with conf. intervals

params = logit1z.params

conf = logit1z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit2z = sm.Logit(y, zscore(stud.AveResEngcon)).fit()

logit2z.summary()
#odds ratio with conf. intervals

params = logit2z.params

conf = logit2z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit3z = sm.Logit(y, zscore(stud.AveResConf)).fit()

logit3z.summary()
#odds ratio with conf. intervals

params = logit3z.params

conf = logit3z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit4z = sm.Logit(y, zscore(stud.AveResFrust)).fit()

logit4z.summary()
#odds ratio with conf. intervals

params = logit4z.params

conf = logit4z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit5z = sm.Logit(y, zscore(stud.AveResOfftask)).fit()

logit5z.summary()
#odds ratio with conf. intervals

params = logit5z.params

conf = logit5z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logit6z = sm.Logit(y, zscore(stud.AveResGaming)).fit()

logit6z.summary()
#odds ratio with conf. intervals

params = logit6z.params

conf = logit6z.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
sm.OLS(y, X).fit().summary() # large R-squared, go for this because also good result for individual predictor OLS result 
# as effect size measures

Xz = zscore(X)

yz = zscore(y)

orzxy = sm.OLS(yz, Xz).fit()

orzxy.summary()
orzx = sm.OLS(y, Xz).fit() #lower Adj. R-squared than both y and x zscore-->no use

orzx.summary()
sm.OLS(y, stud.AveResBored).fit().summary()
sm.OLS(y, stud.AveResEngcon).fit().summary()
sm.OLS(y, stud.AveResConf).fit().summary()
sm.OLS(y, stud.AveResFrust).fit().summary()
sm.OLS(y, stud.AveResOfftask).fit().summary()
sm.OLS(y, stud.AveResGaming).fit().summary()
sm.OLS(y, zscore(stud.AveResBored)).fit().summary() # zscore x1 with negative adj. r-squared
sm.OLS(zscore(y), zscore(stud.AveResBored)).fit().summary() 

# zscore y and x1 with negative adj. r-squared; same value as correlation between y and x1
sm.OLS(zscore(y), zscore(stud.AveResBored)).fit().summary()
sm.OLS(zscore(y), zscore(stud.AveResEngcon)).fit().summary()
sm.OLS(zscore(y), zscore(stud.AveResConf)).fit().summary()
sm.OLS(zscore(y), zscore(stud.AveResFrust)).fit().summary()
sm.OLS(zscore(y), zscore(stud.AveResOfftask)).fit().summary()
sm.OLS(zscore(y), zscore(stud.AveResGaming)).fit().summary()
!pip install eli5
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

#https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())

import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(my_model)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_values = explainer.shap_values(val_X)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_values[1], val_X)
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



#random forest: too high accuracy and R-squared

#https://www.kaggle.com/shrutimechlearn/step-by-step-assumptions-linear-regression?utm_medium=email&utm_source=intercom&utm_campaign=datanotes-2019

from sklearn.metrics import r2_score



rf_tree = RandomForestClassifier(random_state=0)

rf_tree.fit(X,y)

rf_tree_y_pred = rf_tree.predict(X)

print("Accuracy: {}".format(rf_tree.score(X,y)))

print("R squared: {}".format(r2_score(y_true=y,y_pred=rf_tree_y_pred)))
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

 'AveResBored',

 'AveResEngcon',

 'AveResConf',

 'AveResFrust',

 'AveResOfftask',

 'AveResGaming']

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
logitF1 = sm.Logit(yF, female.AveResBored).fit()

logitF1.summary()
#odds ratio with conf. intervals

params = logitF1.params

conf = logitF1.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF2 = sm.Logit(yF, female.AveResEngcon).fit()

logitF2.summary()
#odds ratio with conf. intervals

params = logitF2.params

conf = logitF2.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF3 = sm.Logit(yF, female.AveResConf).fit()

logitF3.summary()
#odds ratio with conf. intervals

params = logitF3.params

conf = logitF3.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF4 = sm.Logit(yF, female.AveResFrust).fit()

logitF4.summary()
#odds ratio with conf. intervals

params = logitF4.params

conf = logitF4.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF5 = sm.Logit(yF, female.AveResOfftask).fit()

logitF5.summary()
#odds ratio with conf. intervals

params = logitF5.params

conf = logitF5.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitF6 = sm.Logit(yF, female.AveResGaming).fit()

logitF6.summary()
#odds ratio with conf. intervals

params = logitF6.params

conf = logitF6.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
sm.OLS(yF, XF).fit().summary()#sig: concentrate positive; game negative 
sm.OLS(zscore(yF), zscore(XF)).fit().summary() # as effect sizes --> x2 become negative and non-significant-->hard to explain
sm.OLS(zscore(yF), zscore(female.AveResBored)).fit().summary()
sm.OLS(zscore(yF), zscore(female.AveResEngcon)).fit().summary()
sm.OLS(zscore(yF), zscore(female.AveResConf)).fit().summary()
sm.OLS(zscore(yF), zscore(female.AveResFrust)).fit().summary()
sm.OLS(zscore(yF), zscore(female.AveResOfftask)).fit().summary()
sm.OLS(zscore(yF), zscore(female.AveResGaming)).fit().summary()
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_XF, val_XF, train_yF, val_yF = train_test_split(XF, yF, random_state=1)

my_modelF = RandomForestClassifier(random_state=0).fit(train_XF, train_yF)

permF = PermutationImportance(my_modelF, random_state=1).fit(val_XF, val_yF)

eli5.show_weights(permF, feature_names = val_XF.columns.tolist())
import shap  # package used to calculate Shap values

# Create object that can calculate shap values

explainerF = shap.TreeExplainer(my_modelF)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_valuesF = explainer.shap_values(val_XF)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_valuesF[1], val_XF)
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

 'AveResBored',

 'AveResEngcon',

 'AveResConf',

 'AveResFrust',

 'AveResOfftask',

 'AveResGaming']

XM = male[feature_colsM]

yM = male.isSTEM
#Compute linear regression standardized coefficient (beta) with Python

#https://stackoverflow.com/questions/33913868/compute-linear-regression-standardized-coefficient-beta-with-python

import statsmodels.api as sm

from scipy.stats.mstats import zscore



#logistic regression result: no significant predictors

logitM = sm.Logit(yM, XM).fit()

logitM.summary()
#odds ratio with conf. intervals; results hard to explain with relative importance

params = logitM.params

conf = logitM.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM1 = sm.Logit(yM, male.AveResBored).fit()

logitM1.summary()
#odds ratio with conf. intervals

params = logitM1.params

conf = logitM1.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM2 = sm.Logit(yM, male.AveResEngcon).fit()

logitM2.summary()
#odds ratio with conf. intervals

params = logitM2.params

conf = logitM2.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM3 = sm.Logit(yM, male.AveResConf).fit()

logitM3.summary()
#odds ratio with conf. intervals

params = logitM3.params

conf = logitM3.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM4 = sm.Logit(yM, male.AveResFrust).fit()

logitM4.summary()
#odds ratio with conf. intervals

params = logitM4.params

conf = logitM4.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM5 = sm.Logit(yM, male.AveResOfftask).fit()

logitM5.summary()
#odds ratio with conf. intervals

params = logitM5.params

conf = logitM5.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
logitM6 = sm.Logit(yM, male.AveResGaming).fit()

logitM6.summary()
#odds ratio with conf. intervals

params = logitM6.params

conf = logitM6.conf_int()

conf['OR'] = params

conf.columns = ['2.5%', '97.5%', 'OR']

np.exp(conf)
sm.OLS(yM, XM).fit().summary() #all non-significant
sm.OLS(zscore(yM), zscore(XM)).fit().summary() # as effect sizes, all non-significant
sm.OLS(zscore(yM), zscore(male.AveResBored)).fit().summary()
sm.OLS(zscore(yM), zscore(male.AveResEngcon)).fit().summary()
sm.OLS(zscore(yM), zscore(male.AveResConf)).fit().summary()
sm.OLS(zscore(yM), zscore(male.AveResFrust)).fit().summary()
sm.OLS(zscore(yM), zscore(male.AveResOfftask)).fit().summary()
sm.OLS(zscore(yM), zscore(male.AveResGaming)).fit().summary()
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_XM, val_XM, train_yM, val_yM = train_test_split(XM, yM, random_state=1)

my_modelM = RandomForestClassifier(random_state=0).fit(train_XM, train_yM)

permM = PermutationImportance(my_modelM, random_state=1).fit(val_XM, val_yM)

eli5.show_weights(permM, feature_names = val_XM.columns.tolist())
import shap 

# Create object that can calculate shap values

explainerM = shap.TreeExplainer(my_modelM)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_valuesM = explainer.shap_values(val_XM)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_valuesM[1], val_XM)
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