#This data has been retrieved from Oura Cloud and it desrcribes average resting heart rate and

# Oura's sleep, activity and readiness scores.
import pandas as pd

#In Kaggle's case we use customized link to download our source data

ouradata = pd.read_csv("../input/oura-health-data-analysis-one-year-period/oura_2019_trends.csv")
# Libraries to import 

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.linear_model import LinearRegression

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split



# Head description of the data for the first 4 rows.

ouradata.head()
ouradata.describe(include='all')
ouradata = ouradata[ouradata["sleep_score"] != "NaN"]

ouradata = ouradata[ouradata["readiness_score"] != "NaN"]

ouradata = ouradata[ouradata["average_rhr"] != "NaN"]

ouradata = ouradata[ouradata["sleep_score"] != "NaN"]
# Pre-processing such as normalizing the data etc.

ouradata.dropna(subset=['sleep_score', 'readiness_score', 'average_rhr', 'readiness_score'])
#Keep the DataFrame with valid entries in the same variable.

ouradata.dropna(axis=0, inplace=True)

ouradata.isnull().sum()
sns.distplot(ouradata['sleep_score'])
q = ouradata['sleep_score'].quantile(0.99)

ouradata = ouradata[ouradata['sleep_score']<q]
sns.distplot(ouradata['sleep_score'])
sns.distplot(ouradata['readiness_score'])
sns.distplot(ouradata['average_rhr'])
# Here we decided to use some matplotlib code, without explaining it

# You can simply use plt.scatter() for each of them (with your current knowledge)

# But since Price is the 'y' axis of all the plots, it made sense to plot them side-by-side (so we can compare them)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(10,3)) #sharey -> share 'Price' as y

ax1.scatter(ouradata['sleep_score'],ouradata['average_rhr'])

ax1.set_title('Sleep and Average RHR')

ax2.scatter(ouradata['readiness_score'], ouradata['average_rhr'])

ax2.set_title('Readiness and Average RHR')

ax3.scatter(ouradata['activity_score'], ouradata['average_rhr'])

ax3.set_title('Activity and Average RHR')
#x = ouradata[['sleep_score','readiness_score']]

#y = ouradata['average_rhr']
#Get natural logarithm 

logaritmi_rhr = np.log(ouradata['average_rhr'])

ouradata['logaritmi_rhr'] = logaritmi_rhr

ouradata
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(10,3)) #sharey -> share 'Price' as y

ax1.scatter(ouradata['sleep_score'],ouradata['logaritmi_rhr'])

ax1.set_title('Sleep and Average RHR')

ax2.scatter(ouradata['readiness_score'], ouradata['logaritmi_rhr'])

ax2.set_title('Readiness and Average RHR')

ax3.scatter(ouradata['activity_score'], ouradata['logaritmi_rhr'])

ax3.set_title('Activity and Average RHR')
ouradata.describe(include='all')
ouradata.columns.values
# sklearn does not have a built-in way to check for multicollinearity

# one of the main reasons is that this is an issue well covered in statistical frameworks and not in ML ones

# surely it is an issue nonetheless, thus we will try to deal with it



# Here's the relevant module

# full documentation: http://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor

from statsmodels.stats.outliers_influence import variance_inflation_factor



# To make this as easy as possible to use, we declare a variable where we put

# all features where we want to check for multicollinearity

# since our categorical data is not yet preprocessed, we will only take the numerical ones

variables = ouradata[['sleep_score','readiness_score','average_rhr','logaritmi_rhr']]



# we create a new data frame which will include all the VIFs

# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)

vif = pd.DataFrame()



# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 

vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

# Finally, I like to include names so it is easier to explore the result

vif["Features"] = variables.columns
vif
#natural logaritm's rhr stays and drop the original average_rhr value out

ouradata = ouradata.drop(['average_rhr'],axis=1)
#confirm, that original average_rhr value has been dropped:

ouradata.columns.values
# To include the categorical data in the regression, let's create dummies

# There is a very convenient method called: 'get_dummies' which does that seemlessly

# It is extremely important that we drop one of the dummies, alternatively we will introduce multicollinearity

data_with_dummies = pd.get_dummies(ouradata, prefix=['readiness_score'], columns=['readiness_score'], drop_first=True)
# Here's the result

data_with_dummies.head()
# To make our data frame more organized, we prefer to place the dependent variable in the beginning of the df

# Since each problem is different, that must be done manually

# We can display all possible features and then choose the desired order

data_with_dummies.columns.values
# To make the code a bit more parametrized, let's declare a new variable that will contain the preferred order

# If you want a different order, just specify it here

# Conventionally, the most intuitive order is: dependent variable, indepedendent numerical variables, dummies

# NOTE! Left date column out since string cannot be converted into float

cols = ['sleep_score', 'activity_score', 'logaritmi_rhr',

       'readiness_score_45.0', 'readiness_score_48.0',

       'readiness_score_52.0', 'readiness_score_55.0',

       'readiness_score_56.0', 'readiness_score_58.0',

       'readiness_score_59.0', 'readiness_score_60.0',

       'readiness_score_61.0', 'readiness_score_62.0',

       'readiness_score_63.0', 'readiness_score_64.0',

       'readiness_score_65.0', 'readiness_score_66.0',

       'readiness_score_67.0', 'readiness_score_68.0',

       'readiness_score_69.0', 'readiness_score_70.0',

       'readiness_score_71.0', 'readiness_score_72.0',

       'readiness_score_73.0', 'readiness_score_74.0',

       'readiness_score_75.0', 'readiness_score_76.0',

       'readiness_score_77.0', 'readiness_score_78.0',

       'readiness_score_79.0', 'readiness_score_80.0',

       'readiness_score_81.0', 'readiness_score_82.0',

       'readiness_score_83.0', 'readiness_score_84.0',

       'readiness_score_85.0', 'readiness_score_86.0',

       'readiness_score_87.0', 'readiness_score_88.0',

       'readiness_score_89.0', 'readiness_score_90.0',

       'readiness_score_91.0', 'readiness_score_92.0',

       'readiness_score_93.0', 'readiness_score_94.0',

       'readiness_score_95.0', 'readiness_score_96.0']
data_preprocessed = data_with_dummies[cols]

data_preprocessed.head()
targets = data_preprocessed['logaritmi_rhr']

inputs = data_preprocessed.drop(['logaritmi_rhr'], axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.35, random_state=365) 
# Luo malli (eli mallin objekti), opetus vasta myöhemmin

reg = LinearRegression()

reg.fit(x_train,y_train)

# Valitse malli tai mallit (yksi tai monta)

# esim. voit vertailla lineaarista ja epälineaarista mallia
y_hat =reg.predict(x_train)
#The closer our scatter plot to this line the better the model 

plt.scatter(y_train, y_hat)

plt.xlabel('Targets (y_train)', size=18)

plt.ylabel('Predictions (y_hat)', size=18)

plt.xlim()

plt.ylim()

plt.show()
# Residuals are the estimate of the errors

sns.distplot(y_train - y_hat)

plt.title("Residuals PDF", size=18)
#Our model representing roughly 73% of our dataset's variability

reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary

#Interpretation: Continuous variables 

#1. A positive weight shows that as a feature increase in value, so do the logaritmi_rhr and "average_rhr" respectively

# 2. A negative weight shows that as a feature increase in value, logaritmi_rhr and "average_rhr" decrease
# Dummy variables:

# 1. A positive weight shows that the respective category (different health measures)

# has bigger impact on average resting heart rate, which is the benchmark 

# 2. A negative weight shows that the respective category has less impact on average resting heart rate, which is the benchmark
y_hat_test = reg.predict(x_test)
#plt.scatter(x,y [, alpha]) create a scatter plot

# alpha: specifies the opacity 

plt.scatter(y_train, y_hat, alpha=0.2)

plt.xlabel('Targets (y_train)', size=18)

plt.ylabel('Predictions (y_hat)', size=18)

plt.xlim()

plt.ylim()

plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])

df_pf.head()
df_pf['Target'] = np.exp(y_test)

df_pf.head()
y_test
y_test = y_test.reset_index(drop=True)

y_test.head()
df_pf['Target'] = np.exp(y_test)

df_pf
df_pf['Target'] = np.exp(y_test)

df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

df_pf
#The closest prediction is 0.259%, while largest gap is 50.88...% 

df_pf.describe()
pd.options.display.max_rows = 999 

pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_pf.sort_values(by=['Difference%'])