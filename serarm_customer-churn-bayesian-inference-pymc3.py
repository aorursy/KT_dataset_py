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

# Standard machine learning models

from sklearn.linear_model import LogisticRegressionCV



# Scikit-learn utilities

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve
# PyMC3 for Bayesian Inference

import pymc3 as pm

print(pm.__version__)

import arviz

import matplotlib.pyplot as plt



%matplotlib inline

plt.style.use('seaborn-darkgrid')

from IPython.core.pylabtools import figsize

import matplotlib.lines as mlines



import seaborn as sns

import itertools



pd.options.mode.chained_assignment = None





from warnings import filterwarnings

filterwarnings('ignore')
telcom = pd.read_csv(r"/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#first few rows

telcom.head()
print ("Rows     : " ,telcom.shape[0])

print ("Columns  : " ,telcom.shape[1])

print ("\nFeatures : \n" ,telcom.columns.tolist())

print ("\nMissing values :  ", telcom.isnull().sum().values.sum())

print ("\nUnique values :  \n",telcom.nunique())
for i in telcom.columns:

    if len(telcom[i].unique())<10:

        print("Column:{},Unique values:{}".format(i,telcom[i].unique()))

    else:

        print("Column:{}Unique values:{}".format(i,len(telcom[i].unique())))



telcom_dummies=pd.DataFrame()

print("Total number of rows before starting copying:{}".format(len(telcom_dummies)))

# len(telcom_dummies[telcom_dummies['TotalCharges'] == " "])

telcom_dummies = pd.get_dummies(telcom[['gender','PaymentMethod','Contract']], columns=['gender','PaymentMethod','Contract'])

telcom_dummies['SeniorCitizen'] =telcom['SeniorCitizen']

telcom_dummies['Partner'] = telcom['Partner'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['Dependents'] = telcom['Dependents'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['tenure']=telcom['tenure']

telcom_dummies['PhoneService'] = telcom['PhoneService'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['MultipleLines'] = telcom['MultipleLines'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['Has_InternetService'] = telcom['InternetService'].map(lambda s :0  if s =='No' else 1)

telcom_dummies['Fiber_optic'] = telcom['InternetService'].map(lambda s :1  if s =='Fiber optic' else 0)

telcom_dummies['DSL'] = telcom['InternetService'].map(lambda s :1  if s =='DSL' else 0)

telcom_dummies['OnlineSecurity'] = telcom['OnlineSecurity'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['OnlineBackup'] = telcom['OnlineBackup'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['DeviceProtection'] = telcom['DeviceProtection'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['TechSupport'] = telcom['TechSupport'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['StreamingTV'] = telcom['StreamingTV'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['StreamingMovies'] = telcom['StreamingMovies'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['PaperlessBilling'] = telcom['PaperlessBilling'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies['MonthlyCharges']=telcom['MonthlyCharges']

telcom_dummies['TotalCharges'] = pd.to_numeric(telcom['TotalCharges'],errors='coerce')

print("Total number of rows after  copying:{}".format(len(telcom_dummies)))

      #Counting number of na

print("Number of NA")

print(len(telcom_dummies) - telcom_dummies.count())

telcom_dummies.dropna(axis=0,inplace=True)

print("Total number of rows after removing NA:{}".format(len(telcom_dummies)))

telcom_dummies['Churn']=telcom['Churn'].map(lambda s :1  if s =='Yes' else 0)

telcom_dummies.rename(columns={"PaymentMethod_Bank transfer (automatic)" :"paymnt_mthd_bank_auto",

"PaymentMethod_Credit card (automatic)"  : "paymnt_mthd_cc_auto",

"PaymentMethod_Electronic check"   :"paymnt_mthd_elc_check",

"PaymentMethod_Mailed check"       :"paymnt_mthd_mailed_check",         

"Contract_Month-to-month":"cont_mnth_to_mnth",                    

"Contract_One year"  :"cont_1_yr",                       

"Contract_Two year"    :"cont_2_yr" },inplace=True)

telcom_dummies.columns
print("Checking if columns are ready to apply ML algorithm")

for i in telcom_dummies.columns:

    if len(telcom_dummies[i].unique())<10:

        print("Column:{},Unique values:{},Type:{}".format(i,telcom_dummies[i].unique(),telcom_dummies[i].dtypes))

    else:

        print("Column:{}Unique values:{},Type:{}".format(i,len(telcom_dummies[i].unique()),telcom_dummies[i].dtypes))
y = telcom_dummies['Churn'].values

X = telcom_dummies.loc[:, telcom_dummies.columns != 'Churn']

from sklearn.preprocessing import MinMaxScaler

features = X.columns.values

scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X)

X = pd.DataFrame(scaler.transform(X))

X.columns = features

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
# Calculate the accuracy and f1 score of a model

def calc_metrics(predictions, y_test):

    accuracy = np.mean(predictions == y_test)

    f1_metric = f1_score(y_test, predictions)



    print('Accuracy of Model: {:.2f}%'.format(100 * accuracy))

    print('F1 Score of Model: {:.4f}'.format(f1_metric))

baseline_pred = [0 for _ in range(len(y_test))]

calc_metrics(baseline_pred, y_test)

lr = LogisticRegressionCV(Cs= 20, cv = 3, scoring = 'f1', 

                          penalty = 'l2', random_state = 42)

lr.fit(X_test, y_test)



# Make predictions and evaluate

lr_pred = lr.predict(X_test)

calc_metrics(lr_pred, y_test)
# Build up a formula

formula = [' %s + ' % variable for variable in X_test.columns]

formula.insert(0, 'y ~ ')

formula = ' '.join(''.join(formula).split(' ')[:-2])

formula
print('Intercept: {:0.4f}'.format(lr.intercept_[0]))

for feature, weight in zip(X_test.columns, lr.coef_[0]):

    print('Feature: {:30} Weight: {:0.4f}'.format(feature, weight))
X_with_labels = X_train.copy()

X_with_labels['y'] = y_train

with pm.Model() as logistic_model:

    priors=dict()

    

    for variable in X_test.columns:

        priors[variable]=pm.Uniform.dist(0,1)

    priors['Intercept']=pm.Normal.dist(mu=0., sigma=100.)

    priors['MonthlyCharges']=pm.Normal.dist(mu=0., sigma=100.)

    priors['TotalCharges'] = pm.Normal.dist(mu=0., sigma=100.)

    # Build the model using the formula and specify the data likelihood 

    pm.GLM.from_formula(formula, data = X_with_labels, family = pm.glm.families.Binomial(),priors=priors)

    

    # Using the no-uturn sampler

    sampler = pm.NUTS()

    

    # Sample from the posterior using NUTS

    trace_log = pm.sample(draws=2000, step = sampler, chains=1, tune=1000, random_seed=100,init='adapt_diag')
import pickle

fileObject = open("all_parameters.pickle",'wb')  

pickle.dump(trace_log, fileObject)

fileObject.close()
trace_log_from_file= pickle.load(open("all_parameters.pickle",'rb')  )

#trace_log=trace_log_from_file   #Uncomment this line if we don't want to run model again
figsize(10, 12)

pm.forestplot(trace_log);
pm.plot_posterior(trace_log);
pm.summary(trace_log)
def evaluate_trace(trace, data, print_model = False):

    means_dict = {}

    std_dict = {}

    

    for var in trace.varnames:

        means_dict[var] = np.mean(trace[var])

        std_dict[var] = np.std(trace[var])

    

    model = 'logit = %0.4f + ' % np.mean(means_dict['Intercept'])

    

    for var in data.columns:

        model += '%0.4f * %s + ' % (means_dict[var], var)

    

    model = ' '.join(model.split(' ')[:-2])

    if print_model:

        print('Final Equation: \n{}'.format(model))

    

    return means_dict, std_dict
means_dict, std_dict = evaluate_trace(trace_log, X_train, print_model=True)
# Find a single probabilty estimate using the mean value of variables in a trace

def find_probs(trace, data):

    

    # Find the means and std of the variables

    means_dict1, std_dict = evaluate_trace(trace, data)

          

    probs = []

       

    

    # Need an intercept term in the data

    data['Intercept'] = 1

    l_means_dict=dict()

    for c in data.columns:

        

        l_means_dict[c]=means_dict1[c]

    

    data = data[list(l_means_dict.keys())]

    mean_array = np.array(list(l_means_dict.values()))

    # Calculate the probability for each observation in the data

    for _, row in data.iterrows():

        # First the log odds

        logit = np.dot(row, mean_array)

        # Convert the log odds to a probability

        probability = 1 / (1 + np.exp(-logit))

        probs.append(probability)

        

    return probs
blr_probs = find_probs(trace_log, X_test.copy())



# Threshold the values at 0.5

predictions = (np.array(blr_probs) > 0.5)

calc_metrics(predictions, y_test)
X_test2=X_test[X_test.columns.difference(['MonthlyCharges', 'paymnt_mthd_cc_auto', 'cont_1_yr', 'cont_2_yr', 'Partner', 'Dependents', 'Tenure', 'PhoneService', 'DSL', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport'])]

# Build up a formula

formula1 = [' %s + ' % variable for variable in X_test2.columns]

formula1.insert(0, 'y ~ ')

formula1 = ' '.join(''.join(formula1).split(' ')[:-2])

formula1
with pm.Model() as logistic_model1:

    

    # Build the model using the formula and specify the data likelihood 

    priors=dict()

    for variable in X_test2.columns:

        priors[variable]=pm.Uniform.dist(0,1)

    priors['Intercept']=pm.Normal.dist(mu=0., sigma=100.)

    priors['MonthlyCharges']=pm.Normal.dist(mu=0., sigma=100.)

    priors['TotalCharges'] = pm.Normal.dist(mu=0., sigma=100.)

              

    pm.GLM.from_formula(formula1, data = X_with_labels, family = pm.glm.families.Binomial(),priors=priors)

    

    # Using the no-uturn sampler

    sampler = pm.NUTS()

    

    # Sample from the posterior using NUTS

    trace_log1 = pm.sample(draws=2000, step = sampler, chains=1, tune=1000, random_seed=100,init='adapt_diag')
pm.plot_posterior(trace_log);
pm.summary(trace_log1)
fileObject = open("sign_parameters.pickle",'wb')  

pickle.dump(trace_log1, fileObject)

fileObject.close()
trace_log1_frm_file= pickle.load(open("sign_parameters.pickle",'rb')  )

#trace_log1=trace_log1_frm_file #Uncomment this line if we want to load the model from static file
means_dict_sign, std_dict_sign = evaluate_trace(trace_log1, X_test2, print_model=True)
blr1_probs = find_probs(trace_log1, X_test2)



# Threshold the values at 0.5

predictions = (np.array(blr1_probs) > 0.5)

calc_metrics(predictions, y_test)
logistic_model.name='all_parm'

logistic_model1.name='sign_parm'

model_trace_dict = {'all_parm':trace_log,

                   'sign_parm':trace_log1}

dfwaic = pm.compare(model_trace_dict)

pm.compareplot(dfwaic);
dfwaicloo = pm.compare(model_trace_dict, ic='LOO')

pm.compareplot(dfwaicloo);
print(dfwaic)

print(dfwaicloo)