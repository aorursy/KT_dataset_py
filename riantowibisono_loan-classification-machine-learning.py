import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import time

import warnings



warnings.filterwarnings('ignore')



startall = time.time()



pd.options.display.max_columns = None

pd.options.display.max_rows = None

pd.options.display.float_format = '{:.2f}'.format # to set the displayed data as in the two decimal format
start = time.time()



df = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv', low_memory = False)



stop = time.time()

duration = stop-start

print('It took {:.2f} seconds to read the entire csv file.'.format(duration))



df.head()
df.describe()
df.info()
# Analyzing the missing value in each columns

df_null = pd.DataFrame({'Count': df.isnull().sum(), 'Percent': round(100*df.isnull().sum()/len(df),2)})

df_null[df_null['Count'] != 0] 
# Visualize the percentage of missing values in columns that have more than 70% missing values

df_null_70up = df_null[df_null['Percent'] >= 70]

df_null_70up = df_null_70up.sort_values(

    by=['Percent'], 

    ascending=False

)



plt.figure(figsize=(15,8))

barchart = sns.barplot(

    df_null_70up.index, 

    df_null_70up['Percent'],

    palette='Set2'

)



barchart.set_xticklabels(barchart.get_xticklabels(), rotation=45, horizontalalignment='right')
# Remove columns which missing values > 70%

df_1 = df.dropna(axis=1, thresh=int(0.70*len(df)))

df_1.head()
print(

    'The number of columns has reduced from {} to {} columns by removing columns with 70% missing values'.

    format(len(df.columns), len(df_1.columns))

)
plt.figure(figsize = (15,5))

plot1 = sns.barplot(df.loan_status.value_counts().index, df.loan_status.value_counts(), palette = 'Set1')

plt.xticks(rotation = 45, horizontalalignment='right')

plt.yticks(fontsize = 12)

plt.title("Loan Status Distribution", fontsize = 20, weight='bold')

plt.ylabel("Count", fontsize = 15)



total = len(df_1)

sizes = []

for p in plot1.patches:

    height = p.get_height()

    sizes.append(height)

    plot1.text(p.get_x() + p.get_width()/2.,

            height + 10000,

            '{:1.3f}%'.format(height/total*100),

            ha = "center", 

            fontsize = 10) 
selected_loan_status = ['Fully Paid', 'Charged Off', 'Default']

df_2 = df_1[df_1.loan_status.isin(selected_loan_status)]

df_2.loan_status = df_2.loan_status.replace({'Fully Paid' : 'Good Loan'})

df_2.loan_status = df_2.loan_status.replace({'Charged Off' : 'Bad Loan'})

df_2.loan_status = df_2.loan_status.replace({'Default' : 'Bad Loan'})
print(

    'The number of rows has been reduced from {:,.0f} to {:,.0f} by filtering the data with the correlated loan status'.

    format(len(df_1), len(df_2))     

)
plt.figure(figsize=(8, 5))

plot2 = sns.countplot(df_2.term, hue = df_2.loan_status)

plt.title("Loan's Term Distribution", fontsize = 20, weight='bold')

plt.ylabel("Count", fontsize = 15)

plt.xlabel("Term", fontsize = 15)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)



total = len(df_2)

sizes = []

for p in plot2.patches:

    height = p.get_height()

    sizes.append(height)

    plot2.text(p.get_x() + p.get_width()/2.,

            height + 10000,

            '{:1.0f}%'.format(height/total*100),

            ha = "center", 

            fontsize = 12) 
plt.figure(figsize = (10,7))

sns.distplot(df.loan_amnt, bins=20)

plt.title('Loan Amount Distribution', fontsize = 20, weight='bold')

plt.xlabel('Loan Amount', fontsize = 15)

plt.ylabel('Frequency', fontsize = 15)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)
plt.figure(figsize = (10,7))

sns.countplot(round(df.int_rate, 0).astype(int))

plt.title('Interest Rate Distribution', fontsize = 20, weight='bold')

plt.xlabel('Interest Rate (Rounded)', fontsize = 15)

plt.ylabel('Count', fontsize = 15)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)
plt.figure(figsize = (16,5))

plot3 = sns.countplot(df_2.sort_values(by='grade').grade, hue = df_2.loan_status)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title("Grade Distribution", fontsize = 20, weight='bold')

plt.xlabel("Grade", fontsize = 15)

plt.ylabel("Count", fontsize = 15)



total = len(df_2)

sizes = []

for p in plot3.patches:

    height = p.get_height()

    sizes.append(height)

    plot3.text(p.get_x() + p.get_width()/2.,

            height + 3000,

            '{:1.2f}%'.format(height/total*100),

            ha = "center", 

            fontsize = 10) 
plt.figure(figsize = (15,5))

plot4 = sns.barplot(df.purpose.value_counts().index, df.purpose.value_counts(), palette = 'Set1')

plt.xticks(rotation = 30, fontsize = 12, horizontalalignment='right')

plt.yticks(fontsize = 12)

plt.title("Loan Purpose Distribution", fontsize = 20, weight='bold')

plt.ylabel("Count", fontsize = 15)



total = len(df_1)

sizes = []

for p in plot4.patches:

    height = p.get_height()

    sizes.append(height)

    plot4.text(p.get_x() + p.get_width()/2.,

            height + 10000,

            '{:1.2f}%'.format(height/total*100),

            ha = "center", 

            fontsize = 10) 
plt.figure(figsize = (20,11))

sns.boxplot(df_2.loan_status, df_2.loan_amnt, hue = df_2.term, palette = 'Paired')

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.xlabel("Loan Status Categories", fontsize = 15)

plt.ylabel("Loan Amount Distribution", fontsize = 15)

plt.title("Loan Status by Loan Amount", fontsize = 20, weight='bold')
plt.figure(figsize = (20,11))

sns.boxplot(df_2.loan_status, round(df_2.int_rate, 0).astype(int), hue = df_2.term, palette = 'Paired')

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.xlabel("Loan Status Categories", fontsize = 15)

plt.ylabel("Interest Rate Distribution", fontsize = 15)

plt.title("Loan Status by Interest Rate", fontsize = 20, weight='bold')
plt.figure(figsize=(12, 7))

plot5 = sns.countplot(df_2.verification_status, hue = df_2.loan_status, palette = 'inferno')

plt.title("Verification Status Distribution", fontsize = 20)

plt.xlabel("Verification Status", fontsize = 15)

plt.ylabel("Count", fontsize = 15)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)



total = len(df_2)

sizes = []

for p in plot5.patches:

    height = p.get_height()

    sizes.append(height)

    plot5.text(p.get_x() + p.get_width()/2.,

            height + 5000,

            '{:1.0f}%'.format(height/total*100),

            ha = "center", 

            fontsize = 12)
most_emp_title = df_2.emp_title.value_counts()[:20].index.values  # get the top 20 most frequent employee job title

cm = sns.light_palette("orange", as_cmap=True)



round(pd.crosstab(df_2[df_2['emp_title'].isin(most_emp_title)]['emp_title'], 

                  df_2[df_2['emp_title'].isin(most_emp_title)]['grade'], 

                  normalize='index') * 100,2).style.background_gradient(cmap = cm)
df_3 = df_2[[

    'loan_status', 'term','int_rate',

    'installment','grade', 'annual_inc',

    'verification_status','dti'  # These features are just initial guess, you can try to choose any other combination

]]

df_3.head()
# Find missing values in the chosen columns

df_null = pd.DataFrame({'Count': df_3.isnull().sum(), 'Percent': round(100*df_3.isnull().sum()/len(df_3),2)})

df_null[df_null['Count'] != 0] 
# Dropping rows with null values

df_clean = df_3.dropna(axis = 0)
print('Number of dropped rows: {} rows'.format(len(df_3)-len(df_clean)))
# The next step is to transform categorical target variable into integer

df_clean.loan_status = df_clean.loan_status.replace({'Good Loan' : 1})

df_clean.loan_status = df_clean.loan_status.replace({'Bad Loan' : 0})

df_clean.loan_status.unique()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()



df_clean['term'] = label.fit_transform(df_clean['term'])

df_clean['grade'] = label.fit_transform(df_clean['grade'])

df_clean['verification_status'] = label.fit_transform(df_clean['verification_status'])
x = df_clean.drop(['loan_status'], axis=1)

y = df_clean['loan_status']
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

import numpy as np 



coltrans = ColumnTransformer(

    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0,3,5])],      # 0,3,5 refers to the column indexes that need to be transformed      

    remainder = 'passthrough'                               

)                                                         



x = np.array(coltrans.fit_transform(x))
from sklearn.model_selection import train_test_split

xtr, xts, ytr, yts = train_test_split(

    x,

    y,

    test_size = .2

)
print(ytr.value_counts())

print(yts.value_counts())
from imblearn.over_sampling import SMOTE



smt = SMOTE()

xtr_2, ytr_2 = smt.fit_sample(xtr, ytr)
np.bincount(ytr_2)
from sklearn.ensemble import RandomForestClassifier



start = time.time()



model = RandomForestClassifier()

model.fit(xtr_2, ytr_2)



stop = time.time()

duration = stop-start

print('The training took {:.2f} seconds.'.format(duration))
print(round(model.score(xts, yts) * 100, 2), '%')
y_pred = model.predict(xts)
from sklearn.metrics import confusion_matrix



confusion_matrix(yts, y_pred)
pd.crosstab(yts, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
from sklearn.metrics import classification_report



target_names = ['Bad Loan', 'Good Loan']

print(classification_report(yts, model.predict(xts), target_names=target_names))
from sklearn.ensemble import RandomForestClassifier



start = time.time()



model2 = RandomForestClassifier()

model2.fit(xtr, ytr)



stop = time.time()

duration = stop-start

print('The training took {:.2f} seconds.'.format(duration))
print(round(model2.score(xts, yts) * 100, 2), '%')
y_pred2 = model2.predict(xts)
from sklearn.metrics import confusion_matrix



confusion_matrix(yts, y_pred2)
pd.crosstab(yts, y_pred2, rownames=['Actual'], colnames=['Predicted'], margins=True)
from sklearn.metrics import classification_report



target_names = ['Bad Loan', 'Good Loan']

print(classification_report(yts, y_pred2, target_names=target_names))
from imblearn.under_sampling import NearMiss



nr = NearMiss()

xtr_3, ytr_3 = nr.fit_sample(xtr, ytr)
np.bincount(ytr_3)
from sklearn.ensemble import RandomForestClassifier



start = time.time()



model3 = RandomForestClassifier()

model3.fit(xtr_3, ytr_3)



stop = time.time()

duration = stop-start

print('The training took {:.2f} seconds.'.format(duration))
print(round(model3.score(xts, yts) * 100, 2), '%')
y_pred3 = model3.predict(xts)
from sklearn.metrics import confusion_matrix



confusion_matrix(yts, y_pred3)
pd.crosstab(yts, y_pred3, rownames=['Actual'], colnames=['Predicted'], margins=True)
from sklearn.metrics import classification_report



target_names = ['Bad Loan', 'Good Loan']

print(classification_report(yts, y_pred3, target_names=target_names))
# First, by knowing what are the features available in the dataframe

df_4 = df_2
# The next step is to transform categorical target variable into integer

df_4.loan_status = df_4.loan_status.replace({'Good Loan' : 1})

df_4.loan_status = df_4.loan_status.replace({'Bad Loan' : 0})
df_4.columns.to_series().groupby(df_clean.dtypes).groups
# First, dropping categorical features (object type) which have too many options available

df_4 = df_4.drop(['emp_title', 'sub_grade', 'issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'hardship_flag', 'debt_settlement_flag'], axis=1)
# Second, to filter numerical features, we can use .corr() function to select only features with high correlation to the target variable

df_4.corr()['loan_status']
df_clean = df_4[[

    'loan_status', # target variable

    # features (object):

    'term', 'grade','home_ownership', 'verification_status', 'pymnt_plan', 'purpose', 

    'initial_list_status', 'application_type', 'disbursement_method',

    # features (int/float):

    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'recoveries',                   

    'collection_recovery_fee', 'last_pymnt_amnt', 'int_rate'

]]
df_null = pd.DataFrame({'Count': df_clean.isnull().sum(), 'Percent': round(100*df_clean.isnull().sum()/len(df_clean),2)})

df_null[df_null['Count'] != 0] 
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()



df_clean['term'] = label.fit_transform(df_clean['term'])

df_clean['grade'] = label.fit_transform(df_clean['grade'])

# df_clean['emp_length'] = label.fit_transform(df_clean['emp_length'])

df_clean['home_ownership'] = label.fit_transform(df_clean['home_ownership'])

df_clean['verification_status'] = label.fit_transform(df_clean['verification_status'])

df_clean['pymnt_plan'] = label.fit_transform(df_clean['pymnt_plan'])

df_clean['purpose'] = label.fit_transform(df_clean['purpose'])

df_clean['initial_list_status'] = label.fit_transform(df_clean['initial_list_status'])

df_clean['application_type'] = label.fit_transform(df_clean['application_type'])

df_clean['disbursement_method'] = label.fit_transform(df_clean['disbursement_method'])
df_clean.head()
x = df_clean.drop(['loan_status'], axis=1)

y = df_clean['loan_status']
x.head()
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

import numpy as np 



coltrans = ColumnTransformer(

    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0,1,2,3,4,5,6,7,8])],        

    remainder = 'passthrough'                               

)                                                         



x = np.array(coltrans.fit_transform(x))
from sklearn.model_selection import train_test_split

xtr, xts, ytr, yts = train_test_split(

    x,

    y,

    test_size = .2

)
from sklearn.ensemble import RandomForestClassifier

import time



start = time.time()



model = RandomForestClassifier()

model.fit(xtr, ytr)



stop = time.time()

duration = stop-start

print('The training took {:.2f} seconds.'.format(duration))
print(round(model.score(xts, yts) * 100, 2), '%')
y_pred = model.predict(xts)
from sklearn.metrics import confusion_matrix



confusion_matrix(yts, y_pred)
pd.crosstab(yts, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
from sklearn.metrics import classification_report



target_names = ['Bad Loan', 'Good Loan']

print(classification_report(yts, model.predict(xts), target_names=target_names))
import sklearn.metrics as metrics



# calculate the fpr and tpr for all thresholds of the classification

probs = model.predict_proba(xts)

preds = probs[:,1]



fpr, tpr, threshold = metrics.roc_curve(yts, y_pred)

roc_auc = metrics.auc(fpr, tpr)



# Plotting the ROC curve

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
import math



stopall = time.time()

durationall = stopall-startall

duration_mins = math.floor(durationall/60)

duration_secs = durationall - (duration_mins*60)



print('The whole notebook runs for {} minutes {:.2f} seconds.'.format(duration_mins, duration_secs))