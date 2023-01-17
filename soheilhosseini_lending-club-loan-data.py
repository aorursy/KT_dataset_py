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
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



from scipy import stats

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score





import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import Ridge



import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier



from sklearn.impute import SimpleImputer



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit





from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import cross_validate

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC



from imblearn.over_sampling import SMOTE



import plotly.express as px

dictionary = pd.read_excel('../input/lending-club-loan-data/LCDataDictionary.xlsx').dropna()

dictionary.style.set_properties(subset=['Description'], **{'width': '1000px'})
#import datatable as dt



#loan = dt.fread("../input/lending-club-loan-data/loan.csv")

loan = pd.read_csv('../input/lending-club-loan-data/loan.csv')
LCDataDictionary =pd.read_excel('../input/lending-club-loan-data/LCDataDictionary.xlsx')

loan.shape
loan.head()
#pd.set_option("max_rows", None)

loan.info(verbose=True)
#Show all columns

pd.set_option('display.max_columns', 500)

loan.describe()
plt.title("Loan applied by the borrower")

sns.distplot(loan['loan_amnt'], color='red')

plt.show()
plt.title("Amount Funded Invested")

sns.distplot(loan['funded_amnt_inv'], color='blue')

plt.show()

plt.title("Amount Funded by the Lender")

sns.distplot(loan['funded_amnt'], color='green')

plt.show()
loan['issue_date'] = pd.to_datetime(loan['issue_d'])

loan['year'] = loan['issue_date'].dt.year
plt.figure(figsize = (6,6))

plt.axis(option='normal')

plt.title("how the loan book was growing")

sns.lineplot(x="issue_date", y="loan_amnt", data=loan, estimator='sum' , ci=None, color = 'green', lw=3)

plt.ticklabel_format(style='plain', axis='y')

plt.show()

#Loan funded in each year

loan[['funded_amnt', 'year']].groupby(['year'], as_index=False).mean().sort_values(by='year', ascending=False)
#Loan funded vs year



sns.barplot(x="year", y="loan_amnt", ci =None, data=loan,palette='plasma')

plt.show()
#Loan amount by status

plt.figure(figsize=(7,7))

plt.title("Loan amount by status")

plt.xticks(rotation=90)

sns.boxplot(x="loan_status", y="loan_amnt", data=loan)

plt.show()
#Loan purpose



plt.title("Loan titles")

plt.xticks(rotation=90)

sns.countplot(y="purpose", data=loan,  palette = 'plasma')

plt.show()
#Loan grade

#Here is the overview of the occurrence of loans of different grades:



plt.title("Loan grades")

plt.xticks(rotation=90)

sns.countplot(y="grade", data=loan,  palette = 'plasma')

plt.show()
#Loan amount by status

plt.figure(figsize=(7,7))

plt.title("Interest rate by grade")

plt.xticks(rotation=90)

sns.boxplot(x="grade", y="int_rate", data=loan)

plt.show()
#Loan amount by status

plt.figure(figsize=(14,7))

plt.title("Interest rate by sub grade")

plt.xticks(rotation=90)

sns.boxplot(x="sub_grade", y="int_rate", data=loan)

plt.show()
loan.loan_status.value_counts(ascending=False)
loan.loan_status.sample(10)
# Determining the loans that are bad from loan_status column



bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period", 

            "Late (16-30 days)", "Late (31-120 days)"]



loan['loan_condition'] = np.nan



loan['loan_condition'] = np.where(((loan['loan_status']=='Charged Off') 

                                  | (loan['loan_status']=='Default') 

                                  

                                  |(loan['loan_status']=='Does not meet the credit policy. Status:Charged Off')  

                                  |   (loan['loan_status']=='In Grace Period')

                                  | (loan['loan_status']=='Late (16-30 days)')

                                   |  (loan['loan_status']=='Late (31-120 days)'))

                                   , 'bad_loan', 'good_loan')
sns.countplot(x="year", hue="loan_condition", data=loan, palette='plasma')

plt.show()
plt.title("Loans issued by credit score")

sns.lineplot(x="year", y="loan_amnt", hue="grade", data=loan,palette ='plasma', ci=None)

plt.show()
plt.title("Interest rates by credit score")

sns.lineplot(x="year", y="int_rate", hue="grade", data=loan,palette ='plasma', ci=None)

plt.show()
plt.title("Distribution of loan amount based on home ownerership (bad loan cases)")

sns.boxplot(x='home_ownership' , y= 'loan_amnt',hue='loan_condition', hue_order =['bad_loan'] ,data= loan, palette ='plasma')

plt.show()

plt.figure(figsize= (18,7))

sns.boxplot(y='loan_amnt' ,x='year', hue= 'home_ownership',palette = 'plasma' ,data= loan)

plt.show()
loan_plot = loan.groupby(['grade', 'loan_condition']).size().reset_index().pivot(columns='loan_condition', index='grade', values=0)

loan_plot.plot(kind='bar', stacked=True, color='rb')

plt.title("Loan conditions based on grade")

plt.show()
#Correlation Matrix

plt.figure(figsize=(15, 8))

plt.title('Pearson Correlation of Features')

sns.heatmap(loan.corr(),cmap='coolwarm' )

plt.show()
'''Remove redundant features, and keep important ones



loan_condition : good loan or bad loan (TARGET)



loan_amnt : The listed amount of the loan applied for by the borrower. 

int_rate : Interest Rate on the loan

grade : Grade of employment

emp_length : Employment length in years. 

home_ownership : Type of ownership of house

annual_inc : Total annual income

term : 36-month or 60-month period'''
loan_features = loan[['loan_amnt','int_rate','grade', 'emp_length','home_ownership','annual_inc','term','loan_condition']]
loan_features['term'] = loan_features['term'].map(lambda x: x.rstrip(' months')).astype(int)
loan_features['emp_length'] = loan_features['emp_length'].str.lstrip('<').str.rstrip('+ years')
#Missing Values:



loan_features.isnull().sum().sort_values(ascending = False)
# change outcomes to Integer values

# Good bad to 0

# Bad loan to 1

loan_features['loan_condition'] = loan_features['loan_condition'].map({'good_loan':0, 'bad_loan':1})
# Remove every row with missing values.

# There are mostly rows where emp_length is NaN.



loan_features.dropna(inplace = True)

#Change employee length from string to int type

loan_features['emp_length'] = loan_features['emp_length'].astype(int)

y = loan_features['loan_condition']
#Convert categorical variable into dummy/indicator variables.

X_dummy = pd.get_dummies(loan_features.drop(['loan_condition'], axis=1), drop_first=True)

#Standardize features by removing the mean and scaling to unit variance

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_dummy)
model ={'Decision Tree':DecisionTreeClassifier( max_depth=2,min_samples_leaf =10, random_state=0, class_weight='balanced'),

       'Random Forest Classifier': RandomForestClassifier(n_jobs=-1, max_depth=2, random_state=0, class_weight='balanced'),

       'Logistic Regression': LogisticRegression(n_jobs=-1, random_state=0, class_weight='balanced')}



for keys, items in model.items():

    cv_results = cross_validate(items, X_scaled, y, cv=5, scoring='roc_auc')

    print(model.keys())

    print("AUC:  ", cv_results['test_score'])

    print("max AUC:  ", max(cv_results['test_score']))

    print("average AUC:  ", np.mean(cv_results['test_score']),"\n")

import keras

keras.__version__
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))
import tensorflow as tf

from sklearn.metrics import roc_auc_score



def auroc(y_true, y_pred):

    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)



model.compile(loss='binary_crossentropy', optimizer='RMSprop',metrics=[ auroc])

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)





history = model.fit(X_scaled,y, epochs=10, batch_size=512, validation_split=0.2, class_weight = class_weights)