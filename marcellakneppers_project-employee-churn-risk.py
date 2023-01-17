#Boilerplate

import pandas as pd

import numpy as np

import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt

#import plotly.graph_objects as go



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



from sklearn.ensemble import IsolationForest

from sklearn.utils.validation import check_is_fitted

from sklearn.utils import check_array
train_data_orig = pd.read_csv(r"../input/traindata/VeryBestCompany_Employees_train.csv")

test_data_orig = pd.read_csv(r"../input/testdata/VeryBestCompany_Employees_test.csv")

train_data_orig.head()
#Checking the info of the df

train_data_orig.info()
#missing data

total = train_data_orig.isnull().sum().sort_values(ascending=False)

percent = (train_data_orig.isnull().sum()/train_data_orig.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#checking for duplicated rows. 

train_data_orig[train_data_orig.duplicated(keep=False)]
train_data_orig.describe()
#Check of the distribution of then numerical variables-

train_data_orig.hist(figsize=(20,20))

plt.show()
 #start by just expoloring the key target first

train_data_orig['Churn_risk'].value_counts()   
#Creating an order for the target categorical values and visualizing distribution in boxplot

category_order = ["low","medium","high",]



sns.countplot(x='Churn_risk',data=train_data_orig, order=category_order)

plt.show()
train_data_orig['Gender'].value_counts()
#Grouping Gender and Churn_Risk to see the distribution per sex

grouped_gender = train_data_orig.groupby(['Gender','Churn_risk'])

grouped_gender.size()
#Checking the precentage per sex in each of the churn_risk classes

pd.crosstab(train_data_orig['Churn_risk'],train_data_orig['Gender']).apply(lambda r: r/r.sum()*100, axis=1)
plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'low', 'Age'], label = 'Low-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'medium', 'Age'], label = 'Medium-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'high', 'Age'], label = 'High-risk churn')

plt.xlim(left=18, right=60)

plt.xlabel('Age (years)')

plt.ylabel('Density')

plt.title('Age Distribution in Churn_Risk');
train_data_orig['Age'].describe()
sns.catplot(x="Churn_risk", y="Days_off", hue="Gender",

            kind="violin", bw=.15, cut=0,

            data=train_data_orig,order=category_order)
sns.catplot(x="Churn_risk", y="Days_off",

            kind="violin", bw=.15, cut=0,

            data=train_data_orig,order=category_order)
train_data_orig['Rotations'].value_counts()
grouped_rotation = train_data_orig.groupby(['Rotations','Churn_risk'])

grouped_rotation.size()
pd.crosstab(train_data_orig['Churn_risk'],train_data_orig['Rotations']).apply(lambda r: r/r.sum()*100, axis=1)
pd.crosstab(train_data_orig['Rotations'],train_data_orig['Churn_risk']).apply(lambda r: r/r.sum()*100, axis=1)
sns.catplot(x="Churn_risk", y="Satis_leader",

            kind="violin", bw=.15, cut=0,

            data=train_data_orig,order=category_order)
train_data_orig['Satis_leader'].describe()
plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'low', 'Satis_leader'], label = 'Low-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'medium', 'Satis_leader'], label = 'Medium-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'high', 'Satis_leader'], label = 'High-risk churn')

plt.xlim(left=0, right=10)

plt.xlabel('Satis_leader')

plt.ylabel('Density')

plt.title('Satis_leader Distribution in Churn_Risk')
train_data_orig['Satis_team'].describe()
plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'low', 'Satis_team'], label = 'Low-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'medium', 'Satis_team'], label = 'Medium-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'high', 'Satis_team'], label = 'High-risk churn')

plt.xlim(left=0, right=10)

plt.xlabel('Satis_team')

plt.ylabel('Density')

plt.title('Satis_team Distribution in Churn_Risk')
sns.distplot(train_data_orig['Emails'])
train_data_orig['Emails'].describe()
#Comparing the amount of E-mails with Churn risk

plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'low', 'Emails'], label = 'Low-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'medium', 'Emails'], label = 'Medium-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'high', 'Emails'], label = 'High-risk churn')

plt.xlim(left=0, right=170)

plt.xlabel('Emails')

plt.ylabel('Density')

plt.title('Emails Distribution in Churn_Risk')
train_data_orig['Tenure'].describe()
sns.distplot(train_data_orig['Tenure'])
#Comparing Tenure with Churn risk

sns.catplot(x="Churn_risk", y="Tenure", kind="boxen",

            data=train_data_orig,order=category_order);
#Comparing Tenure to Gender

sns.catplot(x="Gender", y="Tenure", kind="boxen",

            data=train_data_orig);
train_data_orig['Bonus'].describe()
#Comparing Bonus to Churn risk

plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'low', 'Bonus'], label = 'Low-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'medium', 'Bonus'], label = 'Medium-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'high', 'Bonus'], label = 'High-risk churn')

plt.xlim(left=-1, right=2)

plt.xlabel('Bonus')

plt.ylabel('Density')

plt.title('Bonus Distribution in Churn_Risk')
#Comparing Bonus to Gender

sns.catplot(x="Gender", y="Bonus", kind="boxen",

            data=train_data_orig);
train_data_orig['Distance'].describe()
#Comparing Distance to Churn risk

sns.catplot(x="Churn_risk", y="Distance", kind="violin",

            data=train_data_orig,order=category_order)
train_data_orig['Kids'].describe()
#Comparing having Kids to Churn risk 

sns.catplot(x="Churn_risk", y="Kids", kind="box",

            data=train_data_orig,order=category_order)
train_data_orig['Overtime'].describe()
#Comparing Overtime to Churn risk

plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'low', 'Overtime'], label = 'Low-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'medium', 'Overtime'], label = 'Medium-risk churn')

sns.kdeplot(train_data_orig.loc[train_data_orig['Churn_risk'] == 'high', 'Overtime'], label = 'High-risk churn')

plt.xlim(left=0, right=16)

plt.xlabel('Overtime')

plt.ylabel('Density')

plt.title('Overtime Distribution in Churn_Risk')
#Comparing Overtime to Gender

sns.catplot(x="Gender", y="Overtime", kind="boxen",

            data=train_data_orig);
train_data_orig['Marital_status'].value_counts() 
pd.crosstab(train_data_orig['Churn_risk'],train_data_orig['Marital_status']).apply(lambda r: r/r.sum()*100, axis=1)
#Comparing Churn risk to Marital status

from statsmodels.graphics.mosaicplot import mosaic



plt.rcParams['font.size'] = 11.0

mosaic(train_data_orig, ['Churn_risk', 'Marital_status']);
train_data_orig['Department'].describe()
train_data_orig['Department'].value_counts() 
#Comparing Department to Churn risk

pd.crosstab(train_data_orig['Department'],train_data_orig['Churn_risk']).apply(lambda r: r/r.sum(), axis=1)
#Comparing Department to Churn risk

plt.rcParams['font.size'] = 11.0

mosaic(train_data_orig, ['Churn_risk', 'Department']);
#Comparing Department to Emails

plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(train_data_orig.loc[train_data_orig['Department'] == 'accounting', 'Emails'], label = 'accounting')

sns.kdeplot(train_data_orig.loc[train_data_orig['Department'] == 'computer services', 'Emails'], label = 'computer services')

sns.kdeplot(train_data_orig.loc[train_data_orig['Department'] == 'finances', 'Emails'], label = 'finances')

sns.kdeplot(train_data_orig.loc[train_data_orig['Department'] == 'human resources', 'Emails'], label = 'human resources')

sns.kdeplot(train_data_orig.loc[train_data_orig['Department'] == 'marketing', 'Emails'], label = 'marketing')

sns.kdeplot(train_data_orig.loc[train_data_orig['Department'] == 'sales', 'Emails'], label = 'sales')

plt.xlim(left=0, right=150)

plt.xlabel('Emails')

plt.ylabel('Density')

plt.title('Department to Emails')
# Calculate correlations

corr = train_data_orig.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

# Heatmap

plt.figure(figsize=(15, 10))

sns.heatmap(corr,

            vmax=.5,

            mask=mask,

            # annot=True, fmt='.2f',

            linewidths=.2, cmap="YlGnBu")
#making a copy of the original data frames

Train_DF = train_data_orig.copy()

Test_DF = test_data_orig.copy()
#employee ID as index 

Train_DF.set_index('Employee_ID', inplace=True)

Test_DF.set_index('Employee_ID', inplace=True)
#Missing values was checked in earlier section and defined with below:

missing_data.head(10)
# Create a copy of the current DF and create a new DF called Train_DF2 to remove the missing data values in. 

Train_DF2 = Train_DF.copy()

#Treatment of the missing values with mean

Train_DF2.fillna(Train_DF2.mean(), inplace=True)

#checking changes if there is all treated

Train_DF2.info()

# The initial data frame has 5200 rows, which means that the new DF lost 29 rows (less than 1%)

print(Train_DF2.shape)
print(Train_DF2.describe())
sns.boxplot(x=Train_DF2['Age'])
print(Train_DF2['Age'].quantile(0.95))
#creating a copy of the df 

Train__DF_outlier=Train_DF2.copy()
#Outlier treatment for over age 55 and replace by 3rd quartile = 46 (see code above)

Train__DF_outlier['Age'] =np.where(Train_DF2['Age']>55, 46,Train_DF2['Age'])
sns.boxplot(x=Train__DF_outlier['Age'])
sns.boxplot(x=Train_DF2['Days_off'])
print(Train_DF2['Days_off'].quantile(0.95))
print(Train_DF2[Train_DF2.Days_off>15].Bonus.agg(['count']))
#Outlier treatment for Days_off more than 15 and replace by 95 percentage quartile = 13

#Treating 121 outliers with the 95% quartile 

Train__DF_outlier['Days_off'] =np.where(Train_DF2['Days_off']>15, 13,Train_DF2['Days_off'])
sns.boxplot(x=Train__DF_outlier['Days_off'])
sns.boxplot(x=Train_DF2['Satis_leader'])
data = Train_DF2 

Q1 = data['Satis_leader'].quantile(0.25)

Q3 = data['Satis_leader'].quantile(0.75)

IQR = Q3 - Q1



#calculating the maximum (where the black line shows a horizontal T on the right side of the boxplot)

Q3+1.5*IQR
#How many outliers are we treating from data point 3.1 onwards?

print(Train_DF2[Train_DF2.Satis_leader>3.1].Bonus.agg(['count']))
#Outlier treatment for Satis_leader more than 3.1 are replace by the maximum = 2.705

Train__DF_outlier['Satis_leader'] =np.where(Train_DF2['Satis_leader']>3.1, 2.705,Train_DF2['Satis_leader'])
sns.boxplot(x=Train__DF_outlier['Satis_leader'])
sns.boxplot(x=Train_DF2['Satis_team'])
print(Train_DF2['Satis_team'].quantile(0.95))
#How many outliers are we treating from data point 2 onwards?

print(data[data.Satis_team>2].Satis_leader.agg(['count']))
#Outlier treatment for Satis_team more than 2 and replace by 95% quartile = 1.6

Train__DF_outlier['Satis_team'] =np.where(Train_DF2['Satis_team']>2, 1.616,Train_DF2['Satis_team'])
sns.boxplot(x=Train__DF_outlier['Satis_team'])
sns.boxplot(x=Train_DF2['Emails'])
#how many data points are above 80?

print(data[data.Emails>80].Emails.agg(['count']))
print(Train_DF2['Emails'].quantile(0.95))
data = Train_DF2 

Q1 = data['Emails'].quantile(0.25)

Q3 = data['Emails'].quantile(0.75)

IQR = Q3 - Q1



#calculating the maximum (where the black line shows a horizontal T on the right side of the boxplot)

Q3+1.5*IQR
#Outlier treatment for Emails more than 80 and replace by the maximum = 79.5

Train__DF_outlier['Emails'] =np.where(Train_DF2['Emails']>80, 79.5,Train_DF2['Emails'])
sns.boxplot(x=Train__DF_outlier['Emails'])
sns.boxplot(x=Train_DF2['Bonus'])
fig, ax = plt.subplots(figsize=(10,7))

ax.scatter(Train_DF2['Bonus'], Train_DF2['Churn_risk'])

ax.set_xlabel('Proportion of bonus within the churn-risk')

ax.set_ylabel('Full-proportion churn_risk per bonus')

plt.show()
data = Train_DF2 

Q1 = data['Bonus'].quantile(0.25)

Q3 = data['Bonus'].quantile(0.75)

IQR = Q3 - Q1



#calculating the maximum (where the black line shows a horizontal T on the right side of the boxplot)

Q3+1.5*IQR
#How many values of the outliers are below the 0? 

print(data[data.Bonus<0].Bonus.agg(['count']))
#Outlier treatment for Bonus more than 1.4 and replace by the maximum value = 1.37

Train__DF_outlier['Bonus'] =np.where(Train_DF2['Bonus']>1.4, 1.373,Train_DF2['Bonus'])
sns.boxplot(x=Train__DF_outlier['Bonus'])
sns.boxplot(x=Train_DF2['Overtime'])
#how many data points are above 80?

print(data[data.Overtime>13.9].Overtime.agg(['count']))
Train__DF_outlier['Overtime_year']=(Train__DF_outlier['Overtime']*12)/8
Train__DF_outlier.head()
#creating the new variable in the DF 

Train__DF_outlier['WorkLifeBalance']= Train__DF_outlier['Days_off'] /Train__DF_outlier['Overtime_year']

Train__DF_outlier.head()
Train__DF_outlier['WorkLifeBalance'].describe()
sns.boxplot(x=Train__DF_outlier['Overtime_year'])
sns.boxplot(x=Train__DF_outlier['WorkLifeBalance'])
#Comparing Worklifebalance to Churn risk

plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(Train__DF_outlier.loc[Train__DF_outlier['Churn_risk'] == 'low', 'WorkLifeBalance'], label = 'Low-risk churn')

sns.kdeplot(Train__DF_outlier.loc[Train__DF_outlier['Churn_risk'] == 'medium', 'WorkLifeBalance'], label = 'Medium-risk churn')

sns.kdeplot(Train__DF_outlier.loc[Train__DF_outlier['Churn_risk'] == 'high', 'WorkLifeBalance'], label = 'High-risk churn')

plt.xlim(left=0, right=2)

plt.xlabel('WorkLifeBalance')

plt.ylabel('Density')

plt.title('Worklifebalance Distribution in Churn_Risk')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
Train__DF_outlier.head()
#creating bins for age groups [10,34] = Young, [34, 50]= Adult, [50,79]= Old 

Train__DF_outlier['age_by_group'] = pd.cut(x=Train__DF_outlier['Age'], bins=[10, 34, 50, 79], 

                                            labels=['Young', 'Adult', 'Old'])
#Comparing age_by_decade to Churn risk

plt.rcParams['font.size'] = 11.0

mosaic(Train__DF_outlier, ['Churn_risk', 'age_by_group']);
Train__DF_outlier.groupby(['age_by_group','Churn_risk']).agg('count')
Train__DF_outlier['age_by_group'].value_counts()
#Comparing age_by_group to Churn risk

pd.crosstab(Train__DF_outlier['age_by_group'],Train__DF_outlier['Churn_risk']).apply(lambda r: r/r.sum()*100, axis=1)
#creating an integer of the categorical values 

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Train__DF_outlier['age_by_group'] = LE.fit_transform(Train__DF_outlier['age_by_group'])
#creating bins 

#employees bonus [-0.3, 0.2] = negatives, [0.2, 0.6] = neutral, [0.6, 1]= positives, and [1, 5]= extreme positives 

Train__DF_outlier['bonus_split'] = pd.cut(x=Train__DF_outlier['Bonus'], bins=[-0.3, 0.2, 0.6, 1, 5], 

                                          labels=['Negatives', 'Neutral', 'Positives', 'Extra positives'])
#Comparing age_by_decade to Churn risk

plt.rcParams['font.size'] = 11.0

mosaic(Train__DF_outlier, ['Churn_risk', 'bonus_split']);
Train__DF_outlier['bonus_split'].value_counts()
Train__DF_outlier.groupby(['bonus_split','Churn_risk']).agg('count')
#Comparing Bonus Split to Churn risk

pd.crosstab(Train__DF_outlier['bonus_split'],Train__DF_outlier['Churn_risk']).apply(lambda r: r/r.sum()*100, axis=1)
#creating an integer of the categorical values 

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Train__DF_outlier['bonus_split'] = LE.fit_transform(Train__DF_outlier['bonus_split'])
Train__DF_outlier.info()
Train_DF_Dummy = Train__DF_outlier.copy()
#Import of lable encoder/to make categorical values in to numerical

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Create a label encoder object

le = LabelEncoder()
le_count = 0

for col in Train_DF_Dummy.columns[0:14]:

    if Train_DF_Dummy[col].dtype == 'object':

        if len(list(Train_DF_Dummy[col].unique())) <= 7:

            le.fit(Train_DF_Dummy[col])

            Train_DF_Dummy[col] = le.transform(Train_DF_Dummy[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))
#checking the changed variables though head of the df

Train_DF_Dummy.head()
# Calculate correlations

corr = Train_DF_Dummy.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

# Heatmap

plt.figure(figsize=(15, 10))

sns.heatmap(corr,

            vmax=.5,

            mask=mask,

            annot=True, fmt='.2f',

            linewidths=.5, cmap="YlGnBu")
#Notes of Process: Removing from the train set 

#Train_DF_rem=Train_DF_Dummy.drop(['Tenure', 'Gender', 'Satis_leader','WorkLifeBalance','Overtime_year'], axis = 1)

#Train_DF_rem.info()
#making all features a float with exception for chrun_risk

HR_col = list(Train_DF_Dummy.columns)

HR_col.remove('Churn_risk')

for col in HR_col:

    Train_DF_Dummy[col] = Train_DF_Dummy[col].astype(float)

Train_DF_Dummy.info()
#defining data and target 

data_X = Train_DF_Dummy.drop(['Churn_risk'], axis=1)

target_y = Train_DF_Dummy['Churn_risk']

data_X.info()
#splitting the train DF into a Test and Train set with 20% test size and random state at 25

#By doing this, we create the test.csv file as our validation set 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_X, target_y, test_size = 0.2, random_state=25,

                                                   shuffle = True, stratify=target_y)





print("Number transactions X_train dataset: ", X_train.shape) 

print("Number transactions y_train dataset: ", y_train.shape) 

print("Number transactions X_test dataset: ", X_test.shape) 

print("Number transactions y_test dataset: ", y_test.shape) 
from imblearn.over_sampling import SMOTE 

from imblearn.over_sampling import ADASYN

from imblearn.over_sampling import BorderlineSMOTE

from imblearn.over_sampling import SMOTENC

from imblearn.combine import SMOTEENN



sm = SMOTE(sampling_strategy='not minority', random_state = 12) #resample all classes but the majority class

bsm=BorderlineSMOTE()

adsy=ADASYN(sampling_strategy='all',random_state=1)

smeenn = SMOTEENN()
#Here we will use the SMOTE and ADASYN for balancing the Data(X) and Target(Y)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

X_train_adasyn, y_train_adasyn = adsy.fit_sample(X_train, y_train)
#checking the balance of targets:

print('___________________________________________________________________________________________________________')

print('                                       Before oversampling                                                 ')

print('-----------------------------------------------------------------------------------------------------------')



print('Before OverSampling, the shape of Data: {}'.format(X_train.shape)) 

print('Before OverSampling, the shape of Target: {} \n'.format(y_train.shape))



print("Counts of label on Target 'high': {}".format(sum(y_train== 'high'))) 

print("Counts of label on Target 'medium': {} \n".format(sum(y_train== 'medium'))) 

print("Counts of label on Target 'low': {} \n".format(sum(y_train == 'low'))) 



print('___________________________________________________________________________________________________________')

print('                                        After oversampling Smote                                                 ')

print('-----------------------------------------------------------------------------------------------------------')



print('After OverSampling, the shape of Data: {}'.format(X_train_res.shape)) 

print('After OverSampling, the shape of Target: {} \n'.format(y_train_res.shape))



print("Counts of label on Target 'high': {}".format(sum(y_train_res== 'high'))) 

print("Counts of label on Target 'medium': {} \n".format(sum(y_train_res== 'medium')))

print("Counts of label on Target 'low': {} \n".format(sum(y_train_res == 'low'))) 





print('___________________________________________________________________________________________________________')

print('                                        After oversampling ADASYN                                                 ')

print('-----------------------------------------------------------------------------------------------------------')



print('After OverSampling, the shape of Data: {}'.format(X_train_adasyn.shape)) 

print('After OverSampling, the shape of Target: {} \n'.format(y_train_adasyn.shape))



print("Counts of label on Target 'high': {}".format(sum(y_train_adasyn== 'high'))) 

print("Counts of label on Target 'medium': {} \n".format(sum(y_train_adasyn== 'medium')))

print("Counts of label on Target 'low': {} \n".format(sum(y_train_adasyn == 'low')))
#Before oversampling 

category_order = ["low","medium","high",]



sns.countplot(y_train, order=category_order)

plt.show()
#After oversampling the data set of SMOTE not minority class

category_order = ["low","medium","high",]



sns.countplot(y_train_res, order=category_order)

plt.show()
#After oversampling the data set of ADASYN of resampling all minority classes

category_order = ["low","medium","high",]



sns.countplot(y_train_adasyn, order=category_order)

plt.show()
#creating two types of minmax scaler, one default and on with range of -1 to 1.

from sklearn.preprocessing import MinMaxScaler

MinMax_def = MinMaxScaler()

MinMax= MinMaxScaler(feature_range=(-1,1))
#Creating the Robust 

from sklearn.preprocessing import RobustScaler

robust= RobustScaler()
#Boilerplate for the different libraries/methods for the modelling 

from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import NuSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier



from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.metrics import recall_score

import numpy as np

import matplotlib.pyplot as plt

from sklearn import tree



from sklearn.tree import export_graphviz

import graphviz

import pydotplus





import pandas as pd

import seaborn as sn



import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
#Normalized with MinMax(-1.1) un-balanced train test split: 

MinMax_X_train_org=MinMax.fit_transform(X_train)

MinMax_X_test_org=MinMax.fit_transform(X_test)
#Normalized with Robust un-balanced train test split: 

Robust_X_train_org=robust.fit_transform(X_train)

Robust_X_test_org=robust.fit_transform(X_test)
#Normalized with MinMax(-1.1) SMOTE-balanced train: 

MinMax_X_train_res=MinMax.fit_transform(X_train_res)
#Normalized with Robust SMOTE-balanced train: 

Robust_X_train_res=robust.fit_transform(X_train_res)
#Normalized with MinMax(-1.1) ADASYN-balanced train: 

MinMax_X_train_adasyn=MinMax.fit_transform(X_train_adasyn)
#Normalized with Robust ADASYN-balanced train: 

Robust_X_train_adasyn=robust.fit_transform(X_train_adasyn)
#defining the model

model_DT = DecisionTreeClassifier()
#fittig the model to the balanced x and y train sets (SMOTE)

model_DT.fit(X_train_res, y_train_res)
#creating the Decision Tree graphically

dot_data = tree.export_graphviz(model_DT, out_file=None, 

                     feature_names=X_train_res.columns,  

                     class_names=["Low Churn risk", "Medium Churn risk", "High Churn risk"],  

                     filled=True)  

pydot_graph = pydotplus.graph_from_dot_data(dot_data)

pydot_graph.set_size('"25,25!"')

gvz_graph = graphviz.Source(pydot_graph.to_string())

gvz_graph
#To make predictions with the fitted model and the test data

predictions_DT  = model_DT.predict(X_test)
#obtain the probability estimates for the X_test

model_DT.predict_proba(X_test) 
print(classification_report(y_test, predictions_DT))



#print the output of the testing data on the train data 

print('Accuracy in % on test set: {:.2f}'.format(model_DT.score(X_test,y_test)*100))

#the training set will probably be 1 (or close to 1) because the model trains with this data. Let's find out 

print('Accuracy in % on train set: {:.2f}'.format(model_DT.score(X_train_res, y_train_res)*100))
#creating a confusion matrix 

cm_model_DT=confusion_matrix(y_test, predictions_DT)

cm_model_DT_df = pd.DataFrame(cm_model_DT,

                     index = ['high','medium','low'], 

                     columns = ['high','medium','low'])



sns.heatmap(cm_model_DT_df, annot=True)

plt.title('Decision Tree on Test-set \nAccuracy:{0:.2f}'.format(model_DT.score(X_test, y_test)*100))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#checking the importance of the features 

model_DT.feature_importances_
def plot_feature_importances(model):

    n_features = X_train_res.shape[1]

    plt.figure(figsize=(20,10))

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), X_train_res.columns)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.show()

plot_feature_importances(model_DT)
#Changing the split criteria to entropy to include the gained information 

model_DT_entropy = DecisionTreeClassifier(criterion='entropy')

model_DT_entropy.fit(X_train_adasyn, y_train_adasyn)
#To make predictions with the fitted model and the test data

predictions_DT_entropy  = model_DT_entropy.predict(X_test)
print(classification_report(y_test, predictions_DT_entropy))



#print the output of the testing data on the train data 

print('Accuracy in % on test set: {:.2f}'.format(model_DT_entropy.score(X_test,y_test)*100))

#the training set will probably be 1 (or close to 1) because the model trains with this data. Let's find out 

print('Accuracy in % on train set: {:.2f}'.format(model_DT_entropy.score(X_train_adasyn, y_train_adasyn)*100))
model_DT_maxdepth = DecisionTreeClassifier(max_depth=10)

model_DT_maxdepth.fit(X_train_res, y_train_res)

predictions_DT_maxdepth= model_DT_maxdepth.predict(X_test)
print(classification_report(y_test, predictions_DT_maxdepth))



#print the output of the testing data on the train data 

print('Accuracy in % on test set: {:.2f}'.format(model_DT_maxdepth.score(X_test,y_test)*100))

#the training set accuracy  

print('Accuracy in % on train set: {:.2f}'.format(model_DT_maxdepth.score(X_train_res, y_train_res)*100))
model_DT_l8 = DecisionTreeClassifier(max_leaf_nodes=8)

model_DT_l8.fit(X_train, y_train)
predictions_DT_l8= model_DT_l8.predict(X_test)
print(classification_report(y_test, predictions_DT_l8))



#print the output of the testing data on the train data 

print('Accuracy in % on test set: {:.2f}'.format(model_DT_l8.score(X_test,y_test)*100))

#the training set will probably be 1 (or close to 1) because the model trains with this data. Let's find out 

print('Accuracy in % on train set: {:.2f}'.format(model_DT_l8.score(X_train,y_train)*100))
# 1: LogisticRegression

model_LR_1=OneVsRestClassifier(LogisticRegression(solver='lbfgs'))

model_LR_1.fit(MinMax_X_train_adasyn, y_train_adasyn) 

predictions_LR_1 = model_LR_1.predict(MinMax_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_LR_1))

print('Accuracy in % of LogisticRegression on test set: {:.2f}'.format(model_LR_1.score(MinMax_X_test_org, y_test)*100))

print('Accuracy in % of LogisticRegression on train set: {:.2f}'.format(model_LR_1.score(MinMax_X_train_adasyn, y_train_adasyn)*100))

#2.1: LogisticRegressionCV (Adasyn and MinMax(-1.1) CV=5)

model_LR_2=OneVsRestClassifier(LogisticRegressionCV(solver='saga',max_iter=300,  cv=5))

model_LR_2.fit(MinMax_X_train_adasyn, y_train_adasyn) 

predictions_LR_2 = model_LR_2.predict(MinMax_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_LR_2))

print('Accuracy in % of LogisticRegressionCV on test set: {:.2f}'.format(model_LR_2.score(MinMax_X_test_org, y_test)*100))

print('Accuracy in % of LogisticRegressionCV on ADASYN train set: {:.2f}'.format(model_LR_2.score(MinMax_X_train_adasyn, y_train_adasyn)*100))

 #2.2 LogisticRegressionCV (orignal unbalanced and Robust CV=10)

model_LR_3=OneVsRestClassifier(LogisticRegressionCV(solver='lbfgs',max_iter=500, cv=10))

model_LR_3.fit(Robust_X_train_org, y_train) 

predictions_LR_3= model_LR_3.predict(Robust_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_LR_3))

print('Accuracy in % of LogisticRegressionCV on test set: {:.2f}'.format(model_LR_3.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of LogisticRegressionCV on orignal train set: {:.2f}'.format(model_LR_3.score(Robust_X_train_org, y_train)*100))

#2.1: SGDClassifer (Adasyn and MinMax(-1.1) CV=5)

model_LR_4=OneVsRestClassifier(SGDClassifier(random_state=25, n_jobs=50))

model_LR_4.fit(Robust_X_train_res, y_train_res) 

predictions_LR_4 = model_LR_4.predict(Robust_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_LR_4))

print('Accuracy in % of SGDClassifer on test set: {:.2f}'.format(model_LR_4.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of SGDClassifer on SMOTE train set: {:.2f}'.format(model_LR_4.score(Robust_X_train_res, y_train_res)*100))

#2.1: SGDClassifer (Orignial and MinMax(-1.1))

model_LR_5=OneVsRestClassifier(SGDClassifier(n_jobs=10))

model_LR_5.fit(MinMax_X_train_org, y_train) 

predictions_LR_5 = model_LR_5.predict(MinMax_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_LR_5))

print('Accuracy in % of SGDClassifer on test set: {:.2f}'.format(model_LR_5.score(MinMax_X_test_org, y_test)*100))

print('Accuracy in % of SGDClassifer on Original train set: {:.2f}'.format(model_LR_5.score(MinMax_X_train_org, y_train)*100))

model_rf = RandomForestClassifier(n_estimators=25, random_state=12)

model_rf.fit(Robust_X_train_org, y_train)
#create predictions with fitted model and test data 

Pred_clfRF = model_rf.predict(Robust_X_test_org)

# print classification report 

print(classification_report(y_test, Pred_clfRF)) 

print('Accuracy in % test set: {:.2f}'.format(model_rf.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % train set: {:.2f}'.format(model_rf.score(Robust_X_train_org, y_train)*100))
#creating a confusion matrix 

cm_model_RF=confusion_matrix(y_test, Pred_clfRF)

cm_model_RF_df = pd.DataFrame(cm_model_RF,

                     index = ['high','medium','low'], 

                     columns = ['high','medium','low'])



sns.heatmap(cm_model_RF_df, annot=True)

plt.title('Decision Tree on Test-set \nAccuracy:{0:.2f}'.format(model_rf.score(Robust_X_test_org,y_test)*100))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
model_OvsR_rfclf =OneVsRestClassifier(RandomForestClassifier(criterion='entropy'))

model_OvsR_rfclf.fit(MinMax_X_train_res, y_train_res)
#create predictions with fitted model and test data 

predictions_rfclf = model_OvsR_rfclf.predict(MinMax_X_test_org) 

  

# print classification report 

print(classification_report(y_test, predictions_rfclf)) 

print('Accuracy in % on test set: {:.2f}'.format(model_OvsR_rfclf.score(MinMax_X_test_org, y_test)*100))

print('Accuracy in % on train set: {:.2f}'.format(model_OvsR_rfclf.score(MinMax_X_train_res, y_train_res)*100))
# Model 1: Adasyn with MinMax(-1.1) 

model_GB_1=OneVsRestClassifier(GradientBoostingClassifier(n_estimators=1000, loss='exponential', subsample=0.5, learning_rate=0.01,max_depth=8))

model_GB_1.fit(MinMax_X_train_adasyn, y_train_adasyn) 

predictions_GB_1 = model_GB_1.predict(MinMax_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_GB_1))

print('Accuracy in % of GradientBoost on test set: {:.2f}'.format(model_GB_1.score(MinMax_X_test_org, y_test)*100))

print('Accuracy in % of GradientBoost on train set: {:.2f}'.format(model_GB_1.score(MinMax_X_train_adasyn, y_train_adasyn)*100))
# Model 2: Unbalanced with Robust

model_GB_2=OneVsRestClassifier(GradientBoostingClassifier(n_estimators=1500,max_features=5,max_depth=12))

model_GB_2.fit(Robust_X_train_org, y_train) 

predictions_GB_2 = model_GB_2.predict(Robust_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_GB_2))

print('Accuracy in % of GradientBooston test set: {:.2f}'.format(model_GB_2.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of GradientBoost on train set: {:.2f}'.format(model_GB_2.score(Robust_X_train_org, y_train)*100))
#creating a confusion matrix 

cm_model_GB=confusion_matrix(y_test, predictions_GB_2)

cm_model_GB_df = pd.DataFrame(cm_model_GB,

                     index = ['high','medium','low'], 

                     columns = ['high','medium','low'])



sns.heatmap(cm_model_GB_df, annot=True)

plt.title('Decision Tree on Test-set \nAccuracy:{0:.2f}'.format(model_GB_2.score(Robust_X_test_org,y_test)*100))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
# Model 3: SMOTE with Robust 

model_GB_3=OneVsRestClassifier(GradientBoostingClassifier(n_estimators=2000, subsample=0.5, learning_rate=0.001,max_depth=12,warm_start=True))

model_GB_3.fit(Robust_X_train_res, y_train_res) 

predictions_GB_3 = model_GB_3.predict(Robust_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_GB_3))

print('Accuracy in % of GradientBoost on test set: {:.2f}'.format(model_GB_3.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of GradientBoost on train set: {:.2f}'.format(model_GB_3.score(Robust_X_train_res, y_train_res)*100))
# Model 1: ADASYN with MinMax(-1.1) SVC and GridsreachCV



model_to_set = OneVsRestClassifier(SVC(kernel="rbf"))



parameters = {

    "estimator__C": [1,4,8,10],

    "estimator__kernel": ["poly","rbf"],

    "estimator__degree":[1, 2, 3, 4],

}



model_tunning = GridSearchCV(model_to_set, param_grid=parameters,

                             scoring='f1_weighted')



model_tunning.fit(MinMax_X_train_adasyn, y_train_adasyn)

 

print(model_tunning.best_score_)

print(model_tunning.best_params_)
# Model 1: ADASYN with MinMax(-1.1) SVC and GridsreachCV

predictions_SVM_1 = model_tunning.predict(MinMax_X_test_org) 

print(classification_report(y_test, predictions_SVM_1))

print('Accuracy in % of SVC on test set: {:.2f}'.format(model_tunning.score(MinMax_X_test_org, y_test)*100))

print('Accuracy in % of SVC on train set: {:.2f}'.format(model_tunning.score(MinMax_X_train_adasyn, y_train_adasyn)*100))
# Model 2: Unbalanced with Robust  NuSVC

model_SVM_2=NuSVC(kernel='rbf',nu=0.06,class_weight='balanced',decision_function_shape='ovr',random_state=5)

model_SVM_2.fit(Robust_X_train_org, y_train) 

predictions_SVM_2 = model_SVM_2.predict(Robust_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_SVM_2))

print('Accuracy in % of NuSVC on test set: {:.2f}'.format(model_SVM_2.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of NuSVC on train set: {:.2f}'.format(model_SVM_2.score(Robust_X_train_org, y_train)*100))
# Model 3: ADASYN with Robust LinearSVC 

model_SVM_3= OneVsRestClassifier(LinearSVC())

model_SVM_3.fit(Robust_X_train_adasyn, y_train_adasyn) 

predictions_SVM_3 = model_SVM_3.predict(Robust_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_SVM_3))

print('Accuracy in % of LinearSVC on test set: {:.2f}'.format(model_SVM_3.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of LinearSVC on train set: {:.2f}'.format(model_SVM_3.score(Robust_X_train_adasyn, y_train_adasyn)*100))
model_NB = GaussianNB()

model_NB.fit(MinMax_X_train_res, y_train_res) 
#Performing classification by the .predict() method

labels_test = model_NB.predict(MinMax_X_test_org)
# print classification report 

print(classification_report(y_test, labels_test))

#print the output score of the testing and training data on mean accuracy

print('Accuracy in % of Naive Bayes on test set: {:.2f}'.format(model_NB.score(MinMax_X_test_org,y_test)*100))

print('Accuracy in % of Naive Bayes on train set: {:.2f}'.format(model_NB.score(MinMax_X_train_res, y_train_res)*100))
model_NB2 = GaussianNB()

model_NB2.fit(Robust_X_train_org, y_train) 
#Performing classification by the .predict() method

labels_test = model_NB2.predict(Robust_X_test_org)
# print classification report 

print(classification_report(y_test, labels_test))

#print the output score of the testing and training data on mean accuracy

print('Accuracy in % of Naive Bayes on test set: {:.2f}'.format(model_NB.score(Robust_X_test_org,y_test)*100))

print('Accuracy in % of Naive Bayes on train set: {:.2f}'.format(model_NB.score(Robust_X_train_org, y_train)*100))
#creating a confusion matrix 

cm_model_NB=confusion_matrix(y_test, labels_test)

cm_model_NB_df = pd.DataFrame(cm_model_NB,

                     index = ['high','medium','low'], 

                     columns = ['high','medium','low'])



sns.heatmap(cm_model_NB_df, annot=True)

plt.title('Decision Tree on Test-set \nAccuracy:{0:.2f}'.format(model_NB2.score(Robust_X_test_org,y_test)*100))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
# Model 1: Adasyn with Robust MLPclassifer

model_NN_1=OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(7,7),activation = 'tanh',solver = 'sgd',learning_rate = 'adaptive',learning_rate_init = 0.05, batch_size = 100,max_iter = 250))

model_NN_1.fit(Robust_X_train_adasyn, y_train_adasyn) 

predictions_NN_1 = model_NN_1.predict(Robust_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_NN_1))

print('Accuracy in % of Neural Network on test set: {:.2f}'.format(model_NN_1.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of Neural Network on train set: {:.2f}'.format(model_NN_1.score(Robust_X_train_adasyn, y_train_adasyn)*100))
# Model 2: Unbalanced with Robust

model_NN_2=MLPClassifier(hidden_layer_sizes=(5,2),activation = 'relu',solver = 'adam',learning_rate_init = 0.05, batch_size = 20,max_iter = 500)

model_NN_2.fit(Robust_X_train_org, y_train) 

predictions_NN_2 = model_NN_2.predict(Robust_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_NN_2))

print('Accuracy in % of Neural Network on test set: {:.2f}'.format(model_NN_2.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of Neural Network on train set: {:.2f}'.format(model_NN_2.score(Robust_X_train_org, y_train)*100))
# Model 3: SMOTE with MinMax(-1.1)

model_NN_3=OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(7,20),activation = 'tanh',solver = 'sgd',learning_rate = 'adaptive',learning_rate_init = 0.05, batch_size = 100,max_iter = 250))

model_NN_3.fit(MinMax_X_train_res, y_train_res) 

predictions_NN_3 = model_NN_3.predict(MinMax_X_test_org) 

# print classification report 

print(classification_report(y_test, predictions_NN_3))

print('Accuracy in % of Neural Network on test set: {:.2f}'.format(model_NN_3.score(MinMax_X_test_org, y_test)*100))

print('Accuracy in % of Neural Network on train set: {:.2f}'.format(model_NN_3.score(MinMax_X_train_res, y_train_res)*100))
#Checking the cross validation with CV=10 for the model 3: 

scores = cross_val_score(model_NN_3, MinMax_X_train_res, y_train_res , cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Addiontal model 4: NN with gridsearch ADASYN and ROBUST

model_NN_4 = MLPClassifier()



parameters_nn= {

 'hidden_layer_sizes': [(2,7,3), (5,10), (10,8), (5,2)],

    'activation': ['tanh', 'logistic','relu'],

    'solver': ['sgd', 'adam'],

    'learning_rate': ['constant','adaptive']

}



model_tunning_nn = GridSearchCV(model_NN_4,parameters_nn)



clf_nn4=model_tunning_nn.fit(Robust_X_train_adasyn, y_train_adasyn)

 

print(model_tunning_nn.best_score_)

print(model_tunning_nn.best_params_)



predictions_NN_4 = model_tunning_nn.predict(Robust_X_test_org) 

print(classification_report(y_test, predictions_NN_4))

print('Accuracy in % of Neural Network - GridSearch on test set: {:.2f}'.format(model_tunning_nn.score(Robust_X_test_org, y_test)*100))

print('Accuracy in % of Neural Network - GridSearch on train set: {:.2f}'.format(model_tunning_nn.score(Robust_X_train_adasyn, y_train_adasyn)*100))
#DO NOT RUN takes ages

#scores = cross_val_score(model_tunning_nn,Robust_X_train_adasyn, y_train_adasyn, cv=10)

#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf1 = LogisticRegression(multi_class='multinomial',random_state=5)

clf2 = RandomForestClassifier(n_estimators=50, random_state=12)

clf3 = GaussianNB()

#working with the robustscaled data, unbalanced 

X = Robust_X_test_org

y = y_test
#voting is set to 'hard'

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

eclf1 = eclf1.fit(X, y)

eclf1_pr = eclf1.predict(X)

#print the prediction

print(eclf1_pr)
print(classification_report(y_test, eclf1_pr))

print('Accuracy on test set: {:.2f}'.format(eclf1.score(Robust_X_test_org, y_test)*100))

print('Accuracy on train set: {:.2f}'.format(eclf1.score(Robust_X_train_org, y_train)*100))
eclf3 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1,2,1], flatten_transform=True)

eclf3 = eclf3.fit(X, y)

eclf3_pr = eclf3.predict(X)

#print predictions 

print(eclf3_pr)
print(classification_report(y_test, eclf3_pr))

print('Accuracy on test set: {:.2f}'.format(eclf3.score(Robust_X_test_org, y_test)*100))

print('Accuracy on train set: {:.2f}'.format(eclf3.score(Robust_X_train_org, y_train)*100))
# Classifers to be fit to the Model 4: 

model_GB_2=OneVsRestClassifier(GradientBoostingClassifier(n_estimators=1500,max_features=5,max_depth=12))

model_rf = RandomForestClassifier(n_estimators=25, random_state=12)

model_tunning = GridSearchCV(model_to_set, param_grid=parameters,

                             scoring='f1_weighted')

#working with the robustscaled data, unbalanced 

X = Robust_X_train_org

y = y_train
#voting is set to 'hard' - Model 4:

eclf4 = VotingClassifier(estimators=[('GB', model_GB_2), ('RF', model_rf), ('SVM', model_tunning)], voting='hard')

eclf4.fit(X, y)

eclf1_pr4 = eclf4.predict(Robust_X_test_org)

#print the prediction

print(eclf1_pr4)
print(classification_report(y_test, eclf1_pr4))

print('Accuracy on test set: {:.2f}'.format(eclf4.score(Robust_X_test_org, y_test)*100))

print('Accuracy on train set: {:.2f}'.format(eclf4.score(Robust_X_train_org, y_train)*100))
# Classifers to be fit to the Model 5: 

model_GB_2=OneVsRestClassifier(GradientBoostingClassifier(n_estimators=1500,max_features=5,max_depth=12))

model_rf = RandomForestClassifier(n_estimators=25, random_state=12)

model_tunning = GridSearchCV(model_to_set, param_grid=parameters,scoring='f1_weighted')

model_NN_1=OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(7,7),activation = 'tanh',solver = 'sgd',learning_rate = 'adaptive',learning_rate_init = 0.05, batch_size = 100,max_iter = 250))

#working with the robustscaled data, unbalanced 

X = Robust_X_train_res

y = y_train_res
#voting is set to 'hard'

eclf5 = VotingClassifier(estimators=[('GB', model_GB_2), ('RF', model_rf), ('SVM', model_tunning),('NN', model_NN_1)], voting='hard')

eclf5.fit(X, y)

eclf1_pr5 = eclf5.predict(Robust_X_test_org)

#print the prediction

print(eclf1_pr5)
print(classification_report(y_test, eclf1_pr5))

print('Accuracy on test set: {:.2f}'.format(eclf5.score(Robust_X_test_org, y_test)*100))

print('Accuracy on train set: {:.2f}'.format(eclf5.score(Robust_X_train_res, y_train_res)*100))
# Classifers to be fit to the Model 6: 

model_GB_2=OneVsRestClassifier(GradientBoostingClassifier(n_estimators=1500,max_features=5,max_depth=12))

model_rf = RandomForestClassifier(n_estimators=25, random_state=12)

model_tunning = GridSearchCV(model_to_set, param_grid=parameters,scoring='f1_weighted')

model_NN_1=OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(7,7),activation = 'tanh',solver = 'sgd',learning_rate = 'adaptive',learning_rate_init = 0.05, batch_size = 100,max_iter = 250))

#working with the robustscaled data, unbalanced 

X = Robust_X_train_adasyn

y = y_train_adasyn
#voting is set to 'hard'

eclf6 = VotingClassifier(estimators=[('GB', model_GB_2), ('RF', model_rf), ('NN', model_NN_1)], voting='hard')

eclf6.fit(X, y)

eclf1_pr6 = eclf6.predict(Robust_X_test_org)

#print the prediction

print(eclf1_pr6)
print(classification_report(y_test, eclf1_pr6))

print('Accuracy on test set: {:.2f}'.format(eclf6.score(Robust_X_test_org, y_test)*100))

print('Accuracy on train set: {:.2f}'.format(eclf6.score(Robust_X_train_adasyn, y_train_adasyn)*100))
Test_DF1 = test_data_orig.copy()
Test_DF1['Overtime_year']=(Test_DF1['Overtime']*12)/8
Test_DF1.head()
Test_DF1['WorkLifeBalance']= Test_DF1['Days_off'] / Test_DF1['Overtime_year']

Test_DF1.head()
#creating bins for age groups [10,34] = Young, [34, 50]= Adult, [50,79]= Old 

Test_DF1['age_by_group'] = pd.cut(x=Test_DF1['Age'], bins=[10, 34, 50, 79], labels=['young', 'adult', 'old'])
Test_DF1['age_by_group'].replace(to_replace = 'young', value = 0, inplace = True)

Test_DF1['age_by_group'].replace(to_replace = 'adult', value = 1, inplace = True)

Test_DF1['age_by_group'].replace(to_replace = 'old', value = 3, inplace = True)
Test_DF1.head()
#employees bonus [-0.3, 0.2] = negatives, [0.2, 0.6] = neutral, [0.6, 1]= positives, and [1, 5]= extreme positives 

Test_DF1['bonus_split'] = pd.cut(x=Test_DF1['Bonus'], bins=[-0.3, 0.2, 0.6, 1, 5], 

                                          labels=['Negatives', 'Neutral', 'Positives', 'Extra positives'])
#creating an integer of the categorical values 

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

Test_DF1['bonus_split'] = LE.fit_transform(Test_DF1['bonus_split'])
total = Test_DF1.isnull().sum().sort_values(ascending=False)

percent = (Test_DF1.isnull().sum()/Test_DF1.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#Treatment of the missing values with mean

Test_DF1.fillna(Test_DF1.mean(), inplace=True)

#checking changes if there is all treated

Test_DF1.info()
sns.boxplot(x=Test_DF1['Age'])
print(Test_DF1['Age'].quantile(0.95))
#creating a copy of the df 

Test_DF_outlier = Test_DF1.copy()
#Outlier treatment for over age 45 and replace by 3rd quartile = 45 (see code above)

Test_DF_outlier['Age'] =np.where(Test_DF1['Age']>55, 45,Test_DF1['Age'])
sns.boxplot(x=Test_DF_outlier['Age'])
sns.boxplot(x=Test_DF1['Days_off'])
print(Test_DF1['Days_off'].quantile(0.95))
Test_DF_outlier['Days_off'] =np.where(Test_DF1['Days_off']>13, 10,Test_DF1['Days_off'])
sns.boxplot(x=Test_DF_outlier['Days_off'])
sns.boxplot(x=Test_DF1['Satis_leader'])
print(Test_DF1['Satis_leader'].quantile(0.95))
Test_DF_outlier['Satis_leader'] =np.where(Test_DF1['Satis_leader']>2.5, 2,Test_DF1['Satis_leader'])
sns.boxplot(x=Test_DF1['Satis_team'])
print(Test_DF1['Satis_team'].quantile(0.95))
Test_DF_outlier['Satis_team'] =np.where(Test_DF1['Satis_team']>2, 1.5,Test_DF1['Satis_team'])
sns.boxplot(x=Test_DF1['Emails'])
print(Test_DF1['Emails'].quantile(0.95))
Test_DF_outlier['Emails'] =np.where(Test_DF1['Emails']>75, 60,Test_DF1['Emails'])
sns.boxplot(x=Test_DF1['Tenure'])
print(Test_DF1['Tenure'].quantile(0.95))
Test_DF_outlier['Tenure'] =np.where(Test_DF1['Tenure']>28, 22,Test_DF1['Tenure'])
sns.boxplot(x=Test_DF1['Bonus'])
print(Test_DF1['Bonus'].quantile(0.95))
Test_DF_outlier['Bonus'] =np.where(Test_DF1['Bonus']>1.5, 0.95,Test_DF1['Bonus'])
sns.boxplot(x=Test_DF1['Distance'])
print(Test_DF1['Distance'].quantile(0.95))
Test_DF_outlier['Distance'] =np.where(Test_DF1['Distance']>3.6, 3.4,Test_DF1['Distance'])
sns.boxplot(x=Test_DF1['Overtime'])
print(Test_DF1['Overtime'].quantile(0.95))
Test_DF_outlier['Overtime'] =np.where(Test_DF1['Overtime']>14, 12,Test_DF1['Overtime'])
#LABEL ENCODER (CREATING NUMBERICAL VALUES)

le_count = 0

for col in Test_DF_outlier.columns[1:19]:

    if Test_DF1[col].dtype == 'object':

        if len(list(Test_DF_outlier[col].unique())) <= 7:

            le.fit(Test_DF_outlier[col])

            Test_DF_outlier[col] = le.transform(Test_DF_outlier[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))

Test_DF_outlier.head()
#fit to robust scaler

HR_col = list(Test_DF_outlier.columns)

HR_col.remove('Employee_ID')

for col in HR_col:

    Test_DF_outlier[col] = Test_DF_outlier[col].astype(float)

    Test_DF_outlier[[col]] = robust.fit_transform(Test_DF_outlier[[col]])

Test_DF_outlier.info()
Test_DF2=Test_DF_outlier.copy()
Test_DF2=Test_DF2.drop(['Employee_ID'], axis = 1)

Test_DF2.info()
#making the csv file 

predictions_model =model_OvsR_rfclf.predict(Test_DF2)
csv_pred =pd.DataFrame(data=predictions_model, columns = ['Churn_risk'], index = Test_DF2.index.copy())
Test_DF1['Churn_risk'] = csv_pred

Test_DF1.head()
#After oversampling the data set 

category_order = ["low","medium","high",]



sns.countplot(predictions_model, order=category_order)

plt.show()
Result_rf=Test_DF1[['Employee_ID', 'Churn_risk']]

Result_rf.head()
Result_rf.to_csv('Group20_version27.csv',index=False)
Group20_version21=pd.read_csv('Group20_version21.csv')

Group20_version21.head()