# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
CustomerDF = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv', header=0)
CustomerDF.head(5)
CustomerDF.shape
CustomerDF.info()
CustomerDF.describe(include='all').T
# CustomerDF['TotalCharges'] = CustomerDF['TotalCharges'].astype('float')
CustomerDF['TotalCharges'] = CustomerDF['TotalCharges'].replace(' ', np.nan)
CustomerDF['TotalCharges'].isna().sum()

CustomerDF[CustomerDF['TotalCharges'].isna()]
CustomerDF.loc[CustomerDF['TotalCharges'].isna(), 'tenure'] = 1

CustomerDF[CustomerDF['TotalCharges'].isna()]
CustomerDF['TotalCharges'].replace(np.nan, CustomerDF['MonthlyCharges'], inplace=True)
CustomerDF[CustomerDF['TotalCharges'].isna()]
CustomerDF['TotalCharges'] = CustomerDF['TotalCharges'].astype('float')

CustomerDF.info()
CustomerDF.describe().T
fig = plt.figure(figsize=(8, 8))

pic, label, text = plt.pie(CustomerDF['Churn'].value_counts(), labels=CustomerDF['Churn'].value_counts().index, autopct='%1.2f%%', explode=(0.1,0))

for l in label:

    l.set_size(18)

for t in text:

    t.set_size(18)

plt.title('Churn(Yes OR Not)', fontsize=20)

plt.show()
ChurnDF = CustomerDF['Churn'].value_counts().to_frame().sort_values(by=['Churn'], ascending=False)

ChurnDF.values
fig = plt.figure(figsize=(8, 8))

sns.barplot(x=ChurnDF.index, y=ChurnDF['Churn'], data=ChurnDF)

plt.title('Churn(Yes OR Not)', fontsize=16)
def barplot_percentages(feature, orient='v', axis_name="percentage of customers"):

    ratios = pd.DataFrame()

    g = (CustomerDF.groupby(feature)["Churn"].value_counts() / len(CustomerDF)).to_frame()

    g.rename(columns={"Churn": axis_name}, inplace=True)

    g.reset_index(inplace=True)



    #print(g)

    if orient == 'v':

        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)

        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])

        plt.rcParams.update({'font.size': 13})

        plt.legend(fontsize=12)

    else:

        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)

        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])

        plt.legend(fontsize=12)

    plt.title('Churn(Yes/No) Ratio as {0}'.format(feature))

    plt.show()
barplot_percentages("SeniorCitizen")

barplot_percentages("gender")
CustomerDF['churn_rate'] = CustomerDF['Churn'].replace("No", 0).replace("Yes", 1)

g = sns.FacetGrid(CustomerDF, col="SeniorCitizen", height=4, aspect=.9)

ax = g.map(sns.barplot, "gender", "churn_rate", palette = "Blues_d", order= ['Female', 'Male'])

plt.rcParams.update({'font.size': 13})

plt.show()
gp_partner = (CustomerDF.groupby('Partner')["Churn"].value_counts() / len(CustomerDF)).to_frame()

gp_partner.rename(columns={"Churn": "percentage of customers"}, inplace=True)

gp_partner.reset_index(inplace=True)

gp_partner.head(5)
fig, axis = plt.subplots(1, 2, figsize=(16,8))

axis[0].set_title("Has Partner")

axis[1].set_title("Has Dependents")

axis_y = "percentage of customers"



# Plot Partner column

gp_partner = (CustomerDF.groupby('Partner')["Churn"].value_counts() / len(CustomerDF)).to_frame()

gp_partner.rename(columns={"Churn": axis_y}, inplace=True)

gp_partner.reset_index(inplace=True)

ax1 = sns.barplot(x='Partner', y= axis_y, hue='Churn', data=gp_partner, ax=axis[0])

ax1.legend(fontsize=16)

# ax1.set_xlabel('??????') ???????????????





# Plot Dependents column

gp_dep = (CustomerDF.groupby('Dependents')["Churn"].value_counts() / len(CustomerDF)).to_frame()

gp_dep.rename(columns={"Churn": axis_y} , inplace=True)

gp_dep.reset_index(inplace=True)

ax2 = sns.barplot(x='Dependents', y= axis_y, hue='Churn', data=gp_dep, ax=axis[1])

ax2.legend(fontsize=16)

# ax2.set_xlabel('??????')



# ??????????????????

plt.rcParams.update({'font.size': 20})



plt.show()
# Kernel density estimaton???????????????

def kdeplot(feature,xlabel):

    plt.figure(figsize=(9, 4))

    plt.title("KDE for {0}".format(feature))

    ax0 = sns.kdeplot(CustomerDF[CustomerDF['Churn'] == 'No'][feature].dropna(), color= 'navy', label= 'Churn: No', shade='True') # ????????????????????????????????????????????????????????????

    ax1 = sns.kdeplot(CustomerDF[CustomerDF['Churn'] == 'Yes'][feature].dropna(), color= 'orange', label= 'Churn: Yes', shade='True')

    plt.xlabel(xlabel)

    #??????????????????

    plt.rcParams.update({'font.size': 20})

    plt.legend(fontsize=16)

    plt.show()

    

kdeplot('tenure','tenure')
plt.figure(figsize=(10, 6))

barplot_percentages("MultipleLines", orient='h')
plt.figure(figsize=(10, 6))

barplot_percentages("InternetService", orient="h")
cols = ["PhoneService","MultipleLines","OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df1 = pd.melt(CustomerDF[CustomerDF["InternetService"] != "No"][cols])

df1
plt.figure(figsize=(30, 10))

df1 = CustomerDF[(CustomerDF['InternetService'] != "No") & (CustomerDF['Churn'] == "Yes")]

df1 = pd.melt(df1[cols])

df1.rename(columns={'value': 'Has Service'}, inplace=True)

ax = sns.countplot(data=df1, x='variable', hue='Has Service', hue_order=['No', 'Yes'])

ax.set(xlabel='Internet Additional service', ylabel='Churn Num')

plt.rcParams.update({'font.size':20})

plt.legend( labels = ['No Service', 'Has Service'], fontsize=15)

plt.title('Num of Churn Customers as Internet Additional Service')

plt.show()
plt.figure(figsize=(10, 6))

barplot_percentages("PaymentMethod", orient='h')
g = sns.FacetGrid(CustomerDF, col="PaperlessBilling", height=6, aspect=.9)

ax = g.map(sns.barplot, "Contract", "churn_rate", palette = "Blues_d", order= ['Month-to-month', 'One year', 'Two year'])

plt.rcParams.update({'font.size':12})

plt.show()
kdeplot('MonthlyCharges','MonthlyCharges')

kdeplot('TotalCharges','TotalCharges')

plt.show()
CustomerID = CustomerDF['customerID']

CustomerDF.drop(['customerID'], axis=1, inplace=True) # ??????customerID????????????
cateCols = [c for c in CustomerDF.columns if CustomerDF[c].dtype == 'object' or c == 'SeniorCitizen']

dfCate = CustomerDF[cateCols].copy()

dfCate.head(3)
for col in cateCols:

    if dfCate[col].nunique() == 2: # ??????????????????2?????????Yes Or No

        dfCate[col] = pd.factorize(dfCate[col])[0] # ????????? https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html?highlight=factorize

    else:

        dfCate = pd.get_dummies(dfCate, columns=[col]) # One-Hot



# tenure, MonthlyCharges, TotalCharges ????????????????????????

dfCate['tenure'] = CustomerDF[['tenure']]

dfCate['MonthlyCharges'] = CustomerDF[['MonthlyCharges']]

dfCate['TotalCharges'] = CustomerDF[['TotalCharges']]
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

type(data[['A']])

data.corr()
plt.figure(figsize=(16,8))

dfCate.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')

plt.show()
# ????????????

dropFea = ['gender','PhoneService',

           'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',

           'DeviceProtection_No internet service', 'TechSupport_No internet service',

           'StreamingTV_No internet service', 'StreamingMovies_No internet service',

           #'OnlineSecurity_No', 'OnlineBackup_No',

           #'DeviceProtection_No','TechSupport_No',

           #'StreamingTV_No', 'StreamingMovies_No',

           ]

dfCate.drop(dropFea, inplace=True, axis =1) 

#???????????????????????????

target = dfCate['Churn'].values

#??????????????????1?????????

columns = dfCate.columns.tolist()
# ???????????????

columns.remove('Churn')

# ???????????????DataFrame

features = dfCate[columns].values
from sklearn.model_selection import train_test_split
# 30% ???????????????????????????????????????

# random_state = 1??????????????????????????????????????????????????????

# stratify = target ???????????????????????????????????????????????????????????????????????????????????????

train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify = target, random_state = 1)
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
# ?????????????????????

classifiers = [

    SVC(random_state = 1, kernel = 'rbf'),    

    DecisionTreeClassifier(random_state = 1, criterion = 'gini'),

    RandomForestClassifier(random_state = 1, criterion = 'gini'),

    KNeighborsClassifier(metric = 'minkowski'),

    AdaBoostClassifier(random_state = 1),   

]

# ???????????????

classifier_names = [

            'svc', 

            'decisiontreeclassifier',

            'randomforestclassifier',

            'kneighborsclassifier',

            'adaboostclassifier',

]

# ???????????????

#????????????????????????????????????????????????GridSearchCV???????????????????????????"????????????"+"__"+"?????????"

classifier_param_grid = [

            {'svc__C':[0.1], 'svc__gamma':[0.01]},

            {'decisiontreeclassifier__max_depth':[6,9,11]},

            {'randomforestclassifier__n_estimators':range(1,11)} ,

            {'kneighborsclassifier__n_neighbors':[4,6,8]},

            {'adaboostclassifier__n_estimators':[70,80,90]}

]
# ??????????????????????????? GridSearchCV ????????????

def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score = 'accuracy_score'):

    response = {}

    gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv=3, scoring = score)

    # ????????????????????? ???????????????????????????

    search = gridsearch.fit(train_x, train_y)

    print("GridSearch ???????????????", search.best_params_)

    print("GridSearch ??????????????? %0.4lf" %search.best_score_)

    #??????predict????????????????????????????????????????????????????????????????????????????????????????????????????????????

    predict_y = gridsearch.predict(test_x)

    print(" ????????? %0.4lf" %accuracy_score(test_y, predict_y))

    response['predict_y'] = predict_y

    response['accuracy_score'] = accuracy_score(test_y, predict_y)

    return response
for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):

    #?????? StandardScaler ????????????????????????????????????0????????????1???????????????

    pipeline = Pipeline([

            #('scaler', StandardScaler()),

            #('pca',PCA),

            (model_name, model)

    ])

    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')