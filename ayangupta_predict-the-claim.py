# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.feature_selection import RFE

import sklearn.metrics as metrics

import scipy.stats as ss

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/travel insurance.csv")

df1=df

df.head(5)
df.info()
missingno.matrix(df)
df['Gender'].isnull().sum()
df.fillna('Not Specified',inplace=True)
df.isnull().sum()
df_numerical=df._get_numeric_data()

df_numerical.info()
for i, col in enumerate(df_numerical.columns):

    plt.figure(i)

    sns.distplot(df_numerical[col])
df['Duration'].describe()
df10=df['Duration']<0

df10.sum()
df.loc[df['Duration'] < 0, 'Duration'] = 49.317
df6= df['Net Sales']<df['Commision (in value)']

df6.sum()
df.loc[df['Net Sales'] == 0.0, 'Commision (in value)'] = 0
def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
categorical=['Agency', 'Agency Type', 'Distribution Channel', 'Product Name',  'Destination','Gender','Claim']

cramers=pd.DataFrame({i:[cramers_v(df[i],df[j]) for j in categorical] for i in categorical})

cramers['column']=[i for i in categorical if i not in ['memberid']]

cramers.set_index('column',inplace=True)



#categorical correlation heatmap



plt.figure(figsize=(10,7))

sns.heatmap(cramers,annot=True)

plt.show()
test=[(df[df['Gender']=='Not Specified']['Claim'].value_counts()/len(df[df['Gender']=='Not Specified']['Claim']))[1],(df[df['Gender']=='M']['Claim'].value_counts()/len(df[df['Gender']=='M']['Claim']))[1],

      (df[df['Gender']=='F']['Claim'].value_counts()/len(df[df['Gender']=='F']['Claim']))[1]]

test
fig, axes=plt.subplots(1,3,figsize=(24,9))

sns.countplot(df[df['Gender']=='Not Specified']['Claim'],ax=axes[0])

axes[0].set(title='Distribution of claims for null gender')

axes[0].text(x=1,y=30000,s=f'% of 1 class: {round(test[0],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')

sns.countplot(df[df['Gender']=='M']['Claim'],ax=axes[1])

axes[1].set(title='Distribution of claims for Male')

axes[1].text(x=1,y=6000,s=f'% of 1 class: {round(test[1],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')

sns.countplot(df[df['Gender']=='F']['Claim'],ax=axes[2])

axes[2].set(title='Distribution of claims for Female')

axes[2].text(x=1,y=6000,s=f'% of 1 class: {round(test[2],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')

plt.show()
pd.crosstab(df['Agency'],df['Agency Type'],margins=True)
table1=pd.crosstab(df['Agency'],df['Claim'],margins=True)



table1.drop(index=['All'],inplace=True)

table1=(table1.div(table1['All'],axis=0))*100



table1['mean commision']=df.groupby('Agency')['Commision (in value)'].mean()

table1
table1.columns
fig,ax1=plt.subplots(figsize=(18,9))

sns.barplot(table1.index,table1.Yes,ax=ax1)

plt.xticks(rotation=90)

ax1.set(ylabel='Acceptance %')

ax2=ax1.twinx()

sns.lineplot(table1.index,table1['mean commision'],ax=ax2,linewidth=3)
table2=pd.crosstab(df['Product Name'],df['Claim'],margins=True)

table2=(table2.div(table2['All'],axis=0))*100



table2['mean commision']=df.groupby('Product Name')['Commision (in value)'].mean()

table2.drop(index=['All'],inplace=True)

table2
fig,ax1=plt.subplots(figsize=(20,11))

sns.barplot(table2.index,table2.Yes,ax=ax1)

plt.xticks(rotation=90)

ax1.set(ylabel='Acceptance %')

ax2=ax1.twinx()

sns.lineplot(table2.index,table2['mean commision'],ax=ax2,linewidth=3)
tests=df.copy()

tests['Duration_label']=pd.qcut(df['Duration'],q=35)

table3=pd.crosstab(tests['Duration_label'],tests['Claim'],normalize='index')

table3
table3.columns

plt.figure(figsize=(10,7))

sns.barplot(table3.index,table3.Yes)

plt.xticks(rotation=90)
table4=pd.crosstab(df['Destination'],df['Claim'],margins=True,normalize='index')

table4

table4 = table4.sort_values(by=['Yes'], ascending=[False])

table4
sns.countplot(df['Claim'])
from scipy.stats import chi2_contingency



class ChiSquare:

    def __init__(self, df):

        self.df = df

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = ss.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)
X = df.drop(['Claim'], axis=1)

ct = ChiSquare(df)

for c in X.columns:

    ct.TestIndependence(c, 'Claim')
df.drop(columns=['Distribution Channel','Agency Type'],axis=1,inplace=True)
df.info()
y=df['Claim']

x=df

x.drop(columns='Claim',axis=1,inplace=True)
x_dummy=pd.get_dummies(x,columns=['Agency','Gender','Product Name','Destination'],drop_first=True)
lr = LogisticRegression()

rfe = RFE(estimator=lr, n_features_to_select=10, verbose=3)

rfe.fit(x_dummy, y)

rfe_df1 = rfe.fit_transform(x_dummy, y)
print("Features sorted by their rank:")

print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), x_dummy.columns)))
X=x_dummy[['Agency_EPX','Agency_TST','Gender_Not Specified','Product Name_2 way Comprehensive Plan','Product Name_24 Protect','Product Name_Basic Plan','Product Name_Comprehensive Plan','Product Name_Premier Plan','Product Name_Travel Cruise Protect','Product Name_Value Plan']]
X.head(5)




smote = SMOTE(random_state=7)

X_ov, y_ov = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_ov, y_ov, train_size=0.7, random_state=7)
from sklearn.svm import LinearSVC

algo_dict = {'Random Forest Classifier':RandomForestClassifier(),'DecisionTreeClassifier':DecisionTreeClassifier(),'Linear SVC':LinearSVC()}





                

algo_name=[]

for i in algo_dict:

    algo_name.append(i)



for i in algo_dict.keys():

      

          

        algo = algo_dict[i]

        model = algo.fit(X_train, y_train)

        y_pred = model.predict(X_test)        

        print('Classification report'+'\n',classification_report(y_test, y_pred))

        print('***'*30)

          

        
from sklearn.ensemble import GradientBoostingClassifier

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in learning_rates:

    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)

    gb.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)

    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))

    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))

    print()
gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)

gb.fit(X_train, y_train)

predictions = gb.predict(X_test)



#print("Confusion Matrix:")

#print(confusion_matrix(y_test, predictions))

#print()

print("Classification Report")

print(classification_report(y_test, predictions))