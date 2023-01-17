import matplotlib.pyplot as plt

import seaborn as sns



#Scikit learn librairies

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import scale

from sklearn.metrics import accuracy_score



import pandas as pd

import numpy as np



%matplotlib inline
df_test= pd.read_csv("../input/give-me-some-credit-dataset/cs-test.csv")

df= pd.read_csv("../input/give-me-some-credit-dataset/cs-training.csv")

dftmp=pd.read_csv("../input/give-me-some-credit-dataset/cs-training.csv")

sample_entry=pd.read_csv("../input/give-me-some-credit-dataset/sampleEntry.csv")
df.shape
df.head(10)
df.describe()
sample_entry.head()
df.columns
df.dtypes.value_counts()
df.isnull().sum()
#A function to print every graph with the ID as 

def print_all_values():

    df1=df.drop('Unnamed: 0',axis=1)

    cols=df1.columns

    for col in cols:

        if (df[col].dtypes !='object'):



            fig1=plt.figure()

            ax1=plt.axes()

            plt.scatter(df[[col]],df['Unnamed: 0'],alpha=1,s=0.5)

            plt.title(col)

            ax1 = ax1.set(xlabel=col, ylabel='ID')

            plt.show()

            

            

print_all_values()
print(df.shape)

def delete_absurd_values(df_transformed,cols,max_value,percentage):

        

        

        for col in cols:

            if (df_transformed[col].dtypes !='object'):

                       

                q99=df_transformed[col].quantile(q=percentage)

                q01=df_transformed[col].quantile(q=(1-percentage))

                for i in df_transformed.index:

                    

                    if (df_transformed.loc[i,col]> max_value*q99 or df_transformed.loc[i,col]< q01/max_value):

                        df_transformed=df_transformed.drop(index=i)

        

        return df_transformed



cols=['DebtRatio', 'MonthlyIncome',

       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',

       'NumberRealEstateLoansOrLines',

       'NumberOfDependents']

df=delete_absurd_values(df,cols,4,0.999)

print(df.shape)
df=df[df.RevolvingUtilizationOfUnsecuredLines <30000]

df=df[df.DebtRatio <100000]

df=df[df.MonthlyIncome <15000000]

df=df[df.NumberRealEstateLoansOrLines <40]
df.fillna(df.median(), inplace=True)
df.isnull().sum()
fig11=plt.figure()

ax11=plt.axes()

the_target = dftmp['SeriousDlqin2yrs']

the_target.replace(to_replace=[1,0], value= ['YES','NO'], inplace = True)

plt.title('Target repartition')

ax11 = ax11.set(xlabel='Default proportion', ylabel='Number of people')

the_target.value_counts().plot.pie(startangle=90, autopct='%1.1f%%')

plt.show()
sns.set(style = 'whitegrid', context = 'notebook', rc={'figure.figsize':(20,15)})





cols = ['SeriousDlqin2yrs',

       'RevolvingUtilizationOfUnsecuredLines', 'age',

       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',

       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',

       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',

       'NumberOfDependents']



#sns.pairplot(df[cols])

plt.show()
#Correlation Matrix calcul

corr_mat = df.corr()



fig2=plt.figure()

sns.set(rc={'figure.figsize':(25,15)})

k = 20

cols = corr_mat.nlargest(k, 'SeriousDlqin2yrs')['SeriousDlqin2yrs'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.title('Correlation Matrix')

plt.show()
X = df.drop('SeriousDlqin2yrs',axis=1)

y = df['SeriousDlqin2yrs']  



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100,random_state=0)

logisticRegr.fit(X_train, y_train)
#ERROR

error = (1 - logisticRegr.score(X_test, y_test))*100

print('Score  = ',logisticRegr.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis(solver='svd',shrinkage=None,store_covariance=True)

lda.fit(X_train, y_train)
#ERROR

error = (1 - lda.score(X_test, y_test))*100

print('Score  = ',lda.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=0)

rf.fit(X_train,y_train)
error = (1 - rf.score(X_test, y_test))*100

print('Score  = ',rf.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)
error = (1 - clf.score(X_test, y_test))*100

print('Score  = ',clf.score(X_test, y_test)*100, '%','\nErreur = %f' % error, '%')
print('Taux de réussite par modèle:\n\nRégression Logistique:',logisticRegr.score(X_test, y_test)*100,'%','\n\nLDA:',lda.score(X_test, y_test)*100,'%','\n\nRandom Forest Classifier:',rf.score(X_test, y_test)*100,'%','\n\nDecision Tree Classifier:',clf.score(X_test, y_test)*100,'%')