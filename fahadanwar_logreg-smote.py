# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

from statsmodels.tools import add_constant as add_constant

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/framingham-heart-study-dataset/framingham.csv")

df.info()
100* df.isnull().sum()/df.count()
df.dropna(inplace=True)
df.head()
print("Percentage of People with heart disease: {0:.2f} %".format(100*df.TenYearCHD.value_counts()[1]/df.TenYearCHD.count()))
df = pd.concat([df, pd.get_dummies(df.education, prefix="ed_",drop_first=True)],axis=1)

df.drop(['education'], axis=1, inplace=True)

df.columns
X = df.drop(['TenYearCHD'], axis=1)

Y = df.TenYearCHD



X_const=sm.add_constant(X)

model=sm.Logit(Y, X_const)

result=model.fit()

result.summary()
def back_feature_elim(data_frame,dep_var,col_list):

    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eliminating

    feature with the highest p-value above alpha(0.05) one at a time and returns the regression summary with all 

    p-values below alpha"""



    while len(col_list)>0 :

        model=sm.Logit(dep_var,data_frame[col_list])

        result=model.fit(disp=0)

        largest_pvalue=round(result.pvalues,3).nlargest(1)

        if largest_pvalue[0]<(0.05):

            return result

            break

        else:

            col_list=col_list.drop(largest_pvalue.index)
result=back_feature_elim(df,df.TenYearCHD, X.columns)

result.summary()
params = np.exp(result.params)

conf = np.exp(result.conf_int())

conf['OR'] = params

pvalue=round(result.pvalues,3)

conf['pvalue']=pvalue

conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']

print ((conf))
new_df=df[['male','age','cigsPerDay','prevalentHyp','diabetes','sysBP','diaBP','BMI','heartRate','ed__2.0','ed__3.0','ed__4.0',

           'TenYearCHD']]

X = new_df.drop(['TenYearCHD'], axis=1)

Y = new_df.TenYearCHD

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=5)
logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
sklearn.metrics.accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

print(cm)
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The accuracy of the model [TP+TN/(TP+TN+FP+FN)] = ',(TP+TN)/float(TP+TN+FP+FN),'\n',



'Missclassification [1-Accuracy] = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',



'Sensitivity or True Positive Rate [TP/(TP+FN)] = ',TP/float(TP+FN),'\n',



'Specificity or True Negative Rate [TN/(TN+FP)] = ',TN/float(TN+FP),'\n')
smt = SMOTE()

x_train, y_train = smt.fit_sample(x_train, y_train)
np.bincount(y_train)
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
print(sklearn.metrics.accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)

print(cm)
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The accuracy of the model [TP+TN/(TP+TN+FP+FN)] = ',(TP+TN)/float(TP+TN+FP+FN),'\n',



'Missclassification [1-Accuracy] = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',



'Sensitivity or True Positive Rate [TP/(TP+FN)] = ',TP/float(TP+FN),'\n',



'Specificity or True Negative Rate [TN/(TN+FP)] = ',TN/float(TN+FP),'\n')