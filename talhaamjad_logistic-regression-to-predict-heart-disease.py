



import warnings

warnings.filterwarnings('ignore')


import pandas as pd

import numpy as np

import statsmodels.api as sm

import scipy.stats as st

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import matplotlib.mlab as mlab



#Importing the dataset and dropping the unwanted columns 

heart_df=pd.read_csv("../input/framingham.csv")

heart_df.drop(['education'],axis=1,inplace=True)

heart_df.head()

#Renaming the column name

heart_df.rename(columns={'male':'Sex_male'},inplace=True)



heart_df.head()
# Checking the missing values 

heart_df.isnull().sum()
#Counting the missing values and dropping them

count=0

for i in heart_df.isnull().sum(axis=1):

    if i>0:

        count=count+1

print('Total number of rows with missing values is ', count)

print('since it is only',round((count/len(heart_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')



heart_df.dropna(axis=0,inplace=True)



heart_df.describe()
heart_df.isnull().sum()
#Adding a constant

from statsmodels.tools import add_constant as add_constant

heart_df_constant = add_constant(heart_df)

heart_df_constant.head()





st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)

cols=heart_df_constant.columns[:-1]

model=sm.Logit(heart_df.TenYearCHD,heart_df_constant[cols])

result=model.fit()

result.summary()



def back_feature_elem (data_frame,dep_var,col_list):

    while len(col_list)>0 :

        model=sm.Logit(dep_var,data_frame[col_list])

        result=model.fit(disp=0)

        largest_pvalue=round(result.pvalues,3).nlargest(1)

        if largest_pvalue[0]<(0.05):

            return result

            break

        else:

            col_list=col_list.drop(largest_pvalue.index)



result=back_feature_elem(heart_df_constant,heart_df.TenYearCHD,cols)

#Interpreting the results: Odds Ratio, Confidence Intervals and Pvalues



params = np.exp(result.params)

conf = np.exp(result.conf_int())

conf['OR'] = params

pvalue=round(result.pvalues,3)

conf['pvalue']=pvalue

conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']

print ((conf))



import sklearn

new_features=heart_df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]

x=new_features.iloc[:,:-1]

y=new_features.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)



from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)



from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

print(confusion_matrix)


sklearn.metrics.accuracy_score(y_test,y_pred)
