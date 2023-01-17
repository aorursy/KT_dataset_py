# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
train =pd.read_csv('../input/loanprediction/train.csv')

test =pd.read_csv('../input/loanprediction/test.csv')
import seaborn as sns

%matplotlib inline 
train.head(2)

#test.head(2)
train.describe()
train.info()
x =train.iloc[: ,1:12]
y =train.iloc[: ,12]
x_new =test.iloc[: ,1:]
Id =test.iloc[: ,0] 
total =pd.concat([x ,x_new] , axis =0)
total.head(2)
total.info()
sns.heatmap(total.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Married',data=total,palette='RdBu_r')
#total.drop('Credit_History', axis=1, inplace=True)
total.head(2)
sns.heatmap(total.isnull(),yticklabels=False,cbar=False,cmap='viridis')
total['Loan_Amount_Term'] = total['Loan_Amount_Term'].fillna(total['Loan_Amount_Term'].median())

sns.heatmap(total.isnull(),yticklabels=False,cbar=False,cmap='viridis')

total['LoanAmount'] = total['LoanAmount'].fillna(total['LoanAmount'].median())

sns.heatmap(total.isnull(),yticklabels=False,cbar=False,cmap='viridis')
total.info()
total['Credit_History'] = total['Credit_History'].fillna(total['Credit_History'].median())
sns.heatmap(total.isnull(),yticklabels=False,cbar=False,cmap='viridis')
propertyarea = pd.get_dummies(total['Property_Area'] ,drop_first =True)
propertyarea 
sns.heatmap(total.isnull(),yticklabels=False,cbar=False,cmap='viridis')

gender = pd.get_dummies(total['Gender'] ,drop_first =True)
gender
married = pd.get_dummies(total['Married']  ,drop_first =True)
married
total.info()
dependent  = pd.get_dummies(total['Dependents']  ,drop_first =True)

education  = pd.get_dummies(total['Education']  ,drop_first =True)

selfemployed  = pd.get_dummies(total['Self_Employed']  ,drop_first =True)

#total.drop(['Gender ','Married' ,'Dependents ' ,'Education ', 'Self_Employed' ,'Property_Area'],axis=1,inplace=True)
total.drop(['Married'] , axis =1 ,inplace =True )
total.drop(['Dependents'] , axis =1 ,inplace =True )
total.info()
total.drop(['Education'] , axis =1 ,inplace =True )
total.info()
total.drop(['Self_Employed'] , axis =1 ,inplace =True )
train.info()
total.info()
total.drop(['Property_Area'] , axis =1 ,inplace =True )
total.info()


total = pd.concat([ total,married ,dependent ,education ,propertyarea ,selfemployed],axis=1)

total.info()
total.info()
total.head(2)
sns.heatmap(total.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.shape
train_new = total.iloc[0:614,:]

test_new = total.iloc[614:,:]

train_new.shape
