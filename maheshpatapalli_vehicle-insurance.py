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
import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

dataset.head()
dataset.isnull().sum()
dataset.info()
dataset['Gender'] = dataset['Gender'].replace(to_replace = ['Male','Female'],value = [0,1])

dataset['Vehicle_Damage'] = dataset['Vehicle_Damage'].replace(to_replace = ['Yes','No'], value = [1,0])
dataset.head()
dataset['Vehicle_Age'].value_counts()
sns.barplot('Response', 'Annual_Premium', data = dataset)
sns.barplot("Vehicle_Age", "Response", data = dataset)
sns.barplot("Gender", "Response", data = dataset)
men_in = [1 if dataset["Gender"][i] == 0 and dataset['Response'][i] == 1 else 0 for i in range(len(dataset["Gender"]))]

men_out = [1 if dataset["Gender"][i] == 0 and dataset['Response'][i] == 0 else 0 for i in range(len(dataset["Gender"]))]

women_out = [1 if dataset["Gender"][i] == 1 and dataset['Response'][i] == 0 else 0 for i in range(len(dataset["Gender"]))]

women_in = [1 if dataset["Gender"][i] == 1 and dataset['Response'][i] == 1 else 0 for i in range(len(dataset["Gender"]))]
print("total customers: ", len(dataset["Gender"]))

print("men who insuranced: ",sum(men_in))

print("men who did not buy insurance:", sum(men_out))

print("women who insuranced: ",sum(women_in))

print("women who did not buy insurance:", sum(women_out))
sns.barplot("Driving_License", "Response", data = dataset)
sns.barplot("Previously_Insured", "Response", data = dataset)
sns.FacetGrid(data= dataset, hue="Response", height=5).map(sns.distplot, "Age").add_legend()
plt.figure(figsize = (16,16))

sns.heatmap(dataset.corr(),annot = True, cmap = "Blues")

plt.show()
dataset['Vehicle_Age'] = dataset['Vehicle_Age'].replace(to_replace = ["1-2 Year","< 1 Year","> 2 Years"], value =[1,0,2] )
dataset_new = dataset.drop(['Region_Code', 'Policy_Sales_Channel','Vintage','Annual_Premium'], axis = 1)

dataset_new.head()
x = dataset_new.iloc[:,:-1].values

y = dataset_new.iloc[:,-1].values

print(pd.DataFrame(x)) #coz i am retarded
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
# train_test split is used only for accuracy measurement before submission, has nothing to do with competition

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,random_state = 177013)
XGB = XGBClassifier(n_estimators = 1000, alpha = 0.1, random_state = 177013)

LR = LogisticRegression()

DTC = DecisionTreeClassifier()
XGB.fit(x_train,y_train)

LR.fit(x_train,y_train)

DTC.fit(x_train,y_train)
y_pred_XGB = XGB.predict(x_test)

y_pred_LR = LR.predict(x_test)

y_pred_DTC = DTC.predict(x_test)
from sklearn.metrics import accuracy_score

cm1 = accuracy_score(y_pred_XGB,y_test)

cm2 = accuracy_score(y_pred_LR,y_test)

cm3 = accuracy_score(y_pred_DTC,y_test)

print("XGB score: {0} \nLR score : {1} \nDTC score : {2}".format(cm1,cm2,cm3))
def preprocessing(dataset):

    dataset['Gender'] = dataset['Gender'].replace(to_replace = ['Male','Female'],value = [0,1])

    dataset['Vehicle_Damage'] = dataset['Vehicle_Damage'].replace(to_replace = ['Yes','No'], value = [1,0])

    dataset['Vehicle_Age'] = dataset['Vehicle_Age'].replace(to_replace = ["1-2 Year","< 1 Year","> 2 Years"], value =[1,0,2] )

    dataset = dataset.drop(['Region_Code', 'Policy_Sales_Channel','Vintage','Annual_Premium'], axis = 1)

    return dataset
test_set = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")

testset = preprocessing(test_set)

testset
test = testset.iloc[:,:].values 

test
y_pred = DTC.predict(test)
submission = pd.DataFrame(y_pred,testset['id'])
submission = submission.rename(columns = {0:'Response'})
submission.to_csv('./submission.csv')