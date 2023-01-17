import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics



from sklearn.linear_model import LinearRegression 

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor



from sklearn.metrics import mean_squared_error
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/Train.csv") 

test = pd.read_csv("/kaggle/input/Test.csv")
train.columns
print(train.shape)

train.head()
train.describe()
train.info()
train.isna().any()
#Using Pearson Correlation

plt.figure(figsize=(18,10))

cor = train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Accent)

plt.show()

plt.savefig("main_correlation.png")
label = ["Attrition_rate"]

features = ['VAR7','VAR6','VAR5','VAR1','VAR3','growth_rate','Time_of_service','Time_since_promotion','Travel_Rate','Post_Level','Education_Level']
featured_data = train.loc[:,features+label]

featured_data = featured_data.dropna(axis=0)

featured_data.shape
X = featured_data.loc[:,features]

y = featured_data.loc[:,label]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.55)
df = Ridge(alpha=0.000001)

df.fit(X_train,y_train)

y_pred = df.predict(X_test)

c=[]

for i in range(len(y_pred)):

    c.append((y_pred[i][0].round(5)))

pf=c[:3000]



print(len(c),len(pf),c[0])
score = 100* max(0, 1-mean_squared_error(y_test, y_pred))

print(score)
selected_test = test.loc[:,features]

#selected_test.info()

mean_values = np.mean(selected_test)

selected_test[features].replace(mean_values,np.nan,inplace=True)

for i,val in enumerate(features):

    selected_test[val] = selected_test[val].fillna(mean_values[i])

    

selected_test.head()
#Predicting

import pandas as pd

dff = pd.DataFrame({'Employee_ID':test['Employee_ID'],'Attrition_rate':pf})

#Converting to CSV

dff.to_csv("Predictions.csv",index=False)