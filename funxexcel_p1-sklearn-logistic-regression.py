import pandas as pd

from sklearn.linear_model import LogisticRegression
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.info()
#Get Target data 

y = data['diagnosis']



#Load X Variables into a Pandas Dataframe with columns 

X = data.drop(['id','diagnosis','Unnamed: 32'], axis = 1)
X.head()
#Check size of data

X.shape
X.isnull().sum()

#We do not have any missing values
logModel = LogisticRegression(max_iter=5000)
logModel.fit(X,y)
print (f'Accuracy - : {logModel.score(X,y):.3f}')