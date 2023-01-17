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
import pandas as pd 

df = pd.read_csv('../input/fertility-data-set/fertility.csv')
df.head()
# get some information about data 
df.info()
# data transformation and replacement from string to integer
fertility = df

fertility['Season'].replace({"winter":1, "spring":2, "summer":3, "fall":4 }, inplace= True)
fertility['Childish diseases'].replace({"yes":1, "no":0}, inplace= True)
fertility['Accident or serious trauma'].replace({"yes":1, "no":0}, inplace= True)
fertility['Surgical intervention'].replace({"yes":1,"no":0}, inplace= True)
fertility['High fevers in the last year'].replace({"more than 3 months ago":2,"less than 3 months ago":1,"no":0}, inplace= True)
fertility['Frequency of alcohol consumption'].replace({"once a week":1,"hardly ever or never":0,"several times a week":2 , "every day":3 , "several times a day" : 4}, inplace= True)
fertility['Smoking habit'].replace({"daily":2, "occasional":1,"never":0}, inplace= True)
fertility['Diagnosis'].replace({"Normal":1,"Altered":0}, inplace= True)

fertility.head()
fertility.describe()
# to make some statistical calculations for the data to understand it more 
fertility["Number of hours spent sitting per day"].hist(bins=100)
fertility["Number of hours spent sitting per day"].value_counts()
# from the histogram and the value counts functions we can find that we have an outlire with 342, so we can delete this row
fertility.loc[fertility['Number of hours spent sitting per day'] == 342]
# and then delete this row
fertility.drop([50],axis=0, inplace =True)
fertility.isnull().sum()
# there is no any missing values in our data set and now we are ready to go and explore our data 
# to visualize the correlatin between the data 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

corr = fertility.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize = (10,7))
sns.heatmap(corr,annot=True ,linewidths=.5,mask = mask,square=True)
fertility.hist(bins = 30 , figsize=(15,10))
plt.show()
# split our data to features and output 

X = fertility.drop("Diagnosis" , axis = 1 )
y = fertility["Diagnosis"]
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# split the data to train and test model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

dec = DecisionTreeClassifier(max_depth=3)
ran = RandomForestClassifier(n_estimators=100)
knn = KNeighborsClassifier()
svm = SVC(random_state=1)
naive = GaussianNB()
logReg = LogisticRegression()

models = {"Decision tree" : dec,
          "Random forest" : ran,
          "KNN" : knn,
          "SVM" : svm,
          "Naive bayes" : naive,
          "Logistics regression": logReg}
scores= { }

for key, value in models.items():    
    model = value
    model.fit(X_train, y_train)
    scores[key] = model.score(X_test, y_test)
    
scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T
scores_frame.sort_values(by=["Accuracy Score"], axis=0 ,ascending=False, inplace=True)
scores_frame
