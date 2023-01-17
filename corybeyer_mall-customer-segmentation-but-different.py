import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Checking for normal distribution

from statsmodels.graphics.gofplots import qqplot

from scipy.stats import shapiro



#Cross Validation, Normalization, and adjusted sampling

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE



#My Model

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



#Model Evaluation

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix,classification_report

from sklearn import metrics



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/Mall_Customers.csv')

df.shape
df.head(3)
df.set_index('CustomerID',inplace=True)

df.columns=['Gender','Age','Annual_Income','Spending_Score']

df.dtypes
#creating binary gender categories

df['Gender']=df['Gender'].astype('category').cat.codes

df.head(3)
#checking for null

print(df.isnull().sum())
fig, (ax1,ax2,ax3) = plt.subplots(ncols =3, figsize =(35,10))

sns.boxplot(df.Spending_Score, ax=ax1)

sns.distplot(df.Spending_Score,ax=ax2)

#add ';' at the end to stop repeating charts. this is a notebook issue.

qqplot(df.Spending_Score, line ='s',ax=ax3);



stat, p = shapiro(df.Spending_Score)

print('Statistics=%.3f, p=%.3f' % (stat, p))
df.Spending_Score.describe()
sigma=df.Spending_Score.std()

mean= df.Spending_Score.mean()

Twodev= sigma+mean

Twodev
df['Target']= np.where(df.Spending_Score<Twodev,0,1)
df.Target.value_counts()
chart=df.drop(["Spending_Score"],axis=1)



fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(16,5))



sns.kdeplot(chart.Age[chart.Target==1], label='Target', shade=True, ax=ax1)

sns.kdeplot(chart.Age[chart.Target==0], label='Non_Target', shade=True, ax=ax1)

ax1.set_ylabel('Distribution')

ax1.set_xlabel('Age')



sns.kdeplot(chart.Annual_Income[chart.Target==1], label='Target', shade=True, ax=ax2)

sns.kdeplot(chart.Annual_Income[chart.Target==0], label='Non-Target', shade=True, ax=ax2)

ax2.set_ylabel('Distribution')

ax2.set_xlabel('Annual Income')
sns.lmplot(x='Annual_Income',y='Age', hue='Target',data=chart,fit_reg=False)
#SCALED

scaler = StandardScaler()

model_df=df.drop(['Target','Spending_Score'],axis=1)

scaled_df= scaler.fit_transform(model_df)

scaled_df=pd.DataFrame(scaled_df)

scaled_df.columns=('Gender','Age','Annual_Income')



#Train, Test, Split

X_train, X_test, y_train, y_test = train_test_split(scaled_df,

                                                    df['Target'],

                                                    test_size=0.3,

                                                    random_state=100)
fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(15, 5))

ax1.set_title('Before Scaling')

sns.kdeplot(df['Age'], ax=ax1)

sns.kdeplot(df['Annual_Income'], ax=ax1)



ax2.set_title('After Standard Scaler')

sns.kdeplot(scaled_df['Age'], ax=ax2)

sns.kdeplot(scaled_df['Annual_Income'], ax=ax2)
sm = SMOTE(random_state=100)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
param_grid = {'C': [0.1, 1, 10,100,1000],

              'gamma': [100,10,1,.1,.01], 

              'kernel': ['rbf', 'sigmoid' ]} 

grid_search = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)

grid_search.fit(X_train_res,y_train_res)
grid_predictions = grid_search.predict(X_test)

grid_search.best_params_
print(classification_report(y_test,grid_predictions))

print("Accuracy score: {}".format(accuracy_score(y_test,grid_predictions)))

print("Recall score: {}".format(recall_score(y_test,grid_predictions)))

print("Precision score: {}".format(precision_score(y_test,grid_predictions)))
cm = confusion_matrix(y_test, grid_predictions)

sns.heatmap(cm, annot=True); #annot=True to annotate cells

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion Matrix')