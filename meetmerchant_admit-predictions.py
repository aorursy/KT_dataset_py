# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read the dataset into a dataframe

# df = pd.read_csv('../input/admt-dataset/file3.csv')

# df.head()

df = pd.read_csv('../input/newdata/new_adm_pred.csv')

df = df.drop('Serial No.', axis=1)

df.head()
#Comprehensive description of data

print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)





fig = sns.distplot(df['GRE Score'], kde=False)

plt.title("Distribution of GRE Scores")

plt.show()



fig = sns.distplot(df['TOEFL Score'], kde=False)

plt.title("Distribution of TOEFL Scores")

plt.show()



fig = sns.distplot(df['University Rating'], kde=False)

plt.title("Distribution of University Rating")

plt.show()



fig = sns.distplot(df['CGPA'], kde=False)

plt.title("Distribution of CGPA")

plt.show()



plt.show()



print("The distribution plots prove that this university accepts a diverse range of students")
# Gender Ratio Visualization

from sklearn import preprocessing

import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()

xyz = le.fit_transform(df['Gender'])

f=0

m=0

for i in xyz:

    if (i==0):

        f=f+1

    else:

        m=m+1



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

labels = 'Male', 'Female'

sizes = [m,f]

colors = ['#308CCE', '#E27FD3']

ax.pie(sizes, labels=labels, colors=colors, autopct ='% 1.1f %%', shadow = True)

plt.title('Gender Ratio')

plt.show()
#Nationality Diversity

df['Nation'].value_counts().plot(kind='bar', color = 'orange', title = 'Nationality Distribution')
#Research

df['Research'].value_counts().plot(kind='bar', color = 'brown', title = 'Research Distribution')
#Social Work

df['Social Work'].value_counts().plot(kind='bar', color = 'Green', title = 'Social Work Distribution')
#Work Exp

df['Work Exp'].value_counts().plot(kind='bar', color = 'black', title = 'Work Exp Distribution')
#Avg scores

df.rename(columns={'GRE Score':'GRE','Avg Living Expense':'ALE','TOEFL Score':'TOEFL','University Rating':'UnivRating','Chance of Admit ':'Chance'},inplace=True)

print('Mean CGPA Score is :',int(df[df['CGPA']<=500].CGPA.mean()))

print('Mean GRE Score is :',int(df[df['GRE']<=500].GRE.mean()))

print('Mean TOEFL Score is :',int(df[df['TOEFL']<=500].TOEFL.mean()))

print('Mean University rating is :',int(df[df['UnivRating']<=500].UnivRating.mean()))

print("\nTarget of an aspirant would be get more than the mean scores displayed above.")

print('\nAverage Living Expense (monthly) of students is : $',df['ALE'].mean())
#minimum scores

df_sort=df.sort_values(by=df.columns[-1],ascending=False)

df_sort = df_sort.drop(['Research', 'UnivRating', 'Social Work', 'Work Exp', 'ALE'], axis=1)

df_sort.head()

df_sort[(df_sort['Chance']>0.90)].mean().reset_index()
print("For having a 90% Chance to get admission one should have GRE=333.61,TOEFL=116.28,CGPA=9.53 .If you get scores more than this then your chances of admission are very good.")

df.head()
#Now we dont need the columns Gender, Nation, Avg Living Expense so we drop them

df.drop(['Gender', 'Nation', 'ALE'], axis=1, inplace=True)

df.head()
#Lets split the dataset with training and testing set and prepare the inputs and outputs

from sklearn.model_selection import train_test_split



X = df.drop(['Chance'], axis=1)

y = df['Chance']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)
#Lets use a bunch of different algorithms to see which model performs better

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor

from sklearn.metrics import mean_squared_error



models = [['DecisionTree :',DecisionTreeRegressor()],

           ['Linear Regression :', LinearRegression()],

           ['RandomForest :',RandomForestRegressor()],

           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],

           ['SVM :', SVR()],

           ['AdaBoostClassifier :', AdaBoostRegressor()],

           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],

           ['Xgboost: ', XGBRegressor()],

           ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],

           ['Lasso: ', Lasso()],

           ['Ridge: ', Ridge()],

           ['BayesianRidge: ', BayesianRidge()],

           ['ElasticNet: ', ElasticNet()],

           ['HuberRegressor: ', HuberRegressor()]]



print("Results...")



for name,model in models:

    model = model

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))

    

print("Something as simple as Linear Regression performs the best in this case, which proves that complicated models doesnt always mean better results.")
#Predict random student's chance

reg=LinearRegression()

reg.fit(X_train,y_train)

print("Enter your scores:")

lst = []

for i in range(7): 

    ele = input()

    lst.append(ele)

# Score=['337','118','4','9.65','0','0','1']

Score=pd.DataFrame(lst).T

chance=reg.predict(Score)

print(chance[0]*100)
classifier = RandomForestRegressor()

classifier.fit(X,y)

feature_names = X.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = classifier.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)
# df.head()

import matplotlib.pyplot as plt

plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5, color = 'darkgreen')

plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()

print("Clearly, CGPA is the most factor for graduate admissions followed by GRE Score. \nThe other parameters like Research, Social Work, Work experience have less impact on the chance of admission.")
