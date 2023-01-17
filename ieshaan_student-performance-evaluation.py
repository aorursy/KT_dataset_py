# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression,Ridge

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
df.isna().sum()
df.describe()
plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()
df['total'] = (df['math score']+df['reading score']+df['writing score'])/3

df
sns.barplot(x="gender", y="math score", data=df)

plt.figure(figsize=(8,8))
sns.barplot(x="gender", y="reading score", data=df)

plt.figure(figsize=(8,8))
sns.barplot(x="gender", y="writing score", data=df)

plt.figure(figsize=(8,8))
sns.barplot(x="gender", y="total", data=df)

plt.figure(figsize=(8,8))
new = df[['math score','reading score','writing score']].copy()



math_df = new['math score'].sum()

reading_df = new['reading score'].sum()

writing_df = new['writing score'].sum()



total = [math_df,reading_df,writing_df]

columns = ['Math','Reading','Writing']



fig1, ax1 = plt.subplots()

ax1.pie(total, labels=columns, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
course_gender = df.groupby(['gender','parental level of education']).mean().reset_index()

sns.factorplot(x='gender', y='total', hue='parental level of education', data=course_gender, kind='bar')
bplot = sns.boxplot( y = 'total' ,x ='parental level of education'  ,data = df  )

_ = plt.setp(bplot.get_xticklabels(), rotation=90)
sns.countplot(x="test preparation course", hue="gender", data=df)
sns.catplot(data=df, kind="swarm", x="race/ethnicity", y="total", hue="gender")
sns.catplot(data=df, kind="violin", x="race/ethnicity", y="total", hue="gender")
# Import label encoder 

from sklearn import preprocessing 



label_encoder = preprocessing.LabelEncoder() 



df['gender']= label_encoder.fit_transform(df['gender']) 

df['lunch']= label_encoder.fit_transform(df['lunch'])

df['test preparation course']= label_encoder.fit_transform(df['test preparation course'])
df = pd.get_dummies(df)
X = df.drop(columns=['total'],axis=1)

Y= df['total']



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=177)
scaler = StandardScaler()

X = scaler.fit_transform(X)
models=[LinearRegression(),

        RandomForestRegressor(n_estimators=200,max_depth=5),

        DecisionTreeRegressor(random_state=62,max_depth=5),GradientBoostingRegressor(),

        KNeighborsRegressor(n_neighbors=50),BaggingRegressor()]

model_names=['LinearRegression','RandomForestRegressor','DecisionTree','GradientBoostingRegressor','KNN',

             'BaggingReg']



R2_SCORE=[]

MSE=[]



for model in range(len(models)):

    print('')

    print("*"*40,"\n",model_names[model])

    print('')

    reg=models[model]

    reg.fit(X_train,Y_train)

    pred=reg.predict(X_test)

    R2_score=r2_score(Y_test,pred)

    mse=mean_squared_error(Y_test,pred)

    R2_SCORE.append(R2_score)

    MSE.append(mse)

    print("R2 Score",R2_score)

    print("MSE",mse)