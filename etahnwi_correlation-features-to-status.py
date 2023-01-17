# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
data.head()
data.size
data.gender.value_counts()
female = data.loc[data.gender == 'F']

male =  data.loc[data.gender == 'M']

female.groupby(['degree_t']).count()
female.status.value_counts()
female_placed = 48/76

female_placed
male.groupby(['degree_t']).count()
male.status.value_counts()
male_placed =100/139

male_placed
import seaborn as sns

import matplotlib.pyplot as plt



df =pd.DataFrame({'ratio': [female_placed,male_placed]}, index=['female','male'])

plot = df.plot.pie(y='ratio',figsize=(5,5))

print(df)
fStatusMajor = female.groupby(['degree_t','status']).size().reset_index()

fStatusMajor = fStatusMajor.rename(columns={0: 'count','degree_t':'major'})

fStatusMajor
sns.barplot(data=fStatusMajor, x='status', y='count', hue='major')
plt.scatter(data.degree_p,data.status)
data.head()
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
# Add outcome column, "successful" == 1, others are 0

data = data.assign(outcome=(data['status'] == 'Placed').astype(int))



y = data.outcome

variables = ['ssc_p','hsc_b','degree_p','mba_p','etest_p','degree_t']

X = data[variables]



data.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.30, random_state=1)



#numerical_cols = [col for col in X_train.columns if X_train[col].isnull.any]

s = (X_train.dtypes == 'object')

categorical_cols = list(s[s].index)



#PreProcessing fro numerical data. 



#numerical_transformer = SimpleImputer(strategy='constant')



#PreProcessing for categorcial data. 



categorical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=1000, random_state=42)



from sklearn.metrics import mean_absolute_error



#bundle preprocessing and modeling code



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



#Preprocessing of training data, fit model.

my_pipeline.fit(X_train,y_train)



preds = my_pipeline.predict(X_test)

score = mean_absolute_error(y_test,preds)

print(score)
from sklearn.linear_model import LogisticRegression



model2 = LogisticRegression()



my_pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model2', model2)

                             ])



my_pipeline2.fit(X_train,y_train)

preds = my_pipeline2.predict(X_test)

print(mean_absolute_error(y_test,preds))
data.head()
x_train, x_test, y_train, y_test = train_test_split(data[[

    'ssc_p','hsc_p','degree_p','workex','etest_p','mba_p'

]],data.outcome,test_size=0.1)

s = (x_train.dtypes == 'object')

object_cols = list(s[s].index)



from sklearn.preprocessing import LabelEncoder



label_x_train = x_train.copy()

label_x_test = x_test.copy()



# Apply label encoder to each column wiht categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_x_train[col] = label_encoder.fit_transform(x_train[col])

    label_x_test[col] = label_encoder.transform(x_test[col])



label_x_train.head()
from sklearn.linear_model import LogisticRegression







model = LogisticRegression()



model.fit(label_x_train,y_train)

model.predict(label_x_test)

model.score(label_x_test,y_test)
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators = 1000, random_state=42)

rf.fit(label_x_test,y_test)



rf.score(label_x_test,y_test)
import xgboost as xgb

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(label_x_train)

x_train_std = sc.transform(label_x_train)

x_test_std = sc.transform(label_x_test)



train = xgb.DMatrix(x_train_std, label=y_train)

test = xgb.DMatrix(x_test_std, label=y_test)



param = {

    'max_depth': 4,

    'eta': 0.3,

    'objective': 'multi:softmax',

    'num_class': 3

}

epochs =10



model3 = xgb.train(param,train,epochs)

predictions = model3.predict(test)

predictions