import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



from catboost import CatBoostClassifier, Pool

from catboost.utils import get_confusion_matrix

sns.set(style="darkgrid")
train = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
train.head()
train.info()
train.describe().T
train = train.drop(['id'], axis=1)
numerical = ['Age', 'Region_Code','Annual_Premium','Vintage']

sns.pairplot(train[numerical])
sns.distplot(train.Age)
sns.countplot(train.Response)
sns.countplot(train.Gender)
df_response = train.groupby(['Gender','Response'])['Age'].count().reset_index()

df_response = df_response.rename(columns={'Age':'Count'})

df_response
sns.catplot(x='Gender',y='Count',

            col='Response',

            data=df_response,

           kind='bar')
df_DL = train.groupby(['Gender'])['Driving_License'].count().reset_index()

df_DL = df_DL.rename(columns={'Driving_License':'Driving License'})

df_DL
sns.catplot(x='Gender',y='Driving License',

            data=df_DL,

           kind='bar')
sns.countplot(train.Previously_Insured)
df_PDL = train.groupby(['Gender','Previously_Insured'])['Age'].count().reset_index()

df_PDL = df_PDL.rename(columns={'Age':'count','Previously_Insured':'Previously Insured'})

df_PDL
sns.catplot(x='Gender',y='count',

            col='Previously Insured',

            data=df_PDL,

           kind='bar')
df_vehicleage = train.groupby(['Vehicle_Age','Response'])['Age'].count().reset_index()

df_vehicleage = df_vehicleage.rename(columns={'Age':'count','Vehicle_Age':'Vehicle Age'})

df_vehicleage
sns.catplot(x='Vehicle Age',y='count',

            col='Response',

            data=df_vehicleage,

           kind='bar')
df_damaged = train.groupby(['Vehicle_Damage','Response'])['Age'].count().reset_index()

df_damaged = df_damaged.rename(columns={'Age':'count','Vehicle_Damage':'Vehicle Damage'})

df_damaged
sns.catplot(x='Vehicle Damage',y='count',

            col='Response',

            data=df_damaged,

           kind='bar')
sns.distplot(train.Vintage)
train_copy = train.copy()



lb_make = LabelEncoder()

train_copy["Gender"] = lb_make.fit_transform(train_copy['Gender'])

train_copy['Vehicle_Age'] = lb_make.fit_transform(train_copy['Vehicle_Age'])

train_copy['Vehicle_Damage'] = lb_make.fit_transform(train_copy['Vehicle_Damage'])

train_copy.head()
features = train_copy.iloc[:,:-1]

labels = train_copy.iloc[:,-1:]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
eval_dataset = Pool(x_test,

                    y_test)



model = CatBoostClassifier(learning_rate=0.0001,

                           eval_metric='AUC')
model.fit(x_train,

          y_train,

          eval_set=eval_dataset)
print(model.get_best_score())
cm = get_confusion_matrix(model, eval_dataset)



fig = plt.figure(figsize=(12,7))

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

ax = sns.heatmap(cm, linewidths=0.5, linecolor='white',square=True)

plt.show()



print("catboost Accuracy : ", predict_accuracy_on_test_set*100)
test.head()
lb_make = LabelEncoder()

test["Gender"] = lb_make.fit_transform(test['Gender'])

test['Vehicle_Age'] = lb_make.fit_transform(test['Vehicle_Age'])

test['Vehicle_Damage'] = lb_make.fit_transform(test['Vehicle_Damage'])

test.head()
eval_test = Pool(test)

eval_test
pred = model.predict(eval_test)

pred.shape
submit = pd.DataFrame(index=test.index)

submit["id"] = test.id

submit["Response"] = pred

submit.set_index('id').reset_index(inplace=True)

submit.head()
submit.to_csv("Submission.csv")