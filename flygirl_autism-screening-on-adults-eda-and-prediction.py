import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn .metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier



seed = 143

np.random.seed(seed)



%matplotlib inline
df = pd.read_csv('../input/autism-screening-on-adults/autism_screening.csv')
len(df)
df.head()
df.columns
df.nunique()
df.drop(columns=['age_desc'], inplace=True)
df.isnull().sum()
df['age'].fillna(0, inplace=True)
df['Class/ASD'] = df['Class/ASD'].apply(lambda x: int(x == 'YES'))
sns.swarmplot(x='Class/ASD', y='result', data=df)
sns.distplot(df['age'])
cols = ['gender','jundice','austim','used_app_before']

for i in cols:

    sns.swarmplot(x='Class/ASD', y=i, data=df)

    plt.show()
fig = sns.barplot(y=df['contry_of_res'].value_counts().index[:15], x=df['contry_of_res'].value_counts().values[:15], data=df)

fig.set(xlabel='Count', ylabel='Country')

plt.show()
df['Class/ASD'].value_counts()
df = df.sample(frac=1, random_state=seed)

train_x,test_x,train_y,test_y = train_test_split(df.drop(columns=['Class/ASD']), df['Class/ASD'], test_size=0.2, random_state=seed)
cat_features = [i for i in df.columns if i not in ['Class/ASD', 'age', 'result']]



clf = CatBoostClassifier(

    iterations=10,

    verbose=5,

    class_weights = [1,2]

)



clf.fit(

    train_x, train_y ,

    cat_features=cat_features,

)
pred_y = clf.predict(test_x)
accuracy_score(pred_y, test_y)
f1_score(pred_y, test_y)
clf.get_feature_importance()