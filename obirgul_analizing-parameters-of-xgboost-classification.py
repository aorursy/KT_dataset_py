import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

plt.style.use('seaborn-whitegrid')



import xgboost as xgb



from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/capital-cities-feats/00 df.csv')
char= ['&','<','>','-','=','+',' ']

for col in df.select_dtypes(include ='object') :

    for i in char:

        df[col] = df[col].str.replace(i,'')

df.head()
for col in df:

    print (df[col].unique())
train = df[df['flag']=='train']

test = df[df['flag']=='test']
cat_feats = ['age_bin','capital_gl_bin','education_bin','hours_per_week_bin','msr_bin','occupation_bin','race_sex_bin']



y_train = train['y']

x_train = train[['age_bin','capital_gl_bin','education_bin','hours_per_week_bin','msr_bin','occupation_bin','race_sex_bin']]

x_train = pd.get_dummies(x_train,columns=cat_feats,drop_first=True)



y_test = test['y']

x_test = test[['age_bin','capital_gl_bin','education_bin','hours_per_week_bin','msr_bin','occupation_bin','race_sex_bin']]

x_test = pd.get_dummies(x_test,columns=cat_feats,drop_first=True)
results = []

max_depth = [x for x in range(1,10)]

for depth in max_depth:

    xgb_model = xgb.XGBClassifier(max_depth= depth)

    xgb_model.fit(x_train , y_train)

    y_pred_xgb = xgb_model.predict(x_test)

    accuracy = np.mean(y_test==y_pred_xgb)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, max_depth).plot(kind="bar",color="darkred",ylim=(0.7,0.9))
index_acc = pd.DataFrame({'accuracy': results, 'max_depth':max_depth})

print (index_acc.loc[index_acc.accuracy.idxmax(), 'max_depth'])
results = []

n_estimators = [x for x in range(100,200,10)]

for estimators in n_estimators:

    xgb_model = xgb.XGBClassifier(max_depth=5,n_estimators= estimators , objective="binary:logistic")

    xgb_model.fit(x_train, y_train)

    y_pred_xgb = xgb_model.predict(x_test)

    accuracy = np.mean(y_test==y_pred_xgb)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, n_estimators).plot(color="darkred",marker="o")
results = []

learning_rate = np.arange(0.1, 1, 0.1)

for rate in learning_rate:

    xgb_model = xgb.XGBClassifier(max_depth=5,n_estimators= 140 , learning_rate = rate, objective="binary:logistic")

    xgb_model.fit(x_train, y_train)

    y_pred_xgb = xgb_model.predict(x_test)

    accuracy = np.mean(y_test==y_pred_xgb)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, learning_rate).plot(color="darkred",marker="o")
results = []

tree_method = ['auto', 'exact', 'approx', 'hist'] # 'gpu_exact', 'gpu_hist'

for tree in tree_method:

    xgb_model = xgb.XGBClassifier(max_depth=5,n_estimators= 140 , learning_rate = 0.1, tree_method= tree, objective="binary:logistic")

    xgb_model.fit(x_train, y_train)

    y_pred_xgb = xgb_model.predict(x_test)

    accuracy = np.mean(y_test==y_pred_xgb)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, tree_method).plot(color="darkred",marker="o")
xgb_model = xgb.XGBClassifier(max_depth=5,n_estimators= 140 , learning_rate = 0.1, tree_method= 'auto', objective="binary:logistic")

xgb_model.fit(x_train, y_train)

y_pred_xgb = xgb_model.predict(x_test)
# To get list and number of xgb trees

dump_list=xgb_model.get_booster().get_dump()

len(dump_list)
#xgb.plot_tree(xgb_model,num_trees=0)

#plt.show()
predictions = [round(value) for value in y_pred_xgb] # If objective="logistic", predictions is unnecessary

accuracy_xgb = accuracy_score(y_test, predictions)

print(accuracy_xgb)
print('Classification report: \n',classification_report(y_test,y_pred_xgb))
xgb_cm = confusion_matrix(y_test,y_pred_xgb)

xgb_cm
# visualize with seaborn library

sns.heatmap(xgb_cm,annot=True,fmt="d") 

plt.show()