import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import make_scorer
df_train_path = '/kaggle/input/eval-lab-1-f464-v2/train.csv'

df_test_path = '/kaggle/input/eval-lab-1-f464-v2/test.csv'

df_tr = pd.read_csv(df_train_path)

df_te = pd.read_csv(df_test_path)
df_tr.fillna(value=df_tr.mean(),inplace=True)

df_te.fillna(value=df_te.mean(),inplace=True)
def replace_type_0(row):

    if row.type == 'old':

        row.type = '0';

    return row



def replace_type_1(row):

    if row.type == 'new':

        row.type = '1';

    return row



df_tr = df_tr.apply((lambda x: replace_type_0(x)), axis = 1)

df_tr = df_tr.apply((lambda x: replace_type_1(x)), axis = 1)

df_te = df_te.apply((lambda x: replace_type_0(x)), axis = 1)

df_te = df_te.apply((lambda x: replace_type_1(x)), axis = 1)
sns.countplot(x = 'rating',data = df_tr)
sns.boxplot(x = 'type',y ='rating',data = df_tr)
sns.boxplot(x = 'rating',y ='feature1',data = df_tr)
sns.boxplot(x = 'rating',y ='feature2',data = df_tr)
sns.boxplot(x = 'rating',y ='feature3',data = df_tr)
sns.boxplot(x = 'rating',y ='feature4',data = df_tr)
sns.boxplot(x = 'rating',y ='feature5',data = df_tr)
sns.boxplot(x = 'rating',y ='feature6',data = df_tr)
sns.boxplot(x = 'rating',y ='feature7',data = df_tr)
sns.boxplot(x = 'rating',y ='feature8',data = df_tr)
sns.boxplot(x = 'rating',y ='feature9',data = df_tr)
sns.boxplot(x = 'rating',y ='feature10',data = df_tr)
sns.boxplot(x = 'rating',y ='feature11',data = df_tr)
### Based on the boxplots above, it is clear that no subset of the feature space is significantly helpful to separate the classes

### Therefore, we will not reduce the dimensions of the feature space and will train our model on all features
df = df_tr[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11","rating"]]

# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
feat = ['type','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']

X = df_tr[feat].copy()

y = df_tr['rating'].copy()

scaler = RobustScaler()

X_sc = scaler.fit_transform(X[feat])
reg1 = ExtraTreesRegressor(n_estimators = 4000,random_state = 0)

reg1.fit(X_sc,y.values.ravel())



X_te = df_te[feat]

X_te_sc_1 = scaler.transform(X_te[feat])

y_pred = reg1.predict(X_te_sc_1)

for i in range(len(y_pred)):

    y_pred[i] = round(y_pred[i])

new_id = df_te['id'].copy()

et_output = pd.DataFrame(list(zip(new_id,y_pred)), columns = ['id','rating'])

et_output.to_csv('First_Best.csv', index = False)
## Features were NOT scaled for this model

rf_model1 = RandomForestClassifier()

rf_model1.fit(X, y)



X_te = df_te[feat]

y_pred_2te = rf_model1.predict(X_te)

new_iddd = df_te['id'].copy()

rf_output1 = pd.DataFrame(list(zip(new_iddd,y_pred_2te)), columns = ['id','rating'])

rf_output1.to_csv('Second_Best.csv', index = False)