# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter

from imblearn.over_sampling import RandomOverSampler

from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.decomposition import PCA
train_data = pd.read_csv('../input/train.csv')

train_data = train_data.replace('?',np.NaN)

train_data.head()
miss_variables = ['Worker Class','Enrolled','MIC','MOC','Hispanic','MLU', 'Reason','Area','State','MSA','REG','MOVE','Live',

                 'PREV', 'Teen','Enrolled','COB FATHER','COB MOTHER','COB SELF','Fill']

for i in miss_variables:

    train_data[i].fillna(train_data[i].mode()[0] ,inplace= True)

    



features = train_data.drop(['ID', 'Class'], axis=1)

target = train_data['Class']
# features = pd.get_dummies(features, columns=list(features.select_dtypes(include=['object']).columns))

le = LabelEncoder()



for col in features.select_dtypes(include=['object']).columns:

    le.fit(features[col])

    features[col] = le.transform(features[col])

        

features.head()
# corr = features.corr(method="kendall")



# # Generate a mask for the upper triangle

# mask = np.zeros_like(corr, dtype=np.bool)

# mask[np.triu_indices_from(mask)] = True



# # Set up the matplotlib figure

# f, ax = plt.subplots(figsize=(22, 18))



# # Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# # Draw the heatmap with the mask and correct aspect ratio

# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

#             square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

features.drop(['Live', 'MSA','MOVE'], axis=1, inplace=True)
ros = RandomOverSampler(random_state=42)

features, target = ros.fit_resample(features, target)

# print('Resampled dataset shape %s' % Counter(target))
unscaled_data = pd.DataFrame(features)



scaled_data = StandardScaler().fit_transform(unscaled_data)

scaled_df=pd.DataFrame(scaled_data,columns=unscaled_data.columns)

scaled_df.head()
# pca = PCA(n_components=28)

# pca.fit(scaled_df)

# T1 = pca.transform(scaled_df)

# pca.explained_variance_ratio_.sum()

# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)



# param_grid = {'max_depth': [5,6,9,12], 'n_estimators': [10,100,1000]}



# model = RandomForestClassifier(n_estimators=100, max_depth=6, criterion='gini')

# kfold = KFold(n_splits=5, random_state=0, shuffle=True)

# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold)

# grid_result = grid.fit(features, target)



# means = grid_result.cv_results_['mean_test_score']

# stds = grid_result.cv_results_['std_test_score']

# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):

#     print("%f (%f) with: %r" % (mean, stdev, param))



# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#Train/Validation Set Split

X_train, X_test, y_train, y_test = train_test_split(scaled_df, target, test_size=0.2, random_state=42)



ada = AdaBoostClassifier(n_estimators=100)

model = BalancedBaggingClassifier(base_estimator=ada, n_estimators=10)



# model = RandomForestClassifier(n_estimators=100, max_depth=12, criterion='gini')



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Accuracy : %f' % roc_auc_score(y_test, y_pred))
model.fit(scaled_df, target)
test_data = pd.read_csv('../input/test.csv')

test_data = test_data.replace('?',np.NaN)

test_data.head()
test_data.info()

miss_variables = ['Worker Class','Enrolled','MIC','MOC','Hispanic','MLU', 'Reason','Area','State','MSA','REG','MOVE','Live',

                 'PREV', 'Teen','Enrolled','COB FATHER','COB MOTHER','COB SELF','Fill']

for i in miss_variables:

    test_data[i].fillna(test_data[i].mode()[0] ,inplace= True)

    

features = test_data.drop(['ID','Live','MSA','MOVE'], axis=1)
# features = pd.get_dummies(features, columns=list(features.select_dtypes(include=['object']).columns))

le = LabelEncoder()



for col in features.select_dtypes(include=['object']).columns:

    le.fit(features[col])

    features[col] = le.transform(features[col])

        

features.head()
scaled_data = StandardScaler().fit_transform(features)

scaled_df=pd.DataFrame(scaled_data,columns=features.columns)

scaled_df.head()
# pca = PCA(n_components=28)

# pca.fit(scaled_df)

# T1 = pca.transform(scaled_df)

# pca.explained_variance_ratio_.sum()
y_pred = model.predict(scaled_df)
s = pd.Series(y_pred)



df1 = pd.concat([test_data['ID'], s], axis=1)

df1.columns = ['ID', 'Class']

# df1.tail(10)



df1['Class'].value_counts()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(df1)