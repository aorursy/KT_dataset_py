# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dframe = pd.read_csv('../input/mushrooms.csv')
dframe.head()
y = dframe['class']

X = dframe.drop('class',axis=1)
X.columns
X.info()
import seaborn as sns

import matplotlib as plt

%matplotlib inline
sns.countplot(x='cap-shape',data=dframe)
sns.countplot(x='cap-surface',data=dframe)
sns.countplot(x='cap-color',data=dframe)
sns.countplot(x='bruises',data=dframe)
sns.countplot(x='odor', data=dframe)
sns.countplot(x='gill-attachment',data=dframe)
sns.countplot(x='gill-spacing',data=dframe)
sns.countplot(x='gill-size',data=dframe)
sns.countplot(x='gill-color',data=dframe)
sns.countplot(x='stalk-shape',data=dframe)
sns.countplot(x='stalk-root',data=dframe)
sns.countplot(x='stalk-surface-above-ring',data=dframe)
sns.countplot(x='stalk-surface-below-ring',data=dframe)
sns.countplot(x='stalk-color-above-ring',data=dframe)
sns.countplot(x='stalk-color-below-ring',data=dframe)
sns.countplot(x='veil-type',data=dframe)
sns.countplot(x='veil-color',data=dframe)
sns.countplot(x='ring-number',data=dframe)
sns.countplot(x='ring-type',data=dframe)
sns.countplot(x='spore-print-color',data=dframe)
sns.countplot(x='population',data=dframe)
sns.countplot(x='habitat',data=dframe)
sns.countplot(x='class',data=dframe)
dd = dframe[dframe['stalk-root']=='?']
dd['stalk-root'].value_counts()  #values containing '?' i.e. missing values in stalk root
dframe['class'].value_counts().sum()
dframe = dframe[dframe['stalk-root'] != '?'] #creating new dataframe without missing walues because they cannot be replaceable.
dframe.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dframe['class'] = le.fit_transform(dframe['class']) #label encoding
dframe.head()
y = pd.DataFrame(dframe['class'],columns=['class'])
y.head()
X = dframe.drop('class',axis=1,inplace=True)
X = dframe
X.head()
y.head()
X_enc=pd.get_dummies(X)
X_enc.columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X_enc)
X = pd.DataFrame(X_std,columns=['cap-shape_b', 'cap-shape_c', 'cap-shape_f', 'cap-shape_k',

       'cap-shape_s', 'cap-shape_x', 'cap-surface_f', 'cap-surface_g',

       'cap-surface_s', 'cap-surface_y', 'cap-color_b', 'cap-color_c',

       'cap-color_e', 'cap-color_g', 'cap-color_n', 'cap-color_p',

       'cap-color_w', 'cap-color_y', 'bruises_f', 'bruises_t', 'odor_a',

       'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n', 'odor_p',

       'gill-attachment_a', 'gill-attachment_f', 'gill-spacing_c',

       'gill-spacing_w', 'gill-size_b', 'gill-size_n', 'gill-color_g',

       'gill-color_h', 'gill-color_k', 'gill-color_n', 'gill-color_p',

       'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y',

       'stalk-shape_e', 'stalk-shape_t', 'stalk-root_b', 'stalk-root_c',

       'stalk-root_e', 'stalk-root_r', 'stalk-surface-above-ring_f',

       'stalk-surface-above-ring_k', 'stalk-surface-above-ring_s',

       'stalk-surface-above-ring_y', 'stalk-surface-below-ring_f',

       'stalk-surface-below-ring_k', 'stalk-surface-below-ring_s',

       'stalk-surface-below-ring_y', 'stalk-color-above-ring_b',

       'stalk-color-above-ring_c', 'stalk-color-above-ring_g',

       'stalk-color-above-ring_n', 'stalk-color-above-ring_p',

       'stalk-color-above-ring_w', 'stalk-color-above-ring_y',

       'stalk-color-below-ring_b', 'stalk-color-below-ring_c',

       'stalk-color-below-ring_g', 'stalk-color-below-ring_n',

       'stalk-color-below-ring_p', 'stalk-color-below-ring_w',

       'stalk-color-below-ring_y', 'veil-type_p', 'veil-color_w',

       'veil-color_y', 'ring-number_n', 'ring-number_o', 'ring-number_t',

       'ring-type_e', 'ring-type_l', 'ring-type_n', 'ring-type_p',

       'spore-print-color_h', 'spore-print-color_k', 'spore-print-color_n',

       'spore-print-color_r', 'spore-print-color_u', 'spore-print-color_w',

       'population_a', 'population_c', 'population_n', 'population_s',

       'population_v', 'population_y', 'habitat_d', 'habitat_g', 'habitat_l',

       'habitat_m', 'habitat_p', 'habitat_u'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.decomposition import PCA
pca = PCA(n_components=90)
X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.pyplot.plot(var1)
pca1 = PCA(n_components=59)
X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1
var3.mean()  #variance of the groups captured
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
print("Logistic Regressor Accuracy Score:", lr.score(X_test, y_test)*100)
from sklearn.metrics import classification_report
pred = lr.predict(X_test)
print(classification_report(pred,y_test))