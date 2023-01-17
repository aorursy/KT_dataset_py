# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import cv2

import seaborn

from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split





from scipy.stats import skew



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('ggplot')
df = pd.read_csv('/kaggle/input/ai4all-project/results/classifier/lasso_s5/lassoRandomForest_probs.csv')

df.head()
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
#fill in mean for floats

for c in df.columns:

    if df[c].dtype=='float16' or  df[c].dtype=='float32' or  df[c].dtype=='float64':

        df[c].fillna(df[c].mean())



#fill in -999 for categoricals

df = df.fillna(-999)

# Label Encoding

for f in df.columns:

    if df[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(df[f].values))

        df[f] = lbl.transform(list(df[f].values))

        

print('Labelling done.')
from sklearn.model_selection import train_test_split

# Hot-Encode Categorical features

df = pd.get_dummies(df) 



# Splitting dataset back into X and test data

X = df[:len(df)]

test = df[len(df):]



X.shape
# Save target value for later

y = df.COVID19.values



# In order to make imputing easier, we combine train and test data

df.drop(['COVID19'], axis=1, inplace=True)

df = pd.concat((df, test)).reset_index(drop=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.model_selection import KFold

# Indicate number of folds for cross validation

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)



# Parameters for models

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LassoCV

# Lasso Model

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas = alphas2, random_state = 42, cv=kfolds))



# Printing Lasso Score with Cross-Validation

lasso_score = cross_val_score(lasso, X, y, cv=kfolds, scoring='neg_mean_squared_error')

lasso_rmse = np.sqrt(-lasso_score.mean())

print("LASSO RMSE: ", lasso_rmse)

print("LASSO STD: ", lasso_score.std())
# Training Model for later

lasso.fit(X_train, y_train)
from PIL import Image

im = Image.open("../input/ai4all-project/figures/classifier/lassoRandomForest_5gene_roc.png")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
df1 = pd.read_csv('/kaggle/input/ai4all-project/results/classifier/lasso_s10/lassoRandomForest_probs.csv')

df1.head()
train = '../input/ai4all-project/results/classifier/lasso_s10/lassoRandomForest_probs.csv'

test = '../input/ai4all-project/results/classifier/lasso_s10/lassoRandomForest_probs.csv'



df_train = pd.read_csv(train)

df_test = pd.read_csv(test)
def is_outlier(points, thresh = 3.5):

    if len(points.shape) == 1:

        points = points[:,None]

    median = np.median(points, axis=0)

    diff = np.sum((points - median)**2, axis=-1)

    diff = np.sqrt(diff)

    med_abs_deviation = np.median(diff)



    modified_z_score = 0.6745 * diff / med_abs_deviation



    return modified_z_score > thresh
#plt.style.use('dark_background')

target = df_train[df_train.columns.values[-1]]

target_log = np.log(target)



plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(target, bins=50)

plt.title('Covid-19 CZB')

plt.xlabel('COVID19')



plt.subplot(1,2,2)

sns.distplot(target_log, bins=50)

plt.title('Natural Log of Covid19 CZB')

plt.xlabel('Natural Log of COVID19')

plt.tight_layout()
df_train = df_train[df_train.columns.values[:-1]]

df = df_train.append(df_test, ignore_index = True)
cats = []

for col in df.columns.values:

    if df[col].dtype == 'object':

        cats.append(col)
df_cont = df.drop(cats, axis=1)

df_cat = df[cats]
for col in df_cont.columns.values:

    if np.sum(df_cont[col].isnull()) > 50:

        df_cont = df_cont.drop(col, axis = 1)

    elif np.sum(df_cont[col].isnull()) > 0:

        median = df_cont[col].median()

        idx = np.where(df_cont[col].isnull())[0]

        df_cont[col].iloc[idx] = median



        outliers = np.where(is_outlier(df_cont[col]))

        df_cont[col].iloc[outliers] = median

        

        if skew(df_cont[col]) > 0.75:

            df_cont[col] = np.log(df_cont[col])

            df_cont[col] = df_cont[col].apply(lambda x: 0 if x == -np.inf else x)

        

        df_cont[col] = Normalizer().fit_transform(df_cont[col].reshape(1,-1))[0]
for col in df_cont.columns.values:

    if np.sum(df_cont[col].isnull()) > 50:

        df_cont = df_cont.drop(col, axis = 1)

    elif np.sum(df_cont[col].isnull()) > 0:

        median = df_cont[col].median()

        idx = np.where(df_cont[col].isnull())[0]

        df_cont[col].iloc[idx] = median



        outliers = np.where(is_outlier(df_cont[col]))

        df_cont[col].iloc[outliers] = median

        

        if skew(df_cont[col]) > 0.75:

            df_cont[col] = np.log(df_cont[col])

            df_cont[col] = df_cont[col].apply(lambda x: 0 if x == -np.inf else x)

        

        df_cont[col] = Normalizer().fit_transform(df_cont[col].reshape(1,-1))[0]
df_new = df_cont.join(df_cat)



df_train = df_new.iloc[:len(df_train) - 1]

df_train = df_train.join(target_log)



df_test = df_new.iloc[len(df_train) + 1:]



X_train = df_train[df_train.columns.values[1:-1]]

y_train = df_train[df_train.columns.values[-1]]



X_test = df_test[df_test.columns.values[1:]]
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, False)



clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)

cv_score = np.sqrt(-cross_val_score(estimator=clf, X=X_train, y=y_train, cv=15, scoring = scorer))



plt.figure(figsize=(10,5))

plt.bar(range(len(cv_score)), cv_score)

plt.title('Cross Validation Score')

plt.ylabel('RMSE')

plt.xlabel('Iteration')



plt.plot(range(len(cv_score) + 1), [cv_score.mean()] * (len(cv_score) + 1))

plt.tight_layout()
df_dummies = pd.get_dummies(df_train['CZB_ID'])

df_dummies.head()
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

df_train["CZB_ID"] = encoder.fit_transform(df_train["CZB_ID"].fillna('Nan'))

#df_train["category_name"] = encoder.fit_transform(df_train["category_name"].fillna('Nan'))

df_train.head()
a = "'92.345'\r\n\r\n"

x = float(a[1:6])
unicode_value = u'"0.5"'



string_value = str(unicode_value)



float_value = float(string_value.strip('"'))



print ('float_value')
# Extract the training and test data

data = df.values

X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler

# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# Establish model

model = RandomForestRegressor(n_jobs=-1)
df1.isna().sum()
# Try different numbers of n_estimators - this will take a minute or so

estimators = np.arange(10, 200, 10)

scores = []

for n in estimators:

    model.set_params(n_estimators=n)

    model.fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))

plt.title("Effect of n_estimators")

plt.xlabel("n_estimator")

plt.ylabel("score")

plt.plot(estimators, scores)
scores
from PIL import Image

im = Image.open("../input/ai4all-project/figures/classifier/lassoRandomForest_10gene_roc.png")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
plt.style.use('dark_background')

def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set2')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("COVID19", "COVID19", df,4)