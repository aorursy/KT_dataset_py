import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

SEED = 42
def get_limits_mean_std(data):

    max_v = data.mean() + 2.5 * data.std()

    min_v = data.mean() - 2.5 * data.std()

    return max_v, min_v 
def plot_hist_box(data, column,  figsize = (12, 6)):

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize = figsize)



    max_v, min_v = get_limits_mean_std(data[column])



    ax0.axvline(data[column].mean(), 0, 1, color='r', label='mean')

    ax0.axvline(data[column].median(), 0, 1, color='g', label='median')

    ax0.axvline(max_v, 0, 1, color='y', label='uppper limit')

    ax0.axvline(min_v, 0, 1, color='y', label='bottom limit')

    ax0.legend()



    sns.distplot(data[column], color='b', bins=50, ax=ax0);

    sns.boxplot(x=column, data=data, orient='V', ax=ax1)

    plt.show()



def fit_score_models(models, X_train, X_test, y_train, y_test, folds, metric):

    models_score = []

    for model in models:

        model.fit(X_train, y_train)

        mean_score = cross_val_score(model, X_test, y_test, cv=folds, scoring=metric).mean()

        models_score.append((model, mean_score))

    return models_score



def normal_scaler(data, column):

    return (data[column] - data[column].mean()) / data[column].std()
df_raw = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df_raw.head()
df_raw.tail()
df_raw.shape
df_raw.describe().T
df_raw.info()
sns.pairplot(data=df_raw)
corr = df_raw.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



fig, ax = plt.subplots(figsize=(8,6))



sns.heatmap(data=corr, mask=mask, square=True, ax=ax);
fig, (ax0) = plt.subplots(1, 1, figsize = (12, 6))



sns.countplot(x=df_raw['Pregnancies'], color='b', ax=ax0);
plot_hist_box(df_raw, 'Pregnancies')
plot_hist_box(df_raw, 'Glucose')
plot_hist_box(df_raw, 'BloodPressure')
plot_hist_box(df_raw, 'SkinThickness')
plot_hist_box(df_raw, 'Insulin')
plot_hist_box(df_raw, 'BMI')
plot_hist_box(df_raw, 'DiabetesPedigreeFunction')
plot_hist_box(df_raw, 'Age')
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC 



lr =  LogisticRegression(random_state=SEED, max_iter = 1000)

sgd = SGDClassifier(random_state=SEED)

dtc = DecisionTreeClassifier(random_state=SEED)

svc = SVC(random_state=SEED)
y = df_raw['Outcome']

X = df_raw.drop(labels='Outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)



fit_score_models(

    models=[lr,

        sgd,

        dtc,

        svc

        ], 

    X_train=X_train,

    X_test=X_test, 

    y_train=y_train, 

    y_test=y_test, 

    folds=5,

    metric='f1')
df = df_raw.copy()

for col in df_raw.columns[:-1]:

    top_v, bot_v = get_limits_mean_std(df_raw[col])

    df[col] = df_raw[(df_raw[col] > bot_v) & (df_raw[col] < top_v)][col]



df.dropna(axis=0, inplace=True)

df.shape
y = df['Outcome']

X = df.drop(labels='Outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)



scaler = StandardScaler()

X_train_Scaled = scaler.fit_transform(X=X_train)

X_test_Scaled = scaler.transform(X=X_test)



fit_score_models(

    models=[lr,

        sgd,

        dtc,

        svc

        ], 

    X_train=X_train_Scaled,

    X_test=X_test_Scaled, 

    y_train=y_train, 

    y_test=y_test, 

    folds=5,

    metric='f1')
for col in df.columns[:-1]:

    print(f'Skew {col} - {df[col].skew()}')
df_ib = df.copy()

df_ib = df_ib[df_ib['Insulin'] > 0]

df_ib.shape
y = df_ib['Outcome']

X = df_ib.drop(labels='Outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)



scaler = StandardScaler()

X_train_Scaled = scaler.fit_transform(X=X_train)

X_test_Scaled = scaler.transform(X=X_test)



fit_score_models(

    models=[lr,

        sgd,

        dtc,

        svc

        ], 

    X_train=X_train_Scaled,

    X_test=X_test_Scaled, 

    y_train=y_train, 

    y_test=y_test, 

    folds=5,

    metric='f1')
from sklearn.preprocessing import power_transform

df_cox = df_ib.copy()
y = df_cox['Outcome']

X = power_transform(df_cox.drop(labels='Outcome', axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)



scaler = StandardScaler()

X_train_Scaled = scaler.fit_transform(X=X_train)

X_test_Scaled = scaler.transform(X=X_test)



fit_score_models(

    models=[lr,

        sgd,

        dtc,

        svc

        ], 

    X_train=X_train_Scaled,

    X_test=X_test_Scaled, 

    y_train=y_train, 

    y_test=y_test, 

    folds=5,

    metric='f1')