#!pip install pandas

#!pip install -U scikit-learn



#!python -m pip install -U pip

#!python -m pip install -U matplotlib



#!pip install pandas



#!pip install seaborn
import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



from sklearn.model_selection import KFold

from sklearn.linear_model import RidgeCV

from sklearn.ensemble import RandomForestRegressor



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import minmax_scale

from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import PowerTransformer



import warnings

warnings.filterwarnings('ignore')



print('Pandas:  ' + pd.__version__)

print('Numpy:   ' + np.__version__)

print('Sklearn: ' + sklearn.__version__)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_blind = pd.read_csv('/kaggle/input/ml-for-exploration-geophysics-regression/test.csv')

df_train = pd.read_csv('/kaggle/input/ml-for-exploration-geophysics-regression/train.csv')
#df.head(10) #View first 10 data rows

#df.info()

df_train.describe()
df_blind.describe()
feature_names = df_train.columns[:-1].tolist()

print(feature_names)

label_names = df_train.columns[-1:].tolist()

print(label_names)
sns.set()

sns.set_style("white")

sns.pairplot(df_train, diag_kind="kde")
def make_log_plot(df):

    

    color_list = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

    feature_names = df.columns.tolist()  

    feature_num = len(feature_names)

    Depth = np.linspace(0,len(df[feature_names[0]]),len(df[feature_names[0]]))

   

    f, ax = plt.subplots(nrows=1, ncols=feature_num, figsize=(12, 12))



    for i in range(len(ax)):

        log = df[feature_names[i]]

        ax[i].plot(log, Depth, '-', color=color_list[i])

        ax[i].set_ylim(Depth.min(),Depth.max())

        ax[i].invert_yaxis()

        ax[i].grid()

        ax[i].locator_params(axis='x', nbins=3)

        ax[i].set_xlabel(feature_names[i])

        ax[i].set_xlim(log.min(),log.max())

        if i > 0:

            ax[i].set_yticklabels([]); 

    f.suptitle('Well logs', fontsize=14,y=0.94)
make_log_plot(df_train)
#sns.pairplot(df[['CAL', 'DTC']])

plt.scatter(df_train['HRD'], df_train['DTC'])

plt.show()
df_train_data = np.array(df_train)



X = df_train_data[:,:-1]

y = df_train_data[:,-1]



X_blind = df_blind.values



scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)

X_blind = scaler.transform(X_blind)



print('Size of the X_train dataset: ' + str(X.shape))

print('Size of the y_train dataset: ' + str(y.shape))

print('Size of the X_test dataset: ' + str(X_blind.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)



reg = RandomForestRegressor()



reg.fit(X_train,y_train)



#y_pred = reg.predict(X_test)
print(mean_squared_error(y_test, reg.predict(X_test), squared=False))
sample_submission = pd.read_csv('/kaggle/input/ml-for-exploration-geophysics-regression/sample_submission.csv')



new_submission = sample_submission

new_submission['DTC'] = y_pred



filename = 'new_submission_example.csv'

new_submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
plt.plot(y_pred, label='Predicted')

plt.xlabel('Sample')

plt.ylabel('DTC')

plt.title('DTC Prediction Comparison')

plt.legend(loc='lower right')

plt.show()