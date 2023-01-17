# Libraries setup

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.model_selection import train_test_split

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix, precision_recall_curve



sns.set()
params = {'legend.fontsize': 'x-large',

          'figure.figsize': (25, 15),

          'axes.labelsize': 'x-large',

          'axes.titlesize':'x-large',

          'xtick.labelsize':'x-large',

          'ytick.labelsize':'x-large'}



%matplotlib inline

plt.rcParams.update(params)
bk=pd.read_csv(r'../input/hotel_bookings.csv')
bk = pd.DataFrame(bk)
bk.head()
bk.shape
bk.isnull().sum() 
corrmat = bk.corr()

matrix = np.triu(bk.corr())

sns.heatmap(corrmat, square=True, annot=True, fmt='.1g', vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', 

            cbar=False)
bk.info()
count_classes = pd.value_counts(bk['hotel'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Hotel type")

plt.xlabel("type")

plt.ylabel("Frequency")
count_classes = pd.value_counts(bk['is_canceled'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Hotel type")

plt.xlabel("type")

plt.ylabel("Frequency")
plt.figure(1 , figsize = (20 , 6))

n = 0 

for x in ['is_canceled' , 'arrival_date_year' ,'stays_in_weekend_nights','stays_in_week_nights']:

    n += 1

    plt.subplot(1 , 4 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    sns.distplot(bk[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
bk = bk[bk.previous_cancellations<10] 

bk = bk[bk.booking_changes<10] 

bk = bk[bk.stays_in_weekend_nights<10]

bk = bk[bk.stays_in_week_nights<10]
bk1 = bk.loc[:,('previous_cancellations','booking_changes','stays_in_weekend_nights','stays_in_week_nights','is_canceled')]
bk1.head()
from sklearn.model_selection import train_test_split
X = bk1.iloc[:,0:4]

y = bk1.iloc[:,4]
X.head()
y.head()
y.shape
y.unique()
from sklearn import preprocessing

le = preprocessing.LabelBinarizer()
y = le.fit_transform(y)
y.shape
y = np.ravel(y)

y.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=123,stratify=y)
y_train.shape
def CMatrix(CM,labels =['Not_canceled','Canceled']):

    df = pd.DataFrame( data = CM, index = labels, columns = labels)

    df.index.name ='Real'

    df.columns.name = 'Forecast'

    df.loc['Total']= df.sum()

    df['Total']= df.sum(axis=1)

    return df 
from sklearn.linear_model import LogisticRegression

logistic_regression= LogisticRegression(solver = 'sag', max_iter = 5000)

log = logistic_regression.fit(X_train,y_train)



y_pred_test_log = logistic_regression.predict(X_test) 

acuracia_log = accuracy_score(y_pred=y_pred_test_log,y_true=y_test)

precisao_log = precision_score(y_pred=y_pred_test_log,y_true=y_test)

recall_log = recall_score(y_pred=y_pred_test_log,y_true=y_test)



CM= confusion_matrix(y_pred=y_pred_test_log,y_true=y_test)

CMatrix(CM).T
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=1234)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
count_classes = pd.value_counts(y_train_res, sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Booking_Histogram")

plt.xlabel("Booking_case")

plt.ylabel("Frequency")
from sklearn.linear_model import LogisticRegression

logistic_regression= LogisticRegression(solver = 'sag', max_iter = 5000)

log = logistic_regression.fit(X_train_res,y_train_res)



# teste com resultados after treino

y_pred_test_log = log.predict(X_test) 

acuracia_log = accuracy_score(y_pred=y_pred_test_log,y_true=y_test)

precisao_log = precision_score(y_pred=y_pred_test_log,y_true=y_test)

recall_log = recall_score(y_pred=y_pred_test_log,y_true=y_test)



CM= confusion_matrix(y_pred=y_pred_test_log,y_true=y_test)

CMatrix(CM)
import statsmodels.api as sm  

import statsmodels.formula.api as smf
formula = 'is_canceled ~ previous_cancellations+booking_changes+stays_in_weekend_nights+stays_in_week_nights'

model = smf.glm(formula, data=bk1, family=sm.families.Binomial())

result = model.fit()
print(result.summary())
z = np.exp(result.params)

print(z)