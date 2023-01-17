import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_validate

from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

%matplotlib inline
data=pd.read_csv('../input/airline-delay/DelayedFlights.csv')
data = data.sample(500000)
data.dtypes
data[['a','Year','Month','DayofMonth','DayOfWeek','CRSDepTime','CRSArrTime','FlightNum','Distance','Cancelled']]= data[['a','Year','Month','DayofMonth','DayOfWeek','CRSDepTime','CRSArrTime','FlightNum','Distance','Cancelled']].astype('int32')
data[['DepTime','ArrTime','ActualElapsedTime','CRSElapsedTime','AirTime','ArrDelay','DepDelay','TaxiIn','TaxiOut']]=data[['DepTime','ArrTime','ActualElapsedTime','CRSElapsedTime','AirTime','ArrDelay','DepDelay','TaxiIn','TaxiOut']].astype('float32')
data.head()
data_drop_uni = data.drop(columns=['a','Year','TailNum'],axis=1)
len(data.TailNum.value_counts().values)
data_drop_uni.isna().sum()
# np.array_equal(data_drop_uni[data_drop_uni.ArrTime.isna()].index,data_drop_uni[data_drop_uni.ActualElapsedTime.isna()].index) and np.array_equal(data_drop_uni[data_drop_uni.ArrTime.isna()].index,data_drop_uni[data_drop_uni.AirTime.isna()].index) and np.array_equal(data_drop_uni[data_drop_uni.ArrTime.isna()].index,data_drop_uni[data_drop_uni.ArrDelay.isna()].index) and np.array_equal(data_drop_uni[data_drop_uni.ArrTime.isna()].index,data_drop_uni[data_drop_uni.TaxiIn.isna()].index)



data_drop_uni.drop(index = data_drop_uni[data_drop_uni.ActualElapsedTime.isna()].index, inplace = True)

data_drop_uni.drop(columns=['CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay'],inplace=True)
data_drop_uni.isna().sum().sum()
data_drop_uni['Month'] = data_drop_uni['Month'].astype('object')

data_drop_uni['DayofMonth'] = data_drop_uni['DayofMonth'].astype('object')

data_drop_uni['DayOfWeek'] = data_drop_uni['DayOfWeek'].astype('object')
data_describe = data_drop_uni.describe(percentiles = [.001,.01,.25,.75,.95,.99])

data_describe
data.Diverted.value_counts()
data_drop_uni.drop(columns = ['Diverted'], inplace = True)
outlier_feature = ['ArrDelay','DepDelay','TaxiIn','TaxiOut']

for i in outlier_feature:

    q1 = data_describe[i]['25%']

    q3 = data_describe[i]['75%']

    iqr = q3-q1

    oulier1 = data_drop_uni[data_drop_uni[i]> q3 + 1.5*iqr].index

    oulier2 = data_drop_uni[data_drop_uni[i] < q1-1.5*iqr].index

    oulier = np.concatenate((oulier1, oulier2), axis=0)

    data_drop_uni.drop(oulier,inplace=True)

data_drop_uni.describe(percentiles = [.01,.75,.99])
data_drop_uni.shape
def classify(x):

    if x > 30:

        return 'Yes'

    else:

        return 'No'

data_drop_uni['Late'] = data_drop_uni.ArrDelay.apply(lambda x: classify(x))

data_classify = data_drop_uni.drop(columns='ArrDelay')
data_classify['Late'].value_counts()
labels = ['Yes','No']

values = [data_classify.Late.value_counts().Yes, data_classify.Late.value_counts().No]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
Late = data_classify[data_classify['Late'] == "Yes"]

Ontime = data_classify[data_classify['Late'] == "No"]
data_classify.dtypes
numeric_feature = data_classify.select_dtypes(exclude='object').columns.values

t_val = []

p_val = []

key_factor = []

for i in numeric_feature:

    t, p = stats.ttest_ind(Late[i],Ontime[i])

    t_val.append(t)

    p_val.append(p)

    key_fact = 'No'

    if(p < 0.05):

        key_fact = 'Yes'

    key_factor.append(key_fact)

d = {'name': numeric_feature, 't_val': t_val, 'p_val': p_val, 'Is_keyfactor': key_factor}

df = pd.DataFrame(data=d)

df.sort_values(by=['Is_keyfactor'],ascending=False)
categorical_feature = data_classify.select_dtypes(include='object').columns.values

chi2_val = []

p2_val = []

key_factor2 = []

for i in categorical_feature:

    chi2, p, dof, ex = stats.chi2_contingency(pd.crosstab(data_classify[i],data_classify['Late']))

    chi2_val.append(chi2)

    p2_val.append(p)

    key_fact2 = 'No'

    if(p < 0.05):

        key_fact2 = 'Yes'

    key_factor2.append(key_fact2)
fig = go.Figure(data=[go.Table(

    header=dict(values=['Name', 'Chi2_value','p_val','Is_keyfactor'],

                line_color='darkslategray',

                fill_color='lightskyblue',

                align='left'),

    cells=dict(values=[categorical_feature, # 1st column

                       chi2_val,

                       p2_val,

                       key_factor2], # 2nd column

               line_color='darkslategray',

               fill_color='lightcyan',

               align='left'))

])



fig.update_layout(width=800, height=380)

fig.show()
data_drop_uni
#for i in data_classify.select_dtypes(include='object').columns:

#    print(i,data_classify[i].unique())
X = data_classify.iloc[:,:-1]

Y = data_classify.iloc[:,-1]
X = pd.get_dummies(X)
X.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 ,random_state=1)
scaler = MinMaxScaler()

X_train.iloc[:, :12] = scaler.fit_transform(X_train.iloc[:, :12])

X_test.iloc[:,:12] = scaler.fit_transform(X_test.iloc[:, :12])
logistic = LogisticRegression(max_iter=1000, random_state=0)

logistic.fit(X_train,Y_train)
Y_pred = logistic.predict(X_test)
test_result = np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.values.reshape(len(Y_test),1)), axis=1)

test_result = pd.DataFrame(data = test_result, columns =['Y_Predict','Y_test'] )

test_result.head(30)
from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))