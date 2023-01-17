import pandas as pd
!ls ..
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

extraInfo = pd.read_csv('../input/store.csv')
df.head()
df.dtypes
extraInfo.head()
def _inp(dataframe):

    return dataframe.drop(columns=['Sales', 'Customers'])

_inp(df)
def _out(dataframe):

    return dataframe['Sales']

_out(df)
df = df[df.Open != 0]
df.shape
[(column, sum(df[column].isna())) for column in df.columns]
def splitDate(dataframe):

    dataframe = pd.concat([dataframe, dataframe['Date'].str.split('-', n = 2, expand = True)], axis=1, sort=False)

    dataframe.rename(columns={0:'date_year', 1:'date_month', 2:'date_day'}, inplace=True)

    return dataframe
from sklearn.preprocessing import LabelEncoder



def labelize(dataframe, column='StateHoliday'):

    labelizer = LabelEncoder()

    dataframe[column] = labelizer.fit_transform(dataframe[column].astype('str'))

    return dataframe
def formatDataframe(dataframe):

    dataframe = splitDate(dataframe)

    dataframe = labelize(dataframe)

    return dataframe
df = formatDataframe(df)

df.head()
def _inp(dataframe):

    return dataframe.drop(columns=['Sales', 'Customers', 'Date']).astype('float64')

_inp(df)
from sklearn.model_selection import train_test_split

df_train, df_validation = train_test_split(df, test_size=0.3)

df_validation, df_test = train_test_split(df_validation, test_size=0.15)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
m1 = Pipeline([

    ('normalizer', StandardScaler()),

    ('poli-features', PolynomialFeatures()),

    ('linear-model', LinearRegression())

])

m1.fit(_inp(df_train), _out(df_train))

m1.score(_inp(df_validation), _out(df_validation))
test = formatDataframe(test)

test.head()
[(column, sum(test[column].isna())) for column in test.columns]
test = test.fillna(1)
predictions = m1.predict(test.drop(columns=['Id','Date']).astype('float64'))
predictions.shape
pd.DataFrame(predictions).head()
test[['Id', 'Open']].head()
final_predictions = test[['Id', 'Open']]

final_predictions['Sales'] = pd.DataFrame(predictions)
final_predictions.loc[final_predictions['Open'] == 0, 'Sales'] = 0 #hardcode 0 sales for closed shops
final_predictions[['Id', 'Sales']].to_csv('predictions.csv', index = False)