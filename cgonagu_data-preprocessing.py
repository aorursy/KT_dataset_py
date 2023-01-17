import pandas as pd
import numpy as np

dataTrain = pd.read_csv('../input/train.csv', encoding='iso-8859-1')
dataTest = pd.read_csv('../input/test.csv', encoding='iso-8859-1')
fullData = pd.concat([dataTrain, dataTest], ignore_index=True, sort=False)


print(dataTrain.shape)
print(dataTest.shape)

for column in fullData.columns:
    if fullData[column].isna().sum() > 0:
        print(column)
        print('Número de missings en la variable: %d' % (fullData[column].isna().sum()))
        print()

fullData['EDAD'] = fullData['EDAD'].fillna(dataTrain['EDAD'].mean())
fullData['PROVEEDOR'] = fullData['PROVEEDOR'].fillna('SIN MAIL')
fullData['DOMINIO'] = fullData['DOMINIO'].fillna('SINDOMINIO')
fullData['R_VOCCONS_MISSING'] = fullData['R_VOCCONS'].isna().astype(int)
fullData['R_VOCCONS'] = fullData['R_VOCCONS'].fillna(0)
fullData['LON_MAIL1_MISSING'] = fullData['LON_MAIL1'].isna().astype(int)
fullData['LON_MAIL1'] = fullData['LON_MAIL1'].fillna(0)
columnsToDrop = [column for column in dataTrain.columns if dataTrain[column].nunique() == 1]
print(columnsToDrop)
fullData = fullData.drop(columnsToDrop, axis=1)
for column in fullData.columns:
    if fullData[column].dtype == 'object':
        print(column, dataTrain[column].nunique())
for column in fullData.columns.drop(['ID', 'FRAUDE', 'HORA']): 
    if (fullData[column].dtype == 'object'):
        if (dataTrain[column].nunique() < 40):
            fakeDf = pd.get_dummies(fullData[column], prefix=column)     
            fullData = fullData.drop(column, axis=1)
            fullData = fullData.join(fakeDf)
        else: 
            fakeDf = dataTrain.groupby(column)['ID'].count().reset_index()
            fakeDf = fakeDf[fakeDf['ID'] > 500]
            listValues = fakeDf[column].tolist()
            fullData[column].loc[~(fullData[column].isin(listValues))] = 'RESTO'
            fakeDf2 = pd.get_dummies(fullData[column], prefix=column)     
            fullData = fullData.drop(column, axis=1)
            fullData = fullData.join(fakeDf2)
fullData['HORA'] = fullData['HORA'].str.partition(':')
fullData['HORA'] = fullData['HORA'].astype(int)
corrMatrix = fullData[~(fullData['FRAUDE'].isna())].corr().abs()
upperMatrix = corrMatrix.where(np.triu(
                               np.ones(corrMatrix.shape),
                               k=1).astype(np.bool))
correlColumns = [c for c in upperMatrix.columns
                 if any(upperMatrix[c] > 0.98)]
print(correlColumns)
fullData = fullData.drop(correlColumns, axis=1)
fullData.to_csv('fullData_preProcessed.csv', index=False)