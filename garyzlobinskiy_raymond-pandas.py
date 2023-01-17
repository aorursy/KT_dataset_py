import pandas as pd
data1 = pd.DataFrame({'Column1':['val1', 'val2',2,3,4,"val5",6], 'Column2':['val3','val4',2,3,4,5,6]}, index=["a", "b", "c", "d", "e", "f", "g"])
data1.head()
data1['Column1']
data1
data1.index
data2 = pd.DataFrame([["Bob", 20],["Raymond", 10]], columns=['Name', 'Age'])
data2.head(50)
series1 = pd.Series([1,2,3], index=['a','b','c'], name="Apples")
series1
series1.head()
data1.shape
data1
columna = data1.Column1

columnb = data1['Column2']
data1["Column2"]['e']
data1.iloc[0]
data1.iloc[:,0]
data1.iloc[:4,0]
data1.iloc[1:4,1]
#data1.iloc[[1,4,5]:0]
data1.iloc[-5:]
data1.loc['b','Column2']
data1.loc[:,'Column2']
data1.index
data1.set_index("Column2")
insuranceData = pd.read_csv("../input/data-for-datavis/insurance.csv")
insuranceData.charges >= 2000
insuranceData.loc[insuranceData.charges >= 2000]
insuranceData.loc[(insuranceData.charges >= 10000) & (insuranceData.age >= 30)]
insuranceData.loc[insuranceData.age.isin([18,19,20,21,22,23,24])]
flightData = pd.read_csv("../input/data-for-datavis/flight_delays.csv", index_col="Month")
flightData.loc[flightData.US.notnull()]
spotifyData = pd.read_csv("../input/data-for-datavis/spotify.csv", index_col="Date", parse_dates=True)
spotifyData.loc[spotifyData['Unforgettable'].notnull()]
data1['Column3'] = 10
data1
data1['Column2'] = "hello"
data1
data1['Column2'] = range(0, len(data1), 1)
data1
def smoker():

    return insuranceData.loc[[2,3,14,102],'children']
smoker()
insuranceData['region'].describe()
insuranceData['age'].mean()
insuranceData['region'].unique()
insuranceData['region'].value_counts()
insuranceData['age'].max()
insuranceData['age'].min()
bmiMean = insuranceData['bmi'].mean()

remeanedBMIValues = insuranceData['bmi'].map(lambda p: p - bmiMean)

remeanedBMIValues
func = lambda a, b, c: a+b+c



func(1,2,3) # 6
def remeanBmi(row):

    row['bmi'] = row['bmi'] - insuranceData['bmi'].mean()

    return row

insuranceData.apply(remeanBmi, axis='columns')
insuranceData['bmi'] - insuranceData['bmi'].mean()
insuranceData['smoker-region'] = insuranceData['smoker'] + ',' + insuranceData['region']
insuranceData
insuranceData.age - insuranceData.age.mean()
data1.shape
data1
data1['Column3'] = data1['Column1']
data1
data1.drop(['Column3'], axis=1, inplace=True)
data1
data1.drop('a', axis=0, inplace=True)
data1
data1.loc['a']=['a','b','c']
data1