import numpy as np, pandas as pd, matplotlib.pyplot as plt

import os

import seaborn as sns



from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression



from sklearn.preprocessing import MinMaxScaler



%matplotlib inline
grove_data = pd.read_csv("../input/groove-st-path-city-bookings/Grove St PATH_bookings.csv")
plt.figure(figsize=(20,20))

for yy in [2015,2016,2017,2018,2019,2020]:

    plt.subplot(6,1,yy-2014)

    plt.plot(grove_data[grove_data['YY'] == yy]['Bookings'].values)

    plt.xticks(())

    plt.xlabel(yy)

plt.show()
grove_data.info()
grove_data['Day'] = grove_data['Day'].astype(object)

grove_data['MM'] = grove_data['MM'].astype(object)

grove_data['DD'] = grove_data['DD'].astype(object)
grove_data.info()
grove_dummies = pd.get_dummies(grove_data)
grove_dummies.info()
plt.figure(figsize=(25,20))

sns.heatmap(grove_dummies.corr())

plt.show()
def tt_split(df, split_idx, test_days = None):

    train = df.iloc[:split_idx,:]

    

    if test_days == None:

        test = df.iloc[split_idx:,:]

    else:

        test_end_idx = split_idx + (test_days * 8) 

        test = df.iloc[split_idx:test_end_idx+1,:]

        '''

        I multiplied days with 8 as 8 rows are there for single day as 8 time slots.

        '''

        

        

    X_train = train.drop(columns = ["Bookings"]).values

    X_test = test.drop(columns = ["Bookings"]).values

    y_train = train["Bookings"].values

    y_test = test["Bookings"].values

    

    return X_train, X_test, y_train, y_test
def modelPerformance(model, X_train,X_test, y_train, y_test):

    print("====================================================================================")

    model.fit(X_train, y_train)

    

    train_score = model.score(X_train, y_train)

    test_score = model.score(X_test, y_test)

    

    train_predictions = model.predict(X_train)

    test_predictions = model.predict(X_test)

    

    train_rms = np.sqrt(np.sum(np.square(train_predictions - y_train)) / y_train.shape[0])

    test_rms = np.sqrt(np.sum(np.square(test_predictions - y_test)) / y_test.shape[0])

    

    train_mae = np.sum(np.sqrt(np.square(y_train - train_predictions))) / y_train.shape[0]

    test_mae = np.sum(np.sqrt(np.square(y_test - test_predictions))) / y_test.shape[0]

    

    

    

    perf_matrix = [

        ["Train", str(train_score), str(train_rms), str(train_mae)],

        ["Test", str(test_score), str(test_rms), str(test_mae)]

    ]

    print("Model: ",model,"\n\nPerformance and Predictions Trend: ")

    print(perf_matrix)

    plt.figure(figsize=(20,5))

    plt.plot(y_train, label = "Train Values")

    plt.plot(train_predictions, '--', label = "Predicted Train Values")

    plt.xlabel("Date")

    plt.ylabel("Rentals")

    plt.legend()

    plt.show()

    plt.figure(figsize=(20,5))

    plt.plot(y_test, label = "Test Values")

    plt.plot(test_predictions, '--', label = "Predicted Test Values")



    plt.xlabel("Date")

    plt.ylabel("Rentals")

    

    plt.table(cellText=perf_matrix,

        cellLoc="center", colWidths=None,

        rowLabels=None, rowColours=None, rowLoc="center",

        colLabels=["Data","R^2 Score","RMS Score","MAE Score"], colColours="yyyy", colLoc="center",

        loc='top', bbox=None)

    plt.legend()

    plt.box(True)

    plt.show()

    

    

    return model, test_predictions
def retrieveData(data, yy = None, mm = None, dd = None):

    data = data

    

    if yy != None:

        

        data = data[data['YY'] == yy]

        

        if mm != None:

            

            data = data[data['MM'] == mm]

            

            if dd != None:

                

                data = data[data['DD'] == dd]

                

    return data
yy = 2020

mm = 1

X_train, X_test, y_train, y_test = tt_split(grove_dummies,retrieveData(grove_data, yy, mm).index[0], 

                                            test_days = 31)

_, __ = modelPerformance(KNeighborsRegressor(n_neighbors=2),X_train,X_test,y_train,y_test)
scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
_, __ = modelPerformance(KNeighborsRegressor(n_neighbors=2), X_train_scaled, X_test_scaled, y_train, y_test)
def memoryFeatures(inpt, n_features):

    '''

    n_features will decide how many previous instances to be taken into account.

    '''

    

    outpt = np.zeros((n_features,n_features), dtype = int)

    for i in range(len(inpt)):

        row = np.array(inpt[i-n_features : i])

        outpt = np.append(outpt, row)

    del inpt

    return outpt.reshape(-1,n_features)
outpt = memoryFeatures(grove_data['Bookings'].values, 5)
pd.DataFrame(outpt).tail(10)
def genMemDataset(df, memory):

    return pd.concat([df, pd.DataFrame(memoryFeatures(df['Bookings'].values,

                                                      memory))], axis = 1, ignore_index=False)
models = [LinearRegression(),

          KNeighborsRegressor(n_neighbors=1),

          KNeighborsRegressor(n_neighbors=2), 

          KNeighborsRegressor(n_neighbors=3), 

          KNeighborsRegressor(n_neighbors=4),

          RandomForestRegressor(n_estimators=5), 

          RandomForestRegressor(n_estimators=10), 

          RandomForestRegressor(n_estimators=50),

          RandomForestRegressor(n_estimators=100),

          RandomForestRegressor(n_estimators=500)]
yy = 2020

mm = 1

X_train, X_test, y_train, y_test = tt_split(genMemDataset(grove_dummies, 8),retrieveData(grove_data, yy, mm).index[0], test_days = 31)



for model in models:

    

    _, __ = modelPerformance(model, X_train, X_test, y_train, y_test)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
for model in models:

    _, __ = modelPerformance(model, X_train_scaled, X_test_scaled, y_train, y_test)
memory_dataset = genMemDataset(grove_dummies, 8)
model = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',

          metric_params=None, n_jobs=None, n_neighbors=2, p=2,

          weights='uniform') 
def customPerformance(data, model, td, yy, mm, dd = None):

    yy = yy

    mm = mm

    dd = dd

    td = td

    spidx = retrieveData(grove_data, yy, mm, dd).index[0]

    print(spidx)

    X_train, X_test, y_train, y_test = tt_split(data, spidx, test_days = td)



    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)



    fitted_model, _ = modelPerformance(model,X_train,X_test,y_train,y_test)

    return fitted_model
customPerformance(memory_dataset, model, 7, 2020, 1)
customPerformance(memory_dataset, model, 7, 2020, 1, 15)
customPerformance(memory_dataset, model, 7, 2020, 2)
customPerformance(memory_dataset, model, 31, 2020, 3)
customPerformance(memory_dataset, model, 31, 2019, 6)
holidays = pd.read_csv("../input/groove-st-path-city-bookings/usholidays.csv")
holidays.head()
holidays = holidays['Date'].values
grove_holidays = []

for j in range(grove_data.shape[0]):

    i=0

    date = str()

    for el in grove_data.iloc[j,[0,1,2]].values:

        date = date+str(el)

        if i<2:

            date = date+'-'

        i=i+1

    date = str(pd.to_datetime([date])[0])[:10]

    if date in holidays:

        h = 1

    else:

        h = 0

    grove_holidays.append(h)

grove_holidays = np.array(grove_holidays)
grove_holidays = pd.concat([grove_data,pd.Series(grove_holidays, name = "Holiday")], axis = 1)
grove_holidays.info()
grove_holidays_dummies = pd.get_dummies(grove_holidays)
grove_holidays_dummies.columns
memory_dataset_holidays = genMemDataset(grove_holidays_dummies, 8)
plt.figure(figsize=(25,20))

plt.title("Pearson's Correlation Coefficient")

sns.heatmap(memory_dataset_holidays.corr(), annot=False)

plt.show()
customPerformance(memory_dataset_holidays, model, 31, 2019, 6)
fitted_model = customPerformance(memory_dataset_holidays, RandomForestRegressor(n_estimators=50),31, 2019, 6)
plt.figure(figsize=(20,5))

plt.title("Feaure Importances")

sns.barplot(memory_dataset_holidays.drop(columns = ['Bookings'

                                               ]).columns.astype(str).values,fitted_model.feature_importances_)

plt.xticks(rotation=90)

plt.plot()
customPerformance(memory_dataset_holidays, KNeighborsRegressor(n_neighbors = 5), 30, 2019, 6)
customPerformance(grove_holidays_dummies, KNeighborsRegressor(n_neighbors = 5), 30, 2019, 6)
def genPredictions(data, model, yy, mm, dd = None):

    yy = yy

    mm = mm

    dd = dd

    

    if mm in [1,3,5,7,8,10,12]:

        td = 31

    elif mm in [4,6,9,11]:

        td = 30

    else:

        if yy%4 == 0:

            td = 29

        else:

            td = 28

            

    spidx = retrieveData(grove_data, yy, mm, dd).index[0]

    print("Test Data taking from:", str(yy) + '-' + str(mm), "for",td, "days | Split at index:", spidx)

    X_train, X_test, y_train, y_test = tt_split(data, spidx, test_days = td)



    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)



    fitted_model, predictions = modelPerformance(model,X_train,X_test,y_train,y_test)

    return predictions, y_test
preds, true = genPredictions(memory_dataset_holidays, KNeighborsRegressor(n_neighbors = 5), 2020, 1)
def timeline_forecast(yy_start, mm_start, yy_end, mm_end):

    yy = yy_start

    mm = mm_start

    

    forecasts = np.array([])

    true_vals = np.array([])

    

    while True:

        preds, true = genPredictions(memory_dataset_holidays, KNeighborsRegressor(n_neighbors = 5), yy, mm)



        mm = mm + 1

        

        if mm == 13:

            mm = 1

            yy = yy + 1

            

        forecasts = np.append(forecasts, preds)

        true_vals = np.append(true_vals, true)

        

        if yy>=yy_end and mm>mm_end:

            break

            

            

    return forecasts, true_vals
f, t = timeline_forecast(2019,1,2020,5)
plt.figure(figsize=(30,10))

plt.plot(f)

plt.plot(t)
pd.DataFrame(f).to_csv("timeline_forecasts.csv")

pd.DataFrame(t).to_csv("timeline_true_vals.csv")