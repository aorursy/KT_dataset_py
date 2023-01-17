train_FileName = "../input/atividade-regressao-PMR3508/train.csv"

test_FileName = "../input/atividade-regressao-PMR3508/test.csv"
import pandas



ch_train = pandas.read_csv(train_FileName,

                                names=[

                                "ID", # unique ID of each place

                                "longitude",  # longitude of the place

                                "latitude", # latitude of the place

                                "median_age",  # meadian of the age of houses in the place

                                "total_rooms",  # total number of rooms in the area

                                "total_bedrooms",  # total number of bedrooms in the area

                                "population", # total population in the area

                                "households",  # total number of houses in the area

                                "median_income",  # median income of people in the area

                                "median_house_value" # median of values of houses in the area - Target variable

                                ], # names of columns

                                skiprows=[0], # skip first line (column names in csv), 0-indexed

                                sep=r'\s*,\s*',

                                engine='python',

                                na_values="?") # missing data identified by '?'

ch_train
missing_data_columns = ch_train.columns[ch_train.isnull().any()]

ch_train[ch_train.isnull().any(axis=1)][missing_data_columns]
total_size = ch_train.shape

missing_size = (ch_train[ch_train.isnull().any(axis=1)][missing_data_columns]).shape

print('A conclusao é que existem {} linhas com dados faltantes, presentes em {} colunas.'.format(missing_size[0],missing_size[1]))

print('Isso representa {:.2f}% do total de dados amostrados'.format(100-100*(total_size[0]-missing_size[0])/total_size[0]))
numerical_entries = ["longitude","latitude","median_age","total_rooms","total_bedrooms","population","households","median_income"] # only numerical data
# instalando seaborn versao 0.11.0 para ter acesso a funções extras

!pip install --upgrade seaborn==0.11.0



import seaborn

print(seaborn.__version__)

seaborn.set()
seaborn.pairplot(ch_train, vars=numerical_entries, hue='median_house_value',diag_kind='none')
all_entries = numerical_entries.copy()

all_entries.append("median_house_value")



ch_train[all_entries].describe(include='all')
import numpy

import matplotlib.pyplot as plt



correlation_matrix = (ch_train[all_entries].astype(numpy.int)).corr()



seaborn.set()

plt.figure(figsize=(15,7))

seaborn.heatmap(correlation_matrix, annot=True)
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="longitude",y="median_house_value")



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="latitude",y="median_house_value")



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="longitude",y="latitude",hue="median_house_value");
import geopy

import geopy.distance



geolocator = geopy.geocoders.Nominatim(user_agent="Google Maps")

bigcities = ["San Francisco","Los Angeles","San Diego"]

city_location = {}

for city in bigcities:

    city_geodata = geolocator.geocode(city)

    city_location[city] = (city_geodata.latitude,city_geodata.longitude)

city_location
def distante2closest_bigcity(geoposition):

    #            latitude       longitude

    location = (geoposition[0],geoposition[1])

    distance = None

    

    for city in bigcities:        

        current_distance = geopy.distance.distance(location, city_location[city]).km

        if distance is None:

            distance = current_distance

        elif distance > current_distance:

            distance = current_distance

    return distance



ch_train["distance_to_bigcity"] = ch_train[["latitude","longitude"]].apply(distante2closest_bigcity,axis=1)



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="longitude",y="latitude",hue="distance_to_bigcity");



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="distance_to_bigcity",y="median_house_value");
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="median_age",y="median_house_value");
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="total_rooms",y="median_house_value");



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="total_bedrooms",y="median_house_value");



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="population",y="median_house_value");



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="households",y="median_house_value");
seaborn.pairplot(ch_train, vars=["total_rooms","total_bedrooms","population","households"])
def rooms_per_bedroom(x):

    rooms = x[0]

    bedrooms = x[1]

    population = x[2]

    households = x[3]

    

    value = rooms/bedrooms

    

    return value



ch_train["total_rooms_per_bedroom"] = ch_train[["total_rooms","total_bedrooms","population","households"]].apply(rooms_per_bedroom,axis=1)



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="total_rooms_per_bedroom",y="median_house_value");
def rooms_per_capita(x):

    rooms = x[0]

    bedrooms = x[1]

    population = x[2]

    households = x[3]

    

    value = rooms/population

    value = min(value,4) # cap value off at 4 rooms / person



    return value



ch_train["total_rooms_per_population"] = ch_train[["total_rooms","total_bedrooms","population","households"]].apply(rooms_per_capita,axis=1)



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="total_rooms_per_population",y="median_house_value");
def rooms_per_household(x):

    rooms = x[0]

    bedrooms = x[1]

    population = x[2]

    households = x[3]

    

    value = rooms/households

    value = min(value,10) # cap value off at 10 rooms / house



    return value



ch_train["total_rooms_per_household"] = ch_train[["total_rooms","total_bedrooms","population","households"]].apply(rooms_per_household,axis=1)



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="total_rooms_per_household",y="median_house_value");
def bedrooms_per_population(x):

    rooms = x[0]

    bedrooms = x[1]

    population = x[2]

    households = x[3]

    

    value = bedrooms/population

    value = min(value,1) # cap value off at 1 bedroom / person

    

    return value



ch_train["total_bedrooms_per_population"] = ch_train[["total_rooms","total_bedrooms","population","households"]].apply(bedrooms_per_population,axis=1)



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="total_bedrooms_per_population",y="median_house_value");
def bedrooms_per_household(x):

    rooms = x[0]

    bedrooms = x[1]

    population = x[2]

    households = x[3]

    

    value = bedrooms/households

    value = min(value,2) # cap value off at 2 bedrooms / house

    

    return value



ch_train["total_bedrooms_per_household"] = ch_train[["total_rooms","total_bedrooms","population","households"]].apply(bedrooms_per_household,axis=1)



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="total_bedrooms_per_household",y="median_house_value");
def population_per_households(x):

    rooms = x[0]

    bedrooms = x[1]

    population = x[2]

    households = x[3]

    

    value = population/households

    value = min(value,6) # cap value off at 6 people / house

    

    return value



ch_train["total_population_per_household"] = ch_train[["total_rooms","total_bedrooms","population","households"]].apply(population_per_households,axis=1)



plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="total_population_per_household",y="median_house_value");
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=ch_train,x="median_income",y="median_house_value");
numerical_entries.extend(["distance_to_bigcity","total_rooms_per_bedroom","total_rooms_per_population","total_rooms_per_household","total_bedrooms_per_population","total_bedrooms_per_household","total_population_per_household"])

all_entries = numerical_entries.copy()

all_entries.append("median_house_value")

correlation_matrix = (ch_train[all_entries].astype(numpy.int)).corr()



seaborn.set()

plt.figure(figsize=(15,7))

seaborn.heatmap(correlation_matrix, annot=True)
ch_train
ch_train.describe(include='all')
import sklearn

import sklearn.preprocessing



scaler = sklearn.preprocessing.MinMaxScaler()



ch_train_X = pandas.DataFrame(scaler.fit_transform(ch_train[numerical_entries]), columns = numerical_entries)

ch_train_X
import sklearn.linear_model

import sklearn.model_selection

import sklearn.metrics



model_X = ch_train_X.values

model_Y = ch_train["median_house_value"].values



# use cross-validation to find optimal alpha

ls_reg_model = sklearn.linear_model.LassoCV(cv=5)

ls_reg = ls_reg_model.fit(model_X,model_Y)



print(ls_reg.intercept_)
i = 0

for entry in numerical_entries:

    print("{:30s} coefficient = {:+6.9f}".format(entry,ls_reg.coef_[i]))

    i = i+1
print("(Reference) R^2 = {:.4f}".format(ls_reg.score(model_X,model_Y)))
drop_columns = ["longitude",

                "latitude",

                #"median_age",

                "total_rooms",

                "total_bedrooms",

                "population",

                "households",

                #"median_income",

                #"distance_to_bigcity",

                "total_rooms_per_bedroom",

                "total_rooms_per_population",

                "total_rooms_per_household",

                #"total_bedrooms_per_population",

                "total_bedrooms_per_household",

                #"total_population_per_household"

               ]

model_X = ch_train_X.drop(drop_columns,axis=1).values
# use cross-validation to find optimal alpha

ls_reg_model = sklearn.linear_model.LassoCV(cv=5)

ls_reg = ls_reg_model.fit(model_X,model_Y)



print(ls_reg.intercept_)

print(ls_reg.coef_)

print("(Reference) R^2 = {:.4f}".format(ls_reg.score(model_X,model_Y)))
lr_reg = sklearn.linear_model.LinearRegression()



r2 = sklearn.model_selection.cross_val_score(lr_reg,model_X,model_Y,cv=10)

print("(Cross-validation) R^2 = {:.4f} +- {:.4f}".format(r2.mean(),r2.std()))

print()



#msle = sklearn.model_selection.cross_val_score(lr_reg,model_X,model_Y,cv=10,scoring="neg_mean_squared_log_error")

#rmsle = numpy.sqrt(-msle)

#print("(Cross-validation) RMSLE = {:.4f} +- {:.4f}".format(rmsle.mean(),rmsle.std()))

#print()



lr_reg.fit(model_X,model_Y)

predict_train = lr_reg.predict(model_X)



print("Intercept: ")

print(lr_reg.intercept_)

print("Coefficients: ")

print(lr_reg.coef_)

print()



print("(Trained Model) R^2 = {:.4f}".format(lr_reg.score(model_X,model_Y)))

#print()

#rmsle = numpy.sqrt(sklearn.metrics.mean_squared_log_error(model_Y,predict_train))

#print("(Trained Model) RMSLE = {:.4f}".format(rmsle))
comparison_df = pandas.DataFrame({'Predicted':predict_train,'Real':model_Y})

seaborn.jointplot(data=comparison_df,x='Real',y='Predicted');
model_X_lr = ch_train_X["median_income"].values.reshape(-1, 1)

lr_reg = sklearn.linear_model.LinearRegression() # reset



r2 = sklearn.model_selection.cross_val_score(lr_reg,model_X_lr,model_Y,cv=10)

print("(Cross-validation) R^2 = {:.4f} +- {:.4f}".format(r2.mean(),r2.std()))

print()



msle = sklearn.model_selection.cross_val_score(lr_reg,model_X_lr,model_Y,cv=10,scoring="neg_mean_squared_log_error")

rmsle = numpy.sqrt(-msle)

print("(Cross-validation) RMSLE = {:.4f} +- {:.4f}".format(rmsle.mean(),rmsle.std()))

print()



lr_reg.fit(model_X_lr,model_Y)

predict_train = lr_reg.predict(model_X_lr)



print("Intercept: ")

print(lr_reg.intercept_)

print("Coefficients: ")

print(lr_reg.coef_)

print()



print("(Trained Model) R^2 = {:.4f}".format(lr_reg.score(model_X_lr,model_Y)))

print()

rmsle = numpy.sqrt(sklearn.metrics.mean_squared_log_error(model_Y,predict_train))

print("(Trained Model) RMSLE = {:.4f}".format(rmsle))
comparison_df = pandas.DataFrame({'Predicted':predict_train,'Real':model_Y})

seaborn.jointplot(data=comparison_df,x='Real',y='Predicted');
alpha_grid = numpy.linspace(0,500,1000)

minvalue = []





for alpha in alpha_grid:

    rd_reg = sklearn.linear_model.Ridge(alpha=alpha)

    

    rd_reg.fit(model_X,model_Y)

    predict_train = rd_reg.predict(model_X)

    

    minvalue.append(min(predict_train))

    i = i+1

    

plt.figure(figsize=(15, 7))

plt.plot(alpha_grid,minvalue)

plt.plot([0,500],[0,0],'--')

plt.xlim([0,500])

plt.xlabel('Alpha')

plt.ylabel('Minimum Predicted Value')

plt.title('Analysis of "Alpha" for Ridge Regression')
alpha_opt = 310



rd_reg = sklearn.linear_model.Ridge(alpha=alpha_opt)



r2 = sklearn.model_selection.cross_val_score(rd_reg,model_X,model_Y,cv=10)

print("(Cross-validation) R^2 = {:.4f} +- {:.4f}".format(r2.mean(),r2.std()))

print()



msle = sklearn.model_selection.cross_val_score(rd_reg,model_X,model_Y,cv=10,scoring="neg_mean_squared_log_error")

rmsle = numpy.sqrt(-msle)

print("(Cross-validation) RMSLE = {:.4f} +- {:.4f}".format(rmsle.mean(),rmsle.std()))

print()



rd_reg.fit(model_X,model_Y)

predict_train = rd_reg.predict(model_X)



print("Intercept: ")

print(rd_reg.intercept_)

print("Coefficients: ")

print(rd_reg.coef_)

print()



print("(Trained Model) R^2 = {:.4f}".format(rd_reg.score(model_X,model_Y)))

print()

rmsle = numpy.sqrt(sklearn.metrics.mean_squared_log_error(model_Y,predict_train))

print("(Trained Model) RMSLE = {:.4f}".format(rmsle))
comparison_df = pandas.DataFrame({'Predicted':predict_train,'Real':model_Y})

seaborn.jointplot(data=comparison_df,x='Real',y='Predicted');
import sklearn.neighbors



accuracy_mean = []

accuracy_std = []



k_min = 1

k_max = 50





k_opt = None

max_acc = float('-inf')



i = 0

print('Finding best k...')

for k in range(k_min, k_max+1):

    kNN_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=k)

    

    accuracy  = sklearn.model_selection.cross_val_score(kNN_reg, model_X, model_Y, cv=5,scoring="neg_mean_squared_log_error")

    

    accuracy_mean.append(accuracy.mean())

    accuracy_std.append(accuracy.std())

    

    if accuracy_mean[i] > max_acc:

        k_opt = k

        max_acc = accuracy_mean[i]

    i += 1

    

    if(k%2==0 and k%4!=0):

        print('k = {:2d} - (-MSLE) = {:5.4f} | '.format(k,accuracy_mean[-1]),end='')

    elif(k%2==0):

        print('k = {:2d} - (-MSLE) = {:5.4f}'.format(k,accuracy_mean[-1]))

print('\nBest k: {}'.format(k_opt))
plt.figure(figsize=(15, 7))

plt.errorbar(numpy.arange(k_min, k_max+1), accuracy_mean, accuracy_std,

             marker='o', color='coral', markerfacecolor='purple', ecolor='purple',

             linewidth=3.0, elinewidth=1.5)

plt.plot([k_min-1,k_max+1],[max(accuracy_mean),max(accuracy_mean)],'--')

plt.xlim([k_min-1,k_max+1])

plt.xlabel('k')

plt.ylabel('-MSLE')

plt.title('Analysis of "k" for k-Nearest-Neighbors')
kNN_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=k_opt)



r2 = sklearn.model_selection.cross_val_score(kNN_reg,model_X,model_Y,cv=10)

print("(Cross-validation) R^2 = {:.4f} +- {:.4f}".format(r2.mean(),r2.std()))

print()



msle = sklearn.model_selection.cross_val_score(kNN_reg,model_X,model_Y,cv=10,scoring="neg_mean_squared_log_error")

rmsle = numpy.sqrt(-msle)

print("(Cross-validation) RMSLE = {:.4f} +- {:.4f}".format(rmsle.mean(),rmsle.std()))

print()



kNN_reg.fit(model_X,model_Y)

predict_train = kNN_reg.predict(model_X)



print("(Trained Model) R^2 = {:.4f}".format(kNN_reg.score(model_X,model_Y)))

print()

rmsle = numpy.sqrt(sklearn.metrics.mean_squared_log_error(model_Y,predict_train))

print("(Trained Model) RMSLE = {:.4f}".format(rmsle))
comparison_df = pandas.DataFrame({'Predicted':predict_train,'Real':model_Y})

seaborn.jointplot(data=comparison_df,x='Real',y='Predicted');
ch_test = pandas.read_csv(test_FileName,

                             names=[

                             "ID", # unique ID of each place

                             "longitude",  # longitude of the place

                             "latitude", # latitude of the place

                             "median_age",  # meadian of the age of houses in the place

                             "total_rooms",  # total number of rooms in the area

                             "total_bedrooms",  # total number of bedrooms in the area

                             "population", # total population in the area

                             "households",  # total number of houses in the area

                             "median_income",  # median income of people in the area

                             ], # names of columns

                             skiprows=[0], # skip first line (column names in csv), 0-indexed

                             sep=r'\s*,\s*',

                             engine='python',

                             na_values="?") # missing data identified by '?'

ch_test
missing_data_columns = ch_test.columns[ch_test.isnull().any()]

ch_test[ch_test.isnull().any(axis=1)][missing_data_columns]
total_size = ch_test.shape

missing_size = (ch_test[ch_test.isnull().any(axis=1)][missing_data_columns]).shape

print('A conclusao é que existem {} linhas com dados faltantes, presentes em {} colunas.'.format(missing_size[0],missing_size[1]))

print('Isso representa {:.2f}% do total de dados amostrados'.format(100-100*(total_size[0]-missing_size[0])/total_size[0]))
ch_test["distance_to_bigcity"] = ch_test[["latitude","longitude"]].apply(distante2closest_bigcity,axis=1)

ch_test["total_rooms_per_bedroom"] = ch_test[["total_rooms","total_bedrooms","population","households"]].apply(rooms_per_bedroom,axis=1)

ch_test["total_rooms_per_population"] = ch_test[["total_rooms","total_bedrooms","population","households"]].apply(rooms_per_capita,axis=1)

ch_test["total_rooms_per_household"] = ch_test[["total_rooms","total_bedrooms","population","households"]].apply(rooms_per_household,axis=1)

ch_test["total_bedrooms_per_population"] = ch_test[["total_rooms","total_bedrooms","population","households"]].apply(bedrooms_per_population,axis=1)

ch_test["total_bedrooms_per_household"] = ch_test[["total_rooms","total_bedrooms","population","households"]].apply(bedrooms_per_household,axis=1)

ch_test["total_population_per_household"] = ch_test[["total_rooms","total_bedrooms","population","households"]].apply(population_per_households,axis=1)



ch_test_X = pandas.DataFrame(scaler.transform(ch_test[numerical_entries]), columns = numerical_entries)

ch_test_X = ch_test_X.drop(drop_columns,axis=1)

ch_test_X
model_X = ch_test_X.values

predictions = kNN_reg.predict(model_X)
submission = pandas.DataFrame()

submission[0] = ch_test["ID"].values

submission[1] = predictions

submission.columns = ['Id','median_house_value']

submission.to_csv('submission.csv',index=False)