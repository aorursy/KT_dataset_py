import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt, time

from operator import itemgetter



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



input_data = pd.read_csv("../input/avocado-prices/avocado.csv")

print(input_data.head())



regiongroups_data = {}

region_names = []



date_max_limit = dt.datetime.strptime("2020-01-01", '%Y-%m-%d')

date_max_limit = dt.datetime.timetuple(date_max_limit)

date_max_limit = time.mktime(date_max_limit)*0.001
for i in range(len(input_data)):

    new_line = input_data.iloc[i]

    region = new_line['region']



    #if region == "WestTexNewMexico":

    #   continue



    date_seconds = dt.datetime.strptime(new_line['Date'], '%Y-%m-%d')

    date_seconds = dt.datetime.timetuple(date_seconds)

    date_seconds = time.mktime(date_seconds)



    ave_price = new_line['AveragePrice']

    total_avocados = new_line['Total Volume']

    avo_type = new_line['type']



    if region not in regiongroups_data:

        regiongroups_data[region] = []

        region_names.append(region)



    t = float(date_seconds)#/date_max_limit



    regiongroups_data[region].append((t, ave_price, total_avocados, avo_type))
fig, a = plt.subplots(6, 9, figsize=(40,55))

n = 0

X_in = []

y_in = []



np_data_by_city = np.empty([len(region_names), int(len(regiongroups_data['Albany'])/2)])  # len town, #len of prices



for i in range(6):

    for j in range(9):

        idx = 9*n + j



        label = region_names[idx]

        a[i][j].set_title(label)



        date_conv_idx = [regiongroups_data[label][k][0] for k in range(len(regiongroups_data[label]))

                    if regiongroups_data[label][k][3] == 'conventional']



        date_orga_idx = [regiongroups_data[label][k][0] for k in range(len(regiongroups_data[label]))

                         if regiongroups_data[label][k][3] == 'organic']



        n_orga = len(date_orga_idx)

        n_conv = len(date_conv_idx)



        indices, date_idx_sorted = zip(*sorted(enumerate(date_conv_idx), key=itemgetter(1)))

        indices_sh, date_sh_idx_sorted = zip(*sorted(enumerate(date_orga_idx), key=itemgetter(1)))



        ave_price_conv_idx_sorted = []

        ave_price_orga_idx_sorted = []

        vol_conv_idx_sorted = []

        vol_orga_idx_sorted = []



        shift = len(indices)

        l = 0  # conv index

        p = 0  # org index

        for ix in range(shift):

            vol_orga = 0

            vol_conv = 0

            if date_idx_sorted[l] != date_sh_idx_sorted[p]:

                if n_orga > n_conv :

                    l -= 1

                if n_conv > n_orga :

                    p -= 1



            ave_price_conv_idx_sorted.append(regiongroups_data[label][indices[l]][1])

            vol_conv = regiongroups_data[label][indices[l]][2]

            ave_price_orga_idx_sorted.append(regiongroups_data[label][indices_sh[p] + shift][1])

            vol_orga = regiongroups_data[label][indices_sh[p] + shift][2]



            p += 1

            l += 1



            total_volume = float(vol_orga + vol_conv)

            price_weighted = float((ave_price_conv_idx_sorted[-1] * vol_conv

                                    + ave_price_orga_idx_sorted[-1] * vol_orga)) / total_volume



            # X_in, y_in are used in attempts to predict the city

            X_in.append([price_weighted, total_volume])

            y_in.append(label)



            # np_data_by_city is used for estimating correlation between the cities

            np_data_by_city[idx, ix] = price_weighted



        a[i][j].plot(date_idx_sorted, ave_price_conv_idx_sorted)

        a[i][j].plot(date_idx_sorted, ave_price_orga_idx_sorted)

    n += 1
fig, a = plt.subplots(6, 9, figsize=(40,55))

n = 0

for i in range(6):

    for j in range(9):

        idx = 9*n + j



        label = region_names[idx]

        a[i][j].set_title(label)

        a[i][j].plot(range(len(np_data_by_city[idx,:])), np_data_by_city[idx, :])

    n += 1

plt.show()
y_in = np.array(y_in)



scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

X_in_s = scaler.fit_transform(X_in)



# withinout encoder

X_train, X_test, y_train, y_test = train_test_split(X_in_s, y_in, test_size=0.2, shuffle=True)



for nn in range(2, 50, 2):

    model_knn = KNeighborsClassifier(n_neighbors=nn)

    model_knn.fit(X_train, y_train)



    y_pred = model_knn.predict(X_test)

    print("KNN accuracy for nn= {} is {};".format(nn, metrics.accuracy_score(y_test, y_pred)))
yenc = LabelEncoder()

y_in_1d = np.reshape(y_in, (y_in.size, 1))

y_in_enc = yenc.fit_transform(y_in_1d)

print(y_in_enc)



scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

X_in_s = scaler.fit_transform(X_in)



X_in_train, X_test, y_train, y_test = train_test_split(X_in_s, y_in_enc, test_size=0.1, shuffle=True)

print(X_in_train[0:1])



lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', max_iter=500, tol=1e-6)



lr_model.fit(X_in_train, y_train)



y_pred = lr_model.predict(X_test)



accuracy = np.sum(y_pred == y_test)/y_test.shape[0]

print("accuracy ", accuracy)
corrcoef = np.corrcoef(np_data_by_city)

plt.subplots(figsize=(20,15))

heatm = sns.heatmap(corrcoef, cbar=True, annot=True, fmt='.2f', #annot_kws={'size':15},

                    yticklabels=region_names, xticklabels=region_names)

plt.show()
clustm = sns.clustermap(corrcoef)

#clustm.ax_heatmap.set_xlabel(region_names)

#clustm.ax_heatmap.set_ylabel(region_names)

#plt.setp(clustm.ax_heatmap.get_ylabel(), rotation=0)

#clustm.ax_heatmap.set_ylabel(region_names, rotation=0)

plt.show()



print('Coeff corr all: ', corrcoef)
reg_names_np = np.asarray(region_names)

idx1 = np.where(reg_names_np == 'BuffaloRochester')

idx2 = np.where(reg_names_np == 'Pittsburgh')

idx3 = np.where(reg_names_np == 'Syracuse')



price_1 = np.mean(np_data_by_city[idx1, :])

price_2 = np.mean(np_data_by_city[idx2, :])

price_3 = np.mean(np_data_by_city[idx3, :])

print("Average price over time in BuffaloRochester is ", price_1)

print("Average price over time in Pittsburgh is ", price_2)

print("Average price over time in Syracuse is ", price_3)