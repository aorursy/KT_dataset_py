import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import optimize


# take input data from Phil_loan_meanLabel.csv
def get_data(csv_file):
    df = pd.read_csv(csv_file)
    # drop columns that are not useful
    df = df.drop(['country_code',
                  'activity','use',
                  'country.x',
                  'region',
                  'currency',
                  'geo',
                  'lat',
                  'lon',
                  'mpi_region',
                  'mpi_geo',
                  'posted_time',
                  'disbursed_time',
                  'funded_time',
                  'geocode',
                  'tags',
                  'borrower_genders',
                  'LocationName',
                  'nn.idx',
                  'nn.dists',
                  'ISO',
                  'number',
                  'Unnamed: 0',
                  'forkiva',
                  'names',
                  'Partner.ID',
                  'Field.Partner.Name',
                  'Loan.Theme.ID',
                  'date',
                  'Loan.Theme.Type',
                  'amount',
                  'id',
                  'DHSCLUST',
                  'rural_pct'       #drop this col since it contains NaN
                  ], axis=1)
    df = pd.get_dummies(df, prefix=['sector.x','repayment_interval'])

    X = df.drop(['wealthscore'],axis=1)
    Y = df['wealthscore']
    X = X.values
    Y = np.array(Y)
    df = df.drop(['wealthscore'],axis=1)
    return [X,Y,df]

# find the mean and variance of each cluster
def find_cluster_mean_var():
    dhs = pd.read_csv("../input/philippines/Phil_DHS_info.csv")

    cluster_mean_var_dict = dict()

    num_cluster = 794

    for cluster_num in range(1, num_cluster+1):
        cur_cluster = dhs.loc[dhs["DHSCLUST"] == cluster_num]
        cur_cluster_mean = cur_cluster["wealthscore"].mean()
        cur_cluster_var = cur_cluster["wealthscore"].var()

        if (np.isnan(cur_cluster_mean)):
            cur_cluster_mean = 0.0
        if (np.isnan(cur_cluster_var)):
            cur_cluster_var = 1.0
        cluster_mean_var_dict[cluster_num] = (cur_cluster_mean, cur_cluster_var)
    return cluster_mean_var_dict






# based on the assumption that labels are assigned uniformly within a cluster
def linear_poverty():
    # training
    [X,Y,df] = get_data("../input/philippines/Phil_loan_meanLabel.csv")

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)


    regr = linear_model.LinearRegression()
    regr.fit(X_scaled,Y)

    predict_Y = regr.predict(X_scaled)

    df['naive_value'] = predict_Y

    df.to_csv('naive_value.csv')



# based on the assumption that labels are assigned normly within a cluster
def gaussian_poverty():
    cluster_mean_var_dict = find_cluster_mean_var()
    [X, Y, df] = get_data("../input/philippines/Phil_loan_meanLabel.csv")
    K = 794

    # find the gradient of loss function, take derivative then set to zero
    # optimize over w
    def derivative_m(w_1, m):
        K = 794
        result = 0
        for k in range(1, K+1):
            (mu_k, sig_k) = cluster_mean_var_dict[k]
            for i in range(0, N):
                if sig_k <= 1.0: sig_k = 1.0
                result += (X_scaled[i][m]**2 * w_1 - mu_k*X_scaled[i][m]) / sig_k
                
        return result

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)


    N = len(X_scaled)
    M = len(X_scaled[0])

    ws = []

    for m in range(0, M):
        root = optimize.newton(derivative_m, 50.0, tol=0.001, args = (m,))
        ws.append(root)

    print(ws)
    # ws = [760254.49989362864, 768486.12283061701, 363324.08014879079, 798248.12640516623, 40352.972485067876, 40352.972484749189, 40352.972484983307, 40352.972484801423, 40352.972484807746, 40352.972484819227, 40352.972483169637, 40352.972484795268, 40352.972484850201, 40352.972484722704, 40352.972484746431, 40352.972479315904, 40352.972484980615, 40352.972484980528, 40352.972484808932, 40352.972484769722, 40352.972485522754, 40352.972484685706]


    predict_Y = []

    for i in range(0, N):
        predict_Y.append(np.dot(X[i], ws))

    df['gaussian_value'] = predict_Y

    df.to_csv('gaussian_value.csv')

'''
if __name__ == '__main__':
    linear_poverty()
    gaussian_poverty()
'''

