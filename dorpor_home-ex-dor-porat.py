#Setup Pandas and seaborn

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score

print("Setup Complete")
#Using Pandas to load the data 

data_h = pd.read_csv("../input/home-ex/home_exercise_data.csv") 

data_h.describe()
#Leave only the features and clean y_test nan as observed in the describe

data_h = data_h.dropna(axis=0, subset=["y_test"])

fig, ax = plt.subplots(ncols=4, figsize =(25,10))

sns.boxplot(x="y_test", y="x0",data=data_h, ax=ax[0])

sns.boxplot(x="y_test", y="x1" ,data=data_h, ax=ax[1])

sns.boxplot(x="y_test", y="x2", data=data_h, ax=ax[2])

sns.boxplot(x="y_test", y="x3", data=data_h, ax=ax[3])



fig2, ax2 = plt.subplots(ncols=4, figsize =(25,10))

sns.boxplot(x="y_test", y="x4" ,data=data_h, ax=ax2[0])

sns.boxplot(x="y_test", y="x5", data=data_h, ax=ax2[1])

sns.boxplot(x="y_test", y="x6", data=data_h, ax=ax2[2])

sns.boxplot(x="y_test", y="x7", data=data_h, ax=ax2[3])

plt.show()
# After looking at the plots from above we can tell about which feature effects whice y_test result

#checking for correlation

corr = data_h.drop(["y_test","y_conf", "y_pred","Unnamed: 0"], axis=1).corr()

print(corr)

sns.heatmap(corr,cmap="YlGnBu")
# because of the large corrolation (in abs) between x4, x5 and x6 we will want to ignore at least one of these variables to save time and memory.

# in order to detrmine which viriable is best to ignore - we will sum up all corralations (in abs) and ignore the one with the largest number

corr_sum = (corr.abs()).sum()

corr_sum[['x4','x5','x6']]
only_features = data_h.drop(["y_test","y_conf", "y_pred","Unnamed: 0","x5"], axis=1)

mean = only_features.mean(axis=1)

std = only_features.std(axis=1)

only_features.head()
#centering the data

only_features_centered = only_features.apply(lambda x: (x - mean)/std)

only_features_centered.head()
#using PCA to get more data on the model

#Reducing 8 dimensions to 3

pca = PCA(n_components = 3)

principalComponents = pca.fit_transform(only_features_centered)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2','principal component 3'])

principalDf.head()
finalDf = pd.concat([principalDf, data_h[['y_test']]], axis = 1)

finalDf.head()
ax = plt.axes(projection='3d')

import numpy as np





# Data for three-dimensional scattered points

fig = plt.figure(figsize=(8,8))

zdata = principalDf['principal component 1']

xdata = principalDf['principal component 2']

ydata = principalDf['principal component 3']

ax.scatter3D(xdata, ydata, zdata, c=data_h["y_test"])
# we can see that the yellow points(y_test = 3) are with a smaller amount than the rest and they are not next to the same gravity point

print(pca.explained_variance_ratio_)

# We lost some information here but we are close to the ideal 95%.

print(sum(pca.explained_variance_ratio_)) 
#y_conf in a graph

plt.hist(data_h['y_conf'])

plt.show()
# Creating Accuracy, Coverage and Threshold Arrays

# I am creating a 3d graph with the 3 axis being: threshold, accuracy and coverage



t_df = data_h[["y_pred","y_test","y_conf"]]

t_df['hit'] = np.where(t_df['y_pred']== t_df["y_test"], 1, 0)





def cov(threshold, df):

    count_cov = 0

    total_rows = 12250.0

    coverage_series = df.apply(lambda x: True if x['y_conf'] >= threshold else False , axis=1)

    for case in coverage_series:

        if case is True:

            count_cov += 1

    return float(count_cov)/total_rows

            



def acc(threshold, df):

    count_acc = 0

    count_cov = 0

    accurate_series_correct = df.apply(lambda x: True if  x['y_conf'] >= threshold and x["y_test"] == x["y_pred"] else False, axis=1)

    coverage_series = df.apply(lambda x: True if x['y_conf'] >= threshold else False , axis=1)

    

    for case in coverage_series:

        if case is True:

            count_cov += 1

            

    for case in accurate_series_correct:

        if case is True:

            count_acc += 1

    

    return float(count_acc)/float(count_cov)



cov_array = []

acc_array = []

threshold_array = []



for t in np.linspace(0,1,151):

    cov_array.append(cov(t,t_df))

    acc_array.append(acc(t,t_df))

    threshold_array.append(t)
#3d graph that has accuracy, coverage and the threshold. 

#this graph can give us nice insight regarding the wanted threshold

ax = plt.axes(projection='3d')

ax.scatter(acc_array, cov_array, threshold_array, c=threshold_array, cmap='viridis');
# coverage as a function of threshold

sns.scatterplot(x=cov_array, y=threshold_array)
# accuracy as a function of threshold

sns.scatterplot(x=acc_array, y=threshold_array)