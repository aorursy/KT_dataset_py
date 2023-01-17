# System

import os

import sys



# Numerical

import numpy as np

from numpy import median

import pandas as pd





# NLP

import re



# Tools

import itertools



# Machine Learning - Preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Machine Learning - Model Selection

from sklearn.model_selection import GridSearchCV





# Machine Learning - Models

from sklearn import svm, linear_model

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor





# Machine Learning - Evaluation

from sklearn import metrics 

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score



# Plot

import matplotlib.pyplot as plt

import seaborn as sns



print(os.listdir("../input"))
df1 = pd.read_csv('../input/Admission_Predict.csv', sep='\s*,\s*', header=0, encoding='ascii', engine='python')

df2 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv', sep='\s*,\s*', header=0, encoding='ascii', engine='python')
df = df2

df.head()
df.describe()
def get_plt_params():

    params = {'legend.fontsize': 'x-large',

              'figure.figsize' : (18, 8),

              'axes.labelsize' : 'x-large',

              'axes.titlesize' : 'x-large',

              'xtick.labelsize': 'x-large',

              'ytick.labelsize': 'x-large',

              'font.size'      :  10}

    return params
all_columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance of Admit']

columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

target = 'Chance of Admit'
df['CGPA'] = df['CGPA'].apply(lambda x: ((x/10)*4))
# df.describe().plot(table = True)

# plt.title("General Statistics of Admissions")
# #https://www.kaggle.com/biphili/university-admission-in-era-of-nano-degrees



# f,ax=plt.subplots(1, 2, figsize=(18,6))

# df['Research'].value_counts().plot.pie(ax=ax[0],shadow=True)

# ax[0].set_title('Students Research')

# ax[0].set_ylabel('Student Count')

# sns.countplot('Research',data=df,ax=ax[1])

# ax[1].set_title('Students Research')

# plt.show()
fig = plt.figure(figsize=(22, 10))



params = get_plt_params()

plt.rcParams.update(params)

    

fig.subplots_adjust(hspace=1, wspace=.5)





for i in range(len(columns)-1):

    plt.subplot(2, 3, i+1)

    plt.title("Distribution/ Spread")

    val = df[columns[i]]

    ax = sns.boxplot(x=val)

    plt.xticks(fontsize=12)

    plt.yticks(fontsize=12)

    ax.set_title('Feature Value Spread')



plt.tight_layout()   
print("\n")

for i in range(len(columns)-1):

    print(" " + "*"*50)

    val = df[columns[i]]

    print(" " + columns[i])

    print(" " + "*"*50)

    print(" Minimum                        :{:.2f}".format(val.min()))

    print(" Maximum                        :{:.2f}".format(val.max()))

    print(" Percentile(25%)                :{:.2f}".format(val.quantile(0.25)))

    print(" Percentile(75%)                :{:.2f}".format(val.quantile(0.75)))

    print(" Percentile(50%)/ Median        :{:.2f}".format(val.quantile(0.55)))

    print(" Mean                           :{:.2f}".format(val.mean()))

    print(" Standard Deviation             :{:.2f}".format(val.std()))

    print(" " + "-"*50)

    print("\n")

    
fig = plt.figure(figsize=(18, 9))



params = get_plt_params()

plt.rcParams.update(params)



fig.subplots_adjust(hspace=0.5)



for i in range(len(columns)-1):

    plt.subplot(2, 3, i+1)

    sns.lineplot(x=columns[i], y=target, data=df)

plt.tight_layout()   

plt.plot()
fig = plt.figure(figsize=(18, 12))

fig.subplots_adjust(hspace=0.6)



params = get_plt_params()

plt.rcParams.update(params)



for i in range(len(columns)-1):

    plt.subplot(3, 3, i+1)

    sns.regplot(x=columns[i], y=target, data=df)

plt.tight_layout()   

plt.plot()
fig = plt.figure(figsize=(18, 12))



params = get_plt_params()

plt.rcParams.update(params)



g = sns.PairGrid(df[all_columns])

g.map(sns.lineplot);
sns.set(style="white")

fig = plt.figure(figsize=(18, 12))



d = df[all_columns]



params = get_plt_params()

plt.rcParams.update(params)





corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(len(all_columns)*2, len(all_columns)*2))



cmap = sns.diverging_palette(h_neg=220, h_pos=0, s=75, l=50, sep=10, n=len(all_columns), center='light', as_cmap=True)



ax = sns.heatmap(

    corr,

    cmap=cmap,

    center=0,

    robust=True,

    annot=True,

    linewidths=0.5,

    linecolor='white',

    cbar=True,

    cbar_kws={"shrink": .5},

    square=True,

    mask=mask)



plt.yticks(rotation=0)

plt.xticks(rotation=90)
def print_performance(model, X_test, y_test):

    preds = model.predict(X_test)



    explained_variance_score = metrics.explained_variance_score(y_test, preds)

    mean_absolute_error = metrics.mean_absolute_error(y_test, preds)

    mean_squared_log_error = metrics.mean_squared_log_error(y_test, preds)

    median_absolute_error = metrics.median_absolute_error(y_test, preds)

    r2_score = metrics.r2_score(y_test, preds)



    print(" " + "-"*55)

    print(" Performance")

    print(" " + "-"*55)

    print(" {} : {:.4f} ".format("Explained Variance Score ", explained_variance_score))

    print(" {} : {:.4f} ".format("Mean Absolute Error      ", mean_absolute_error))

    print(" {} : {:.4f} ".format("Mean Squared Error       ", mean_squared_log_error))

    print(" {} : {:.4f} ".format("Median Squared Error     ", median_absolute_error))

    print(" {} : {:.4f} ".format("R2 Score                 ", r2_score))

    print(" " + "-"*55)

    print("\n\n")

    

    return preds
X = df[columns].values

y = df[target].values



# split the data into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# Standardize the features

# scaler = StandardScaler()

scaler = preprocessing.MinMaxScaler()

scaler = scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
print("Training")

model = LinearRegression(normalize=True)

# model = svm.SVR(kernel='linear')

# model = linear_model.Ridge()

# model = BaggingRegressor(model, n_estimators=100)



model.fit(X_train, y_train)
print("\n Model Score: {:.2f}%\n".format(model.score(X_test, y_test)*100))



preds = print_performance(model, X_test, y_test)
diff = y_test-preds



print("Mean difference in prediction  : {:0.4f}".format(diff.mean()))

print("Median difference in prediction: {:0.4f}".format(median(diff)))



sns.lineplot(data=preds)

sns.lineplot(data=y_test)

plt.title("Actual vs. Estimated Admision Chance")

plt.show()

sns.lineplot(data=diff)

plt.title("Difference in estimation of admision chance")

plt.show()
GRE_Score = 313

TOEFL_Score = 102

University_Rating = 5

SOP = 3

LOR = 3

CGPA = 3.80

Research = 1



sample1 = [GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]

sample1 = np.array(sample1).reshape(1, -1)



sample1 = scaler.transform(sample1)



probab = model.predict(sample1)

print("Chance of admission: {:.2f}%".format(probab[0]*100))
# # ! pip install ipywidgets

# from ipywidgets import widgets

# from ipywidgets import interact, interactive, fixed, interact_manual





# while(True):

#     print("\n\nPlease Enter -1 to Exit or Enter to Continue: ")

#     b = input()

#     if b=='-1':

#         print("Exiting Admission Calcultor")

#         break

#     print("\nPlease Enter GRE Score: ")

#     GRE_Score = float(input())

#     print("\nPlease Enter TOEFL Score: ")

#     TOEFL_Score = float(input()) 

#     print("\nPlease Enter University Rating:")

#     University_Rating = float(input())  

#     print("\nPlease Enter SOP: ")

#     SOP = float(input()) 

#     print("\nPlease Enter LOR: ")

#     LOR = float(input())  

#     print("\nPlease Enter CGPA: ")

#     CGPA = float(input())  

#     print("\nPlease Enter Research: ")

#     Research = float(input())  



#     sample2 = [GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]

#     print(sample2)

#     sample2 = np.array(sample1).reshape(1, -1)



#     sample2 = scaler.transform(sample2)

#     print(sample2)

#     probab2 = model.predict(sample2)

#     print("Chance of admission: {:.2f}%".format(probab2[0]*100))

    