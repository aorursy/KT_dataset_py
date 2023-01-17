import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing, cross_validation

from sklearn.cluster import KMeans

from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/creditcard.csv')

df.head()
def plot_corr(df):

    #### Checking correlation of PCA variables

    corr = df.corr()

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    

    fig1, ax1 = plt.subplots()

    corrplot = sns.heatmap(corr, mask=mask, square=True, ax=ax1)

    plt.yticks(rotation=0)

    plt.xticks(rotation=90)



plot_corr(df)
def plot_time(df):

##### Checking the time series (fraud)

    df1 = df.loc[df['Class']==1, :]

    df2 = df.loc[df['Class']==0, :]

    

    fig2, ax2 = plt.subplots()

    x1 = df1['Time']

    y1 = df1['Amount']

    x2 = df2['Time']

    y2 = df2['Amount']

    ax2.scatter(x2, y2, alpha=0.2, color = 'blue')

    ax2.scatter(x1, y1, alpha=0.8, color='red')

    ax2.set_xlim((0, max(x2)))

    ax2.set_ylim((0, max(y2)))

    

plot_time(df)
def plot_dist(df):

    ##### Boxplot Violin plot to show distribution

    fig3, (ax3_1, ax3_2) = plt.subplots(ncols=2, sharey=True)

    sns.violinplot(x=df['Class'], y=df['Amount'], ax=ax3_1)

    sns.boxplot(x=df['Class'], y=df['Amount'], ax=ax3_2)

    

plot_dist(df)
def preproc(df):

    ##### Preprocessing

    X = np.array(df.drop(['Time', 'Class'], 1).astype(float))

    Y = np.array(df['Class'])

    

    X = preprocessing.scale(X)

    

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.3)



    return(X_train, X_test, Y_train, Y_test)



def mod_predict(x, y, model):

    prediction = model.predict(x)

    acc = abs(sum(prediction==y)/len(y))

    if acc <= 0.5:

        acc = 1 - acc

    return(acc)



def KMs(df):

    X_train, X_test, Y_train, Y_test = preproc(df)

    

    clf = KMeans(n_clusters=2)

    clf.fit(X_train)

    

    in_sample_acc = mod_predict(X_train, Y_train, clf)

    out_sample_acc = mod_predict(X_test, Y_test, clf)

    

    print('In-Sample accuracy : ' + str(round(in_sample_acc, 4)) +'\n' + 'Out-Sample accuracy : ' + str(round(out_sample_acc, 4)))



def LogReg(df):

    X_train, X_test, Y_train, Y_test = preproc(df)

    

    glm = linear_model.LogisticRegression()

    glm.fit(X_train, Y_train)

    

    in_sample_acc = mod_predict(X_train, Y_train, glm)

    out_sample_acc = mod_predict(X_test, Y_test, glm)

    

    print('In-Sample accuracy : ' + str(round(in_sample_acc, 4)) +'\n' + 'Out-Sample accuracy : ' + str(round(out_sample_acc, 4)))

    

def randomforest(df):

    X_train, X_test, Y_train, Y_test = preproc(df)

    

    rf = RandomForestClassifier()

    rf.fit(X_train, Y_train)

    

    in_sample_acc = mod_predict(X_train, Y_train, rf)

    out_sample_acc = mod_predict(X_test, Y_test, rf)

    

    print('In-Sample accuracy : ' + str(round(in_sample_acc, 4)) +'\n' + 'Out-Sample accuracy : ' + str(round(out_sample_acc, 4)))

    

    imp = rf.feature_importances_

    fig4, ax4 = plt.subplots()

    sns.barplot(list(range(len(imp))), imp)

    fig4.suptitle('PCA feature importance')

    

print('--------KMeans--------')

KMs(df)

print('--------Logistic Regression--------')

LogReg(df)

print('--------Random Forest--------')

randomforest(df)