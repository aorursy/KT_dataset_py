import os

print(os.listdir("../input/heart-disease-uci"))
from sklearn.neural_network import MLPClassifier

from sklearn.manifold import TSNE

from sklearn.svm import LinearSVC,NuSVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from time import time

from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("../input/heart-disease-uci/heart.csv")

df.head()
df.describe()

df[['cp', 'thal', 'slope','fbs','exang','target']] = df[['cp', 'thal', 'slope','fbs','exang','target']].astype('category')

sexdic = {0:'Female',1:'Male'}

df['sex'] = df['sex'].map(sexdic)
fig,axs = plt.subplots(1,2,figsize=(12,5))

sns.countplot(df['sex'],ax=axs[0]).set_title('SEX CATEGORY ')

sns.countplot(df['target'],hue=df['sex'],ax=axs[1]).set_title('HD FOR EACH CATEGORY')

plt.show()
Male_age_ = df[df['sex']=='Male']['age'] 

Fmale_age_ = df[df['sex']=='Female']['age'] 

fig, axs = plt.subplots(2, 2, figsize=(12, 4),sharex=True,gridspec_kw={"height_ratios": (0.2, 0.8)})

sns.boxplot(Male_age_, ax=axs[0,0]).set_title("Male's age Distribution")

sns.distplot(Male_age_,ax=axs[1,0])

sns.boxplot(Fmale_age_, ax=axs[0,1]).set_title("Female's age Distribution")

sns.distplot(Fmale_age_,ax=axs[1,1],color='Orange')

plt.show()
def find_outliers(data,var):

    Q1 = data[var].quantile(0.25)

    Q3 = data[var].quantile(0.75)

    treshold_height = Q3 + 1.5*(Q3 - Q1)

    treshold_low = Q1 - 1.5*(Q3 - Q1)

    return len(data[data[var] > treshold_height]),len(data[data[var] < treshold_low])

MOH,MOL   = find_outliers(df,'trestbps')

print(f"The size of the data: {df.shape[0]}")

print(f'Number outliers of resting blood pressure height:{MOH},low :{MOL}')
fig, axs = plt.subplots(2, 1, figsize=(12, 4),sharex=True)

sns.boxplot(df["trestbps"], ax=axs[0]).set_title("resting blood pressure  Distribution")

axs[0].set_xlabel(" ")

sns.boxenplot(df["trestbps"], ax=axs[1])

plt.show()
num_col = df[['age','trestbps','chol','thalach','oldpeak']]

corr_m = num_col.corr()

fig , axs = plt.subplots(1,1,figsize=(12,4))

sns.heatmap(corr_m,cmap='RdYlGn_r',fmt ='.2f%')

plt.show()
X_tsne  = TSNE(learning_rate=500,n_components=3).fit_transform(num_col)

X_pca   = PCA(n_components=3).fit_transform(num_col)

fig,axs = plt.subplots(1,2,figsize = (12,4))

axs[0].scatter(X_tsne[: ,0],X_tsne[:,1],c=df['target'])

axs[0].set_title('TENS')

axs[1].scatter(X_pca[: ,0],X_pca[:,1],c=df['target'])

axs[1].set_title('PCA')

plt.show()
sexdic = {'Female':0,'Male':1}

df['sex'] = df['sex'].map(sexdic)

X = df

y = df.pop('target')
def Process_data_fit_model(classifier,X,y):

    print("Start Prepross...")

    start = time()

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, shuffle=True)

    #Normalization

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)    

    X_test = scaler.transform(X_test)

    mmscaler = MinMaxScaler()

    X_train = mmscaler.fit_transform(X_train)    

    X_test = mmscaler.fit_transform(X_test)

    end = time()

    print("End Prepross ( {0:.2f} seconds )".format(end-start))

    

    print("Start Model fitting...")

    start = time()

    classifier.fit(X_train, y_train)

    end = time()

    print(f"End Model fitting {end-start} seconds...")

    

    print("\nPredicting...")

    start = time()

    y_predicted = classifier.predict(X_test)

    end = time()

    print(f"Model fitting {end-start} seconds...")

    print("\nPredicting...")

    start = time()

    y_predicted = classifier.predict(X_test)

    end = time()

    print(f"End Predicting {end-start} seconds...")

    

    print("\nReporting...\n")

    print(classification_report(y_test, y_predicted),"\n")

    print("Confusion matrix:\n")

    print(confusion_matrix(y_test, y_predicted),"\n")

    print("Cohen's Kappa score : ",cohen_kappa_score(y_test, y_predicted),"\n")

    

    if len(np.unique(y_test)) == 2:

        print("AUC score : {0:.3f}".format(roc_auc_score(y_test, y_predicted)))

        fpr, tpr, thresholds = roc_curve(y_test, y_predicted)

        plt.plot([0, 1], [0, 1], linestyle='--',color='Red')

        plt.plot(fpr, tpr, marker='*')

        plt.title("ROC Curve")

        plt.show()

    
dt = DecisionTreeClassifier(criterion='gini', max_depth=10)   

Process_data_fit_model(dt,X,y)
LR = LogisticRegression()

Process_data_fit_model(LR,X,y)
svclf = LinearSVC()

Process_data_fit_model(svclf,X,y)
mlp = MLPClassifier(activation='logistic', alpha=1e-03, batch_size=32,\

                    beta_1=0.9, beta_2=0.999, early_stopping=False,\

                    epsilon=1e-08, hidden_layer_sizes=(200,200,200),\

                    learning_rate='constant', learning_rate_init=0.0001,\

                    max_iter=500, momentum=0.9, n_iter_no_change=10,\

                    nesterovs_momentum=True, power_t=0.5,\

                    shuffle=True, solver='adam', tol=0.00001,\

                    validation_fraction=0.1, verbose=True, warm_start=False)

Process_data_fit_model(mlp,X,y)