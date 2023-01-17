import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

sns.set_style('whitegrid')



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots





from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from scipy.stats import uniform

from scipy import interp





from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans





#metrics

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from sklearn.metrics import silhouette_samples, silhouette_score

from bayes_opt import BayesianOptimization





import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/german-credit-data-with-risk/german_credit_data.csv')
#show data frame

df.head()
def show_info(data):

    print('DATASET SHAPE: ', data.shape, '\n')

    print('-'*50)

    print('FETURE DATA TYPES:')

    print(data.info())

    print('\n', '-'*50)

    print('NUMBER OF UNIQUE VALUES PER FEATURE:', '\n')

    print(data.nunique())

    print('\n', '-'*50)

    print('NULL VALUES PER FEATURE')

    print(data.isnull().sum())
show_info(df)
fig, ax = plt.subplots(1,3,figsize=(20,5))

plt.suptitle('DISTRIBUTION PLOTS')

sns.distplot(df['Credit amount'], bins=40, ax=ax[0]);

sns.distplot(df['Duration'], bins=40, ax=ax[1], color='salmon');

sns.distplot(df['Age'], bins=40, ax=ax[2], color='darkviolet');



fig, ax = plt.subplots(1,3,figsize=(20,5))

plt.suptitle('BOX PLOTS')

sns.boxplot(df['Credit amount'], ax=ax[0]);

sns.boxplot(df['Duration'], ax=ax[1], color='salmon');

sns.boxplot(df['Age'], ax=ax[2], color='darkviolet');
fig, ax = plt.subplots(1,2,figsize=(15,5))



sns.countplot(df['Sex'], ax=ax[0], palette='magma');

sns.countplot(df.Risk, ax=ax[1], palette='spring');
#Show basic stats

df[['Age', 'Duration', 'Credit amount']].describe()
fig, ax = plt.subplots(3,1,figsize=(10,10))

plt.suptitle('BIVARIATE ANALYSIS (HUE=SEX)', fontsize=20)

plt.tight_layout(2)



sns.lineplot(data=df, x='Age', y='Credit amount', hue='Sex', lw=2, ax=ax[0]);

sns.lineplot(data=df, x='Duration', y='Credit amount', hue='Sex', lw=2, ax=ax[1]);

sns.lineplot(data=df, x='Age', y='Duration', hue='Sex', lw=2, ax=ax[2]);

ig, ax = plt.subplots(3,1,figsize=(10,10))

plt.suptitle('BIVARIATE ANALYSIS (HUE=RISK)', fontsize=20)

plt.tight_layout(2)



sns.lineplot(data=df, x='Age', y='Credit amount', hue='Risk', lw=2, ax=ax[0], palette='deep');

sns.lineplot(data=df, x='Duration', y='Credit amount', hue='Risk', lw=2, ax=ax[1], palette='deep');

sns.lineplot(data=df, x='Age', y='Duration', hue='Risk', lw=2, ax=ax[2], palette='deep');
sns.pairplot(df);
fig, ax =plt.subplots(3,1,figsize=(10,10))

plt.tight_layout(2)



sns.countplot(df['Saving accounts'], hue=df.Risk, ax=ax[0], palette='Greens');

sns.boxenplot(df['Saving accounts'], df['Credit amount'], hue=df.Risk, ax=ax[1], palette='Greens');

sns.violinplot(df['Saving accounts'], df['Job'], hue=df.Risk, ax=ax[2], palette='Greens');
df.groupby('Saving accounts')[['Duration', 'Job', 'Credit amount']].describe().T
fig, ax =plt.subplots(3,1,figsize=(15,10))

plt.tight_layout(4)



for i in range(3):

    ax[i].set_xticklabels(ax[i].get_xticklabels(),rotation=10)





sns.countplot(df['Purpose'], hue=df.Risk, ax=ax[0], palette='muted');

sns.boxenplot(df['Purpose'], df['Credit amount'], hue=df.Risk, ax=ax[1], palette='muted');

sns.violinplot(df['Purpose'], df['Job'], hue=df.Risk, ax=ax[2], palette='muted');
fig, ax =plt.subplots(3,1,figsize=(10,10))

plt.tight_layout(2)



sns.countplot(df['Housing'], hue=df.Risk, ax=ax[0], palette='magma');

sns.boxenplot(df['Housing'], df['Credit amount'], hue=df.Risk, ax=ax[1], palette='magma');

sns.violinplot(df['Housing'], df['Job'], hue=df.Risk, ax=ax[2], palette='magma');
# replace null values with unknown

df= df.fillna('unknown')
#check the null values again

df.isnull().sum()
#drop the unnamed feature

df.drop('Unnamed: 0', axis=1, inplace=True)

categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']

#labelencode the categorical features

for i, cat in enumerate(categorical_features):

    df[cat] = LabelEncoder().fit_transform(df[cat])
#show new df

df.head()
num_df = df[['Age', 'Duration', 'Credit amount']]

num_df = np.log(num_df)
fig, ax = plt.subplots(1,3,figsize=(20,5))

plt.suptitle('DISTRIBUTION PLOTS AFTER LOG TRANSFORMATION')

sns.distplot(num_df['Credit amount'], bins=40, ax=ax[0]);

sns.distplot(num_df['Duration'], bins=40, ax=ax[1], color='salmon');

sns.distplot(num_df['Age'], bins=40, ax=ax[2], color='darkviolet');
scaler = StandardScaler()

num_df_scaled = scaler.fit_transform(num_df)
#show new values

print(num_df_scaled.shape)

num_df_scaled
inertias = []



for i in range(2,16):

    kmeans = KMeans(n_clusters=i, random_state=0).fit(num_df_scaled)

    inertias.append(kmeans.inertia_)



plt.figure(figsize=(10,5))

plt.title('ELBOW METHOD')

plt.plot(np.arange(2,16),inertias, marker='o', lw=2, color='steelblue');
results = []



for i in range(2,16):

    for r in range(20):

        kmeans = KMeans(n_clusters=i, random_state=r)

        c_labels = kmeans.fit_predict(num_df_scaled)

        sil_ave = silhouette_score(num_df_scaled, c_labels)

        results.append([i, r, sil_ave])

        

res_df = pd.DataFrame(results, columns=['num_cluster', 'seed', 'sil_score'])

pivot_kmeans = pd.pivot_table(res_df, index='num_cluster', columns='seed', values='sil_score')



plt.figure(figsize=(15,6))

plt.tight_layout

sns.heatmap(pivot_kmeans, annot=True, linewidths=0.5, fmt='.3f', cmap='magma', annot_kws={"size":8});
km = KMeans(n_clusters=3, random_state=0)

clusters = km.fit_predict(num_df_scaled)
#show a 3D plot of clusters

from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111, projection='3d')



for i in range(3):

    ax.scatter(num_df_scaled[clusters ==i,0], num_df_scaled[clusters ==i,1], num_df_scaled[clusters ==i,2])

    

fig, ax  = plt.subplots(1,3,figsize=(20,5))

sns.scatterplot(df['Duration'], df['Credit amount'], hue=clusters, ax=ax[0], palette='cividis');

sns.scatterplot(df['Age'], df['Credit amount'], hue=clusters, ax=ax[1], palette='cividis');

sns.scatterplot(df['Age'], df['Duration'], hue=clusters, ax=ax[2], palette='cividis');
df_clustered = df[['Age', 'Duration', 'Credit amount']]

df_clustered['cluster'] = clusters
df_clustered.groupby('cluster').mean()
num_df_scaled = pd.DataFrame(num_df_scaled, columns=['Age', 'Duration', 'Credit Amount'])

cat_df = df[categorical_features]



data = pd.concat([cat_df, num_df_scaled], axis=1)
#show new dataframe

data.head()
x = data.drop('Risk', axis=1)

y = data['Risk']



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=101)



print('xtrain shape: ', x_train.shape)

print('xtest shape: ', x_test.shape)

print('ytrain shape: ', y_train.shape)

print('ytest shape: ', y_test.shape)

#RandomSearchCV

# define the parameters to tune

param_dist = {"learning_rate": uniform(0, 2),

              "gamma": uniform(1, 0.000001),

              "max_depth": range(1,50),

              "n_estimators": range(1,300),

              "min_child_weight": range(1,10),

              'n_jobs': range(1,5)}

#instance of RandomSearchCV

rs = RandomizedSearchCV(XGBClassifier(), param_distributions=param_dist, n_iter=25) #25 iterations
rs.fit(x_train, y_train)
predictions = rs.predict(x_test)



print(classification_report(y_test, predictions))
def plot_roc(X, y, estemator,n_splits, lns = 100):

    #creating an instance of KFold

    kfold = StratifiedKFold(n_splits=n_splits,shuffle=False)

    #define estemator

    rf = estemator

    #deifne figuresize

    plt.rcParams['figure.figsize'] = (10,5)

    

    tprs = []

    aucs = []

    mean_fpr = np.linspace(0,1,lns)

    i = 1



    for train,test in kfold.split(X,y):

        #get prediction

        prediction = rf.fit(X.iloc[train],y.iloc[train]).predict_proba(X.iloc[test])

        #get the true pos. rate, false positive rate and thresh 

        fpr, tpr, t = roc_curve(y[test], prediction[:, 1])

        tprs.append(interp(mean_fpr, fpr, tpr))

        #get the area under the curve

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        #plot the tpr and fpr

        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i= i+1



    #plot the mean ROC

    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')

    mean_tpr = np.mean(tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color='gold',

    label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)



    #setup the labels

    plt.legend(bbox_to_anchor=(1, 1))

    plt.title('ROC PLOT', fontsize=16)

    plt.xlabel('False Positive Rate', fontsize=12)

    plt.ylabel('True Positive Rate', fontsize=12)
xgb_model = XGBClassifier()

xgb_model.set_params(**rs.best_params_)



plot_roc(x,y, xgb_model, n_splits=10)