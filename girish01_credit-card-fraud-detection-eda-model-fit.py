import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data.describe()
data.info()
data.shape
percentage_fraud=data['Class'].value_counts()/len(data)*100
percentage_fraud
print('Transactions without fraud in dataset are {}%'.format(round(percentage_fraud[0],2)))

print('Transactions with fraud detected in dataset are {}%'.format(round(percentage_fraud[1],2)))
import matplotlib.pyplot as plt

import matplotlib as mlt

import seaborn as sns

%matplotlib inline

plt.rcParams["figure.figsize"]=(16,9)

plt.style.use('fivethirtyeight')


sns.countplot('Class',data=data,palette='RdBu')

plt.xticks([0,1],('No Fraud','Fraud'))

plt.show()
Fraud=data[data['Class']==1]

Normal=data[data['Class']==0]
Fraud.shape
Normal.shape
# Analyzing Time and Amount features

fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)

amount_val=data['Amount'].values

Time_val=data['Time'].values



bins=50



ax1.hist(Fraud.Time,bins=bins)

ax1.set_title('Time Analysis of Fraud Transactions')



ax2.hist(Normal.Time,bins=bins)

ax2.set_title('Time Analysis of Normal Distribution')



plt.xlabel('Time(in seconds)')

plt.ylabel('Number of transaction')



plt.show()
# Analyse fraud and normal transacions distribution through summary statistics

print('Fraud time analysis','\n',Fraud.Time.describe(),'\n','Normal Time Analysis','/n',Normal.Time.describe())
print('Amount info of fraud transaction','\n',Fraud.Amount.describe())

print('Amount infor of normal transaction \n',Normal.Amount.describe())
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)



bins=10

ax1.hist(Fraud.Amount,bins=bins)

ax1.set_title('Amount transacted in Fraudalent transactions')



ax2.hist(Normal.Amount,bins=bins)

ax2.set_title('Amount transacted in Normal transaction')



plt.xlabel('Amount')

plt.ylabel('Transactions')

plt.yscale('log')



plt.show()
# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

df=data.copy()



data.drop(['Time','Amount'], axis=1, inplace=True)
data.head()
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold,KFold



print('No Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')

y=data['Class']

X=data.drop('Class',axis=1)

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-' * 100)

print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
#Shuffle the data first to random selection of samples.



data=data.sample(frac=1)



fraud=data[data['Class']==1]

normal=data[data['Class']==0][:492]



normal_distributed_df = pd.concat([fraud, normal])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



new_df.head()
print('Distribution of the Classes in the subsample dataset')

print(new_df['Class'].value_counts()/len(new_df))







sns.countplot('Class', data=new_df, palette='RdBu')

plt.title('Equally Distributed Classes', fontsize=14)

plt.xticks([0,1],('Non_fraud','Fraud'))

plt.show()
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(24,36))

corr_df=df.corr()

sns.heatmap(corr_df,cmap='hot',ax=ax1)

ax1.set_title('Imbalanced dataset correlation')



corr_mat=new_df.corr()



sns.heatmap(corr_mat,cmap='hot',ax=ax2)

ax2.set_title('New truncated dataset correlation')

fig.tight_layout()

plt.show()

print(new_df.corr()['Class'].drop('Class').sort_values(ascending=False).head())

print(new_df.corr()['Class'].drop('Class').sort_values(ascending=False).tail())
f, axes = plt.subplots(ncols=4, figsize=(20,4),sharex=True)



# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)

sns.boxplot(x="Class", y="V17", data=new_df, palette='YlGn', ax=axes[0])

axes[0].set_title('V17 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V14", data=new_df, palette='YlGn', ax=axes[1])

axes[1].set_title('V14 vs Class Negative Correlation')





sns.boxplot(x="Class", y="V12", data=new_df, palette='YlGn', ax=axes[2])

axes[2].set_title('V12 vs Class Negative Correlation')





sns.boxplot(x="Class", y="V10", data=new_df, palette='YlGn', ax=axes[3])

axes[3].set_title('V10 vs Class Negative Correlation')



plt.xticks([0,1],('Non Fraud','Fraud'))

plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4),sharex=True)



# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)

sns.boxplot(x="Class", y="V11", data=new_df, palette='RdBu', ax=axes[0])

axes[0].set_title('V11 vs Class Positive Correlation')



sns.boxplot(x="Class", y="V4", data=new_df, palette='RdBu', ax=axes[1])

axes[1].set_title('V4 vs Class Positive Correlation')





sns.boxplot(x="Class", y="V2", data=new_df, palette='RdBu', ax=axes[2])

axes[2].set_title('V2 vs Class Positive Correlation')





sns.boxplot(x="Class", y="V19", data=new_df, palette='RdBu', ax=axes[3])

axes[3].set_title('V19 vs Class Positive Correlation')



plt.xticks([0,1],('Non Fraud','Fraud'))

plt.show()
from scipy.stats import norm
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))



v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)



v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values

sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')

ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)





v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values

sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')

ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)



plt.show()
f, (ax1, ax2, ax3,ax4) = plt.subplots(1,4, figsize=(20, 6))



v11_fraud_dist = new_df['V11'].loc[new_df['Class'] == 1].values

sns.distplot(v11_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V11 Distribution \n (Fraud Transactions)', fontsize=14)



v4_fraud_dist = new_df['V4'].loc[new_df['Class'] == 1].values

sns.distplot(v4_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')

ax2.set_title('V4 Distribution \n (Fraud Transactions)', fontsize=14)





v2_fraud_dist = new_df['V2'].loc[new_df['Class'] == 1].values

sns.distplot(v2_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')

ax3.set_title('V2 Distribution \n (Fraud Transactions)', fontsize=14)



v9_fraud_dist = new_df['V9'].loc[new_df['Class'] == 1].values

sns.distplot(v9_fraud_dist,ax=ax4, fit=norm, color='#C5B3F9')

ax4.set_title('V9 Distribution \n (Fraud Transactions)', fontsize=14)



plt.show()
q25,q75=25,75



# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)

v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25

print('iqr: {}'.format(v14_iqr))



v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))



outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]

print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V14 outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)

print('----' * 44)



# -----> V12 removing outliers from fraud transactions

v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)

v12_iqr = q75 - q25



v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 outliers: {}'.format(outliers))

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)





# Removing outliers V10 Feature

v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)

v10_iqr = q75 - q25



v10_cut_off = v10_iqr * 1.5

v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off

print('V10 Lower: {}'.format(v10_lower))

print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('V10 outliers: {}'.format(outliers))

print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))



colors = ['#B3F9C5', '#f9c5b3']

# Boxplots with outliers removed

# Feature V14

sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)

ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)

ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature 12

sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)

ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)

ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature V10

sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)

ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)

ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA,TruncatedSVD

import matplotlib.patches as mpatches



#classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



#train-test split

from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
X=new_df.drop('Class',axis=1)

y=new_df['Class']



#T-sne implementation

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)



#PCA implementationX_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)

X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)



#TruncatedSVD implementation

X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
classifiers={

            'LogisticRegression':LogisticRegression(),

            'SVC':SVC(),

            'DecisionTree':DecisionTreeClassifier(),

            'RandomForest':RandomForestClassifier()

}
#Setup cross-validation model to improve accuracy score

from sklearn.model_selection import cross_val_score



for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)

# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_
#estimating score with best estimator

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')





knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, X_train, y_train, cv=5)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
# now we should check the score by implementing this dataset on original dataset



# first removing outliers from original dataset as observed above

# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)

v14_fraud = data['V14'].loc[data['Class'] == 1].values

q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25

print('iqr: {}'.format(v14_iqr))



v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))



outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]

print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V14 outliers:{}'.format(outliers))



data = data.drop(new_df[(data['V14'] > v14_upper) | (data['V14'] < v14_lower)].index)

print('----' * 44)



# -----> V12 removing outliers from fraud transactions

v12_fraud = data['V12'].loc[data['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)

v12_iqr = q75 - q25



v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 outliers: {}'.format(outliers))

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

data = data.drop(data[(new_df['V12'] > v12_upper) | (data['V12'] < v12_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)





# Removing outliers V10 Feature

v10_fraud = data['V10'].loc[data['Class'] == 1].values

q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)

v10_iqr = q75 - q25



v10_cut_off = v10_iqr * 1.5

v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off

print('V10 Lower: {}'.format(v10_lower))

print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('V10 outliers: {}'.format(outliers))

print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

data = data.drop(new_df[(data['V10'] > v10_upper) | (data['V10'] < v10_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))
# now split the original dataset and create train and test dataset

from sklearn.model_selection import ShuffleSplit

X=data.drop('Class',axis=1)

y=data['Class']

# from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeClassifier



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

print('train data dimensions: ', train_X.shape)

print('validation data dimensions: ', val_X.shape)



# fd_model = RandomForestRegressor(random_state=1)

tree_model = DecisionTreeClassifier(random_state=0)



# fd_model.fit(train_X, train_y)

tree_model.fit(train_X, train_y)



# fd_predicted = fd_model.predict(val_X)

tree_predicted = tree_model.predict(val_X)



predicted_nonfraud_falsely = 0

predicted_fraud_truely = 0

index = 0

for record in val_y.iteritems():

    if(record[1] == 1):

        if(tree_predicted[index] == 0):

            predicted_nonfraud_falsely = predicted_nonfraud_falsely + 1

        else:

            predicted_fraud_truely = predicted_fraud_truely + 1

    index = index + 1

print('Rate of true predictions of fraud transactions as fraud: ', predicted_fraud_truely / (predicted_fraud_truely + predicted_nonfraud_falsely))

predicted_fraud_falsely = 0

predicted_nonfraud_truely = 0

index = 0

for record in val_y.iteritems():

    if(record[1] == 0):

        if(tree_predicted[index] == 1):

            predicted_fraud_falsely = predicted_fraud_falsely + 1

        else:

            predicted_nonfraud_truely = predicted_nonfraud_truely + 1

    index = index + 1

print('Rate of false predictions of nonfraud transactions as fraud: ', predicted_fraud_falsely / (predicted_fraud_falsely + predicted_nonfraud_truely))

# print('forest model MAE: ', mean_absolute_error(val_y, fd_predicted))

# print('tree model MAE: ', mean_absolute_error(val_y, tree_predicted))