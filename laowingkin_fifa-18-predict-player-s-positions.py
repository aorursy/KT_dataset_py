import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV



%matplotlib inline 
df = pd.read_csv('../input/CompleteDataset.csv')

df.head()
df.columns
# GK attributes are not our interest

columns_needed = ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',

       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',

       'Free kick accuracy', 'Heading accuracy', 'Interceptions',

       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',

       'Positioning', 'Reactions', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision', 'Volleys', 'Preferred Positions']



# attack attribute first, then defence, then mixed

columns_needed_rearranged = ['Aggression','Crossing', 'Curve', 'Dribbling', 'Finishing',

       'Free kick accuracy', 'Heading accuracy', 'Long shots','Penalties', 'Shot power', 'Volleys', 

       'Short passing', 'Long passing',

       'Interceptions', 'Marking', 'Sliding tackle', 'Standing tackle',

       'Strength', 'Vision', 'Acceleration', 'Agility', 

       'Reactions', 'Stamina', 'Balance', 'Ball control','Composure','Jumping', 

       'Sprint speed', 'Positioning','Preferred Positions']



df = df[columns_needed_rearranged]

df.head()
df['Preferred Positions'] = df['Preferred Positions'].str.strip()

df = df[df['Preferred Positions'] != 'GK']

df.head()



df.isnull().values.any()
p = df['Preferred Positions'].str.split().apply(lambda x: x[0]).unique()

p
# copy a structure

df_new = df.copy()

df_new.drop(df_new.index, inplace=True)



for i in p:

    df_temp = df[df['Preferred Positions'].str.contains(i)]

    df_temp['Preferred Positions'] = i

    df_new = df_new.append(df_temp, ignore_index=True)

    

df_new.iloc[::500, :]

            



cols = [col for col in df_new.columns if col not in ['Preferred Positions']]



for i in cols:

    df_new[i] = df_new[i].apply(lambda x: eval(x) if isinstance(x,str) else x)



df_new.iloc[::500, :]
fig, ax = plt.subplots()

df_new_ST = df_new[df_new['Preferred Positions'] == 'ST'].iloc[::200,:-1]

df_new_ST.T.plot.line(color = 'black', figsize = (15,10), legend = False, ylim = (0, 110), title = "ST's attributes distribution", ax=ax)



ax.set_xlabel('Attributes')

ax.set_ylabel('Rating')



ax.set_xticks(np.arange(len(cols)))

ax.set_xticklabels(labels = cols, rotation=90)



for ln in ax.lines:

    ln.set_linewidth(1)



ax.axvline(0, color='red', linestyle='--')   

ax.axvline(12.9, color='red', linestyle='--')



ax.axvline(13, color='blue', linestyle='--')

ax.axvline(17, color='blue', linestyle='--')



ax.axvline(17.1, color='green', linestyle='--')

ax.axvline(28, color='green', linestyle='--')



ax.text(5, 100, 'Attack Attributes', color = 'red', weight = 'bold')

ax.text(13.5, 100, 'Defend Attributes', color = 'blue', weight = 'bold')

ax.text(22, 100, 'Mixed Attributes', color = 'green', weight = 'bold')
df_new_ST_normalized = df_new_ST.div(df_new_ST.sum(axis=1), axis=0)



fig, ax = plt.subplots()

df_new_ST_normalized.T.plot.line(color = 'black', figsize = (15,10), ylim = (0, 0.08), legend = False, title = "ST's attributes distribution (normalized)", ax=ax)



ax.set_xlabel('Attributes')

ax.set_ylabel('Normalized Rating')



ax.set_xticks(np.arange(len(cols)))

ax.set_xticklabels(labels = cols, rotation=90)



for ln in ax.lines:

    ln.set_linewidth(1)



ax.axvline(0, color='red', linestyle='--')   

ax.axvline(12.9, color='red', linestyle='--')



ax.axvline(13, color='blue', linestyle='--')

ax.axvline(17, color='blue', linestyle='--')



ax.axvline(17.1, color='green', linestyle='--')

ax.axvline(28, color='green', linestyle='--')



ax.text(5, 0.07, 'Attack Attributes', color = 'red', weight = 'bold')

ax.text(13.5, 0.07, 'Defend Attributes', color = 'blue', weight = 'bold')

ax.text(22, 0.07, 'Mixed Attributes', color = 'green', weight = 'bold') 

df_new_normalized = df_new.iloc[:,:-1].div(df_new.iloc[:,:-1].sum(axis=1), axis=0)

mapping = {'ST': 1, 'RW': 1, 'LW': 1, 'RM': 1, 'CM': 1, 'LM': 1, 'CAM': 1, 'CF': 1, 'CDM': 0, 'CB': 0, 'LB': 0, 'RB': 0, 'RWB': 0, 'LWB': 0}

df_new_normalized['Preferred Positions'] = df_new['Preferred Positions']

df_new_normalized = df_new_normalized.replace({'Preferred Positions': mapping})



df_new_normalized.iloc[::1000,]

X_train, X_test, y_train, y_test = train_test_split(df_new_normalized.iloc[:,:-1], df_new_normalized.iloc[:,-1], random_state=0)



print('X train shape: {}'.format(X_train.shape))

print('X test shape: {}'.format(X_test.shape))

print('y train shape: {}'.format(y_train.shape))

print('y test shape: {}'.format(y_test.shape))

clf_d = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)

acc_d = clf_d.score(X_test, y_test)

print ('Dummy Classifier (most frequent class): {}'.format(acc_d))



clf = LogisticRegression().fit(X_train, y_train)

acc = clf.score(X_test, y_test)

print ('Logistic Regression Accuracy: {}'.format(acc))

Coef_list = list(sorted(zip(X_train.columns, abs(clf.coef_[0])),key=lambda x: -x[1]))

Coef_table = pd.DataFrame(np.array(Coef_list).reshape(-1,2), columns = ['Attributes', 'Coef'])



print (Coef_table)
target_cols = Coef_table[:10]['Attributes'].tolist()



clf_2 = LogisticRegression().fit(X_train[target_cols], y_train)

acc_2 = clf_2.score(X_test[target_cols], y_test)

print ('Logistic Regression Accuracy (10 features): {}'.format(acc_2))
f, ax = plt.subplots(figsize=(20, 20))



plt.title('Pearson Correlation of Player attributes')



sns.heatmap(df_new.corr(),linewidths=0.25,vmax=1.0, square=True, cmap = 'PuBu', linecolor='black', annot=True)
cov_mat = np.cov(df_new.iloc[:,:-1].T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance



print(list(zip(range(29),cum_var_exp)))



# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 

plt.figure(figsize=(10, 10))

plt.bar(range(len(var_exp)), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')

plt.step(range(len(var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.show()



pca = PCA(n_components=17)



X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df_new.iloc[:,:-1], df_new.iloc[:,-1], random_state=0)



X_train_2_pca = pca.fit_transform(X_train_2)

X_train_2_pca = pd.DataFrame(X_train_2_pca)



x_test_2_pca = pca.transform(X_test_2)



clf_17d = LogisticRegression().fit(X_train_2_pca, y_train_2)

acc_17d = clf_17d.score(x_test_2_pca, y_test_2)

print ('Logistic Regression Accuracy with PCA (17 components): {}'.format(acc_17d))

lda = LDA(n_components=None)



X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(df_new.iloc[:,:-1], df_new.iloc[:,-1], random_state=0)



X_lda = lda.fit(X_train_3, y_train_3)



lda_var_ratios = lda.explained_variance_ratio_



# get number of components needed to explain 95% variance

def select_n_components(var_ratio, goal_var: float) -> int:

    

    total_variance = 0.0

    n_components = 0

    

    for explained_variance in var_ratio:

        total_variance += explained_variance

        n_components += 1

        if total_variance >= goal_var:

            break



    return n_components



print('Number of components needed to explain 95% variance: {}'.format(select_n_components(lda_var_ratios, 0.95)))
lda_n = LDA(n_components=3)

X_train_3_lda = lda_n.fit_transform(X_train_3, y_train_3)

X_train_3_lda = pd.DataFrame(X_train_3_lda)



X_test_3_lda = lda_n.transform(X_test_3)



clf_3d = LogisticRegression().fit(X_train_3_lda, y_train_3)

acc_3d = clf_3d.score(X_test_3_lda, y_test_3)

print ('Logistic Regression Accuracy with LDA (3 components): {}'.format(acc_3d))
df_new_normalized_all = df_new.copy()

mapping_all = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 'CF': 7, 'CDM': 8, 'CB': 9, 'LB': 10, 'RB': 11, 'RWB': 12, 'LWB': 13}



df_new_normalized_all = df_new_normalized_all.replace({'Preferred Positions': mapping_all})

df_new_normalized_all.iloc[::1000,]

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(df_new_normalized_all.iloc[:,:-1], df_new_normalized_all.iloc[:,-1], random_state=0)



print('X train shape: {}'.format(X_train_all.shape))

print('X test shape: {}'.format(X_test_all.shape))

print('y train shape: {}'.format(y_train_all.shape))

print('y test shape: {}'.format(y_test_all.shape))
clf_d_all = DummyClassifier(strategy = 'most_frequent').fit(X_train_all, y_train_all)

acc_d_all = clf_d_all.score(X_test_all, y_test_all)

print ('Dummy Classifier (most frequent class): {}'.format(acc_d_all))



clf_all = LogisticRegression().fit(X_train_all, y_train_all)

acc_all = clf_all.score(X_test_all, y_test_all)

print ('Logistic Regression Accuracy: {}'.format(acc_all))
clf_all_for = RandomForestClassifier(random_state=0).fit(X_train_all, y_train_all)

acc_all_for = clf_all_for.score(X_test_all, y_test_all)

print ('Random Forest Accuracy (Default parameters): {}'.format(acc_all_for))

parameters_f = [{'max_depth': range(2,10), 'n_estimators': range(2,8,2), 'max_features': range(10,20)}]

clf_all_for_g = GridSearchCV(RandomForestClassifier(random_state=0), parameters_f)

clf_all_for_g.fit(X_train_all, y_train_all)



print('Best score for train data:', clf_all_for_g.best_score_)

print('Best depth:',clf_all_for_g.best_estimator_.max_depth)

print('Best n trees:',clf_all_for_g.best_estimator_.n_estimators)

print('Best n features:',clf_all_for_g.best_estimator_.max_features)

print('Score for test data:',clf_all_for_g.score(X_test_all, y_test_all))



clf_all_nn = MLPClassifier(random_state=0).fit(X_train_all, y_train_all)

acc_all_nn = clf_all_nn.score(X_test_all, y_test_all)

print ('Neural Networks Accuracy (Default parameters): {}'.format(acc_all_nn))



parameters_n = [{'alpha': [0.0001, 0.001, 0.01, 0.1], 'hidden_layer_sizes':[(10,),(20,),(100,)]}]

clf_all_nn_g = GridSearchCV(MLPClassifier(random_state=0), parameters_n)

clf_all_nn_g.fit(X_train_all, y_train_all)



print('Best score for train data:', clf_all_nn_g.best_score_)

print('Best alpha:',clf_all_nn_g.best_estimator_.alpha)

print('Best hidden_layer_sizes:',clf_all_nn_g.best_estimator_.hidden_layer_sizes)

print('Score for test data:',clf_all_nn_g.score(X_test_all, y_test_all))
