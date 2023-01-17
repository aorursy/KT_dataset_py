import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve, accuracy_score, f1_score,roc_auc_score
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")



df.info()
print(df[['sex','smoking','diabetes','high_blood_pressure','anaemia','DEATH_EVENT']].head(5))
for feature in ['sex','smoking','diabetes','high_blood_pressure','anaemia','DEATH_EVENT']:

        df[feature]=df[feature].astype(bool)

df.info()
print(df.describe())
print(df["DEATH_EVENT"].describe())
print(df.loc[df['DEATH_EVENT']==0]['time'].mean())

print(df.loc[df['DEATH_EVENT']==0]['time'].median())

print(df.loc[df['DEATH_EVENT']==1]['time'].mean()) 

print(df.loc[df['DEATH_EVENT']==1]['time'].median()) 

print(df.loc[df['DEATH_EVENT']==1]['time'].max())

print(df.loc[df['DEATH_EVENT']==1]['time'].max())
best_correlation = 0

for threshold in np.arange(20,240,1):

    df['time_new'] = (df['time'] >= threshold)

    correlation = df[['time_new','DEATH_EVENT']].corr().to_numpy()[0,1]

    if np.abs(correlation) >  np.abs(best_correlation):

            best_correlation = df[['time_new','DEATH_EVENT']].corr().to_numpy()[0,1]

            best_threshold = threshold

                

print("Best threshold = " + str(best_threshold) + ' days' )

print("linear correlation = " + str(best_correlation) )



df['time_new'] = (df['time']>= best_threshold)

df = df.drop(['time'], axis=1)
print(df.corr()['DEATH_EVENT'])

plt.barh(np.arange(len(df.corr()['DEATH_EVENT'])), df.corr()['DEATH_EVENT'],align = 'center',tick_label = df.columns)

plt.xlim((-1,1))

plt.grid(axis='x')

plt.title('Pearson Correlation of features with response')

plt.show()



plt.clf()
selected_features = ['time_new','age','ejection_fraction','serum_creatinine','serum_sodium']

print(selected_features)
X = df.drop(['DEATH_EVENT'],axis=1)

y = df['DEATH_EVENT']







sc = StandardScaler()

X = sc.fit_transform(X)





rfc = RandomForestClassifier(n_estimators=1000, random_state=0)



rfc.fit(X, y)





importances = rfc.feature_importances_



print(df.drop('DEATH_EVENT', axis=1).columns)



plt.barh(np.arange(len(df.drop('DEATH_EVENT', axis=1).columns)), importances,align = 'center',tick_label = df.drop('DEATH_EVENT', axis=1).columns)

plt.xlim((0,0.5))

plt.grid(axis='x')

plt.title('Random Forest importances')

plt.show()



plt.clf()
selected_features.append('creatinine_phosphokinase')

selected_features.append('platelets')

print(selected_features)
df['new_feature']=df['smoking']&df['high_blood_pressure']

correlation = df[['new_feature','DEATH_EVENT']].corr().to_numpy()[0,1]

print('smoking and high_blood_pressure correlation='+str(correlation))



df['new_feature']=df['anaemia']&df['sex']

correlation = df[['new_feature','DEATH_EVENT']].corr().to_numpy()[0,1]

print('anaemia and male correlation='+str(correlation))



df['new_feature']=df['smoking']&df['sex']

correlation = df[['new_feature','DEATH_EVENT']].corr().to_numpy()[0,1]

print('smoking and male='+str(correlation))



df['new_feature']=df['smoking']&(~df['sex'])

correlation = df[['new_feature','DEATH_EVENT']].corr().to_numpy()[0,1]

print('smoking and female='+str(correlation))



df['new_feature']=df['high_blood_pressure']&df['anaemia']

correlation = df[['new_feature','DEATH_EVENT']].corr().to_numpy()[0,1]

print('high_blood_pressure and anaemia correlation='+str(correlation))



df['new_feature']=df['smoking']&df['high_blood_pressure']&df['sex']

correlation = df[['new_feature','DEATH_EVENT']].corr().to_numpy()[0,1]

print('smoking, high_blood_pressure and male correlation='+str(correlation))



df['new_feature']=df['smoking']&df['high_blood_pressure']&df['anaemia']

correlation = df[['new_feature','DEATH_EVENT']].corr().to_numpy()[0,1]

print('smoking, high_blood_pressure and anaemia correlation='+str(correlation))



df['new_feature']=df['smoking']&df['high_blood_pressure']&df['anaemia']&df['sex']

correlation = df[['new_feature','DEATH_EVENT']].corr().to_numpy()[0,1]

print('smoking, high_blood_pressure, anaemia and male correlation='+str(correlation))

plt.figure(figsize=(10,10))

sns.heatmap(df[selected_features].corr(), vmin=-1, vmax=1, cmap='seismic', annot=True)

plt.title('Correlation matrix')

plt.yticks(rotation=0)

plt.show()

plt.clf()
# 2d scatter plot





plt.scatter(df.loc[df['DEATH_EVENT'] == 0]['age'],df.loc[df['DEATH_EVENT'] == 0]['ejection_fraction'], color='blue')

plt.scatter(df.loc[df['DEATH_EVENT'] == 1]['age'],df.loc[df['DEATH_EVENT'] == 1]['ejection_fraction'], color='red')

plt.legend(['DEATH_EVENT=1','DEATH_EVENT=0'])

plt.xlabel('age')

plt.ylabel('ejection_fraction')

plt.show()

plt.clf()



# 3d scatter plot



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')



for farbe,b in [('blue',0),('red',1)]:

    xs = df.loc[df['DEATH_EVENT'] == b]['serum_creatinine']

    ys = df.loc[df['DEATH_EVENT'] == b]['ejection_fraction']

    zs = df.loc[df['DEATH_EVENT'] == b]['age']

    ax.scatter(xs, ys, zs, color = farbe)



ax.set_xlabel('serum_creatinine')

ax.set_ylabel('ejection_fraction')

ax.set_zlabel('age')



plt.show()

plt.clf()
X = df[selected_features]

y = df['DEATH_EVENT']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 6)



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
C_range = np.logspace(-2,3,11)

gamma_range = np.logspace(-4,1,11)





param_grid = dict(C=C_range,gamma=gamma_range)



# initialize the grid

svc = SVC(kernel='rbf')

grid = GridSearchCV(svc, param_grid, cv=10, scoring = 'f1')







#fit grid to training data

grid.fit(X_train,y_train)



cv_results_df = pd.DataFrame.from_dict(grid.cv_results_)



# ###########################################################################

# Plot heatmap of F1 score

# ###########################################################################

# Utility function to move the midpoint of a colormap to be around

# the values of interest.

# ###########################################################################

class MidpointNormalize(Normalize):



    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):

        self.midpoint = midpoint

        Normalize.__init__(self, vmin, vmax, clip)



    def __call__(self, value, clip=None):

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))

# ############################################################################

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),

                                                     len(gamma_range))

# ############################################################################

# The parameters vmin and midpoint control the colorbar of the heatmap

plt.figure(figsize=(8, 6))

plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)

plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,

           norm=MidpointNormalize(vmin=0.2, midpoint=0.6))

plt.xlabel('gamma')

plt.ylabel('C')

plt.colorbar()



C_range_aux = np.logspace(-2,3,6)

gamma_range_aux = np.logspace(-4,1,6)

plt.xticks(np.arange(0,len(gamma_range),2), gamma_range_aux, rotation=45)

plt.yticks(np.arange(0,len(C_range),2), C_range_aux)

plt.title('Cross-validation F1 score')

plt.show()

plt.clf()
print('Best F1 score = ' + str(grid.best_score_)+ 'at')

print(grid.best_params_)
C_range = np.logspace(-1,4,200)

gamma_range = np.logspace(-4, 0,200)



param_dist = dict(C=C_range,gamma=gamma_range)



rand = RandomizedSearchCV(svc, param_dist, cv=10, scoring = 'f1',n_iter = 150 , random_state=42)

rand.fit(X_train,y_train)



print('Best F1 score = ' + str(rand.best_score_) + 'at')

print(rand.best_params_)
pred_svc = rand.predict(X_test)

print(confusion_matrix(y_test, pred_svc))
print(classification_report(y_test, pred_svc))
plot_roc_curve(rand, X_test, y_test)
svc_best = SVC(kernel='rbf', C=0.4500557675700499 , gamma=0.017027691722258997)



f1_scores = []

accuracy_scores = []

roc_auc_scores = []



for i in range(1, 200):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = i)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)

    X_test = sc.fit_transform(X_test)

    svc_best.fit(X_train,y_train)

    pred_svc = svc_best.predict(X_test)

    f1_scores.append(f1_score(y_test,pred_svc))

    accuracy_scores.append(accuracy_score(y_test,pred_svc))

    roc_auc_scores.append(roc_auc_score(y_test,pred_svc))

    

    

plt.hist(f1_scores, bins=15)

plt.title('Distribution of F1 scores')

plt.show()

plt.clf()





plt.hist(roc_auc_scores, bins=15)

plt.title('Distribution of AUC scores')

plt.show()

plt.clf()



plt.hist(accuracy_scores, bins=15)

plt.title('Distribution of accuracy_scores')

plt.show()

plt.clf()



# randomly shuffle rows

df.sample(frac=1) 

X = df[selected_features].astype('float64').to_numpy()

y = df['DEATH_EVENT'].astype('float64').to_numpy()



error_train = []

error_test = []



I = range(50, 298)



for i in I:

    X_train, X_test, y_train, y_test = train_test_split(X[0:i,:], y[0:i], test_size = 0.2, random_state = 42)

    X_train = sc.fit_transform(X_train)

    X_test = sc.fit_transform(X_test)

    svc_best.fit(X_train,y_train)

    pred_svc_train = svc_best.predict(X_train)

    pred_svc_test  = svc_best.predict(X_test)

    error_train.append(np.linalg.norm(pred_svc_train-y_train)/(np.shape(X_train)[0]))

    error_test.append(np.linalg.norm(pred_svc_test-y_test)/(np.shape(X_test)[0]))



    

plt.plot(I,error_train, color='blue')

plt.plot(I,error_test,color='red')

plt.ylabel('Errors')

plt.xlabel('Number of data points')

plt.legend(['Error on training set','Error on test set'])

plt.show()

plt.clf()


