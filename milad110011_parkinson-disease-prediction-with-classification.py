import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


dataset = pd.read_csv('../input/parkinsons-disease-speech-signal-features/pd_speech_features.csv')

x = dataset.drop(['id','class'],axis=1)
y = dataset['class'].values

y.shape

unique_elements, counts_elements = np.unique(y, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements))) 


x.describe()

X_train, X_test, y_train, y_test = train_test_split(
     x, y, test_size=0.2, random_state=0)
clfRF=RandomForestClassifier(n_estimators=20,
                       criterion='gini',
                       max_depth=None,
                       min_weight_fraction_leaf=0.0,
                       max_features=10, max_leaf_nodes=4,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       bootstrap=True, oob_score=False,
                       n_jobs=5, random_state=None,
                       verbose=0, warm_start=False, class_weight=None,
                       )

clfRF=clfRF.fit(X_train,y_train)

import eli5 
from eli5.sklearn import PermutationImportance



perm = PermutationImportance(clfRF, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

feature_scores = pd.Series(clfRF.feature_importances_, index=X_train.columns).sort_values(ascending=False)

for i,v in enumerate(feature_scores):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(feature_scores))], feature_scores)
pyplot.show()
features_score=clfRF.feature_importances_

features = x.columns

indices = np.argsort(features_score)[-30:]


plt.title('Feature Importances')
plt.barh(range(len(indices)), features_score[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


f, ax = plt.subplots(figsize=(30, 24))
ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=x)
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()
x.isnull().sum().sort_values(ascending=False)

x.numPulses[(x.numPulses==0)].count()


import seaborn as sns

z=x['numPulses']
y=x['numPeriodsPulses']


sns.scatterplot(z,y)
# Detect Outliers with isolation forest algorithm
from sklearn.ensemble import IsolationForest

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(x)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

scaler_pipeline = Pipeline([('rob_scale', RobustScaler())])

x_scaled = scaler_pipeline.fit_transform(x)

x_scaled = pd.DataFrame(x_scaled,columns=x.columns,index=x.index)
x_scaled.describe()
import seaborn as sns

z=x_scaled['numPulses']
y=x_scaled['numPeriodsPulses']


sns.scatterplot(z,y)
sns.countplot(dataset['class'].values)
plt.xlabel('class Values')
plt.ylabel('class Counts')
plt.show()

print('No parkinson disease', round(dataset['class'].value_counts()[0]/len(dataset) * 100,2), '% of the dataset')
print('parkinson disease', round(dataset['class'].value_counts()[1]/len(dataset) * 100,2), '% of the dataset')


fig, ax = plt.subplots(1, 2, figsize=(18,4))

ppe_val = dataset['PPE'].values
dfa_val = dataset['DFA'].values

sns.distplot(ppe_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of ppe', fontsize=14)
ax[0].set_xlim([min(ppe_val), max(ppe_val)])

sns.distplot(dfa_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of dfa', fontsize=14)
ax[1].set_xlim([min(dfa_val), max(dfa_val)])


transformer = RobustScaler().fit(x)
x_scaled = transformer.transform(x)


sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(x, y, test_size=0.2, random_state=42)

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)


print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))
from imblearn.under_sampling import TomekLinks
 

tl = TomekLinks(return_indices=True, ratio='majority')

x_tl, y_tl, id_tl = tl.fit_sample(x, y)

y_tl.count()

sns.countplot(dataset['class'].values)
plt.xlabel('status Values')
plt.ylabel('status Counts')
plt.show()




correlation_values=dataset.corr()['status']
print(correlation_values.abs().sort_values(ascending=False))

g = sns.FacetGrid(dataset, hue="status", height=15)
g.map(plt.scatter, "MDVP:Fo(Hz)","MDVP:Flo(Hz)", alpha=.7)
g.add_legend()



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)




# sns.heatmap(x_train_std[x_train_std.columns[0:10]].corr(),annot=True)
# sns.heatmap(x_train_std[x_train_std.columns[10:20]].corr(),annot=True)
#Number of pulses and numPeriod pulses looks like each other. Corr test maybe good for this type of problem..
# print(x_drop[['number_of_pulses','number_of_periods']].corr())
# print(x_drop[['jitter_local','jitter_rap']].corr())
# This 2 Attiributes is too similar. We should discard one of these. I choose number_of_periods
# del  x_drop['number_of_periods']
# # jitter_local and jitter_rap have highly correlated also. We should delete one of these columns. I delete jitter_rap
# del x_drop['jitter_rap']
# # shimmer_local and shimmer_local_db is also highly correlated.I delete shimmer_local_db
# del x_drop['shimmer_local_db']
# # jitter_ddp and jitter_local is also highly correlated.I delete jitter_ddp
# del x_drop['jitter_ddp']
# # # ac and htn is also highly correlated.I delete htn
# # del x_drop['htn']



# # train_col=[]
# # for col in dataset.columns:
# #     train_col.append(col)
# # print(train_col)

# # print(dataset.shape)



# # def assess_NA(dataset):
# #  null_sum = dataset.isnull().sum()
# #  total = null_sum.sort_values(ascending=False)
# #  percent = ((null_sum / len(dataset.index))*100).round(2).sort_values(ascending=False)
# #  df_na = pd.concat([total,percent],axis=1,keys=['Number of NAn','percent of Nan'])
# #  df_na = df_na[(df_na.T != 0).any()]
# #  return(df_na)

# # df_na = assess_NA(dataset)
# # df_na





# # scaler1 = MinMaxScaler(feature_range=(-1, 1))
# # x_tr_minmax = scaler1.fit_transform(x_train)
# sc=StandardScaler()
# sc.fit(x_train)
# x_tr_std = sc.fit_transform(x_train)
# x_tst_std  = sc.transform(x_test)

# pca = PCA(n_components = 16)
# x_tr_std_pca = pca.fit_transform(x_tr_std)
# x_tst_std_pca = pca.transform(x_tst_std)
# x_tr_std_pca[0:3,:]

 
# # x_train_std =pd.DataFrame(x_tr_std)
# # x_train_std.rename(columns={0:'jitter_local',
# #                             1:'jitter_local_absolute',
# #                             2:'jitter_rap',
# #                             3:'jitter_ppq5',
# #                             4:'jitter_ddp',
# #                             5:'shimmer_local',
# #                             6:'shimmer_local_db',
# #                             7:'shimmer_apq3',
# #                             8:'shimmer_apq5',
# #                             9:'shimmer_apq11',
# #                             10:'shimmer_dda',
# #                             11:'ac',
# #                             12:'nth',
# #                             13:'htn',
# #                             14:'median_pitch',
# #                             15:'mean_pitch',
# #                             16:'standard_deviation',
# #                             17:'minimum_pitch',
# #                             18:'maximum_pitch',
# #                             19:'number_of_pulses',
# #                             20:'number_of_periods',
# #                             21:'mean_period',
# #                             22:'standard_deviation_of_period',
# #                             23:'fraction_of_locally_unvoiced_frames',
# #                             24:'number_of_voice_breaks',
# #                             25:'degree_of_voice_breaks',
# #                             26:'updrs'}, 
# #                             inplace=True)

# # # fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(10, 9))
# # # ax1.set_title('Before Scaling')
# # sns.kdeplot(x['number_of_periods'], ax=ax1)
# # ax2.set_title('After Standard Scaler')
# # sns.kdeplot(x_train_std['number_of_periods'], ax=ax2)
# # ax3.set_title('Before Scaling')
# # sns.kdeplot(x['number_of_voice_breaks'], ax=ax3)
# # ax4.set_title('After Standard Scaler')
# # sns.kdeplot(x_train_std['degree_of_voice_breaks'], ax=ax4)
# # plt.show()






# # Classification With Random Forest


# # print(clf.score(x_train_std_array,y_train))





clf=SVC(probability=True,random_state=1)
clf.fit(x_tr_std_pca,y_train)
print(clf.score(x_tst_std_pca,y_test))
print(clf.score(x_tr_std_pca,y_train))

clf2=RandomForestClassifier(n_estimators=20,
                       criterion='gini',
                       max_depth=None,
                       min_weight_fraction_leaf=0.0,
                       max_features=10, max_leaf_nodes=4,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       bootstrap=True, oob_score=False,
                       n_jobs=5, random_state=None,
                       verbose=0, warm_start=False, class_weight=None,
                       )
clf2.fit(x_tr_std_pca,y_train)
print(clf.score(x_tst_std_pca,y_test))
print(clf.score(x_tr_std_pca,y_train))
clf=DecisionTreeClassifier(criterion='entropy', splitter='random',
                           max_depth=3, min_samples_split=5,
                           min_samples_leaf=3, min_weight_fraction_leaf=0.0,
                           max_features=14, random_state=0, 
                           max_leaf_nodes=None, min_impurity_decrease=0.3, 
                           class_weight=None,
                           presort='auto')
clf.fit(x_tr_std_pca,y_train)
print(clf.score(x_tst_std_pca,y_test))
print(clf.score(x_tr_std_pca,y_train))

