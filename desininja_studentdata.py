import xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
train = pd.read_excel('../input/research_student (1).xlsx')
train.head(5)
g = train.groupby('Branch')
g
ece = g.get_group('ECE')
ece.head(5)
ece = ece.drop(['Marks[10th]', 'Board[10th]'], axis=1)


ece.info()
ece.head(5)

feature_cols = ['Branch','Marks[12th]','Normalized Rank','Category']
x = ece.loc[:,feature_cols]
x.shape
#feature_cols = ['Marks[12th]','Category']
#x = ece.loc[:,feature_cols]
#x.shape
x.info()

new =  ['Marks[12th]','GPA 1','Rank','Normalized Rank','Current Back','CGPA','GPA 3','GPA 4','GPA 5','GPA 6','Olympiads Qualified','Technical Projects','Tech Quiz','Engg. Coaching','NTSE Scholarships','Miscellany Tech Events']
vis = ece.loc[:,new]
vis.shape
vis.head(4)
vis.rename(columns={'Marks[12th]': 'Marks','GPA 1':'GPA1', 'Normalized Rank': 'Normalized_Rank','Current Back':'CurrentBack','GPA 3':'GPA3','GPA 4':'GPA4','GPA 5':'GPA5','GPA 6':'GPA6','Olympiads Qualified':'OlympiadsQualified','Technical Projects':'TechnicalProjects','Tech Quiz':'TechQuiz','Engg. Coaching':'EnggCoaching','NTSE Scholarships':'NTSEScholarships','Miscellany Tech Events':'MiscellanyTechEvents'}, inplace=True)
vis.head()
x.rename(columns={'Marks[12th]': 'Marks', 'Normalized Rank': 'Normalized_Rank'}, inplace=True)
x.info()
encoding_list = ['Category']
x[encoding_list] = x[encoding_list].apply(LabelEncoder().fit_transform)
x.head(1)
color_list = ['red' if i=='Abnormal' else 'green' for i in x.loc[:,'Marks']]
pd.plotting.scatter_matrix(x.loc[:, x.columns != 'Marks'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '#',
                                       edgecolor= "black")
plt.show()


z = x.Normalized_Rank.reshape(-1,1)



w  = (x.Marks).reshape(-1,1)
plt.plot(w,'g') #marks
plt.plot(z,'b')

plt.legend()
plt.title('comparison')

sns.distplot(x.Category, bins=50, kde=True, rug=False);
sns.kdeplot(x.Category, shade=True);

scale_list = ['Marks','GPA1','Rank','Normalized_Rank','CurrentBack','CGPA','GPA3','GPA4','GPA5','GPA6','OlympiadsQualified','TechnicalProjects','TechQuiz','EnggCoaching','NTSEScholarships','MiscellanyTechEvents']
sc = vis[scale_list]
sc.head(3)
scaler = StandardScaler()
sc = scaler.fit_transform(sc)
vis[scale_list] = sc
vis[scale_list].head()
vis.head()

ax = sns.heatmap(vis,vmax = 5,vmin = -2)
vis.head(10)
#y = vis['Normalized_Rank']
#x = vis.drop('Normalized_Rank', axis=1)
#X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
#X_train.shape
x.shape

logreg=LinearRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


y_test
x.head(2)
vis.head()
encoding_list = ['Branch','Gender','Board[12th]','Category']
ece[encoding_list] = ece[encoding_list].apply(LabelEncoder().fit_transform)
ece.head()

ece2 = ece.drop(['Branch','Marks[12th]','Gender','Board[12th]','Category'], axis=1)
ece2.head()
x.head(4)
y = ece['Rank']
x = ece.drop('Rank', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
X_train.shape
logreg=LinearRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
y_test
y_pred
print(metrics.mean_squared_error(y_test, y_pred))
#xgb = xgboost.XGBRegressor(n_estimators=25000, learning_rate=0.06, gamma=0, subsample=0.6,
 #                          colsample_bytree=0.7, min_child_weight=4, max_depth=3)
                           
#xgb.fit(X_train,y_train)
#predictions = xgb.predict(X_test)
#print(metrics.mean_squared_error(y_test, predictions))
#y = ece['Rank']
#x = ece.drop('Rank', axis=1)
#X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
#X_train.shape
# find best scored 5 features
#select_feature = SelectKBest(chi2, k=10).fit(x_train, y_train)
#print('Score list:', select_feature.Rank)
#print('Feature list:', x_train.Normalized_Rank)
#from sklearn.feature_selection import RFE
#y = ece2['Rank']
#x = ece2.drop('Rank', axis=1)
#X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
#X_train.shape
#for column in ece.columns:
    #if ece[column].dtype == type(object):
        #le = LabelEncoder()
        #ece[column] = le.fit_transform(dataset[column])

# Create the RFE object and rank each pixel
#clf_rf_3 = RandomForestClassifier()      
#rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
#rfe = rfe.fit(x_train, y_train)
#print('Chosen best 5 feature by rfe:',x_train.columns[rfe.support_])
y = ece2['Rank']
x = ece2.drop('Rank', axis=1)
x.shape
X_new = SelectKBest(chi2, k=2).fit_transform(x, y)
X_new.shape




ece2.head(4)
x.head(5)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)
ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(x, y)
selector.support_ 
selector.ranking_
clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#normalization
x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_train_N)

plt.figure(1, figsize=(14, 13))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')

