#import all the necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
winedb=pd.read_csv(r"/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
winedb.head()
winedb.describe()
#Plot the corr plot to see the corelation between variables

d=pd.DataFrame(data=winedb)
corr=d.corr()
f, ax = plt.subplots(figsize=(10, 6))
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="gray",fmt='.2f',
            linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation HeatMap', fontsize=14)
from sklearn.preprocessing import StandardScaler, LabelEncoder
#The bins parameter tells you the number of bins that your data will be divided into.
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
winedb['quality'] = pd.cut(winedb['quality'], bins = bins, labels = group_names)
#Assign labels
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
winedb['quality'] = label_quality.fit_transform(winedb['quality'])
winedb['quality'].value_counts()
#Plot the counts 
sns.countplot(winedb['quality'])
#Seperate the dataset
Y=winedb['quality']
X = winedb.iloc[:,:11]
#Create test train split data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size = 0.2, random_state = 42)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func  = chi2, k ='all')
fitted = test.fit(X,Y)
list(fitted.scores_)
colnames =  list(X.columns)
X  = np.array(X)
df_1 = pd.DataFrame({'Feature_name':list(colnames),'Feature_score':list(fitted.scores_)}) 
df_1
df_1.sort_values(['Feature_score'],ascending = False)
from sklearn.feature_selection import RFE  ##runs on top of some algorithm ..we can run on top of randomforest and DT also)
from sklearn.svm import LinearSVC #(running on SVM)
svm = LinearSVC()
rfe= RFE(svm, 10)  ## keep increasing the number to get the importance in order

rfe.fit(X, Y)
print(rfe.support_)
df2 = pd.DataFrame( { "Feature Names": colnames , "Importance" : list(rfe.support_)})
df2.sort_values(['Importance'], ascending = False )
#Trying the machine learning algorithms
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
pred_rfc = rfc.predict(X_test)
#Let's see how our model performed
print(classification_report(Y_test, pred_rfc))
print(confusion_matrix(Y_test, pred_rfc))
#SVM
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, Y_train)
pred_svc = svc.predict(X_test)
print(classification_report(Y_test, pred_svc))