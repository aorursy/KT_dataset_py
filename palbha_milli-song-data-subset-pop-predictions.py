# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from pylab import rcParams





sp_data = pd.read_csv("../input/SongPred.csv")

# Taking a look at data and its features

print(sp_data.info())

print(sp_data.columns) #Lets take a look at all the column names in our dataset
#Since most of us are familiar with R and ggplot and its visualizations seems more appealing

# Lets have the stylesheet as ggplot

plt.style.use('ggplot')

fig=sns.scatterplot(x="artist_familiarity",y="song_hotttnesss",data=sp_data)

plt.xlabel("Artist Familiarity Quotient")

plt.ylabel("Song Hotness")

plt.title("Artist Familiarity v/s Song Hotness")

plt.show(fig)

#We can see that artist familarity and song hotness are somewhat related whenever

#artist has familairty quotient high the song has higher chance of being popular
fig=sns.scatterplot(x="artist_longitude",y="song_hotttnesss",data=sp_data)

plt.xlabel("Artist Longitude ")

plt.ylabel("Song Hotness")

plt.title("Artist Longitude v/s Song Hotness")

plt.show(fig)

#As per our analysis we can say that there is generally more activity in the regions

#that also produce hits, we can see that the hits are centralized around these specific areas.

#Most of the activity is coming from the western side of the world, 

#and on North America (as per the longitude values
sp_data_hit=sp_data[sp_data["bbhot"]==1]



fig=sns.distplot(sp_data_hit[["duration"]],hist_kws=dict(edgecolor="k", linewidth=2),rug=True, kde=False)

#fig=sns.countplot(x="duration",data=sp_data_hit)

plt.xlabel("Duration")

plt.xlim(0,650)

plt.ylabel("No. of Songs")

plt.title("Most Common Duration of HIT songs")

plt.show(fig)



fig=sns.distplot(sp_data_hit[["tempo"]],hist_kws=dict(edgecolor="k", linewidth=2),rug=True, kde=False)

plt.xlabel("Tempo")

plt.xlim(0,280)

plt.ylabel("No. of Songs")

plt.title("Most Common Tempo of HIT songs")

plt.show(fig)



sp_data_hit_year=sp_data[(sp_data["bbhot"]==1 )& (sp_data["year"]!=0)]

sp_data_year=sp_data[sp_data["year"]!=0]



df_year_hit=pd.DataFrame(sp_data_hit_year["year"].astype("category").value_counts())

df_year_hit.columns=["hit_count"]

df_year=pd.DataFrame(sp_data_year["year"].astype("category").value_counts())

df_year.columns=["count"]

df_year_final=pd.concat([df_year_hit, df_year], axis=1, join='inner')

df_year_final["year"]=df_year_final.index

sns.lineplot(x="year",y="hit_count",data=df_year_final,)



sns.lineplot(x="year",y="count",data=df_year_final)

plt.xlabel("Year")

plt.ylabel("No. of Hit songs")

plt.title("Total No.of Songs per Year")

plt.legend(['Hit songs','All songs'],loc=1)

plt.show()

# checking the class of key and mode 

print(type(sp_data['key']))

print(type(sp_data['mode']))



# changing the datatype of variables as its depcited by numeric values but internally categorical 

sp_data['key']=sp_data['key'].astype("category")

sp_data['mode']=sp_data['mode'].astype("category")





print("key and its count",sp_data['key'].value_counts())
fig=sns.catplot(x="key", y="end_of_fade_in", data=sp_data)

plt.title("Key v/s End of Fade in")

plt.xlabel("Key")

plt.ylabel("End of Fade in")

plt.show(fig)

rcParams['figure.figsize'] = 10, 8

corr = sp_data.corr()

print("Correlation value is ",corr)

# Plot that gives correlation values 

corr.style.background_gradient()

# Heatmap

sns.heatmap(corr)

#Columns to be included in dataset for model building ..



"""Duration and start_of_fade_out have high correlation 

and hence lets keep only one of them

"""





sp_data_col=[ 'artist_hotttnesss','duration', 'end_of_fade_in', 'key', 'key_confidence', 'loudness',

       'mode', 'mode_confidence','tempo', 'time_signature','time_signature_confidence']





sp_data[['artist_familiarity', 'artist_hotttnesss','duration', 'end_of_fade_in', 'key_confidence', 'loudness',

       'mode_confidence','tempo', 'time_signature','time_signature_confidence']].plot(kind="box")



sp_data['duration'].plot(kind="box")

sp_data['duration'].describe()

sp_data['duration'].quantile([.25,.5,.75,.8,.9,.95,.97,.98,.99,1])



sp_data[sp_data['duration']>618]['duration'].count()

# total we have 100 outliers as per the duration value ..
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection  import train_test_split

from sklearn import metrics as sm

from sklearn.preprocessing import scale

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler





sp_data=sp_data.drop_duplicates(subset=None, keep='first', inplace=False) # dropping duplicate rows from the data ..

#sp_data.dropna(subset=['artist_familiarity'],inplace=True)

sp_data_final=sp_data[sp_data_col]



target=sp_data["bbhot"] # bbhot target value ... i.e song is popular or not



print(sp_data_final.isnull().sum()) # Checking Null Values

print(sp_data_final.duplicated().sum()) # Checking if data is duplicate

# we found there were two duplicate rows in our data 

from imblearn.over_sampling import SMOTE

#We need to scale data before applying any algorithim

sp_data_s = sp_data_final.copy()

for col in sp_data_s.columns:

    if sp_data_s[col].dtype == np.float64 or sp_data_s[col].dtype == np.int64:

        sp_data_s[col] = scale(sp_data_s[col])



#print(deep.head())

#print(sp_data_s.isnull().sum())

#print(sp_data_s.fillna(value=0,inplace=True))

print(sp_data_s.isnull().sum())

print("sp_data",sp_data_s.head())

#sp_data_s=scale(sp_data_final)



sp_train,sp_test,tar_train,tar_test=train_test_split(sp_data_s,target,test_size=0.10,random_state=10)

sme = SMOTE(random_state=2)

sp_train_res, tar_train_res = sme.fit_sample(sp_train, tar_train.ravel())



knn=KNeighborsClassifier(metric='minkowski', weights='uniform',p=2,n_neighbors=5).fit(sp_train_res,tar_train_res)

knn

predict=knn.predict(sp_test)



print(sm.accuracy_score(tar_test,predict))

ac_knn=sm.accuracy_score(tar_test,predict)

fpr, tpr, threshold = sm.roc_curve(tar_test,predict)

roc_auc = sm.auc(fpr, tpr)

print("Roc/auc",roc_auc)

import matplotlib.pyplot as plt



# Plotting the roc curve ..



plt.title('Receiver Operating Characteristic for KNeighborsClassifier')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix



# this has been done above ..

#cer_train,cer_test,tar_train,tar_test=train_test_split(cer_s,target,test_size=0.2,random_state=18)

logr=LogisticRegression().fit(sp_train_res,tar_train_res)



pred=logr.predict(sp_test)

#To check the porbbility wise prediction of each class ...

print(logr.predict_proba(sp_test))



print("Accuracy score",sm.accuracy_score(pred,tar_test))

fpr, tpr, threshold = sm.roc_curve(tar_test,predict)

roc_auc = sm.auc(fpr, tpr)

ac_log=59.6



print("Logistic Regression Auc value ",roc_auc)



print(confusion_matrix(tar_test,pred))
import matplotlib.pyplot as plt



# Plotting the roc curve ..



plt.title('Receiver Operating Characteristic for Logistic Regression')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



from sklearn.model_selection import cross_val_score

from xgboost.sklearn import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

# Applying logistic Regression along with cross validation 

lr = LogisticRegression()

lr.fit(sp_train_res,tar_train_res)



# Function for applying cross validation on different models

def model_accuracy(model,train,target,cross_val):

    cv_scores = cross_val_score(model,train,target, cv=cross_val)

    sns.distplot(cv_scores)

    plt.title('Average score: {}'.format(np.mean(cv_scores)))





model_accuracy(lr, sp_data_s, target,20)

#################################

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold



dtc = DecisionTreeClassifier()



parameter_grid = {'criterion': ['gini', 'entropy'],

                  'splitter': ['best', 'random'],

                  'max_depth': [1, 2, 3, 4, 5],

                  'max_features': [1, 2, 3, 4]}



cross_validation = StratifiedKFold( n_splits=10).get_n_splits(sp_data_s, target)



grid_search = GridSearchCV(dtc, param_grid=parameter_grid, cv=cross_validation)



grid_search.fit(sp_data_s, target)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))



dtc = grid_search.best_estimator_

cv_scores = cross_val_score(dtc,sp_data_s, target)

sns.distplot(cv_scores)

plt.title('Average score: {}'.format(np.mean(cv_scores)))



#####################

acc_cross=cv_scores.max()
sp_data_ohe=pd.get_dummies(sp_data_final, columns=["key","mode"])

sp_data_ohe_s=scale(sp_data_ohe)



sp_train,sp_test,tar_train,tar_test=train_test_split(sp_data_ohe_s,target,test_size=0.3,random_state=20)

#Applying smote on training dataset so as to overcome skewness in the data

sme = SMOTE(random_state=2)

sp_train_res, tar_train_res = sme.fit_sample(sp_train, tar_train.ravel())



xgb_m=XGBClassifier(learning_rate =0.1,n_estimators=100, max_depth=5,

                    min_child_weight=1, gamma=0, subsample=0.8,

                    colsample_bytree=0.8, objective= 'binary:logistic',

                    nthread=4, scale_pos_weight=1, seed=27).fit(sp_train_res,tar_train_res)





predict=xgb_m.predict(sp_test)



print("Accuracy score of XGB Classifier",sm.accuracy_score(tar_test,predict))





fpr, tpr, threshold = sm.roc_curve(tar_test,predict)

roc_auc = sm.auc(fpr, tpr)



fpr, tpr, threshold = sm.roc_curve(tar_test,predict)

roc_auc = sm.auc(fpr, tpr)

print("ROC/AUC value for XGB Classifier",roc_auc)
import matplotlib.pyplot as plt



# Plotting the roc curve ..



plt.title('Receiver Operating Characteristic for XGB Classifier')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Lets apply XGB classifier along with cross validaton and see if there is any improvement

xgb_m=XGBClassifier(learning_rate =0.1,n_estimators=100, max_depth=5,

                    min_child_weight=1, gamma=0, subsample=0.8,

                    colsample_bytree=0.8, objective= 'binary:logistic',

                    nthread=4, scale_pos_weight=1, seed=27)





model_accuracy(xgb_m,sp_train,tar_train,10)