import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/cardiac-arrhythmia-database/data_arrhythmia.csv', delimiter = ';',na_values = ['?'])
df
df.diagnosis.value_counts()
sum(df['diagnosis']!=1) #check how many occurences of non-normal arrythmia exists
df1 = df.copy() #create a copy of the original dataset



#setting prediction values to "Normal" or "Risk" based on column scores values

df1.loc[df1["diagnosis"] == 1,"diagnosis"] = "Normal"         #class 1 is normal arrythmia

df1.loc[df1["diagnosis"] != "Normal","diagnosis"] = "Risk"    #other classes are risk classes

df1.diagnosis.value_counts()
familiar_features = ['age','sex','height','weight','heart_rate'] #list of well known features with missing data



#function that creates an histogram for a feature

def print_hist(df,feature,nbins):

    print("Histogram for " + feature + ":")

    column = df[feature]

    plt.hist(column,bins=nbins)

    plt.show()



for feature in familiar_features: print_hist(df1,feature,30)
df1['age'].value_counts().sort_index() #get different values and count occurences for each value
df1[df1['age']==0] #select occurence with Age=0
df1.loc[df1["age"] == 0, "height"] = 61
df1['height'].value_counts().sort_index()
df1[df1['height']==780] #select occurence with Height=780
df1.loc[df1["height"] == 780, "height"] = 78
one_levels = [] #create an empty list to append features with one level



ncol = df1.shape[1] # store number of columns



#for loop to add features names with one level to "one_levels" list 

for index in range(ncol):

    if len(df1.iloc[:,index].unique()) == 1:

        one_levels.append(df1.columns[index])



one_levels
df2 = df1.copy()

df2 = df1.drop(columns=one_levels) # delete one_level columns 
df2.isnull().sum().sum() #total number of missing values is 408
#print columns with missing values and its occurences

missing = df2.isnull().sum()

missing_df = pd.DataFrame(missing)

missing_df[missing_df[0]!=0]
df3 = df2.copy()

df3 = df3.drop(columns=['J'])

df3
missing_features = ['P','T','QRST','heart_rate'] #list pf features with missing data



for feature in missing_features: print_hist(df3,feature,30)
from sklearn.impute import SimpleImputer



df4 = df3.copy() #create a new copy of the dataset

X_df = df4.loc[:,df4.columns != 'diagnosis'] #select all features except target feature

X = np.array(X_df) #convert it to array (Simple Imputer doesn't work with dataframes)

imp = SimpleImputer(missing_values=np.nan, strategy='median') #create imp object to impute median in all missing values 

imp = imp.fit(X) #calculate median values of the features with missing values

X_imp = imp.transform(X) #fill dataset with median values wherever finds missing values
np.isnan(X_imp).sum().sum() #total number of missing values in the dataset
from sklearn.model_selection import train_test_split



y = df4.iloc[:,-1] #subset target label



# splitting train and test data with same seed (random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.3, random_state=1) #70%/30% splitting



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
print('Normal instances in test set: ' + str(sum(y_test=="Normal")))

print('Risk instances in test set: ' + str(sum(y_test=="Risk"))) 
#create classifier object

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



clf1 = RandomForestClassifier(n_estimators=10,random_state=1) #create classifier object

fit_model1 = clf1.fit(X_train,y_train) #train classifier

y_pred = fit_model1.predict(X_test) #create predictions vector



cm_m1 = metrics.confusion_matrix(y_test,y_pred)

print("Confusion matrix:")

print(cm_m1)



print("Accuracy: " + str(round(metrics.accuracy_score(y_test, y_pred)*100,2))+"%")
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler() #MinMaxScaler default is [0,1] normalization of features

X_norm = scaler.fit_transform(X_imp)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



feat_test = SelectKBest(score_func=chi2, k=100) # k = 100 is not important at this stage (I just want to check the scores)



feat_fit = feat_test.fit(X_norm,y) # chi2 test over all normalised dataset



np.set_printoptions(precision=6, suppress=True) # set precision and avoid scientific notation for scores print

scores = feat_fit.scores_ 

print(scores)
f = plt.figure(figsize=(19, 15))

plt.matshow(df4.corr(), fignum=f.number)

#plt.xticks(range(df4.shape[1]), df4.columns, fontsize=14, rotation=45)

#plt.yticks(range(df4.shape[1]), df4.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
# splitting train and test data with same seed (random_state=1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_norm, y, test_size=0.3, random_state=1) #70%/30% splitting



print('Normal instances in test set: ' + str(sum(y_test2=="Normal")))

print('Risk instances in test set: ' + str(sum(y_test2=="Risk")))



fit_model2 = clf1.fit(X_train2,y_train2) #train classifier

y_pred2 = fit_model2.predict(X_test2) #create predictions vector



cm_m2 = metrics.confusion_matrix(y_test2,y_pred2)

print("Confusion matrix (normalized dataset):")

print(cm_m2)



print("Accuracy: " + str(round(metrics.accuracy_score(y_test2, y_pred2)*100,2))+"%")
from sklearn.feature_selection import RFE



#define a function to train and evaluate a random forest algorithm in a reduced feature space varying the number of features

# and recursively selecting the optimal subset of features using RFE (Feature ranking with recursive feature elimination) 

#from skelearn



def evaluate_feature_reduction_RFE(nfeat):



    selector = RFE(clf1,n_features_to_select=nfeat) # n_features_to_select (number of features) will vary



    best_feats = selector.fit(X_train2, y_train2) #get k best features from normalized training dataset (not all dataset!)



    cols = best_feats.get_support(indices=True) # get best columns indexes to use later when selecting test subset



    X_train_reduced = best_feats.transform(X_train2)  # get train dataframe with feature reduction



    fit_model3 = clf1.fit(X_train_reduced, y_train2) # train model with feature reduction



    X_test3 = X_test2[:,cols] #select same best columns from the unseen test set using indexes stored previously



    y_pred3 = fit_model3.predict(X_test3) #make predictions with unseen instances and those that weren't used in feature selection



    accuracy = round(metrics.accuracy_score(y_test2,y_pred3)*100,3) #compute accuracy for model with k subset of features

    

    return accuracy



accuracies = [evaluate_feature_reduction_RFE(i) for i in range(1,X_train2.shape[1]+1)]



print(accuracies)
best_nfeat = np.argmax(np.array(accuracies))+1

print("Highest accuracy obtained with " + str(best_nfeat) + " features")

selector = RFE(clf1,n_features_to_select=best_nfeat) #best number of features

best_feats = selector.fit(X_train2, y_train2) #get best features from normalized training dataset (not all dataset!)

cols = best_feats.get_support(indices=True) # get best columns indexes to use later when selecting test subset

X_train_reduced = best_feats.transform(X_train2)  # get train dataframe with feature reduction

fit_model3 = clf1.fit(X_train_reduced, y_train2) # train model with feature reduction

X_test3 = X_test2[:,cols] #select same best columns from the unseen test set using indexes stored previously

y_pred3 = fit_model3.predict(X_test3) #make predictions with unseen instances and those that weren't used in feature selection

accuracy = round(metrics.accuracy_score(y_test2,y_pred3)*100,3) #compute accuracy for model with 40 features



cm_m3 = metrics.confusion_matrix(y_test2,y_pred3)

print("Confusion matrix (reduced dataset):")

print(cm_m3)

print("Accuracy: " + str(accuracy)+"%")

from sklearn.metrics import roc_auc_score



y_pred_prob = fit_model3.predict_proba(X_test3) #prediction probabilities vector [P(Normal), P(Risk)]



print('ROC:' + str(round(roc_auc_score(y_test2, y_pred_prob[:,1]),3)))
