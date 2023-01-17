import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

rawData = pd.read_csv("../input/ncdb-2014/NCDB_2014.csv", 

                   na_values=['U', 'UU', 'UUU', 'UUUU',

                            'N', 'NN', 'NNN', 'NNNN',

                            'X', 'XX', 'XXX', 'XXXX',

                            'Q', 'QQ', 'QQQ', 'QQQQ'])
rawData.shape
#create death attribute 

col_p_isev = rawData['P_ISEV']

col_death = []

for j in range(len(col_p_isev)):

    if col_p_isev[j] == 3:

        col_death.append(1)

    else: 

        col_death.append(0)

rawData.insert(20, "P_DEATH", col_death, True)
rawData.head(5)
rawData_nonan = rawData.dropna()
print('With null values, row_count is: %d'%rawData.shape[0])

print('Without null values, row_count is: %d'%rawData_nonan.shape[0])
#reindex data

rawData_nonan = rawData_nonan.reset_index()

rawData_nonan = rawData_nonan.drop('index',1)
#change Sex to binary 

#Map Sex from F/M 0/1

col_p_sex = rawData_nonan['P_SEX']



col_sex = []

for m in range(len(col_p_sex)):

    if col_p_sex[m] == 'M':

        col_sex.append(1)

    else: 

        col_sex.append(0)

rawData_nonan['P_SEX'] = col_sex

rawData_nonan.head(5)
#create df of clean data 

colsToConvert = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS', 'C_CONF',

       'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF', 'V_ID', 'V_TYPE',

       'V_YEAR', 'P_ID', 'P_SEX', 'P_AGE', 'P_PSN', 'P_ISEV', 'P_DEATH',

       'P_DEATH', 'P_SAFE', 'P_USER']

rawData_nonan[colsToConvert] = rawData_nonan[colsToConvert].applymap(np.int64)
# split data into training and validation set 

All_Data = rawData_nonan

P_DEATH = All_Data.iloc[:,20] 
#reindex all data 

All_Data = All_Data.reset_index()

All_Data = All_Data.drop('index',1)
# chi square test for independence 

from scipy import stats

import statsmodels.api as sm



for column in All_Data.columns:

    if column != 'P_DEATH' :

        table =  pd.crosstab(All_Data[column],All_Data['P_DEATH'])

        stat, p, dof, expected = stats.chi2_contingency(table)

        if p < 0.05:

            print('The %s and P_DEATH are dependent'%column)

        else:

            print('The %s and P_DEATH are independent'%column)
#plot correlation coefficient with P_DEATH

P_DEATH = All_Data['P_DEATH'].values



# calculate correlation coefficient of each feature with survival

features =  ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS', 'C_CONF',

       'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF', 'V_ID', 'V_TYPE',

       'V_YEAR', 'P_ID', 'P_SEX', 'P_AGE', 'P_PSN', 'P_ISEV', 'P_SAFE', 'P_USER']



corrdeath = []

for l in range(len(features)):

    P_DEATH = All_Data['P_DEATH'].values 

    C_WDAY = All_Data[features[l]].values

    corr_matrix = np.corrcoef(P_DEATH,C_WDAY)

    corr = corr_matrix[0]

    corr_coef = corr[1]

    corrdeath.append(corr_coef)

plt.barh(features, corrdeath) 
#correlation of features with each other 



#get correlations of each features in dataset

corrmat = All_Data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(All_Data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#select Features using Random Forests 

data_no_p_death = All_Data.drop('P_DEATH',1)

clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

clf.fit(data_no_p_death, P_DEATH)
feat_labels = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS', 'C_CONF',

       'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF', 'V_ID', 'V_TYPE',

       'V_YEAR', 'P_ID', 'P_SEX', 'P_AGE', 'P_PSN', 'P_ISEV', 'P_SAFE', 'P_USER']

print('**** Columns and their Gini importance ranked *****')

list_of_gini_imp = []

for feature in zip(feat_labels, clf.feature_importances_):

    list_of_gini_imp.append(feature)

list_of_gini_imp.sort(key = lambda  tup: tup[1], reverse=True)  # sorts in place

for e in range(len(list_of_gini_imp)):

    print(list_of_gini_imp[e])

    

from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBest

data_no_p_death_p_isev  = data_no_p_death.drop('P_ISEV', axis = 1)



print("*** top 5 columns out of SelectKBest***")

#X_new = SelectKBest(chi2, k=7).fit_transform(X1, Y1)

#print(X_new.shape) 



#top 5 columns (not ranked) 

selector = SelectKBest(chi2, k=5)

selector.fit(data_no_p_death_p_isev, P_DEATH)

All_Data.columns[selector.get_support(indices=True)]



#  get the list

col_names = list(All_Data.columns[selector.get_support(indices=True)])

#cheaking at print(col_names)

print(col_names)

#Taken the common featues - considering Headmap(ignored the strongly correlated features) , Randon Forest, KBest feature selection models



#C_HOUR

C_hourdummy = pd.get_dummies(All_Data['C_HOUR'])

chour_label = []

for i in range(24):

    label = 'C_hour_%d' %i

    chour_label.append(label)

C_hourdummy.columns = chour_label







#C_SEV

C_sevdummy = pd.get_dummies(All_Data['C_SEV'])

C_sevdummy.columns = ['C_SEV_1', 'C_SEV_2']





#C_CONF

C_confdummy = pd.get_dummies(All_Data['C_CONF'])

C_confdummy.columns = ['C_CONF_1','C_CONF_2','C_CONF_3','C_CONF_4','C_CONF_5','C_CONF_6','C_CONF_21','C_CONF_22','C_CONF_23','C_CONF_24','C_CONF_25','C_CONF_31','C_CONF_32','C_CONF_33','C_CONF_34','C_CONF_35','C_CONF_36','C_CONF_41']





#C_RALN

C_ralndummy = pd.get_dummies(All_Data['C_RALN'])

raln_label = []

for i in range(1,7):

    label = 'C_RALN_%d' %i

    raln_label.append(label)

C_ralndummy.columns = raln_label





# V_TYPE

V_typedummy = pd.get_dummies(All_Data['V_TYPE'])

V_typedummy.columns = ['V_TYPE_1','V_TYPE_5','V_TYPE_6','V_TYPE_7','V_TYPE_8','V_TYPE_9','V_TYPE_10','V_TYPE_11','V_TYPE_14','V_TYPE_17','V_TYPE_18','V_TYPE_21','V_TYPE_23']



# P_AGE

P_age = All_Data.loc[:,'P_AGE'] 





ModelAllData = pd.concat([C_hourdummy, C_sevdummy, C_confdummy, C_ralndummy,  V_typedummy, P_age], axis=1, sort=False)





print(ModelAllData.columns)
#Further eliminating features after creating dummies(for categorical columns)

from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBest

#data_no_p_death_p_isev  = data_no_p_death.drop('P_ISEV', axis = 1)



print("*** top 20 columns out of SelectKBest***")



P_Death=All_Data['P_DEATH']



selector = SelectKBest(chi2, k=20)

selector.fit(ModelAllData, P_Death)

#sel_feature=ModelAllData

#sel_feature.columns[selector.get_support(indices=True)]



#  get the list

col_names = list(ModelAllData.columns[selector.get_support(indices=True)])

print(col_names)







print(ModelAllData.shape)



#data_selected_features=sel_feature[['C_hour_1', 'C_hour_2', 'C_hour_3', 'C_hour_4', 'C_hour_5', 'C_SEV_1', 'C_SEV_2', 'C_CONF_3', 'C_CONF_4', 'C_CONF_6', 'C_CONF_21', 'C_CONF_31', 'C_CONF_36', 'C_RALN_1', 'C_RALN_3', 'C_RALN_4', 'C_RALN_6', 'V_TYPE_14', 'V_TYPE_18', 'P_AGE']]

ModelTrainData, ModelTestData, P_DeathTrain, P_DeathTest = train_test_split(ModelAllData,P_Death, test_size=0.2,random_state=1)



Selected_features_train_all = ModelTrainData[['C_hour_1', 'C_hour_2', 'C_hour_3', 'C_hour_4', 'C_hour_5', 'C_SEV_1', 'C_SEV_2', 'C_CONF_3', 'C_CONF_4', 'C_CONF_6', 'C_CONF_21', 'C_CONF_31', 'C_CONF_36', 'C_RALN_1', 'C_RALN_3', 'C_RALN_4', 'C_RALN_6', 'V_TYPE_14', 'V_TYPE_18', 'P_AGE']].copy()

Selected_features_test_all = ModelTestData[['C_hour_1', 'C_hour_2', 'C_hour_3', 'C_hour_4', 'C_hour_5', 'C_SEV_1', 'C_SEV_2', 'C_CONF_3', 'C_CONF_4', 'C_CONF_6', 'C_CONF_21', 'C_CONF_31', 'C_CONF_36', 'C_RALN_1', 'C_RALN_3', 'C_RALN_4', 'C_RALN_6', 'V_TYPE_14', 'V_TYPE_18', 'P_AGE']].copy()

print(Selected_features_train_all.shape)
#implement the model 

import statsmodels.api as sm

model=sm.Logit(P_DeathTrain,Selected_features_train_all)

result = model.fit()

yPred = result.predict(Selected_features_test_all)

print(yPred.head(5))

result.summary2()
#further features are dropped because , they are insignificant for being p value>0.05 in above summary

Selected_features_train = Selected_features_train_all.drop(columns=[ 'C_hour_2', 'C_hour_3', 'C_hour_4', 'C_SEV_2','C_CONF_6', 'C_CONF_21','C_CONF_36', 'C_RALN_1', 'C_RALN_3','C_RALN_6', 'V_TYPE_18'])

Selected_features_test = Selected_features_test_all.drop(columns=['C_hour_2', 'C_hour_3', 'C_hour_4', 'C_SEV_2','C_CONF_6', 'C_CONF_21','C_CONF_36', 'C_RALN_1', 'C_RALN_3','C_RALN_6', 'V_TYPE_18'])



model=sm.Logit(P_DeathTrain,Selected_features_train)

result = model.fit()

yPred = result.predict(Selected_features_test)

print(yPred.head(5))

result.summary2()
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg_model = LogisticRegression()

result=logreg_model.fit(Selected_features_train, P_DeathTrain)

y_pred = logreg_model.predict(Selected_features_test)

# predict binary outcome 

# now the Prdicted values are binary as expected

accuracy_of_pred = round(100*(logreg_model.score(Selected_features_test, P_DeathTest)),3)

print('the accuracy of the prediction is ' + repr(accuracy_of_pred) + '%')
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(P_DeathTest, y_pred)

print(confusion_matrix)

TN = confusion_matrix[0,0]

print('True Negative is ' + repr(TN)) 

FN = confusion_matrix[1,0]

print('False Negative is ' + repr(FN))

FP = confusion_matrix[0,1]

print('False Positive is ' + repr(FP))

TP = confusion_matrix[1,1]

print('True Positive is ' + repr(TP))

sensetivity = round(TP/(TP+FN),3)

specificity = round(TN/(TN+FP),3)

pos_pred_val_prec = round(TP/(TP+FP),3)

neg_pred_val = round(TN/(TN+FN),3)

print('The sensetivivty of the model is ' + repr(sensetivity))

print('The specificty of the model is ' + repr(specificity))

print('The positive predictive/ precision value of the model is ' + repr(pos_pred_val_prec))

print('The negative predictive value of the model is ' + repr(neg_pred_val))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(P_DeathTest, logreg_model.predict(Selected_features_test))

fpr, tpr, thresholds = roc_curve(P_DeathTest, logreg_model.predict_proba(Selected_features_test)[:,1])



#plotting ROC curve 

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#first get the probability of getting death 

import math as math

prob_pred_binary = logreg_model.predict_proba(Selected_features_test)

prob_pred = []

for i in range(len(prob_pred_binary)):

    prob_pred.append(prob_pred_binary[i][1])



log_odds_pred = []

for j in range(len(prob_pred)):

    log_odds_pred.append(math.log(prob_pred[j]/(1-prob_pred[j])))



log_odds_pred_dict = {'Log_odds_pred' : log_odds_pred}

log_odds_pred_df = pd.DataFrame(log_odds_pred_dict)    

Model_diag_data = pd.concat([Selected_features_test, log_odds_pred_df], axis=1, sort=False)



import seaborn as sns

for i in Model_diag_data.columns:

    sns.lmplot(x=i,y='Log_odds_pred',data=Model_diag_data,fit_reg=True) 
y_pred_df = pd.DataFrame({'y_pred' : y_pred})

plt.scatter(y_pred_df.index, y_pred)

plt.title('P_DEATH Vs. Order of Prediction')

plt.ylabel('Variable P_Death')

plt.xlabel('Order of Prediction')