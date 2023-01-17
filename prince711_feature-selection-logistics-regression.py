import warnings # Ignoreing all the warning
warnings.filterwarnings("ignore")
# Loading all the library
import os
import numpy
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score
%matplotlib inline
file_list = os.listdir('../input')
file_list[0]
Data = pd.read_csv(r'../input/'+file_list[0])
Data.columns # # column Name of Data
Data.head(10)
Data['target'].value_counts() # Get the value_counts for Heart Diseases class and not Heart Diseases 
class_dist_df = pd.DataFrame(Data['target'].value_counts()).reset_index(drop=True)# Reseting the index and get the counts for each class 
class_dist_df['class']= ['HD','WHD'] # HD Means Heart diseases and WHD means NO diseases
sns.barplot(y = 'target', x = class_dist_df['class'], data=class_dist_df) # Plotting Class frequency vs Class
pyplot.title('Class Frequency Vs Class Name')
pyplot.show()
DF_HD = Data[Data['target']==1] # Dataframe of Heart diseases
DF_WHD = Data[Data['target']==0] # Dataframe of Without Heart diseases
DF_HD.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
DF_WHD.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
DF_WHD['age'].describe() #This will output the basic statsitics of age in population
# Hist of Age
DF =DF_HD.append(DF_WHD) 
pd.crosstab(DF['age'],DF['target']).plot(kind="bar",figsize=(20,6))
pyplot.title('Heart Disease Frequency for Ages')
pyplot.xlabel('Age')
pyplot.ylabel('Frequency')
pyplot.savefig('heartDiseaseAndAges.png')
pyplot.show()
print ('male count with the heart dieases ='), DF_HD[DF_HD['sex']==1]['sex'].value_counts() # 1 means Male in Sex Column
print ('male count without the heart dieases ='), DF_WHD[DF_WHD['sex']==1]['sex'].count() # 0 means Female in Sex column
print ('Female count with the heart dieases ='), len(DF_HD[DF_HD['sex']==0]['sex'])
print ('Female count without the heart dieases ='), len(DF_WHD[DF_WHD['sex']==0]['sex'])
Gender_HD = pd.DataFrame(DF_HD['sex'].value_counts()).reset_index() # Value counts for sex with Heart diseases
Gender_WHD = pd.DataFrame(DF_WHD['sex'].value_counts()).reset_index() # Value counts for sex with withot Heart diseases
Gender_HD['class'] = "HD"
Gender_WHD['class'] = "WHD"
Gender_HD['index'] =['Male','Female']
Gender_WHD['index'] =['Male','Female']
Gender_DF = Gender_HD.append(Gender_WHD)
Gender_DF.columns = ['Category','Gender_count','class']
sns.barplot(y='Gender_count', x='Category', data=Gender_DF, hue='class')
pyplot.title('Gender Frequency vs Gender_Name')
pyplot.show()
print ('male age with heart dieases ='), DF_HD[DF_HD['sex']==1]['age'].describe()
print ('male age without heart dieases ='), DF_WHD[DF_WHD['sex']==1]['age'].describe()
print ('Female age without heart dieases ='), DF_WHD[DF_WHD['sex']==0]['age'].describe()
print ('Female age without heart dieases ='), DF_HD[DF_HD['sex']==0]['age'].describe()
DF_HD['cp'].value_counts() # Get value counts for each class 
DF_WHD['cp'].value_counts() # Get value counts for each class 
cp_dist_df = pd.DataFrame(DF_HD['cp'].value_counts()).reset_index() # Value counts for CP with Heart diseases
cp_dist_Wdf = pd.DataFrame(DF_WHD['cp'].value_counts()).reset_index()# Value counts for CP without Heart diseases
cp_dist_df['class'] = "HD"
cp_dist_Wdf['class'] = "WHD"
cp_dist_df_copy = cp_dist_df.copy()
cp_dist_df_copy = cp_dist_df_copy.append(cp_dist_Wdf)
cp_dist_df_copy.columns = [u'index', u'cp_frequency', u'class']
sns.barplot(y='cp_frequency', x='index', data=cp_dist_df_copy, hue='class')
pyplot.show()
DF_HD['trestbps'].describe() # Basic statsitics for trestbps for Heart Diseases cases
DF_WHD['trestbps'].describe() # Basic statsitics for trestbps for without Heart Diseases cases
pyplot.plot(DF[DF['target']==0]['trestbps'].values,'ro',label='WHD') # Heart Diseases cases vs trestbps
pyplot.plot(DF[DF['target']!=0]['trestbps'].values,'bo',label='HD') # Without Heart Diseases cases vs trestbps
pyplot.xlabel('Index')
pyplot.ylabel('trestbps')
pyplot.title('Heart_disease VS trestbps')
pyplot.legend()
pyplot.show()
# Plot between age vs trestbps
pyplot.plot(DF_WHD['age'],DF_WHD['trestbps'],'bo')
pyplot.plot(DF_HD['age'],DF_HD['trestbps'],'ro')
pyplot.title('age VS trestbps')
pyplot.ylabel('age')
pyplot.xlabel('trestbps')
pyplot.show()
DF_HD['chol'].describe()# Basic statsitics for chol for Heart Diseases cases
DF_WHD['chol'].describe() # Basic statsitics for chol for without Heart Diseases cases
# Plot of target variable with chol level
pyplot.plot(DF[DF['target']==0]['chol'].values,'ro',label='WHD')
pyplot.plot(DF[DF['target']!=0]['chol'].values,'bo',label='HD')
pyplot.xlabel('Dummy_Index')
pyplot.ylabel('Chol_values')
pyplot.title('Heart_dieases VS Chol')
pyplot.legend()
pyplot.show()
# Plot of age value with chol level
pyplot.plot(DF_WHD['age'],DF_WHD['chol'],'bo',label='WHD')
pyplot.plot(DF_HD['age'],DF_HD['chol'],'ro',label='HD')
pyplot.title('age VS chol')
pyplot.ylabel('age')
pyplot.xlabel('chol')
pyplot.legend()
pyplot.show()
DF_HD['fbs'].value_counts() # Value counts for categorical data (FBS) for heart disease
DF_WHD['fbs'].value_counts() # Value counts for categorical data (FBS) for without heart disease
fbs_dist_df = pd.DataFrame(DF_HD['fbs'].value_counts()).reset_index()
fbs_dist_Wdf = pd.DataFrame(DF_WHD['fbs'].value_counts()).reset_index()
fbs_dist_df['class'] = "HD"
fbs_dist_Wdf['class'] = "WHD"
fbs_dist_df_copy = fbs_dist_df.copy()
fbs_dist_df_copy = fbs_dist_df_copy.append(fbs_dist_Wdf)
fbs_dist_df_copy.columns = [u'index', u'fbs_frequency', u'class']
sns.barplot(y='fbs_frequency', x='index', data=fbs_dist_df_copy, hue='class')
pyplot.title('fbs_frequency vs class ')
pyplot.show()
DF_HD['restecg'].value_counts()  # Value counts for categorical data (restecg) for heart disease
DF_WHD['restecg'].value_counts() # Value counts for categorical data (restecg) for without heart disease
restecg_dist_df = pd.DataFrame(DF_HD['restecg'].value_counts()).reset_index() # value counts for restecg for Heart diseases Cases
restecg_dist_Wdf = pd.DataFrame(DF_WHD['restecg'].value_counts()).reset_index()# value counts for restecg for without Heart diseases Cases
restecg_dist_df['class'] = "HD"
restecg_dist_Wdf['class'] = "WHD"
restecg_dist_df_copy = restecg_dist_df.copy()
restecg_dist_df_copy = restecg_dist_df_copy.append(restecg_dist_Wdf)
restecg_dist_df_copy.columns = [u'index', u'restecg_frequency', u'class']
sns.barplot(y='restecg_frequency', x='index', data=restecg_dist_df_copy, hue='class')
pyplot.title('restecg_frequency vs index')
pyplot.show()
DF_HD['thalach'].describe() # Basic statstics for thalach for Heart diseases 
DF_WHD['thalach'].describe() # Basic statstics for thalach for without Heart diseases 
# Plot between target vs thalach
pyplot.plot(DF[DF['target']==0]['thalach'].values,'ro',label='WHD')
pyplot.plot(DF[DF['target']!=0]['thalach'].values,'bo',label='HD')
pyplot.xlabel('Dummy_Index')
pyplot.ylabel('thalach')
pyplot.title('thalach vs Heart_dieases')
pyplot.legend()
pyplot.show()
DF_HD['exang'].value_counts()  # Value counts for categorical data (exang) for heart disease
DF_WHD['exang'].value_counts() # Value counts for categorical data (exang) for without heart disease
exang_dist_df = pd.DataFrame(DF_HD['exang'].value_counts()).reset_index()
exang_dist_Wdf = pd.DataFrame(DF_WHD['exang'].value_counts()).reset_index()
exang_dist_df['class'] = "HD"
exang_dist_Wdf['class'] = "WHD"
exang_dist_df_copy = exang_dist_df.copy()
exang_dist_df_copy = exang_dist_df_copy.append(exang_dist_Wdf)
exang_dist_df_copy.columns = ['index','exang_frequency','class']
sns.barplot(y='exang_frequency', x='index', data=exang_dist_df_copy, hue='class')
pyplot.title('exang_frequency vs index')
pyplot.show()
DF_HD[u'oldpeak'].describe() # Basic statstics for oldpeak for Heart diseases 
DF_WHD[u'oldpeak'].describe() # Basic statstics for oldpeak for without Heart diseases 
pyplot.plot(DF[DF['target']==0]['oldpeak'].values,'ro',label='WHD')
pyplot.plot(DF[DF['target']!=0]['oldpeak'].values,'bo',label='HD')
pyplot.xlabel('Dummy_Index')
pyplot.ylabel('oldpeak')
pyplot.title('oldpeak vs Heart_dieases')
pyplot.legend()
pyplot.show()
DF_WHD[u'slope'].value_counts() # Value counts for categorical data (slope) for heart disease
DF_HD[u'slope'].value_counts()  # Value counts for categorical data (slope) for without heart disease
slope_dist_df = pd.DataFrame(DF_HD['slope'].value_counts()).reset_index()
slope_dist_Wdf = pd.DataFrame(DF_WHD['slope'].value_counts()).reset_index()
slope_dist_df['class'] = "HD"
slope_dist_Wdf['class'] = "WHD"
slope_dist_df_copy = slope_dist_df.copy()
slope_dist_df_copy = slope_dist_df_copy.append(slope_dist_Wdf)
slope_dist_df_copy.columns = ['index','slope_frequency','class']
sns.barplot(y='slope_frequency', x='index', data=slope_dist_df_copy, hue='class')
pyplot.show()
DF_HD['ca'].value_counts() # Value counts for categorical data (ca) for heart disease
DF_WHD['ca'].value_counts() # Value counts for categorical data (ca) for without heart disease
ca_dist_df = pd.DataFrame(DF_HD['ca'].value_counts()).reset_index()
ca_dist_Wdf = pd.DataFrame(DF_WHD['ca'].value_counts()).reset_index()
ca_dist_df['class'] = "HD"
ca_dist_Wdf['class'] = "WHD"
ca_dist_df_copy = ca_dist_df.copy()
ca_dist_df_copy = ca_dist_df_copy.append(ca_dist_Wdf)
ca_dist_df_copy.columns =['index','ca_frequency','class']
sns.barplot(y='ca_frequency', x='index', data=ca_dist_df_copy, hue='class')
pyplot.title('ca_frequency vs index')
pyplot.show()
DF_HD['thal'].value_counts()# Value counts for categorical data (thal) for heart disease
DF_WHD['thal'].value_counts()# Value counts for categorical data (thal) for without heart disease
thal_dist_df = pd.DataFrame(DF_HD['thal'].value_counts()).reset_index()
thal_dist_Wdf = pd.DataFrame(DF_WHD['thal'].value_counts()).reset_index()
thal_dist_df['class'] = "HD"
thal_dist_Wdf['class'] = "WHD"
thal_dist_df_copy = thal_dist_df.copy()
thal_dist_df_copy = thal_dist_df_copy.append(thal_dist_Wdf)
thal_dist_df_copy.columns = ['index','thal_frequency','class']
sns.barplot(y='thal_frequency', x='index', data=thal_dist_df_copy, hue='class')
pyplot.show()
Features = ['thal','ca','slope','exang','restecg','trestbps','cp','sex'] # we will choose only these based on above analysis
Data[Features].head(5) # Print the prepared Data
# This Function will convert categorical data to numerical data and put the data of different category into different column
def one_hot_coding(Data,column_name):
    thal_value = Data[column_name].tolist()
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(thal_value)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    new_df = pd.DataFrame(onehot_encoded)
    new_df = rename_column(new_df,column_name)
    return new_df
# This function rename the columns created by above column 
def rename_column(new_df,column_name):
    col_list = []
    for i in range(0,len(new_df.columns)):
        col_list.append(column_name+'_'+str(i))
    new_df.columns = col_list
    return new_df
        

# converting the categorical data to numerical data
new_df_thal = one_hot_coding(Data,'thal')
new_df_ca = one_hot_coding(Data,'ca')
new_df_slope = one_hot_coding(Data,'slope')
new_df_exang = one_hot_coding(Data,'exang')
new_df_restecg = one_hot_coding(Data,'restecg')
new_df_cp = one_hot_coding(Data,'cp')
new_df_sex = one_hot_coding(Data,'sex')
new_df_thalach = Data['thalach']
new_df_oldpeak = Data['oldpeak']
# Merging all the feature Dataframe into single Dataframe
Merged_df = pd.concat([new_df_thal, new_df_ca,new_df_slope,new_df_exang,new_df_restecg,new_df_cp,new_df_sex,new_df_thalach,new_df_oldpeak], axis=1)
# Normalizing the numerical data and bring them in range 0 to 1
Merged_df['thalach'] = (Merged_df['thalach'] - np.min(Merged_df['thalach'])) / (np.max(Merged_df['thalach']) - np.min(Merged_df['thalach']))
Merged_df['oldpeak'] = (Merged_df['oldpeak'] - np.min(Merged_df['oldpeak'])) / (np.max(Merged_df['oldpeak']) - np.min(Merged_df['oldpeak']))
(Merged_df.columns)
# Divide the data into input and Output data 
Merged_df['Output_variable'] = Data['target']
Input_DF = Merged_df.drop(['Output_variable'],axis =1)
# Divide the data into train and test data sets 
X_train, X_test, y_train, y_test = train_test_split(Input_DF, Merged_df['Output_variable'], test_size=0.20, random_state=42)
len(X_train)
len(X_test)
# Intialization of classifier 
classifiers =[]
model1 = LogisticRegression()
classifiers.append(model1)
model2 = SVC()
classifiers.append(model2)
model3 = DecisionTreeClassifier()
classifiers.append(model3)
model4 = RandomForestClassifier()
classifiers.append(model4)
model5 = AdaBoostClassifier()
classifiers.append(model5)
# List of models 
model_name = ['LogisticRegression','Support Vector Machine','DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier']
Training_score ,Testing_score,TP,FP,FN,Precision,Recall,classifiers_list = [],[],[],[],[],[],[],[]
# Running for differnent classifier and Save scores for different classfiers into model
for i in range(0,len(classifiers)):
    clf = classifiers[i]
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    classifiers_list.append(model_name[i])
    Training_score.append(clf.score(X_train,y_train))
    Testing_score.append(clf.score(X_test,y_test))
    TP.append(cm[1][1])
    FP.append(cm[0][1])
    FN.append(cm[1][0])
    Precision.append( precision_score(y_test,y_pred))
    Recall.append(recall_score(y_test,y_pred))
    
Score_DF = pd.DataFrame()
Score_DF['classifiers'] = classifiers_list
Score_DF['Training_score'] = Training_score
Score_DF['Testing_score'] = Testing_score
Score_DF['True_positive'] = TP
Score_DF['False_positive'] = FP
Score_DF['False_negative'] = FN
Score_DF['Precision'] = Precision
Score_DF['Recall'] = Recall
Score_DF
# Since from above LogisticRegression was performing best between among the model and we will try with logistic model with different value of C ()
c =[0.0001,0.001,0.01,0.1,1,10,20,30,40,50] # c is inverse of Regularization Coefficient
Training_score ,Testing_score,TP,FP,FN,Precision,Recall,classifiers_list = [],[],[],[],[],[],[],[]
for i in range(0,len(c)):
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c[i], fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, warm_start=False, n_jobs=None)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    Training_score.append(clf.score(X_train,y_train))
    Testing_score.append(clf.score(X_test,y_test))
    TP.append(cm[1][1])
    FP.append(cm[0][1])
    FN.append(cm[1][0])
    Precision.append( precision_score(y_test,y_pred))
    Recall.append(recall_score(y_test,y_pred))

Score_DF = pd.DataFrame()
Score_DF['C value'] = c
Score_DF['Training_score'] = Training_score
Score_DF['Testing_score'] = Testing_score
Score_DF['True_positive'] = TP
Score_DF['False_positive'] = FP
Score_DF['False_negative'] = FN
Score_DF['Precision'] = Precision
Score_DF['Recall'] = Recall
Score_DF
# plot accuracy vs Regularization Coefficient
pyplot.plot(c,Score_DF['Testing_score'],'r-',label='Testing_Accuracy')
pyplot.plot(c,Score_DF['Training_score'],'b-',label='Trainig_Accuracy')
pyplot.xlabel('Regularization Coefficient')
pyplot.ylabel('Accuracy')
pyplot.legend()
axes = pyplot.gca()
axes.set_ylim([0.70,1])
pyplot.legend()
pyplot.show()
# plot scores(Precision,Recall) vs Regularization Coefficient
pyplot.plot(c,Score_DF['Precision'],'g-',label='Precision')
pyplot.plot(c,Score_DF['Recall'],'y-',label='Recall')
pyplot.xlabel('Regularization Coefficient')
pyplot.ylabel('scores')
pyplot.legend()
pyplot.show()
# we will try with L1 Penalty and different value of Regularization Coefficient
c =[0.1,1,10,20,30,40,50]
Training_score ,Testing_score,TP,FP,FN,Precision,Recall,classifiers_list = [],[],[],[],[],[],[],[]
for i in range(0,len(c)):
    clf = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=c[i], fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, warm_start=False, n_jobs=None)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    Training_score.append(clf.score(X_train,y_train))
    Testing_score.append(clf.score(X_test,y_test))
    TP.append(cm[1][1])
    FP.append(cm[0][1])
    FN.append(cm[1][0])
    Precision.append( precision_score(y_test,y_pred))
    Recall.append(recall_score(y_test,y_pred))

Score_DF = pd.DataFrame()
Score_DF['C value'] = c
Score_DF['Training_score'] = Training_score
Score_DF['Testing_score'] = Testing_score
Score_DF['True_positive'] = TP
Score_DF['False_positive'] = FP
Score_DF['False_negative'] = FN
Score_DF['Precision'] = Precision
Score_DF['Recall'] = Recall
Score_DF
# plot accuracy vs Regularization Coefficient
pyplot.plot(c,Score_DF['Testing_score'],'r-',label='Testing_Accuracy')
pyplot.plot(c,Score_DF['Training_score'],'b-',label='Trainig_Accuracy')
pyplot.xlabel('Regularization Coefficient')
pyplot.ylabel('Accuracy')
pyplot.legend()
axes = pyplot.gca()
axes.set_ylim([0.70,1])
pyplot.legend()
pyplot.show()
# plot scores(Precision,Recall) vs Regularization Coefficient
pyplot.plot(c,Score_DF['Precision'],'g-',label='Precision')
pyplot.plot(c,Score_DF['Recall'],'y-',label='Recall')
pyplot.xlabel('Regularization Coefficient')
pyplot.ylabel('scores')
pyplot.legend()
pyplot.show()