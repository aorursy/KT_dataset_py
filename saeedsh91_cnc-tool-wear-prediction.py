#Mixed Naive bayes library installation

!pip install git+https://github.com/remykarem/mixed-naive-bayes#egg=mixed_naive_bayes
# Related Libraries Importing

import pandas as pd

import numpy as np

import glob as gl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from mixed_naive_bayes import MixedNB

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import  train_test_split

import warnings

from sklearn.tree import export_graphviz

import graphviz

from sklearn.tree.export import export_text

warnings.filterwarnings("ignore")
#Phase (1): Data Acquirement

#Objective: Partial datasets grouping and operating condition attachment

"""Note: Please make sure that you uploaded the 18 sub-datasets

named with experiment_#.csv (#=1-18) as well as the train from

the data folder file before running this cell to avoid error"""



#Partial datasets grouping

Data_File_Paths= [i for i in gl.glob("../input/*.csv")]

Data_File_Paths.sort()

y=[pd.read_csv(x) for x in Data_File_Paths]

Combined_Dataset=pd.concat([pd.read_csv(x) for x in Data_File_Paths[:-1]],

                           ignore_index=True)



#Extracting the lengths of each sub-dataset 

length=[]

for x in Data_File_Paths[:-1]:

  length.append(len(pd.read_csv(x)))

Operating_Conditions=pd.read_csv("../input/train.csv")



#Extracting the operation conditions and class label Values

Feed_Rate_Val=list(Operating_Conditions['feedrate'])

Clamping_Pressure_Val=list(Operating_Conditions['clamp_pressure'])

Tool_Condition_Val=list(Operating_Conditions['tool_condition'])



#Equating the opration condition and class label values to the sub-datasets length



for i in range(18):

  Feed_Rate_Val+=Feed_Rate_Val[i:i+1]*length[i]

  Clamping_Pressure_Val+=Clamping_Pressure_Val[i:i+1]*length[i]

  Tool_Condition_Val+=Tool_Condition_Val[i:i+1]*length[i]

del Feed_Rate_Val[0:18]

del Clamping_Pressure_Val[0:18]

del Tool_Condition_Val[0:18]



#Operation conditions and class labeles attachment to the full dataset

Combined_Dataset.insert(loc=46,column='Feed_Rate',value=Feed_Rate_Val)

Combined_Dataset.insert(loc=47,column='Clamp',value=Clamping_Pressure_Val)

Combined_Dataset.insert(loc=50,column='Class',value=Tool_Condition_Val)



Combined_Dataset.head()
#Phase (2): Data Preprocessing 

#Objective: To prapare the data for model muilding phase



#checking the discriptive statistics of the dataset

Combined_Dataset.describe()



#Features with zero StDev extraction

ZeroStdev=Combined_Dataset.describe().T['std'].loc[Combined_Dataset.describe().T['std']==0]

print('the zero stdev features are:\n\n',ZeroStdev,'\n')



#Checking the correlation between features

mask = np.zeros_like(Combined_Dataset.corr().round(2), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(25,25))

sns.heatmap(Combined_Dataset.corr().round(2), mask=mask,vmax=1,

            center=0,square=True, linewidths=.5,annot=True)



#Highly correlated features extraction

corr_matrix = Combined_Dataset.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),

                                      k=1).astype(np.bool))

Highly_Corr = [column for column in upper.columns if any(upper[column] > 0.95)]

print('The highly Correlated features are:\n\n',Highly_Corr)



#Elimination of the attributes with constant values

Combined_Dataset.drop(['Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent',

                              'Z1_OutputVoltage','S1_SystemInertia']+Highly_Corr,axis=1,

                       inplace=True)
#Check for Null - NaN Values

if Combined_Dataset.isnull().sum().sum() == 0:

  print("there is no null Values")

if Combined_Dataset.isna().sum().sum() == 0:

    print("there is no NaN Values")
#Check for erroneous values

print('The number of instances with X1_ActualPosition==198 is :')

print(len(Combined_Dataset.loc[Combined_Dataset.X1_ActualPosition==198]))

print('The number of instances with M1_CURRENT_FEEDRATE ==50 is :')

print(len(Combined_Dataset.loc[Combined_Dataset.M1_CURRENT_FEEDRATE ==50]))

print('The number of instances with X1_ActualPosition==198 and M1_CURRENT_FEEDRATE ==50 is :')

print(len(Combined_Dataset[(Combined_Dataset.X1_ActualPosition==198) &

                     (Combined_Dataset.M1_CURRENT_FEEDRATE ==50)]))

print('the possible values of M1_CURRENT_FEEDRATE is:')

print(list(Combined_Dataset['M1_CURRENT_FEEDRATE'].unique()))

#Erroneous values replacement

Combined_Dataset['X1_ActualPosition'] = Combined_Dataset['X1_ActualPosition'].mask(Combined_Dataset['X1_ActualPosition']>196,196)

Combined_Dataset['M1_CURRENT_FEEDRATE'] = Combined_Dataset['M1_CURRENT_FEEDRATE'].mask(Combined_Dataset['M1_CURRENT_FEEDRATE']==50,20)

Combined_Dataset.head()
#Dataset Scaling

#Check for attributes Range

print(Combined_Dataset.describe().T[['min','max']].round().head(35))

list(Combined_Dataset.describe().T[['max']].values-Combined_Dataset.describe().T[['min']].values)



# Scaling 

Scaler=MinMaxScaler()

Numirical_Var=Combined_Dataset.drop(['Machining_Process','Class'],axis=1)

Categorical_Var=Combined_Dataset[['Machining_Process','Class']]

header={0:'X1_ActualPosition',1:'X1_ActualVelocity',2:'X1_ActualAcceleration',3:'X1_CommandAcceleration',

        4:'X1_CurrentFeedback',5:'X1_DCBusVoltage',6:'X1_OutputCurrent',7:'X1_OutputVoltage',8:'X1_OutputPower',

        9:'Y1_ActualPosition',10:'Y1_ActualVelocity',11:'Y1_ActualAcceleration',12:'Y1_CommandAcceleration',13:'Y1_CurrentFeedback',

        14:'Y1_DCBusVoltage',15:'Y1_OutputCurrent',16:'Y1_OutputVoltage ',17:'Y1_OutputPower ',18:'Z1_ActualPosition',19:'Z1_ActualVelocity',

        20:'Z1_ActualAcceleration',21:'Z1_CommandAcceleration',22:'S1_ActualPosition',23:'S1_ActualVelocity',24:'S1_ActualAcceleration',

        25:'S1_CommandAcceleration',26:'S1_CurrentFeedback',27:'S1_OutputCurrent',28:'M1_CURRENT_PROGRAM_NUMBER',29:'M1_sequence_number',

        30:'Feed_Rate',31:'Clamp',32:'M1_CURRENT_FEEDRATE',33:'Machining_Process',34:'Class'}



Scaled=pd.DataFrame(Scaler.fit_transform(Numirical_Var))

Scaled_Dataset=pd.concat([Scaled,Categorical_Var],ignore_index=True,axis=1,names=header)

Scaled_Dataset.rename(columns=header, inplace=True)

Scaled_Dataset.head()
#Categorial attributes and class Encoding

# Machining process Labels extarction

# lowerCase substitution In end machining process

Scaled_Dataset['Machining_Process'].replace(to_replace='end', value='End',inplace=True)

print(Scaled_Dataset['Machining_Process'].unique())



#Class label encoding

Labels={"Class":{'worn':0,

                 'unworn':1}}

Scaled_Dataset.replace(Labels,inplace=True)



# Machining Crocess binary encoding for Each value

Scaled_Dataset = pd.get_dummies(Scaled_Dataset,columns=['Machining_Process'],)

Scaled_Dataset = Scaled_Dataset.reindex(['X1_ActualPosition','X1_ActualVelocity','X1_ActualAcceleration','X1_CommandAcceleration',

        'X1_CurrentFeedback','X1_DCBusVoltage','X1_OutputCurrent','X1_OutputVoltage','X1_OutputPower','Y1_ActualPosition',

        'Y1_ActualVelocity','Y1_ActualAcceleration','Y1_CommandAcceleration','Y1_CurrentFeedback','Y1_DCBusVoltage',

        'Y1_OutputCurrent','Y1_OutputVoltage ','Y1_OutputPower ','Z1_ActualPosition','Z1_ActualVelocity',

        'Z1_ActualAcceleration','Z1_CommandAcceleration','S1_ActualPosition','S1_ActualVelocity',

        'S1_ActualAcceleration','S1_CommandAcceleration','S1_CurrentFeedback','S1_OutputCurrent',

        'M1_CURRENT_PROGRAM_NUMBER','M1_sequence_number','Feed_Rate','Clamp','M1_CURRENT_FEEDRATE',

        'Machining_Process_Starting','Machining_Process_Prep','Machining_Process_Layer 1 Up',

        'Machining_Process_Layer 1 Down','Machining_Process_Repositioning','Machining_Process_Layer 2 Up',

        'Machining_Process_Layer 2 Down','Machining_Process_Layer 3 Up','Machining_Process_Layer 3 Down',

        'Machining_Process_End','Class'], axis=1)

Scaled_Dataset
#Phase (3): Model Buliding

#Objective: To fit the data into the selected models



#Splitting The date

y=np.array(Scaled_Dataset['Class'])

x=Scaled_Dataset.drop("Class", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=241)



# Decesion tree Classifier fitting

DT_Model=DecisionTreeClassifier(criterion="entropy",splitter='best',min_samples_split=2,random_state=241)

DT_Model.fit(x_train,y_train)

DT_Predict=DT_Model.predict(x_test)

print("Decision tree model has been correctly fitted.'\n")



# Support vector Machine fitting

SVM_Model=SVC(kernel="rbf", gamma=0.1, C=2000,random_state=241)

SVM_Model.fit(x_train,y_train)

SVM_Predict=SVM_Model.predict(x_test)

print("Support vector machine model has been correctly fitted.'\n")



# Multi layer perceptron fitting

MLP_Model=MLPClassifier((100,100,100),activation='relu',solver='adam',max_iter=1000,random_state=241)

MLP_Model.fit(x_train,y_train)

MLP_Predict=MLP_Model.predict(x_test)

print("Multi-layer perceptron model has been correctly fitted.'\n")



KNN_Model=KNeighborsClassifier(n_neighbors=9,weights="uniform",p=2)

KNN_Model.fit(x_train,y_train)

KNN_Predict=KNN_Model.predict(x_test)

print("K-nearest neighbor model has been correctly fitted.'\n")



# Logistic Regression  fitting

LR_Model=LogisticRegression(random_state=241,C=1000)

LR_Model.fit(x_train,y_train)

LR_Predict=LR_Model.predict(x_test)

print("Logistic regression model has been correctly fitted.'\n")



# Mixed Naive Bayes fitting

MNB_Model=MixedNB(categorical_features=[33,34,35,36,37,38,39,40,41,42])

MNB_Model.fit(x_train,y_train)

MNB_Predict=MNB_Model.predict(x_test)

print("Mixed naive bayes model has been correctly fitted.'\n")
#Phase(4): Results presentation

#Objective: To present the evaluation metrics 



#Decesion tree evaluation mitrics

print('Decision tree classifier mitrics:\n')

DT_Acc = round(accuracy_score(y_test, DT_Predict)*100,1)

DT_Conf = confusion_matrix(y_test, DT_Predict)

DT_Sen = round((DT_Conf[0,0]/(DT_Conf[0,0]+DT_Conf[0,1]))*100,1)

DT_Spec = round((DT_Conf[1,1]/(DT_Conf[1,0]+DT_Conf[1,1]))*100,1)

print("Accuracy score:",DT_Acc)

print("Sensitivity score:",DT_Sen)

print("Specificity Score:",DT_Spec,'\n')

sns.heatmap(DT_Conf,annot=True,fmt="0000.0f",

            xticklabels=["worn","unworn"],yticklabels=['worn','unworn'])

plt.show()

print("--------------------------------------------------------------------")



#Support vector machine evaluation mitrics

print('Support vector machine classifier mitrics:\n')

SVM_Acc = round(accuracy_score(y_test, SVM_Predict)*100,1)

SVM_Conf=confusion_matrix(y_test, SVM_Predict)

SVM_Sen = round((SVM_Conf[0,0]/(SVM_Conf[0,0]+SVM_Conf[0,1]))*100,1)

SVM_Spec = round((SVM_Conf[1,1]/(SVM_Conf[1,0]+SVM_Conf[1,1]))*100,1)

print("Accuracy score:",SVM_Acc)

print("Sensitivity score:",SVM_Sen)

print("Specificity Score:",SVM_Spec,'\n')

sns.heatmap(SVM_Conf,annot=True,fmt="0000.0f",

            xticklabels=["worn","unworn"],yticklabels=['worn','unworn'])

plt.show()

print("--------------------------------------------------------------------")



#Multi-layer perceptron evaluation mitrics

print('Multi-layer perceptron classifier mitrics:\n')

MLP_Acc = round(accuracy_score(y_test, MLP_Predict)*100,1)

MLP_Conf = confusion_matrix(y_test, MLP_Predict)

MLP_Sen = round((MLP_Conf[0,0]/(MLP_Conf[0,0]+MLP_Conf[0,1]))*100,1)

MLP_Spec = round((MLP_Conf[1,1]/(MLP_Conf[1,0]+MLP_Conf[1,1]))*100,1)

print("Accuracy score:",MLP_Acc)

print("Sensitivity score:",MLP_Sen)

print("Specificity Score:",MLP_Spec,'\n')

sns.heatmap(MLP_Conf,annot=True,fmt="0000.0f",

            xticklabels=["worn","unworn"],yticklabels=['worn','unworn'])

plt.show()

print("--------------------------------------------------------------------")



#K-nearest neighbor evaluation mitrics

print('K-nearest neighbor classifier mitrics:\n')

KNN_Acc = round(accuracy_score(y_test, KNN_Predict)*100,1)

KNN_Conf=confusion_matrix(y_test, KNN_Predict)

KNN_Sen = round((KNN_Conf[0,0]/(KNN_Conf[0,0]+KNN_Conf[0,1]))*100,1)

KNN_Spec = round((KNN_Conf[1,1]/(KNN_Conf[1,0]+KNN_Conf[1,1]))*100,1)

print("Accuracy score:",KNN_Acc)

print("Sensitivity score:",KNN_Sen)

print("Specificity Score:",KNN_Spec,'\n')

sns.heatmap(KNN_Conf,annot=True,fmt="0000.0f",

            xticklabels=["worn","unworn"],yticklabels=['worn','unworn'])

plt.show()

print("--------------------------------------------------------------------")



#Logistic Regression evaluation mitrics

print('Logistic regression classifier mitrics:\n')

LR_Acc = round(accuracy_score(y_test, LR_Predict)*100,1)

LR_Conf = confusion_matrix(y_test, LR_Predict)

LR_Sen = round((LR_Conf[0,0]/(LR_Conf[0,0]+LR_Conf[0,1]))*100,1)

LR_Spec = round((LR_Conf[1,1]/(LR_Conf[1,0]+LR_Conf[1,1]))*100,1)

print("Accuracy score:",LR_Acc)

print("Sensitivity score:",LR_Sen)

print("Specificity Score:",LR_Spec,'\n')

sns.heatmap(LR_Conf,annot=True,fmt="0000.0f",

            xticklabels=["worn","unworn"],yticklabels=['worn','unworn'])

plt.show()

print("--------------------------------------------------------------------")



#Mixed naive bayes evaluation mitrics

print('Mixed naive bayes classifier mitrics:\n')

MNB_Acc = round(accuracy_score(y_test, MNB_Predict)*100,1)

MNB_Conf=confusion_matrix(y_test, MNB_Predict)

MNB_Sen = round((MNB_Conf[0,0]/(MNB_Conf[0,0]+MNB_Conf[0,1]))*100,1)

MNB_Spec = round((MNB_Conf[1,1]/(MNB_Conf[1,0]+MNB_Conf[1,1]))*100,1)

print("Accuracy score:",MNB_Acc)

print("Sensitivity score:",MNB_Sen)

print("Specificity Score:",MNB_Spec,'\n')

sns.heatmap(MNB_Conf,annot=True,fmt="0000.0f",

            xticklabels=["worn","unworn"],yticklabels=['worn','unworn'])

plt.show()



# overall accuracy scores plotting

Accuracy=[DT_Acc,MLP_Acc,KNN_Acc,SVM_Acc,LR_Acc,MNB_Acc]

Models=['DT','MLP','KNN','SVM','LR','MNB']

sns.barplot(Models,Accuracy)
# Features importance

features = [(Scaled_Dataset.columns[i], v) for i,v in enumerate(DT_Model.feature_importances_)]

features.sort(key=lambda x: x[1], reverse = True)

for item in features[0:10]:

    print("{0}: {1:0.4f}".format(item[0], item[1]))
# Decesion tree plotting

Combined_Dataset= pd.get_dummies(Combined_Dataset,columns=['Machining_Process'],)

Y=np.array(Combined_Dataset['Class'])

X=Combined_Dataset.drop("Class", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.8, random_state=241)



DT_Mod=DecisionTreeClassifier(criterion="entropy",splitter='best',

                              min_samples_split=2,random_state=241)

DT_Mod.fit(x_train,y_train)

DT_Pred=DT_Mod.predict(x_test)



dot_data = export_graphviz(DT_Mod, out_file=None,

                           feature_names = (Combined_Dataset.drop('Class', axis = 1)).columns)

graph = graphviz.Source(dot_data) 

graph
#Decesion rules extraction

tree_rules = export_text(DT_Mod, feature_names=['X1_ActualPosition','X1_ActualVelocity','X1_ActualAcceleration','X1_CommandAcceleration',

        'X1_CurrentFeedback','X1_DCBusVoltage','X1_OutputCurrent','X1_OutputVoltage','X1_OutputPower','Y1_ActualPosition',

        'Y1_ActualVelocity','Y1_ActualAcceleration','Y1_CommandAcceleration','Y1_CurrentFeedback','Y1_DCBusVoltage',

        'Y1_OutputCurrent','Y1_OutputVoltage ','Y1_OutputPower ','Z1_ActualPosition','Z1_ActualVelocity',

        'Z1_ActualAcceleration','Z1_CommandAcceleration','S1_ActualPosition','S1_ActualVelocity',

        'S1_ActualAcceleration','S1_CommandAcceleration','S1_CurrentFeedback','S1_OutputCurrent',

        'M1_CURRENT_PROGRAM_NUMBER','M1_sequence_number','Feed_Rate','Clamp','M1_CURRENT_FEEDRATE',

        'Machining_Process_Starting','Machining_Process_Prep','Machining_Process_Layer 1 Up',

        'Machining_Process_Layer 1 Down','Machining_Process_Repositioning','Machining_Process_Layer 2 Up',

        'Machining_Process_Layer 2 Down','Machining_Process_Layer 3 Up','Machining_Process_Layer 3 Down',

        'Machining_Process_End',"Class"])

print(tree_rules)