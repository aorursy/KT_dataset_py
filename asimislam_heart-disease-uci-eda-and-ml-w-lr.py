import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/heart.csv')

df.shape
#  check for NULL values

print('\n--- NULL count ---\n{}'.format(df.isnull().sum()))

#df.dropna(inplace=True)             #  drop NULLs



#  check for DUPLICATES

print('\n1st DUPLICATE count:\t{}'.format(df.duplicated().sum()))

df.drop_duplicates(inplace = True)  #  drop duplitcates

print('2nd DUPLICATE count:\t{}'.format(df.duplicated().sum()))
df.rename(columns={

        'cp':'chest_pain_type', 'trestbps':'resting_blood_pressure',

        'chol':'cholestoral','fbs':'fasting_blood_sugar',

        'restecg':'resting_electrocardiographic','thalach':'maximum_heart_rate',

        'exang':'exercise_induced_angina','oldpeak':'ST_depression',

        'slope':'slope_peak_exercise_ST','ca':'number_of_major_vessels'},

    inplace=True)



print(df.columns.tolist())
df['sex'] = df['sex'].map({0:'female', 1:'male'})

df['chest_pain_type'] = df['chest_pain_type'].map({

        0:'typical angina', 1:'atypical angina',

        2:'non-anginal',    3:'asymptomatic'})

df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({

        0:'> 120 mg/dl', 1:'< 120 mg/dl'})

df['resting_electrocardiographic'] = df['resting_electrocardiographic'].map({

        0:'normal', 1:'ST-T wave abnormality', 2:'ventricular hypertrophy'})

df['exercise_induced_angina'] = df['exercise_induced_angina'].map({

        0:'no', 1:'yes'})

df['slope_peak_exercise_ST'] = df['slope_peak_exercise_ST'].map({

        0:'upsloping', 1:'flat', 2:'downsloping'})

df['thal'] = df['thal'].map({

        0:'normal 0',     1:'normal 1',

        2:'fixed defect', 3:'reversable defect'})

df['target'] = df['target'].map({0:'no disease', 1:'disease'})



df.head()
print(df.info())         #  dataset size and types

print('\nData Shape:  {}'.format(df.shape))
df.describe()   #  NUMERICAL DATA
df.describe(include=['O'])   #  CATEGORICAL DATA
df.head()
#  Separate out Categorical and Numeric data

colCAT = []

colNUM = []

for i in df.columns:

    if (len(df[i].unique())) > 5:

        colNUM.append(i)

    else:

        colCAT.append(i)

    print('unique values:  {}\t{}'.format(len(df[i].unique()),i))



dataCAT = df[colCAT]     #  Categorical columns

colNUM.append('target')  #  add target column to Numeric

dataNUM = df[colNUM]     #  Numeric columns
dataCAT.head()  # categorical dataframe from Section 3.3
diseaseCAT    = df[(df['target'] == 'disease')]

no_diseaseCAT = df[(df['target'] == 'no disease')]



#  fig.add_subplot([# of rows] by [# of columns] by [plot#])

subNumOfRow = len(dataCAT.columns)

subNumOfCol = 3     # three columns: overall, no disease, disease

subPlotNum  = 1     # initialize plot number



fig = plt.figure(figsize=(16,60))



for i in colCAT:

    # overall

    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)

    plt.title('OVERALL - {}'.format(i), fontsize=14)

    plt.xlabel(i, fontsize=12)

    sns.swarmplot(data=df, x=df[i],y=df.age,hue=df.target)

    subPlotNum = subPlotNum + 1

    # no_diseaseCAT

    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)

    plt.title('NO DISEASE, target = 0', fontsize=14)

    plt.xlabel(i, fontsize=12)

    sns.swarmplot(data=no_diseaseCAT, x=no_diseaseCAT[i],y=no_diseaseCAT.age,color='darkorange')

    subPlotNum = subPlotNum + 1

    # diseaseCAT

    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)

    plt.title('DISEASE, target = 1', fontsize=14)

    plt.xlabel(i, fontsize=12)

    #sns.countplot(diseaseCAT[i], hue=df.sex)#,color='darkred')

    sns.swarmplot(data=diseaseCAT, x=diseaseCAT[i],y=diseaseCAT.age,color='blue')

    subPlotNum = subPlotNum + 1

plt.show()
dataNUM.head()  # numeric dataframe from Section 3.3
#  assign NUM dataframe for "no disease" and "disease"

no_diseaseNUM = dataNUM[(df['target'] == 'no disease')]

diseaseNUM    = dataNUM[(df['target'] == 'disease')]



#  fig.add_subplot([# of rows] by [# of columns] by [plot#])

subNumOfRow = len(dataNUM.columns)-1   #  x='age' in plots, drop column

subNumOfCol = 3     # three columns: overall, no disease, disease

subPlotNum  = 1     # initialize plot number



fig = plt.figure(figsize=(16,30))



for i in dataNUM.columns.drop(["age","target"]):

    # overall

    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)

    plt.title('OVERALL', fontsize=14)

    plt.xlabel(i, fontsize=12)

    sns.distplot(df[i],color='black')

    subPlotNum = subPlotNum + 1

    # no_diseaseNUM

    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)

    plt.title('NO DISEASE, target = 0', fontsize=14)

    plt.xlabel(i, fontsize=12)

    sns.distplot(no_diseaseNUM[i],color='darkorange')

    subPlotNum = subPlotNum + 1

    # diseaseNUM

    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)

    plt.title('DISEASE, target = 1', fontsize=14)

    plt.xlabel(i, fontsize=12)

    sns.distplot(diseaseNUM[i],color='darkblue')

    subPlotNum = subPlotNum + 1



plt.show()
#  one hot encoding works on type 'object'

for i in colCAT:

    df[i] = df[i].astype(object)

    

df_OHE = df[colCAT]               #  dataframe with categorical values

df_OHE = pd.get_dummies(df_OHE)   #  one-hot encoding

df_OHE = df_OHE.join(df[colNUM])  #  add numeric columns



#  change target data to 0/1

df_OHE['target'] = df_OHE['target'].map({'no disease':0,'disease':1})

df_OHE = df_OHE.drop(['target_disease', 'target_no disease'], axis=1)



df_OHE.head()
from sklearn.preprocessing import MinMaxScaler

norm = MinMaxScaler().fit_transform(df_OHE)

norm[0:2]
#  dataframe with the One Hot Encoding and Normalized data

df = pd.DataFrame(norm, index=df_OHE.index, columns=df_OHE.columns)

df.head()
dataCorr = df.corr()

plt.figure(figsize=(20,20))

plt.title('Heart Disease - CORRELATION, Overall', fontsize=14)

sns.heatmap(dataCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')
corrALL = dataCorr['target'].sort_values(ascending=False)

corrALL = corrALL.drop(['target'])

corrALL.to_frame()
plt.figure(figsize=(16,16))

plt.title('Heart Disease - CORRELATION, Overall', fontsize=14)

ax = sns.barplot(y=corrALL.index,x=corrALL.values)

for p in ax.patches:

    ax.annotate("%.4f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()))

plt.show()
dataFemale = df[(df['sex_female'] == 1)]

dataFemaleCorr = dataFemale.drop(['sex_female','sex_male'], axis=1).corr()

dataFemaleCorr = dataFemaleCorr['target'].sort_values(ascending=False)

dataFemaleCorr['number_of_major_vessels_4'] = 0  # -7.9e-17  all numbers will be exp if not set to 0

dataFemaleCorr.to_frame()

dataFemaleCorr = dataFemaleCorr.drop(['target'])  # for barplot



plt.figure(figsize=(16,16))

plt.title('Heart Disease - CORRELATION, Female', fontsize=14)

ax = sns.barplot(y=dataFemaleCorr.index,x=dataFemaleCorr.values)

for p in ax.patches:

    ax.annotate("%.4f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()))

plt.show()
dataMale   = df[(df['sex_male'] == 1)]

dataMaleCorr = dataMale.drop(['sex_female','sex_male'], axis=1).corr()

dataMaleCorr = dataMaleCorr['target'].sort_values(ascending=False)

dataMaleCorr = dataMaleCorr.drop(['target'])



plt.figure(figsize=(16,16))

plt.title('Heart Disease - CORRELATION, Male', fontsize=14)

ax = sns.barplot(y=dataMaleCorr.index,x=dataMaleCorr.values)

for p in ax.patches:

    ax.annotate("%.4f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()))

plt.show()
from sklearn.model_selection import train_test_split



X = df.drop(['target'], axis = 1)

y = df['target']



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:  ', X_train.shape,  y_train.shape)

print ('Test set:   ', X_test.shape,  y_test.shape)
from sklearn.linear_model import LogisticRegression



LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

LR
#  predict

y_predict = LR.predict(X_test)

y_predict[0:10]
from sklearn.model_selection import cross_val_score



print(cross_val_score(LR, X_train, y_train, cv=5, scoring='accuracy'))

print('Cross Validation Score (mean):  {:3.4%}'.format(cross_val_score(LR, X_train, y_train, cv=5, scoring='accuracy').mean()))
from sklearn.metrics import accuracy_score



print('Accuracy Score:  {:3.4%}'.format(accuracy_score(y_test,y_predict)))
from sklearn.metrics import f1_score



f1score = f1_score(y_test, y_predict)

print('F1 Score:  {:3.4%}'.format(f1score))
from sklearn.metrics import confusion_matrix



conf_matrix = confusion_matrix(y_test, y_predict)



sns.heatmap(conf_matrix, annot=True,cmap='Blues',annot_kws={"size": 30})

plt.title("Confusion Matrix, F1 Score: {:3.4%}".format(f1score))

plt.show()



print('True Positive:\t{}'.format(conf_matrix[0,0]))

print('True Negative:\t{}'.format(conf_matrix[0,1]))

print('False Positive:\t{}'.format(conf_matrix[1,0]))

print('False Negative:\t{}'.format(conf_matrix[1,1]))
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



LR.probability = True   # need for predict_proba to work

LR.fit(X_train,y_train)

y_predita = LR.predict_proba(X_test)

y_predita = y_predita[:,1]   # positive values only

    

ROC_AUC = roc_auc_score(y_test, y_predita)

fpr, tpr, thresholds = roc_curve(y_test, y_predita)



plt.plot([0,1],[0,1], linestyle='--')

plt.plot(fpr, tpr, marker='.')

plt.title("ROC Curve, ROC_AUC Score: {:3.4%}".format(ROC_AUC))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
from sklearn.metrics import classification_report



print(classification_report(y_test,y_predict))
from sklearn.metrics import log_loss



#  predict_proba returns estimates for all classes

y_predict_prob = LR.predict_proba(X_test)

print(y_predict_prob[0:5])



print("\nLog Loss:  {:3.4}".format(log_loss(y_test, y_predict_prob)))