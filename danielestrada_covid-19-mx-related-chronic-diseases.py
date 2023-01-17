import numpy as np   #numpy for lineal algebra
import pandas as pd  # pandas for series and dataframes

from sklearn import preprocessing  #sklearn preprocessing raw data 
import matplotlib.pyplot as plt     #matplotlib visualization, graphs
plt.rc("font", size=14)
import seaborn as sns                #seaborn, built on mathplotlib, graphics for statistical data 
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.simplefilter(action='ignore')

#SOURCING DATA 
#train_df is a dataframe containing the dataset 
train_df = pd.read_csv("/kaggle/input/COVID_CONDICIONES_SUBYACENTES.csv")  

train_df.head()
print('The number of rows (samples) into the dataset is {}.'.format(train_df.shape[0]))  
# check missing values in train data
print('There are not empty cells in the dataset')
train_df.isnull().sum()

#POSITIVE CASES: 1
#NEGATIVE CASES: 2
train_df['RESULTADO'].value_counts()
train_df["NEUMONIA"].value_counts()
#Filling missing values

train_data = train_df.copy()
train_data["NEUMONIA"].replace( to_replace=99, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["DIABETES"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["EPOC"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["ASMA"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["INMUSUPR"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["HIPERTENSION"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["CARDIOVASCULAR"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["OBESIDAD"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["RENAL_CRONICA"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
train_data["TABAQUISMO"].replace( to_replace=98, value=2, inplace=True, limit=None, regex=False, method='pad')
#train_data["RESULTADO"].replace( to_replace=3, value=2, inplace=True, limit=None, regex=False, method='pad')
#create categorical variables (using 1 to indicate presence of health conditions), and drop some variables .
#se crean variables categoricas para indicar de forma discreta (0-1) la presencia de cierta condicion de salud, el sexo o un caso confirmado
training=pd.get_dummies(train_data, columns=["SEXO","NEUMONIA","DIABETES","EPOC","ASMA","INMUSUPR","HIPERTENSION","CARDIOVASCULAR","OBESIDAD","RENAL_CRONICA","TABAQUISMO","RESULTADO"])
training.drop('ENTIDAD_RES', axis=1, inplace=True)
training.drop('SEXO_1', axis=1, inplace=True)
training.drop('NEUMONIA_2', axis=1, inplace=True)
training.drop('DIABETES_2', axis=1, inplace=True)
training.drop('EPOC_2', axis=1, inplace=True)
training.drop('ASMA_2', axis=1, inplace=True)
training.drop('INMUSUPR_2', axis=1, inplace=True)
training.drop('HIPERTENSION_2', axis=1, inplace=True)
training.drop('CARDIOVASCULAR_2', axis=1, inplace=True)
training.drop('OBESIDAD_2', axis=1, inplace=True)
training.drop('RENAL_CRONICA_2', axis=1, inplace=True)
training.drop('TABAQUISMO_2', axis=1, inplace=True)
training.drop('RESULTADO_2', axis=1, inplace=True)
final_train = training
final_train.head()
# 1 ->yes , 0->no
#SEXO_2 -> Men: 1, Women:0
#RESULTADO_1 -> 1:Positive, 0: negative
#descriptive statistics summary
final_train['RESULTADO_1'].describe()
print('COVID-19: SUSPICIOUS CASES REGISTERED IN MEXICO')
print('Negative vs Positive Results')
#RESULT DISTRIBUTION
ax = final_train["RESULTADO_1"].hist(bins=2, color='gray', alpha=0.8) #grafica histograma
ax.set(xlabel='RESULTS')
ticks = ax.get_xticks()
ax.set_xticks(ticks[2::3])
ax.set_xticklabels(labels=['Negative (0)','Positive (1)'])
plt.show()

print('AGE DISTRIBUTION')
print('SUSPICIOUS CASES VS CONFIRMED CASES')
print('Note 1: Suspicious cases include both negative and positive results')
print('Note 2: Observe how the proportion of confirmed cases increases for older people')
Positive_df=final_train.copy()
Positive_df=Positive_df.query('RESULTADO_1==1')
pos_susp=int(Positive_df.shape[0])

plt.subplot(1, 2, 1)
ax1 = final_train["EDAD"].hist(bins=10, density=True, stacked=True, color='teal', alpha=0.6) #grafica histograma
final_train["EDAD"].plot(kind='density', color='teal')  #grafica linea de tendencia de densidad
ax1.set(xlabel='EDAD')      
plt.xlim(-10,85)

plt.subplot(1, 2, 2)
ax2 = Positive_df["EDAD"].hist(bins=10, density=True, stacked=True, color='teal', alpha=0.6) #grafica histograma
Positive_df["EDAD"].plot(kind='density', color='teal')  #grafica linea de tendencia de densidad
ax2.set(xlabel='EDAD')      
plt.xlim(-10,85)


#plt.show()
plt.tight_layout()
print('SUSPICIOUS CASES: SEX FREQUENCY')
print(final_train['SEXO_2'].value_counts())
ax=sns.countplot(x='SEXO_2', data=final_train, palette='Set2')
ticks = ax.get_xticks()
ax.set_xticklabels(labels=['Woman','Man'])
plt.show()
print('CONFIRMED CASES: SEX FREQUENCY')
print(Positive_df['SEXO_2'].value_counts())
ax=sns.countplot(x='SEXO_2', data=Positive_df, palette='Set2')
ticks = ax.get_xticks()
ax.set_xticklabels(labels=['Woman','Man'])
plt.show()
condiciones=list(final_train.columns)
condiciones=condiciones[2:12]

enf_pre_df=final_train.copy()
enf_pre_df=enf_pre_df.query('(NEUMONIA_1==1) or (DIABETES_1==1) or (EPOC_1==1) or (ASMA_1==1) or (INMUSUPR_1==1) or (HIPERTENSION_1==1) or (CARDIOVASCULAR_1==1) or (OBESIDAD_1==1) or (RENAL_CRONICA_1==1) or (TABAQUISMO_1==1) ')
num_enf=int(enf_pre_df.shape[0]) #number of people with preexisting  health conditions
num_susp=int(train_df.shape[0]) #total number of suspected cases
#print(num_enf,num_susp-num_enf)

#confirmed cases
Pos_df=Positive_df.copy()
Pos_df=Pos_df.query('(NEUMONIA_1==1) or (DIABETES_1==1) or (EPOC_1==1) or (ASMA_1==1) or (INMUSUPR_1==1) or (HIPERTENSION_1==1) or (CARDIOVASCULAR_1==1) or (OBESIDAD_1==1) or (RENAL_CRONICA_1==1) or (TABAQUISMO_1==1) ')
pos_enf=int(Pos_df.shape[0]) #number of people with preexisting  health conditions
pos_enf
print( 'Percentage of confirmed cases with an underlying condition: %',(pos_enf/pos_susp)*100)
print('SUSPICIOUS CASES: Reported Chronic Diseases  vs Not Preexisting Health Problems')
print('People having chronic diseases:',num_enf)
print('People without reported health problems:',num_susp-num_enf)
labels1=['Chronic Diseases', 'Not Conditions Reported'] #Rename labels for graphics
sizes1= [num_enf, num_susp-num_enf]
colors1= ['gold', 'yellowgreen']
explode1= (0.1, 0)  # explode 1st slice
# Plot
plt.pie(sizes1, explode=explode1, labels=labels1, colors=colors1)


plt.axis('equal')

plt.show()
print('CONFIRMED CASES: Reported Chronic Diseases  vs Not Preexisting Health Problems')
print('People having chronic diseases:',pos_enf)
print('People without reported health problems:',pos_susp-pos_enf)
labels1=['Chronic Diseases', 'Not Conditions Reported'] #Rename labels for graphics
sizes1= [pos_enf, pos_susp-pos_enf]
colors1= ['gold', 'yellowgreen']
explode1= (0.1, 0)  # explode 1st slice
# Plot
plt.pie(sizes1, explode=explode1, labels=labels1, colors=colors1)


plt.axis('equal')

plt.show()
sizes=[]
for item in condiciones:
    size=final_train[item].value_counts()
    sizes.append(int(size[1]))
#sizes
labels=[item[:-2]  for item in condiciones] #Rename labels for graphics
#sizes = sizes
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'gray', 'red', 'orange','cyan', 'pink', 'teal','indigo']
explode = (0.1, 0,0,0,0,0,0,0,0,0)  # explode 1st slice

print('SUSPICIOUS CASES: Chronic Disease Distribution')
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors)
#autopct=('%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()

#agregar distribucion para casos confirmados
sizes=[]
for item in condiciones:
    size=Positive_df[item].value_counts()
    sizes.append(int(size[1]))
#sizes
print('CONFIRMED CASES: Chronic Disease Distribution')
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors)
#autopct=('%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()

#chronic disease distribution for confirmed cases
final_train.describe()

# SARS-COVID 19  CONFIRMED CASES BY AGE GROUP
#PROBABILITIES
print('Confirmed Cases by Age Groups')
print('Note: This graph shows the proportion of confirmed cases for each age group')
print('As an example; it can be said that around 40% of suspicious cases among people in their fifties ended in confirmed cases')
ax=sns.barplot('EDAD', 'RESULTADO_1', data=final_train,color="aquamarine")
ticks = ax.get_xticks()
labels = ax.get_xticklabels()
ax.set_xticks(ticks[4::5])
ax.set_xticklabels(labels[4::5])
plt.show()

#TENDENCY, INCREASE IN INFECTION PROBABIITY WITH AGE
#CASOS CONFIRMADOS DE SARS-COVID 19 POR SEXO (MUJER=0, HOMBRE=1)
#Calculo de point estimates, central mean
#Probabilidad de un resultado positivo asociada con el sexo  
print('AGE AND SARS-COVID 19')
print('SARS-COVID 19: PROBABILITY OF POSITIVE RESULTS (WOMEN=0, MEN=1)')
sns.pointplot('SEXO_2', 'RESULTADO_1',data=final_train, color="aquamarine")
plt.show()
#SE APRECIA QUE LOS HOMBRES TIENEN 9% MAS PROBABILIDADES DE DAR POSITIVO
#CONFIRMED CASES , PNEUMONIA(no NEUM=0, NEUM=1)
#PROBABILITIES
print('PNEUMONIA AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT PNEU=0, WITH PNEU=1)')
sns.pointplot('NEUMONIA_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
# SARS-COVID 19 AND DIABETES ,  (No DIAB=0, DIAB=1)
#PROBABILITIES
print('DIABETES AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT DIAB=0, WITH DIAB=1)')
sns.pointplot('DIABETES_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#TENDENCY
#SARS-COVID 19 , COPD (NO EPOC=0, EPOC=1)
#PROBABILITIES
print('COPD AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT COPD=0, WITH COPD=1)')
print('Note: The tendency doesn´t seem statistically relevant (few COPD cases were registered)')
sns.barplot('EPOC_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#TENDENCY NOT RELEVANT
#SARS-COVID 19 AND ASTHMA (SIN ASMA=0, CON ASMA=1)
#PROBABILITIES
print('ASTHMA AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT ASTH=0, WITH ASTH=1)')
print('Note: People with asthma tend to have less probability to be diagnosed with COVID-19')
sns.pointplot('ASMA_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#Negative tendency
#SARS-COVID 19 AND INMUNOSPRESION (SIN INMUNOSPR=0, CON INMUNOSUP=1)
#PROBABILITIES
print('IMMUNOSUPPRESSION AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT IMMU=0, WITH IMMU=1)')
sns.pointplot('INMUSUPR_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#NEGATIVE TENDENCY
#SARS-COVID 19 AND CARDIOVASCULAR (SIN CARD=0, CON CARD=1)
#PROBABILITIES
print('CARDIOVASCULAR AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT CARD=0, WITH CARD=1)')
sns.pointplot('CARDIOVASCULAR_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#NOT RELEVANT 
#SARS-COVID 19 AND OBESITY
#PROBABILITIES
print('OBESITY AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT OBES=0, WITH OBES=1)')
sns.pointplot('OBESIDAD_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#TENDENCY
#SARS-COVID 19 AND KIDNEY FAILURE
#PROBABILITIES
print('KIDNEY FAILURE AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT FAILURE=0, WITH FAILURE=1)')
print('Note: It is not statistically relevant')
sns.barplot('RENAL_CRONICA_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#NOT RELEVANT
#SARS-COVID 19 AND SMOKING
#PROBABILITIES
print('SMOKING AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (NOT SMOKERS=0, SMOKERS=1)')
sns.pointplot('TABAQUISMO_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#NOT RELEVANT
#SARS COVID AND HYPERTENSION
#PROBABIITIES
print('HYPERTENSION AND SARS-COVID 19')
print('PROBABILITY OF POSITIVE RESULTS (WITHOUT HYP=0, WITH HYP=1)')
sns.pointplot('HIPERTENSION_1', 'RESULTADO_1', data=final_train, color="aquamarine")
plt.show()
#TENDENCY
depured_data=final_train
depured_data.drop('EPOC_1', axis=1, inplace=True)
depured_data.drop('ASMA_1', axis=1, inplace=True)
#depured_data.drop('HIPERTENSION_1', axis=1, inplace=True)
depured_data.drop('CARDIOVASCULAR_1', axis=1, inplace=True)
depured_data.drop('INMUSUPR_1', axis=1, inplace=True)
depured_data.drop('RENAL_CRONICA_1', axis=1, inplace=True)
depured_data.drop('TABAQUISMO_1', axis=1, inplace=True)

depured_data.head()
# 1 ->YES , 0->NO 
#SEXO_2 ->MAN
#RESULTADO_1 ->CONFIRMED CASE (POSITIVE)
#PREDICTIONS
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# x stores the inputs to out model
X = np.array(depured_data.drop(['RESULTADO_1'],1))  
# y stores de desirible outputs for the model
y = np.array(depured_data['RESULTADO_1'])
X.shape

model = linear_model.LogisticRegression(max_iter=2000) #logistic regression is the selected model
model.fit(X,y)

validation_size = 0.20 # 20% of the dataset is used to validate the model, then 80% is usd to train it
seed = 3 #a seed is given, in case this exact model needs to be replicated
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=seed)

cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_validation)
print("Right Answers")
print(accuracy_score(Y_validation, predictions))

print("Confusion Matrix")
print(confusion_matrix(Y_validation, predictions))

print("Metrics")
print(classification_report(Y_validation, predictions))


X_new = pd.DataFrame({'EDAD': [50], 'SEXO_2': [1], 'NEUMONIA_1':[1],'DIABETES':[0],'HIPERTENSION_1':[0],'OBESIDAD_1':[1]})
print("Predicción del dato de prueba")
print(model.predict(X_new))