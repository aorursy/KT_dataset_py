import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv"]).decode("utf8"))
df = pd.read_csv("../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv")
df.head(13)
df.tail(13)
df.shape
df.info()
df.isnull().any() # Buscamos posibles valores NULL
df.columns
print(df.shape)

count_row = df.shape[0]  # row

print(count_row)

count_col = df.shape[1] # col

print(count_col)
df_na = df.replace('?', np.nan)
df_na.isnull().sum() #check NaN counts in different columns
df_na.head(13)
df = df_na #df temporal

df_na.head(13)
df_na.info()
#df = df.convert_objects(convert_numeric=True) #DEPRECATED

df = df.apply(pd.to_numeric, errors='coerce')

df.head(13)
df.info()
# for continuous variable

df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())

df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())

df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())

df['Smokes'] = df['Smokes'].fillna(1)

df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())

df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())

df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)

df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())

df['IUD'] = df['IUD'].fillna(0) # Under suggestion

df['IUD (years)'] = df['IUD (years)'].fillna(0) #Under suggestion

df['STDs'] = df['STDs'].fillna(1)

df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())

df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())

df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].median())

df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].median())

df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].median())

df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())

df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].median())

df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())

df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].median())

df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())

df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())

df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())

df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())

df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(df['STDs: Time since first diagnosis'].median())

df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].median())
# Variables categoricas a numericas

df = pd.get_dummies(data=df, columns=['Smokes','Hormonal Contraceptives','IUD','STDs',

                                      'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller'])
df.isnull().sum() #No null left~
df.head(13)
df.describe()
df_data = df #making temporary save
fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,figsize=(20,40))

sns.countplot(x='Age', data=df, ax=ax1)

sns.countplot(x='Number of sexual partners', data=df, ax=ax2)

sns.countplot(x='Num of pregnancies', data=df, ax=ax3)

sns.countplot(x='Smokes (years)', data=df, ax=ax4)

sns.countplot(x='Hormonal Contraceptives (years)', data=df, ax=ax5)

sns.countplot(x='IUD (years)', data=df, ax=ax6)

sns.countplot(x='STDs (number)', data=df, ax=ax7)
#Shuffle

np.random.seed(42)

df_data_shuffle = df_data.iloc[np.random.permutation(len(df_data))]



df_train = df_data_shuffle.iloc[1:686, :] #Parametros

df_test = df_data_shuffle.iloc[686: , :] #Parametros
df_train_feature = df_train[['Age', 'Number of sexual partners', 'First sexual intercourse',

       'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',

       'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',

       'STDs:condylomatosis', 'STDs:cervical condylomatosis',

       'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',

       'STDs:syphilis', 'STDs:pelvic inflammatory disease',

       'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',

       'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',

       'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 

       'Smokes_0.0', 'Smokes_1.0',

       'Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0',

       'IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1',

       'Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1',

       'Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1','Schiller_0','Schiller_1']]



train_label = np.array(df_train['Biopsy'])



df_test_feature = df_test[['Age', 'Number of sexual partners', 'First sexual intercourse',

       'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',

       'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',

       'STDs:condylomatosis', 'STDs:cervical condylomatosis',

       'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',

       'STDs:syphilis', 'STDs:pelvic inflammatory disease',

       'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',

       'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',

       'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 

       'Smokes_0.0', 'Smokes_1.0',

       'Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0',

       'IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1',

       'Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1',

       'Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1','Schiller_0','Schiller_1']]



test_label = np.array(df_test['Biopsy'])
df_train_feature.head(7)
df_test_feature.head(7)
#Normalization

from sklearn import preprocessing



minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))



train_feature = minmax_scale.fit_transform(df_train_feature) #Arreglos numpy

test_feature = minmax_scale.fit_transform(df_test_feature)
print(train_feature)
print(train_feature[0])





print(train_label[0])
print(test_feature[0])

print(test_label[0])
print(train_feature.shape)

print(test_feature.shape)
import matplotlib.pyplot as plt





def show_train_history(train_history,train,validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Train History')

    plt.ylabel(train)

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='best')

    plt.show()





######################### Model designing

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



model = Sequential() 



#Input layer

model.add(Dense(units=500, 

                input_dim=46, 

                kernel_initializer='uniform', 

                activation='relu'))

model.add(Dropout(0.5))



#Hidden layer 1

model.add(Dense(units=200,  

                kernel_initializer='uniform', 

                activation='relu'))

model.add(Dropout(0.5))



#Output layer

model.add(Dense(units=1,

                kernel_initializer='uniform', 

                activation='sigmoid'))



print(model.summary()) #for showing the structure and parameters



# Defining how to measure performance

model.compile(loss='binary_crossentropy',   

              optimizer='adam', metrics=['accuracy'])



# Train the model

# Verbose=2, showing loss and accuracy change timely

train_history = model.fit(x=train_feature, y=train_label,  

                          validation_split=0.2, epochs=20, 

                          batch_size=200, verbose=2) 



#visualize the loss and accuracy after each epoch

show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss','val_loss')



#For saving weights

#model.save_weights("Savemodels/Cervical_ca(Kaggles)_MLP.h5")

#print('model saved to disk')
scores = model.evaluate(test_feature, test_label)

print('\n')

print('accuracy=',scores[1])
prediction = model.predict_classes(test_feature)

# print(prediction)
# Create a dataframe for prediction and correct answer

df_ans = pd.DataFrame({'Biopsy' :test_label})

df_ans['Prediction'] = prediction
df_ans.head(5) #Nuevo dataframe
df_ans
df_ans[ df_ans['Biopsy'] != df_ans['Prediction'] ]
df_ans['Prediction'].value_counts()
df_ans['Biopsy'].value_counts()
cols = ['Biopsy_1','Biopsy_0']  #Gold standard

rows = ['Prediction_1','Prediction_0'] #diagnostic tool (our prediction)



#Los valores de cada cuadro en la matriz

B1P1 = len(df_ans[(df_ans['Prediction'] == df_ans['Biopsy']) & (df_ans['Biopsy'] == 1)])



B1P0 = len(df_ans[(df_ans['Prediction'] != df_ans['Biopsy']) & (df_ans['Biopsy'] == 1)])

B0P1 = len(df_ans[(df_ans['Prediction'] != df_ans['Biopsy']) & (df_ans['Biopsy'] == 0)])



B0P0 = len(df_ans[(df_ans['Prediction'] == df_ans['Biopsy']) & (df_ans['Biopsy'] == 0)])



conf = np.array([[B1P1,B0P1],[B1P0,B0P0]])

df_cm = pd.DataFrame(conf, columns = [i for i in cols], index = [i for i in rows])



f, ax= plt.subplots(figsize = (5, 5))

sns.heatmap(df_cm, annot=True, ax=ax) 

ax.xaxis.set_ticks_position('top') #Making x label be on top is common in textbooks.



print('total test case number: ', np.sum(conf))
print(conf)
def model_efficacy(conf):

    total_num = np.sum(conf)

    

    

    sen = conf[0][0]/(conf[0][0]+conf[1][0]) # 4/ (4+4(de abajo))

    spe = conf[1][1]/(conf[1][0]+conf[1][1])

    false_positive_rate = conf[0][1]/(conf[0][1]+conf[1][1])

    false_negative_rate = conf[1][0]/(conf[0][0]+conf[1][0])

    

    print('total_num: ',total_num)

    print('G1P1: ',conf[0][0]) #G = gold standard; P = prediction

    print('G0P1: ',conf[0][1])

    print('G1P0: ',conf[1][0])

    print('G0P0: ',conf[1][1])

    print('##########################')

    print('sensitivity: ',sen)

    print('specificity: ',spe)

    print('false_positive_rate: ',false_positive_rate)

    print('false_negative_rate: ',false_negative_rate)

    

    return total_num, sen, spe, false_positive_rate, false_negative_rate



model_efficacy(conf)
import seaborn as sns

sns.jointplot(x='Age', y='Biopsy', data=df, alpha=0.1) 

#By adding alpha, we can see the density of the scattered spots clearly.
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(15,12))

sns.countplot(x='Age', data=df, ax=ax1)

sns.countplot(x='Biopsy', data=df, ax=ax2)

sns.barplot(x='Age', y='Biopsy', data=df, ax=ax3)



#Estratificado

facet = sns.FacetGrid(df, hue='Biopsy',aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, df['Age'].max()))

facet.add_legend()
import seaborn as sns

sns.jointplot(x='Number of sexual partners', y='Biopsy', data=df, alpha=0.1) 

#By adding alpha, we can see the density of the scattered spots clearly.
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15,8))

sns.countplot(x='Number of sexual partners', data=df, ax=ax1)

sns.barplot(x='Number of sexual partners', y='Biopsy', data=df, ax=ax2) #categorical to categorical



#continuous to categorical

facet = sns.FacetGrid(df, hue='Biopsy',aspect=4)

facet.map(sns.kdeplot,'Number of sexual partners',shade= True)

facet.set(xlim=(0, df['Number of sexual partners'].max()))

facet.add_legend()
import seaborn as sns

sns.jointplot(x='Num of pregnancies', y='Biopsy', data=df, alpha=0.1) 
sns.factorplot('Num of pregnancies','Biopsy',data=df, size=5, aspect=3)
#continuous to categorical

facet = sns.FacetGrid(df, hue='Biopsy',aspect=4)

facet.map(sns.kdeplot,'Num of pregnancies',shade= True)

facet.set(xlim=(0, df['Num of pregnancies'].max()))

facet.add_legend()
import seaborn as sns

sns.jointplot(x='Citology_1', y='Biopsy', data=df, alpha=0.1) 

# Hard do see anything...
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Citology_1', data=df, ax=axis1)

sns.countplot(x='Biopsy', data=df, ax=axis2)

sns.barplot(x='Citology_1', y='Biopsy', data=df, ax=axis3)  #categorical to categorical
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Schiller_1', data=df, ax=axis1)

sns.countplot(x='Biopsy', data=df, ax=axis2)

sns.barplot(x='Schiller_1', y='Biopsy', data=df, ax=axis3) #categorical to categorical
corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, square=True, cmap='rainbow')
df['STDs:cervical condylomatosis'].value_counts()
df['STDs:AIDS'].value_counts()
k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Biopsy')['Biopsy'].index

cm = np.corrcoef(df[cols].values.T)



plt.figure(figsize=(9,9))



sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},

                 yticklabels = cols.values, xticklabels = cols.values)

plt.show()