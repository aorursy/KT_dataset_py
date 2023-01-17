# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/kag_risk_factors_cervical_cancer.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_full = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv')
df_full
df_full.info()
df_fullna = df_full.replace('?', np.nan)
df_fullna.isnull().sum() #check NaN counts in different columns
df = df_fullna  #making temporary save
df = df.convert_objects(convert_numeric=True) #turn data into numeric type for computation
df.info() # Now it's all numeric type, and we are ready for computation and fill NaN.
# for continuous variable

df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].mean())

df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].mean())

df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].mean())

df['Smokes'] = df['Smokes'].fillna(1)

df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].mean())

df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].mean())

df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)

df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].mean())

df['IUD'] = df['IUD'].fillna(1)

df['IUD (years)'] = df['IUD (years)'].fillna(df['IUD (years)'].mean())

df['STDs'] = df['STDs'].fillna(1)

df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].mean())

df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].mean())

df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].mean())

df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].mean())

df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].mean())

df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].mean())

df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].mean())

df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].mean())

df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].mean())

df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].mean())

df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].mean())

df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].mean())

df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].mean())

df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(df['STDs: Time since first diagnosis'].mean())

df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].mean())

df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].mean())
# for categorical variable

df = pd.get_dummies(data=df, columns=['Smokes','Hormonal Contraceptives','IUD','STDs',

                                      'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology'])
df.isnull().sum() #No null left~
df 
df_data = df #making temporary save
#Shuffle

np.random.seed(42)

df_data_shuffle = df_data.iloc[np.random.permutation(len(df_data))]



df_train = df_data_shuffle.iloc[1:686, :]

df_test = df_data_shuffle.iloc[686: , :]
#分類feature/label

df_train_feature = df_train[['Age', 'Number of sexual partners', 'First sexual intercourse',

       'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',

       'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',

       'STDs:condylomatosis', 'STDs:cervical condylomatosis',

       'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',

       'STDs:syphilis', 'STDs:pelvic inflammatory disease',

       'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',

       'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',

       'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',

       'Schiller', 'Smokes_0.0', 'Smokes_1.0',

       'Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0',

       'IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1',

       'Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1',

       'Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1']]



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

       'Schiller', 'Smokes_0.0', 'Smokes_1.0',

       'Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0',

       'IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1',

       'Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1',

       'Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1']]



test_label = np.array(df_test['Biopsy'])
#Normalization

from sklearn import preprocessing

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

train_feature = minmax_scale.fit_transform(df_train_feature)

test_feature = minmax_scale.fit_transform(df_test_feature)
#Make sure if it's the shape what we want!

print(train_feature[0])

print(train_label[0])

print(test_feature[0])

print(test_label[0])
train_feature.shape
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

model.add(Dense(units=1000, 

                input_dim=45, 

                kernel_initializer='uniform', 

                activation='relu'))

model.add(Dropout(0.5))



#Hidden layer 1

model.add(Dense(units=500,  

                kernel_initializer='uniform', 

                activation='relu'))

model.add(Dropout(0.5))



#Hidden layer 2

model.add(Dense(units=500,  

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

                          validation_split=0.2, epochs=40, 

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
# Answer sheet

prediction = model.predict_classes(test_feature)
# Create a dataframe for prediction and correct answer

df_ans = pd.DataFrame({'Biopsy' :test_label})

df_ans['Prediction'] = prediction
df_ans
df_ans[ df_ans['Biopsy'] != df_ans['Prediction'] ]
df_ans['Prediction'].value_counts()
df_ans['Biopsy'].value_counts()
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
import seaborn as sns

sns.jointplot(x='Age', y='Biopsy', data=df, alpha=0.1) 

#By adding alpha, we can see the density of the scattered spots clearly.
import seaborn as sns

sns.jointplot(x='Number of sexual partners', y='Biopsy', data=df, alpha=0.1) 

#By adding alpha, we can see the density of the scattered spots clearly.
import seaborn as sns

sns.jointplot(x='Num of pregnancies', y='Biopsy', data=df, alpha=0.1) 
corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, square=True, cmap='rainbow')
k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Biopsy')['Biopsy'].index

cm = np.corrcoef(df[cols].values.T)



plt.figure(figsize=(9,9)) #可以調整大小



sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},

                 yticklabels = cols.values, xticklabels = cols.values)

plt.show()