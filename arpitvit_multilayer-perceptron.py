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
cancer_ds = pd.read_csv('C:/Users/ARPIT/Desktop/Notes 2/PROJECT SEM 2/DM PROJECT/data/kag_risk_factors_cervical_cancer.csv')
cancer_ds.head()
cancer_ds
cancer_ds.info()
cancer_ds_na = cancer_ds.replace('?', np.nan)
cancer_ds_na.isnull().sum()
temp_ds=cancer_ds_na
temp_ds = temp_ds.convert_objects(convert_numeric=True)
temp_ds.info()
temp_ds['Number of sexual partners'] = temp_ds['Number of sexual partners'].fillna(temp_ds['Number of sexual partners'].median())
temp_ds['First sexual intercourse'] = temp_ds['First sexual intercourse'].fillna(temp_ds['First sexual intercourse'].median())
temp_ds['Num of pregnancies'] = temp_ds['Num of pregnancies'].fillna(temp_ds['Num of pregnancies'].median())
temp_ds['Smokes'] = temp_ds['Smokes'].fillna(1)
temp_ds['Smokes (years)'] = temp_ds['Smokes (years)'].fillna(temp_ds['Smokes (years)'].median())
temp_ds['Smokes (packs/year)'] = temp_ds['Smokes (packs/year)'].fillna(temp_ds['Smokes (packs/year)'].median())
temp_ds['Hormonal Contraceptives'] = temp_ds['Hormonal Contraceptives'].fillna(1)
temp_ds['Hormonal Contraceptives (years)'] = temp_ds['Hormonal Contraceptives (years)'].fillna(temp_ds['Hormonal Contraceptives (years)'].median())
temp_ds['IUD'] = temp_ds['IUD'].fillna(0) # Under suggestion
temp_ds['IUD (years)'] = temp_ds['IUD (years)'].fillna(0) #Under suggestion
temp_ds['STDs'] = temp_ds['STDs'].fillna(1)
temp_ds['STDs (number)'] = temp_ds['STDs (number)'].fillna(temp_ds['STDs (number)'].median())
temp_ds['STDs:condylomatosis'] = temp_ds['STDs:condylomatosis'].fillna(temp_ds['STDs:condylomatosis'].median())
temp_ds['STDs:cervical condylomatosis'] = temp_ds['STDs:cervical condylomatosis'].fillna(temp_ds['STDs:cervical condylomatosis'].median())
temp_ds['STDs:vaginal condylomatosis'] = temp_ds['STDs:vaginal condylomatosis'].fillna(temp_ds['STDs:vaginal condylomatosis'].median())
temp_ds['STDs:vulvo-perineal condylomatosis'] = temp_ds['STDs:vulvo-perineal condylomatosis'].fillna(temp_ds['STDs:vulvo-perineal condylomatosis'].median())
temp_ds['STDs:syphilis'] = temp_ds['STDs:syphilis'].fillna(temp_ds['STDs:syphilis'].median())
temp_ds['STDs:pelvic inflammatory disease'] = temp_ds['STDs:pelvic inflammatory disease'].fillna(temp_ds['STDs:pelvic inflammatory disease'].median())
temp_ds['STDs:genital herpes'] = temp_ds['STDs:genital herpes'].fillna(temp_ds['STDs:genital herpes'].median())
temp_ds['STDs:molluscum contagiosum'] = temp_ds['STDs:molluscum contagiosum'].fillna(temp_ds['STDs:molluscum contagiosum'].median())
temp_ds['STDs:AIDS'] = temp_ds['STDs:AIDS'].fillna(temp_ds['STDs:AIDS'].median())
temp_ds['STDs:HIV'] = temp_ds['STDs:HIV'].fillna(temp_ds['STDs:HIV'].median())
temp_ds['STDs:Hepatitis B'] = temp_ds['STDs:Hepatitis B'].fillna(temp_ds['STDs:Hepatitis B'].median())
temp_ds['STDs:HPV'] = temp_ds['STDs:HPV'].fillna(temp_ds['STDs:HPV'].median())
temp_ds['STDs: Time since first diagnosis'] = temp_ds['STDs: Time since first diagnosis'].fillna(temp_ds['STDs: Time since first diagnosis'].median())
temp_ds['STDs: Time since last diagnosis'] = temp_ds['STDs: Time since last diagnosis'].fillna(temp_ds['STDs: Time since last diagnosis'].median())
temp_ds = pd.get_dummies(data=temp_ds, columns=['Smokes','Hormonal Contraceptives','IUD','STDs',
                                      'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller'])
temp_ds.isnull().sum()
temp_ds
ds_data = temp_ds
ds_data.describe()
np.random.seed(42)
ds_data_shuffle = ds_data.iloc[np.random.permutation(len(ds_data))]

ds_train = ds_data_shuffle.iloc[1:686, :]
ds_test = ds_data_shuffle.iloc[686: , :]
ds_train_feature = ds_train[['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
       'STDs:condylomatosis', 'STDs:cervical condylomatosis',
       'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
       'STDs:syphilis', 'STDs:pelvic inflammatory disease',
       'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',
       'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
       'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 
       'Smokes_0.0', 'Smokes_1.0','Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 
       'IUD_0.0','IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1','Dx:CIN_0', 
       'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1','Hinselmann_0', 'Hinselmann_1', 
       'Citology_0', 'Citology_1','Schiller_0','Schiller_1']]

train_label = np.array(ds_train['Biopsy'])
ds_test_feature = ds_test[['Age', 'Number of sexual partners', 'First sexual intercourse',
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

test_label = np.array(ds_test['Biopsy'])
from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
train_feature = minmax_scale.fit_transform(ds_train_feature)
test_feature = minmax_scale.fit_transform(ds_test_feature)
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential() 
model.add(Dense(units=500, 
                input_dim=46, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=200,  
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,
                kernel_initializer='uniform', 
                activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy',   
              optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=train_feature, y=train_label,  
                          validation_split=0.2, epochs=20, 
                          batch_size=200, verbose=2)
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
scores = model.evaluate(test_feature, test_label)
print('\n')
print('accuracy=',scores[1])
prediction = model.predict_classes(test_feature)
ds_ans = pd.DataFrame({'Biopsy' :test_label})
ds_ans['Prediction'] = prediction
ds_ans
ds_ans[ ds_ans['Biopsy'] != ds_ans['Prediction'] ]
ds_ans['Prediction'].value_counts()
ds_ans['Biopsy'].value_counts()
cols = ['Biopsy_1','Biopsy_0']  #Gold standard
rows = ['Prediction_1','Prediction_0'] #diagnostic tool (our prediction)

B1P1 = len(ds_ans[(ds_ans['Prediction'] == ds_ans['Biopsy']) & (ds_ans['Biopsy'] == 1)])
B1P0 = len(ds_ans[(ds_ans['Prediction'] != ds_ans['Biopsy']) & (ds_ans['Biopsy'] == 1)])
B0P1 = len(ds_ans[(ds_ans['Prediction'] != ds_ans['Biopsy']) & (ds_ans['Biopsy'] == 0)])
B0P0 = len(ds_ans[(ds_ans['Prediction'] == ds_ans['Biopsy']) & (ds_ans['Biopsy'] == 0)])

conf = np.array([[B1P1,B0P1],[B1P0,B0P0]])
ds_cm = pd.DataFrame(conf, columns = [i for i in cols], index = [i for i in rows])

f, ax= plt.subplots(figsize = (5, 5))
sns.heatmap(ds_cm, annot=True, ax=ax) 
ax.xaxis.set_ticks_position('top') #Making x label be on top is common in textbooks.

print('total test case number: ', np.sum(conf))
def model_efficacy(conf):
    total_num = np.sum(conf)
    sen = conf[0][0]/(conf[0][0]+conf[1][0])
    spe = conf[1][1]/(conf[1][0]+conf[1][1])
    false_positive_rate = conf[0][1]/(conf[0][1]+conf[1][1])
    false_negative_rate = conf[1][0]/(conf[0][0]+conf[1][0])
    
    print('total_num: ',total_num)
    print('G1P1: ',conf[0][0]) 
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