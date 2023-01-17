import numpy as np
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
import keras
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras.models import Model
from keras import layers
from keras import Input 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import random
from keras import regularizers
from pandas.plotting import andrews_curves
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, GRU, Dropout
from keras import regularizers
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix as CM
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
scaler = preprocessing.StandardScaler() # Here I initialize scaler in order to call it later
# It is used to scale column values to Normal(0,1) distribution
le = preprocessing.LabelEncoder() # Label encoder will convert string values to numbers
def plot_pie(table, column):
    '''
    Function that plots single pie chart
    '''
    labels = []
    for (key, value) in table[column].value_counts().items():
        labels.append(key)
    #plt.figure(figsize = (6,5))
    plt.pie(table[column].value_counts())
    plt.legend(labels, loc = 'best', bbox_to_anchor=(0.1,0.9))
    plt.title(str(column))
def plotting_pies(table, columns):
    '''
    Plots multiple pie chart using the above function
    '''
    n = len(columns)
    plt.figure(figsize = (7*2,7*int(n/2+1)))
    for i in range(len(columns)):
        plt.subplot(int(n/2+1), 2, i+1)
        plot_pie(table, columns[i])
    plt.show()
    plt.close('All')
def plot_nans(table):
    '''
    Presents nan values in dataframe
    '''
    n = len(table.columns)
    zeros = []
    for i in table.columns:
        zeros.append(table[i].isnull().sum())
    zeros = np.array(zeros)
    indi = np.argsort(zeros)[::-1]
    plt.figure(figsize = (n/3,5))
    plt.title("Nans Over Columns")
    plt.bar(range(n), zeros[indi],
           color='teal', align="center")
    plt.xticks(np.arange(n), table.columns[indi], rotation='vertical')
    plt.xlim([-1, n])
    plt.show()
    plt.close("all")
def Aggregation(table1, table2, ID, summing, averaging, counting, maximum, minimum, suff):
    ''' 
    table1 and table2 are two dataframes we want to merge, ID is column according to which we perform aggregation,
    and summing, averaging, counting, maximum and minimum are column on wich we plan to perform specified operation;
    suff is string to be added to each newly created column's name
    '''
    dictionary = {}
    for col in summing:
        dictionary[col] = 'sum'
    for col in averaging:
        dictionary[col] = 'mean'
    for col in counting:
        dictionary[col] = 'count'
    for col in maximum:
        dictionary[col] = 'max'
    for col in minimum:
        dictionary[col] = 'min'
    indexi = table1[ID].values
    Aggr = np.zeros((len(indexi),len(dictionary)))
    for i in range(len(indexi)):
        Aggr[i] = table2[table2[ID] == indexi[i]].agg(dictionary).values
    Aggr = pd.DataFrame(Aggr, columns = [i+suff for i in dictionary.keys()])
    
    return pd.concat([table1, Aggr], axis=1, join='inner') 
application_train = pd.read_csv('../input/application_train.csv') 
application_test = pd.read_csv('../input/application_test.csv') 
application_train.head() 
plot_nans(application_train)
strColumns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','NAME_TYPE_SUITE']
numColumns = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE','DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'HOUR_APPR_PROCESS_START', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'DAYS_LAST_PHONE_CHANGE' ]
Categorical = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','LIVE_CITY_NOT_WORK_CITY','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
plotting_pies(application_train, strColumns)
for col in numColumns:
    application_train[col] = application_train[col].fillna(application_train[col].mean()) 
    application_test[col] = application_test[col].fillna(application_train[col].mean()) 
    scaler.fit(application_train[col].values.reshape(-1, 1))
    application_train[col] = scaler.transform(application_train[col].values.reshape(-1, 1))
    application_test[col] = scaler.transform(application_test[col].values.reshape(-1, 1))    
application_train[np.array(strColumns)] = application_train[np.array(strColumns)].fillna('nan') 
application_test[np.array(strColumns)] = application_test[np.array(strColumns)].fillna('nan')

for col in strColumns:
    le.fit(application_train[col])
    application_train[col] = le.transform(application_train[col])
    application_test[col] = le.transform(application_test[col])
application_train.head()
application_train = application_train[:1000]
application_test = application_test[:100]
bureau = pd.read_csv('../input/bureau.csv')
bureau.head()
# pd.get_dummies(bureau)
# moglo je i ovako, ali nema veze
plot_nans(bureau)
burNumeric = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT','AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG','AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY'] 
burCateg = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE']
plotting_pies(bureau, burCateg)
bureau['marker'] = np.ones(len(bureau)) # I added this column so that I could later count number of occurances for each user

Bad_Debt = np.zeros(len(bureau)) 
Bad_Debt[np.where(bureau.CREDIT_ACTIVE == 'Bad debt')[0]] = 1
Sold = np.zeros(len(bureau))
Sold[np.where(bureau.CREDIT_ACTIVE == 'Sold')[0]] = 1
bureau['Bad_debt'] = Bad_Debt
bureau['Sold'] = Sold
del Bad_Debt
del Sold
# Again fill nans in numerical columns with mean value and in categorical with string 'nan'
for col in burNumeric: 
    bureau[col] = scaler.fit_transform(bureau[col].fillna(bureau[col].mean()).values.reshape(-1, 1))
for col in burCateg:
    bureau[col] = le.fit_transform(bureau[col].fillna('nan'))
# dataframe after processing
bureau.head()
# As I mentione abowe, becouse of computer capacity I will take only fraction of data
bureau = bureau[:1000]
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
bureau_balance.head()
plotting_pies(bureau_balance, ['STATUS'])
# Here I divided column MONTHS_BALANCE to scale it on 0-1 interval
bureau_balance.MONTHS_BALANCE /= -1*bureau_balance.MONTHS_BALANCE.min() 
# get_dummies makes 'flag' columns for each STATUS value, which is good practice when working with categorical data
bureau_balance = pd.get_dummies(bureau_balance)
bureau_balance.head()
bureau_balance = bureau_balance[:1000]
minimum = ['MONTHS_BALANCE']
averaging = [bureau_balance.columns[i] for i in range(2, len(bureau_balance.columns))]
bureau = Aggregation(bureau, bureau_balance, 'SK_ID_BUREAU', summing= [], averaging = averaging, counting =[], maximum=[], minimum =minimum, suff = '_b_b')
del bureau_balance # I don't need this table any more, so I'll delete it to save some working memory
bureau.head()
minimum = ['AMT_CREDIT_MAX_OVERDUE']
maximum = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'CNT_CREDIT_PROLONG', 'DAYS_CREDIT_UPDATE']
averaging = ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY']
summing = ['marker']
TRAIN = Aggregation(application_train, bureau[bureau.CREDIT_ACTIVE == 0], 'SK_ID_CURR', summing= summing, averaging = averaging, counting =[], maximum=maximum, minimum =minimum, suff = '_b_b_b')
del application_train # delete to release space
TEST = Aggregation(application_test, bureau[bureau.CREDIT_ACTIVE == 0], 'SK_ID_CURR', summing= summing, averaging = averaging, counting =[], maximum=maximum, minimum =minimum, suff = '_b_Act')
del application_test # delete to release space
minimum = ['AMT_CREDIT_MAX_OVERDUE']
maximum = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'CNT_CREDIT_PROLONG', 'DAYS_CREDIT_UPDATE','DAYS_ENDDATE_FACT']
averaging = ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY']
summing = ['marker', 'Bad_debt', 'Sold']
TRAIN = Aggregation(TRAIN, bureau[bureau.CREDIT_ACTIVE == 0], 'SK_ID_CURR', summing= summing, averaging = averaging, counting =[], maximum=maximum, minimum =minimum, suff = '_b_NAct')
TEST = Aggregation(TEST, bureau[bureau.CREDIT_ACTIVE != 0], 'SK_ID_CURR', summing= summing, averaging = averaging, counting =[], maximum=maximum, minimum =minimum, suff = '_b_NAct')
TRAIN.head() 
del bureau
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
plot_nans(POS_CASH_balance)
POS_CASH_balance.head()
plotting_pies(POS_CASH_balance, ['NAME_CONTRACT_STATUS'])
for col in ['CNT_INSTALMENT_FUTURE', 'CNT_INSTALMENT', 'MONTHS_BALANCE']:
    POS_CASH_balance[col] = scaler.fit_transform(POS_CASH_balance[col].fillna(POS_CASH_balance[col].mean()).values.reshape(-1, 1))
POS_CASH_balance.SK_DPD /= POS_CASH_balance.SK_DPD.max()
POS_CASH_balance.SK_DPD_DEF /= POS_CASH_balance.SK_DPD_DEF.max()
POS_CASH_balance =pd.get_dummies(POS_CASH_balance)
POS_CASH_balance.head()
maximum = ['MONTHS_BALANCE']
averaging = ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF']
summing = POS_CASH_balance.columns[7:].tolist()
TRAIN = Aggregation(TRAIN, POS_CASH_balance, 'SK_ID_CURR', summing= summing, averaging = averaging, counting =[], maximum=maximum, minimum =[], suff = '_PCb')
TEST = Aggregation(TEST, POS_CASH_balance, 'SK_ID_CURR', summing= summing, averaging = averaging, counting =[], maximum=maximum, minimum =[], suff = '_PCb')
del POS_CASH_balance
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
credit_card_balance.head()
plot_nans(credit_card_balance)
plotting_pies(credit_card_balance, ['NAME_CONTRACT_STATUS'])
credit_card_balance = pd.get_dummies(credit_card_balance)
for col in credit_card_balance.columns[2:15]:
    credit_card_balance[col] = scaler.fit_transform(credit_card_balance[col].fillna(credit_card_balance[col].mean()).values.reshape(-1, 1))
for col in credit_card_balance.columns[15:22]:
    credit_card_balance[col] /= credit_card_balance[col].max()
credit_card_balance.head()
averaging = credit_card_balance.columns[3:22].tolist()
maximum = ['MONTHS_BALANCE']
summing = credit_card_balance.columns[22:].tolist()
counting = ['NAME_CONTRACT_STATUS_Active']
TRAIN = Aggregation(TRAIN, credit_card_balance, 'SK_ID_CURR', summing= summing, averaging = averaging, counting =counting, maximum=maximum, minimum =[], suff = '_ccb')
TEST = Aggregation(TEST, credit_card_balance, 'SK_ID_CURR', summing= summing, averaging = averaging, counting =counting, maximum=maximum, minimum =[], suff = '_ccb')
del credit_card_balance
previous_application = pd.read_csv('../input/previous_application.csv')
previous_application.head()
plot_nans(previous_application)
strColumns = ['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'PRODUCT_COMBINATION', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION', 'NAME_GOODS_CATEGORY']
numColumns = ['AMT_ANNUITY','AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE','RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'DAYS_DECISION', 'SELLERPLACE_AREA', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE','DAYS_TERMINATION', 'SELLERPLACE_AREA', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
plotting_pies(previous_application, strColumns)
previous_application=pd.get_dummies(previous_application)
for col in numColumns:
    previous_application[col] = scaler.fit_transform(previous_application[col].fillna(previous_application[col].mean()).values.reshape(-1, 1))
for col in  ['HOUR_APPR_PROCESS_START', 'CNT_PAYMENT']:
    previous_application[col] /= previous_application[col].max()
previous_application.head()
averaging = [col for col in previous_application.columns[2:] if col in numColumns or col in ['HOUR_APPR_PROCESS_START', 'CNT_PAYMENT']]
summing = [col for col in previous_application.columns[2:] if col not in averaging]
TRAIN = Aggregation(TRAIN, previous_application, 'SK_ID_CURR', summing= summing, averaging = averaging, counting =[], maximum=[], minimum =[], suff = '_pa')
TEST = Aggregation(TEST, previous_application, 'SK_ID_CURR', summing= summing, averaging = averaging, counting =[], maximum=[], minimum =[], suff = '_pa')
del previous_application
installments_payments = pd.read_csv('../input/installments_payments.csv')
installments_payments.head()
plot_nans(installments_payments)
for col in ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER']:
    installments_payments[col] /= installments_payments[col].max()
for col in installments_payments.columns[4:]:
    installments_payments[col] = scaler.fit_transform(installments_payments[col].fillna(installments_payments[col].mean()).values.reshape(-1, 1))  
installments_payments.head()
averaging = [col for col in installments_payments.columns[2:]]
TRAIN = Aggregation(TRAIN, installments_payments, 'SK_ID_CURR', summing= [], averaging = averaging, counting =[], maximum=[], minimum =[], suff = '_ip')
TEST = Aggregation(TEST, installments_payments, 'SK_ID_CURR', summing= [], averaging = averaging, counting =[], maximum=[], minimum =[], suff = '_ip')
del installments_payments
TRAIN.head()
TRAIN = TRAIN.fillna(0) # in case that some nan values still remained
TEST = TEST.fillna(0)
TRAIN.head()
TRAIN.shape
TEST.shape
TEST.head()
marker = [] # list that will keep track of rows I wont to keep
for i in range(len(TRAIN)):
    if TRAIN.TARGET[i] == 0: # if application is rejected
        if np.random.rand()<=0.1: # take it with probability 10 % (so the ratio between positive and negative targets is close to 1)
            marker.append(i)
    else:
        marker.append(i) # keep all approved loans
X = TRAIN.iloc[np.array(marker),3:].values
Y = TRAIN.iloc[np.array(marker)].TARGET.values
Y = keras.utils.to_categorical(Y, 2) # for neural networks, it is better to encode labels (from 0, 1 to [1,0], [0,1])
limit = int(0.8*len(X)) # for training take 80% of data, and 20 for validation
trainX, valX, trainY, valY = X[:limit], X[limit:], Y[:limit], Y[limit:]
model = Sequential()
model.add(Dense(1024, input_shape=(trainX.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(trainX, trainY,
                    epochs=3, # it should be way more epochs, so if your computer is able to carry this out, increase it
                    batch_size=8,validation_data=(valX, valY))
sns.heatmap(CM(np.argmax(model.predict(valX), axis=1), np.argmax(valY, axis = 1)), annot = True)
plt.show()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(X, Y,
                    epochs=3, 
                    batch_size=8)
Solution = pd.DataFrame(TEST.SK_ID_CURR)
Solution['TARGET'] = np.argmax(model.predict(TEST.iloc[:,2:].values).tolist(), axis = 1)
# the next line saves solution in CSV format, and it is ready for submitting
# Solution.set_index('SK_ID_CURR').to_csv('solution.csv')
