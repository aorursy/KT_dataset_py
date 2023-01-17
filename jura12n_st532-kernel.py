import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn

from pandas import Series,DataFrame

from sklearn import preprocessing

from sklearn.preprocessing import Imputer

from sklearn.decomposition import PCA

import seaborn as sns

import os

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, recall_score, precision_score, precision_recall_curve, average_precision_score

from sklearn.utils.fixes import signature

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.utils import class_weight

import keras

from keras.datasets import mnist

from keras.models import Sequential, Model, model_from_json

from keras.layers.core import Dense, Activation, Dropout

from keras.layers import Input

from keras.utils import plot_model, np_utils

from keras.utils import to_categorical



from pandas.tools.plotting import scatter_matrix

import statsmodels.api as sm

print(os.listdir("../input"))
#create dataframe of 2015

df_import = pd.read_csv('../input/2015.csv', header=0)

pd.set_option('display.max_rows', 400)

pd.set_option('display.max_columns', 400)
#delete variables not used

del df_import['_STATE']

del df_import['FMONTH']

del df_import['IDATE']

del df_import['IMONTH']

del df_import['IDAY']

del df_import['IYEAR']

del df_import['DISPCODE']

del df_import['SEQNO']

del df_import['_PSU']

del df_import['CTELENUM']

del df_import['PVTRESD1']

del df_import['COLGHOUS']

del df_import['STATERES']

del df_import['CELLFON3']

del df_import['LADULT']

del df_import['NUMADULT']

del df_import['NUMMEN']

del df_import['NUMWOMEN']

del df_import['CTELNUM1']

del df_import['CELLFON2']

del df_import['CADULT']

del df_import['PVTRESD2']

del df_import['CCLGHOUS']

del df_import['CSTATE']

del df_import['LANDLINE']

del df_import['HHADULT']

del df_import['PERSDOC2']

del df_import['MEDCOST']

del df_import['BPHIGH4']

del df_import['BPMEDS']

del df_import['BLOODCHO']

del df_import['CHOLCHK']

del df_import['TOLDHI2']

del df_import['CVDINFR4']

del df_import['CVDCRHD4']

del df_import['CVDSTRK3']

del df_import['ASTHMA3']

del df_import['ASTHNOW']

del df_import['CHCSCNCR']

del df_import['CHCOCNCR']

del df_import['CHCCOPD1']

del df_import['HAVARTH3']

del df_import['ADDEPEV2']

del df_import['CHCKIDNY']

del df_import['EDUCA']

del df_import['RENTHOM1']

del df_import['NUMHHOL2']

del df_import['NUMPHON2']

del df_import['CPDEMO1']

del df_import['CHILDREN']

del df_import['INCOME2']

del df_import['INTERNET']

del df_import['WEIGHT2']

del df_import['HEIGHT3']

del df_import['PREGNANT']

del df_import['QLACTLM2']

del df_import['USEEQUIP']

del df_import['BLIND']

del df_import['DECIDE']

del df_import['DIFFWALK']

del df_import['DIFFDRES']

del df_import['DIFFALON']

del df_import['SMOKE100']

del df_import['SMOKDAY2']

del df_import['STOPSMK2']

del df_import['LASTSMK2']

del df_import['ALCDAY5']

del df_import['AVEDRNK2']

del df_import['DRNK3GE5']

del df_import['MAXDRNKS']

del df_import['FRUITJU1']

del df_import['FRUIT1']

del df_import['FVBEANS']

del df_import['FVGREEN']

del df_import['FVORANG']

del df_import['VEGETAB1']

del df_import['EXERANY2']

del df_import['EXRACT11']

del df_import['EXEROFT1']

del df_import['EXERHMM1']

del df_import['EXRACT21']

del df_import['EXEROFT2']

del df_import['EXERHMM2']

del df_import['STRENGTH']

del df_import['LMTJOIN3']

del df_import['ARTHDIS2']

del df_import['ARTHSOCL']

del df_import['JOINPAIN']

del df_import['SEATBELT']

del df_import['FLSHTMY2']

del df_import['IMFVPLAC']

del df_import['HIVTSTD3']

del df_import['WHRTST10']

del df_import['PDIABTST']

del df_import['PREDIAB1']

del df_import['INSULIN']

del df_import['BLDSUGAR']

del df_import['FEETCHK2']

del df_import['DOCTDIAB']

del df_import['CHKHEMO3']

del df_import['FEETCHK']

del df_import['EYEEXAM']

del df_import['DIABEYE']

del df_import['DIABEDU']

del df_import['PAINACT2']

del df_import['QLMENTL2']

del df_import['QLSTRES2']

del df_import['QLHLTH2']

del df_import['CAREGIV1']

del df_import['CRGVREL1']

del df_import['CRGVLNG1']

del df_import['CRGVHRS1']

del df_import['CRGVPRB1']

del df_import['CRGVPERS']

del df_import['CRGVHOUS']

del df_import['CRGVMST2']

del df_import['CRGVEXPT']

del df_import['VIDFCLT2']

del df_import['VIREDIF3']

del df_import['VIPRFVS2']

del df_import['VINOCRE2']

del df_import['VIEYEXM2']

del df_import['VIINSUR2']

del df_import['VICTRCT4']

del df_import['VIGLUMA2']

del df_import['VIMACDG2']

del df_import['CIMEMLOS']

del df_import['CDHOUSE']

del df_import['CDASSIST']

del df_import['CDHELP']

del df_import['CDSOCIAL']

del df_import['CDDISCUS']

del df_import['WTCHSALT']

del df_import['LONGWTCH']

del df_import['DRADVISE']

del df_import['ASTHMAGE']

del df_import['ASATTACK']

del df_import['ASERVIST']

del df_import['ASDRVIST']

del df_import['ASRCHKUP']

del df_import['ASACTLIM']

del df_import['ASYMPTOM']

del df_import['ASNOSLEP']

del df_import['ASTHMED3']

del df_import['ASINHALR']

del df_import['HAREHAB1']

del df_import['STREHAB1']

del df_import['CVDASPRN']

del df_import['ASPUNSAF']

del df_import['RLIVPAIN']

del df_import['RDUCHART']

del df_import['RDUCSTRK']

del df_import['ARTTODAY']

del df_import['ARTHWGT']

del df_import['ARTHEXER']

del df_import['ARTHEDU']

del df_import['TETANUS']

del df_import['HPVADVC2']

del df_import['HPVADSHT']

del df_import['SHINGLE2']

del df_import['HADMAM']

del df_import['HOWLONG']

del df_import['HADPAP2']

del df_import['LASTPAP2']

del df_import['HPVTEST']

del df_import['HPLSTTST']

del df_import['HADHYST2']

del df_import['PROFEXAM']

del df_import['LENGEXAM']

del df_import['BLDSTOOL']

del df_import['LSTBLDS3']

del df_import['HADSIGM3']

del df_import['HADSGCO1']

del df_import['LASTSIG3']

del df_import['PCPSAAD2']

del df_import['PCPSADI1']

del df_import['PCPSARE1']

del df_import['PSATEST1']

del df_import['PSATIME']

del df_import['PCPSARS1']

del df_import['PCPSADE1']

del df_import['PCDMDECN']

del df_import['SCNTMNY1']

del df_import['SCNTMEL1']

del df_import['SCNTPAID']

del df_import['SCNTWRK1']

del df_import['SCNTLPAD']

del df_import['SCNTLWK1']

del df_import['SXORIENT']

del df_import['TRNSGNDR']

del df_import['RCSGENDR']

del df_import['RCSRLTN2']

del df_import['CASTHDX2']

del df_import['CASTHNO2']

del df_import['EMTSUPRT']

del df_import['LSATISFY']

del df_import['ADPLEASR']

del df_import['ADDOWN']

del df_import['ADSLEEP']

del df_import['ADENERGY']

del df_import['ADEAT1']

del df_import['ADFAIL']

del df_import['ADTHINK']

del df_import['ADMOVE']

del df_import['MISTMNT']

del df_import['ADANXEV']

del df_import['QSTVER']

del df_import['QSTLANG']

del df_import['EXACTOT1']

del df_import['EXACTOT2']

del df_import['MSCODE']

del df_import['_STSTR']

del df_import['_STRWT']

del df_import['_RAWRAKE']

del df_import['_WT2RAKE']

del df_import['_CHISPNC']

del df_import['_CRACE1']

del df_import['_CPRACE']

del df_import['_CLLCPWT']

del df_import['_DUALUSE']

del df_import['_DUALCOR']

del df_import['_LLCPWT']

del df_import['_RFHLTH']

del df_import['_HCVU651']

del df_import['_RFHYPE5']

del df_import['_CHOLCHK']

del df_import['_RFCHOL']

del df_import['_MICHD']

del df_import['_LTASTH1']

del df_import['_CASTHM1']

del df_import['_ASTHMS1']

del df_import['_DRDXAR1']

del df_import['_PRACE1']

del df_import['_MRACE1']

del df_import['_HISPANC']

del df_import['_RACE']

del df_import['_RACEG21']

del df_import['_RACE_G1']

del df_import['_AGEG5YR']

del df_import['_AGE65YR']

del df_import['_AGE_G']

del df_import['HTIN4']

del df_import['HTM4']

del df_import['WTKG3']

del df_import['_BMI5CAT']

del df_import['_RFBMI5']

del df_import['_CHLDCNT']

del df_import['_SMOKER3']

del df_import['FTJUDA1_']

del df_import['FRUTDA1_']

del df_import['BEANDAY_']

del df_import['GRENDAY_']

del df_import['ORNGDAY_']

del df_import['VEGEDA1_']

del df_import['_MISFRTN']

del df_import['_MISVEGN']

del df_import['_FRTRESP']

del df_import['_VEGRESP']

del df_import['_FRTLT1']

del df_import['_VEGLT1']

del df_import['_FRT16']

del df_import['_VEG23']

del df_import['METVL11_']

del df_import['METVL21_']

del df_import['MAXVO2_']

del df_import['FC60_']

del df_import['ACTIN11_']

del df_import['ACTIN21_']

del df_import['PADUR1_']

del df_import['PADUR2_']

del df_import['PAFREQ1_']

del df_import['PAFREQ2_']

del df_import['_MINAC11']

del df_import['_MINAC21']

del df_import['PAMISS1_']

del df_import['PAMIN11_']

del df_import['PAMIN21_']

del df_import['PAVIG11_']

del df_import['PAVIG21_']

del df_import['_PACAT1']

del df_import['_PA150R2']

del df_import['_PA300R2']

del df_import['_PA30021']

del df_import['_PAREC1']

del df_import['_PASTAE1']

del df_import['_LMTACT1']

del df_import['_LMTWRK1']

del df_import['_LMTSCL1']

del df_import['_RFSEAT2']

del df_import['_RFSEAT3']

del df_import['_FLSHOT6']

del df_import['_PNEUMO2']

del df_import['_AIDTST3']

del df_import['USENOW3']
df = pd.DataFrame(df_import)
#different variables have different indicators of missing value

#in order to treat missing value consistently, transform those indicators to one indicator



df.loc[(df['GENHLTH'] == 7) | (df['GENHLTH'] == 9), 'GENHLTH'] = np.NaN

df.loc[(df['PHYSHLTH'] == 77) | (df['PHYSHLTH'] == 99), 'PHYSHLTH'] = np.NaN

df.loc[(df['MENTHLTH'] == 77) | (df['MENTHLTH'] == 99), 'MENTHLTH'] = np.NaN

df.loc[(df['POORHLTH'] == 77) | (df['POORHLTH'] == 99), 'POORHLTH'] = np.NaN

df.loc[(df['HLTHPLN1'] == 7) | (df['HLTHPLN1'] == 9), 'HLTHPLN1'] = np.NaN

df.loc[(df['CHECKUP1'] == 7) | (df['CHECKUP1'] == 9), 'CHECKUP1'] = np.NaN

df.loc[(df['DIABETE3'] == 7) | (df['DIABETE3'] == 9), 'DIABETE3'] = np.NaN

df.loc[(df['DIABAGE2'] == 98) | (df['DIABAGE2'] == 99), 'DIABAGE2'] = np.NaN

df.loc[df['MARITAL'] == 9, 'MARITAL'] = np.NaN

df.loc[(df['VETERAN3'] == 7) | (df['VETERAN3'] == 9), 'VETERAN3'] = np.NaN

df.loc[df['EMPLOY1'] == 9, 'EMPLOY1'] = np.NaN

df.loc[(df['FLUSHOT6'] == 7) | (df['FLUSHOT6'] == 9), 'FLUSHOT6'] = np.NaN

df.loc[(df['PNEUVAC3'] == 7) | (df['PNEUVAC3'] == 9), 'PNEUVAC3'] = np.NaN

df.loc[(df['HIVTST6'] == 7) | (df['HIVTST6'] == 9), 'HIVTST6'] = np.NaN

df.loc[df['_RACEGR3'] == 9, '_RACEGR3'] = np.NaN

df.loc[df['_EDUCAG'] == 9, '_EDUCAG'] = np.NaN

df.loc[df['_INCOMG'] == 9, '_INCOMG'] = np.NaN

df.loc[df['_RFSMOK3'] == 9, '_RFSMOK3'] = np.NaN

df.loc[(df['DRNKANY5'] == 7) | (df['DRNKANY5'] == 9), 'DRNKANY5'] = np.NaN

df.loc[df['DROCDY3_'] == 900, 'DROCDY3_'] = np.NaN

df.loc[df['_RFBING5'] == 9, '_RFBING5'] = np.NaN

df.loc[df['_TOTINDA'] == 9, '_TOTINDA'] = np.NaN

df.loc[df['_DRNKWEK'] == 99900, '_DRNKWEK'] = np.NaN

df.loc[df['_RFDRHV5'] == 9, '_RFDRHV5'] = np.NaN

df.loc[df['STRFREQ_'] == 99000, 'STRFREQ_'] = np.NaN

df.loc[df['_PAINDX1'] == 9, '_PAINDX1'] = np.NaN

df.loc[df['_PASTRNG'] == 9, '_PASTRNG'] = np.NaN
#impute/preprocessing



#Healthy days

df.loc[df['PHYSHLTH'] == 88, 'PHYSHLTH'] = 0

df.loc[df['MENTHLTH'] == 88, 'MENTHLTH'] = 0

df.loc[df['POORHLTH'] == 88, 'POORHLTH'] = 0

df.loc[(df['PHYSHLTH'] == 0) & (df['MENTHLTH'] == 0), 'POORHLTH'] = 0



#DIABETES

df.loc[(df['DIABETE3'] == 2)|(df['DIABETE3'] == 3)|(df['DIABETE3'] == 4), 'DIABETE3'] = 2

df.loc[df['DIABETE3'] == 2, 'DIABAGE2'] = 0

df['DIABDUR'] = 0

df.loc[df['DIABETE3'] == 2, 'DIABDUR'] = 0

df.loc[df['DIABETE3'] == 1, 'DIABDUR'] = df['_AGE80'] - df['DIABAGE2']

df.loc[df['DIABDUR'] < 0, 'DIABDUR'] = np.NaN

df['DIABDUR_G'] = 0

df.loc[df['DIABDUR'] <= 1, 'DIABDUR_G'] = 1

df.loc[(df['DIABDUR'] >  1) & (df['DIABDUR'] <= 5), 'DIABDUR_G'] = 5

df.loc[(df['DIABDUR'] >  5) & (df['DIABDUR'] <= 10), 'DIABDUR_G'] = 10

df.loc[df['DIABDUR'] >  10, 'DIABDUR_G'] = "10>"

df.loc[df['DIABETE3'] == 2, 'DIABDUR_G'] = 99



df.loc[df['DIABETE3'] == 1, 'DIABETE3'] = 3

df['DIABETE3'] += -2



#Exercise (Physical Activity)

df.loc[df['_TOTINDA'] == 2, 'PA1MIN_'] = 0

df.loc[df['_TOTINDA'] == 2, 'PA1VIGM_'] = 0

del df['_TOTINDA']



#Fruit&Vegetables

df.loc[df['_FRUITEX'] < 1, '_FRUITEX'] = 0

df.loc[df['_VEGETEX'] < 1, '_VEGETEX'] = 0



df = df[(df['_FRUITEX'] == 0) & (df['_VEGETEX'] == 0)]



del df['_FRUITEX']

del df['_VEGETEX']
df.loc[df['GENHLTH'] == 1 , 'GENHLTH'] = "1 Excellent"

df.loc[df['GENHLTH'] == 2 , 'GENHLTH'] = "2 Very good"

df.loc[df['GENHLTH'] == 3 , 'GENHLTH'] = "3 Good"

df.loc[df['GENHLTH'] == 4 , 'GENHLTH'] = "4 Fair"

df.loc[df['GENHLTH'] == 5 , 'GENHLTH'] = "5 Poor"



df.loc[df['HLTHPLN1'] == 1 , 'HLTHPLN1'] = "have health care coverage"

df.loc[df['HLTHPLN1'] == 2 , 'HLTHPLN1'] = "dont have health care coverage"



df.loc[df['CHECKUP1'] == 1 , 'CHECKUP1'] = "0 ~ 1"

df.loc[df['CHECKUP1'] == 2 , 'CHECKUP1'] = "1 ~ 2"

df.loc[df['CHECKUP1'] == 3 , 'CHECKUP1'] = "2 ~ 5"

df.loc[df['CHECKUP1'] == 4 , 'CHECKUP1'] = "5 or more"

df.loc[df['CHECKUP1'] == 8 , 'CHECKUP1'] = "Never"



df.loc[df['DIABETE3'] == 0 , 'DIABETE3'] = "No"

df.loc[df['DIABETE3'] == 1 , 'DIABETE3'] = "Yes"



df.loc[df['SEX'] == 1 , 'SEX'] = "Male"

df.loc[df['SEX'] == 2 , 'SEX'] = "Female"



df.loc[df['MARITAL'] == 1 , 'MARITAL'] = "Married"

df.loc[df['MARITAL'] == 2 , 'MARITAL'] = "Divorced"

df.loc[df['MARITAL'] == 3 , 'MARITAL'] = "Widowed"

df.loc[df['MARITAL'] == 4 , 'MARITAL'] = "Separated"

df.loc[df['MARITAL'] == 5 , 'MARITAL'] = "Never married"

df.loc[df['MARITAL'] == 6 , 'MARITAL'] = "A member of an unmarried couple"



df.loc[df['VETERAN3'] == 1 , 'VETERAN3'] = "Yes(ever)"

df.loc[df['VETERAN3'] == 2 , 'VETERAN3'] = "Never"



df.loc[df['EMPLOY1'] == 1 , 'EMPLOY1'] = "Employed for wages"

df.loc[df['EMPLOY1'] == 2 , 'EMPLOY1'] = "Self-employed"

df.loc[df['EMPLOY1'] == 3 , 'EMPLOY1'] = "Out of work for 1 year or more"

df.loc[df['EMPLOY1'] == 4 , 'EMPLOY1'] = "Out of work for less than 1 year"

df.loc[df['EMPLOY1'] == 5 , 'EMPLOY1'] = "A homemaker"

df.loc[df['EMPLOY1'] == 6 , 'EMPLOY1'] = "A student"

df.loc[df['EMPLOY1'] == 7 , 'EMPLOY1'] = "Retired"

df.loc[df['EMPLOY1'] == 8 , 'EMPLOY1'] = "Unable to work"



df.loc[df['FLUSHOT6'] == 1 , 'FLUSHOT6'] = "Yes"

df.loc[df['FLUSHOT6'] == 2 , 'FLUSHOT6'] = "No"



df.loc[df['PNEUVAC3'] == 1 , 'PNEUVAC3'] = "Yes"

df.loc[df['PNEUVAC3'] == 2 , 'PNEUVAC3'] = "No"



df.loc[df['HIVTST6'] == 1 , 'HIVTST6'] = "Yes"

df.loc[df['HIVTST6'] == 2 , 'HIVTST6'] = "No"



df.loc[df['_RACEGR3'] == 1 , '_RACEGR3'] = "White"

df.loc[df['_RACEGR3'] == 2 , '_RACEGR3'] = "Black"

df.loc[df['_RACEGR3'] == 3 , '_RACEGR3'] = "Other race only"

df.loc[df['_RACEGR3'] == 4 , '_RACEGR3'] = "Multiracial"

df.loc[df['_RACEGR3'] == 5 , '_RACEGR3'] = "Hispanic"



df.loc[df['_EDUCAG'] == 1 , '_EDUCAG'] = "1 Did not graduate High School"

df.loc[df['_EDUCAG'] == 2 , '_EDUCAG'] = "2 Graduated High School"

df.loc[df['_EDUCAG'] == 3 , '_EDUCAG'] = "3 Attended College or Technical School"

df.loc[df['_EDUCAG'] == 4 , '_EDUCAG'] = "4 Graduated from College or Technical School"



df.loc[df['_INCOMG'] == 1 , '_INCOMG'] = " < $15,000"

df.loc[df['_INCOMG'] == 2 , '_INCOMG'] = " $15,000 ~ 25,000"

df.loc[df['_INCOMG'] == 3 , '_INCOMG'] = " $25,000 ~ 35,000"

df.loc[df['_INCOMG'] == 4 , '_INCOMG'] = " $35,000 ~ 50,000"

df.loc[df['_INCOMG'] == 5 , '_INCOMG'] = " $50,000 ~ "



df.loc[df['_RFSMOK3'] == 1 , '_RFSMOK3'] = "Not Smorker"

df.loc[df['_RFSMOK3'] == 2 , '_RFSMOK3'] = "Smorker"



df.loc[df['DRNKANY5'] == 1 , 'DRNKANY5'] = "Yes"

df.loc[df['DRNKANY5'] == 2 , 'DRNKANY5'] = "No"



df.loc[df['_RFBING5'] == 1 , '_RFBING5'] = "NOT BINGE DRINKER"

df.loc[df['_RFBING5'] == 2 , '_RFBING5'] = "BINGE DRINKER"



df.loc[df['_RFDRHV5'] == 1 , '_RFDRHV5'] = "NOT HEAVY DRINKER"

df.loc[df['_RFDRHV5'] == 2 , '_RFDRHV5'] = "HEAVY DRINKER"



df.loc[df['_PAINDX1'] == 1 , '_PAINDX1'] = "Meet Aerobic Recom."

df.loc[df['_PAINDX1'] == 2 , '_PAINDX1'] = "Not Meet Aerobic Recom."



df.loc[df['_PASTRNG'] == 1 , '_PASTRNG'] = "Meet muscle strengthening Recom."

df.loc[df['_PASTRNG'] == 2 , '_PASTRNG'] = "Not Meet muscle strengthening Recom."
df_ex_na = df.dropna()

df_ex_na.to_csv("df_ex_na.csv")

#distribution (numeric variables)

print(df_ex_na.describe())

plt.rcParams["font.size"] = 24

df_ex_na.hist(bins=50,figsize=(50,50))

plt.show()
#distribution of diabetes patients(numeric variables)

plt.rcParams["font.size"] = 24

df_ex_na[df_ex_na['DIABETE3'] == "Yes"].hist(bins=50,figsize=(50,50))

plt.show()
plt.rcParams["font.size"] = 18

plt.rcParams["figure.figsize"] = (50,90)

plt.subplots_adjust(wspace=0.2, hspace=0.4)



plt.subplot(11, 2, 1)

sns.countplot(x='GENHLTH',data=df_ex_na)

plt.subplot(11, 2, 2)

sns.countplot(x='HLTHPLN1',data=df_ex_na)

plt.subplot(11, 2, 3)

sns.countplot(x='CHECKUP1',data=df_ex_na)

plt.subplot(11, 2, 4)

sns.countplot(x='DIABETE3',data=df_ex_na)

plt.subplot(11, 2, 5)

sns.countplot(x='SEX',data=df_ex_na)

plt.subplot(11, 2, 6)

sns.countplot(x='MARITAL',data=df_ex_na)

plt.subplot(11, 2, 7)

sns.countplot(x='VETERAN3',data=df_ex_na)

plt.subplot(11, 2, 8)

sns.countplot(x='EMPLOY1',data=df_ex_na)

plt.subplot(11, 2, 9)

sns.countplot(x='FLUSHOT6',data=df_ex_na)

plt.subplot(11, 2, 10)

sns.countplot(x='PNEUVAC3',data=df_ex_na)

plt.subplot(11, 2, 11)

sns.countplot(x='HIVTST6',data=df_ex_na)

plt.subplot(11, 2, 12)

sns.countplot(x='_RACEGR3',data=df_ex_na)

plt.subplot(11, 2, 13)

sns.countplot(x='_EDUCAG',data=df_ex_na)

plt.subplot(11, 2, 14)

sns.countplot(x='_INCOMG',data=df_ex_na)

plt.subplot(11, 2, 15)

sns.countplot(x='_RFSMOK3',data=df_ex_na)

plt.subplot(11, 2, 16)

sns.countplot(x='DRNKANY5',data=df_ex_na)

plt.subplot(11, 2, 17)

sns.countplot(x='_RFBING5',data=df_ex_na)

plt.subplot(11, 2, 18)

sns.countplot(x='_RFDRHV5',data=df_ex_na)

plt.subplot(11, 2, 19)

sns.countplot(x='_PAINDX1',data=df_ex_na)

plt.subplot(11, 2, 20)

sns.countplot(x='_PASTRNG',data=df_ex_na)

plt.subplot(11, 2, 21)

sns.countplot(x='DIABDUR_G',data=df_ex_na)



plt.show()
#standardization

ss = preprocessing.StandardScaler()

cols_to_norm = ['PHYSHLTH','MENTHLTH','POORHLTH','_AGE80','_BMI5',

                'DROCDY3_','_DRNKWEK','STRFREQ_','PA1MIN_','PA1VIGM_','_VEGESUM','_FRUTSUM']

df_ex_na_std = df_ex_na

df_ex_na_std[cols_to_norm] = ss.fit_transform(df_ex_na_std[cols_to_norm])
df_ex_na_std_dummies = pd.get_dummies(df_ex_na_std,drop_first=False)
#Correlation between variables used for PCA/Clustering



df_ex_na_std_dummies_PCA = df_ex_na_std_dummies[['DIABETE3_Yes',

                                                   'SEX_Male',

                                                   '_BMI5',

                                                   '_RFSMOK3_Smorker',

                                                   'DROCDY3_',

                                                   '_DRNKWEK',

                                                   '_FRUTSUM',

                                                   '_VEGESUM',

                                                   'STRFREQ_',

                                                   'PA1MIN_',

                                                   'PA1VIGM_',

                                                   '_AGE80']]

df_ex_na_std_dummies_PCA_corr = df_ex_na_std_dummies_PCA.dropna().corr()



plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (10,10)



plt.subplot(1, 1, 1)

sns.heatmap(df_ex_na_std_dummies_PCA_corr, square=True, vmax=1, vmin=-1, center=0,annot=True,fmt="1.1f")



plt.show
#Correlation between variables used for PCA/Clustering



df_ex_na_std_dummies_PCA = df_ex_na_std_dummies[df_ex_na_std_dummies['DIABETE3_Yes']==1][['SEX_Male',

                                                                                           '_BMI5',

                                                                                           '_RFSMOK3_Smorker',

                                                                                           'DROCDY3_',

                                                                                           '_DRNKWEK',

                                                                                           '_FRUTSUM',

                                                                                           '_VEGESUM',

                                                                                           'STRFREQ_',

                                                                                           'PA1MIN_',

                                                                                           'PA1VIGM_',

                                                                                           '_AGE80',

                                                                                           'DIABDUR']]

df_ex_na_std_dummies_PCA_corr = df_ex_na_std_dummies_PCA.dropna().corr()



plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (10,10)



plt.subplot(1, 1, 1)

sns.heatmap(df_ex_na_std_dummies_PCA_corr, square=True, vmax=1, vmin=-1, center=0,annot=True,fmt="1.1f")



plt.show
#PCA

df_ex_na_std_dummies_PCA = df_ex_na_std_dummies[['DIABETE3_Yes','_AGE80','_BMI5','_RFSMOK3_Smorker','DROCDY3_',

                                                   '_DRNKWEK', '_FRUTSUM','_VEGESUM','STRFREQ_','PA1MIN_','PA1VIGM_',

                                                 'DIABDUR_G_1','DIABDUR_G_5','DIABDUR_G_10','DIABDUR_G_10>']]



pca = PCA()

X = pca.fit_transform(df_ex_na_std_dummies_PCA[[ '_BMI5','_RFSMOK3_Smorker','DROCDY3_','_DRNKWEK','_FRUTSUM','_VEGESUM','STRFREQ_','PA1MIN_','PA1VIGM_']])



X = np.insert(X, 8, df_ex_na_std_dummies_PCA['DIABETE3_Yes'],axis=1)

X = np.insert(X, 9,df_ex_na_std_dummies_PCA['DIABDUR_G_1'],axis=1)

X = np.insert(X,10,df_ex_na_std_dummies_PCA['DIABDUR_G_5'],axis=1)

X = np.insert(X,11,df_ex_na_std_dummies_PCA['DIABDUR_G_10'],axis=1)

X = np.insert(X,12,df_ex_na_std_dummies_PCA['DIABDUR_G_10>'],axis=1)
#2D visualization of PCA above 

plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (12,12)

plt.subplots_adjust(wspace=0.6, hspace=0.6)



fig = plt.figure()



ax = fig.add_subplot(221)

ax.scatter(X[:,0]*(1-X[:,8]),X[:,1]*(1-X[:,8]),c='blue', alpha=0.2)

ax.set_xlim(- 5,20)

ax.set_ylim(-10,15)

plt.grid(color='gray')

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

plt.title('not_diabetes')



ax = fig.add_subplot(222)

ax.scatter(X[:,0]*(1-X[:,8]),X[:,1]*(1-X[:,8]),c='blue', alpha=0.2, label="non-diabetes")

ax.scatter(X[:,0]*X[:,8],X[:,1]*X[:,8],c='red', alpha=0.2, label="diabetes")

ax.set_xlim(- 5,20)

ax.set_ylim(-10,15)

plt.grid(color='gray')

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

plt.title('Total')



ax = fig.add_subplot(223)

ax.scatter(X[:,0]*X[:,8],X[:,1]*X[:,8],c='red', alpha=0.2)

ax.set_xlim(- 5,20)

ax.set_ylim(-10,15)

plt.grid(color='gray')

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

plt.title('diabetes')





plt.show()
#2D visualization of PCA above for diabetes patients



plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (12,12)

plt.subplots_adjust(wspace=0.2, hspace=1)

fig = plt.figure()



ax = fig.add_subplot(221)

ax.scatter(X[:,0]*X[:,8],X[:,1]*X[:,8],c='red', alpha=0.2)

ax.set_xlim(- 5,20)

ax.set_ylim(-10,15)

plt.grid(color='gray')

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

plt.title('diabetes')





ax = fig.add_subplot(222)

ax.scatter(X[:,0]*X[:,8]*X[:,9]

           ,X[:,1]*X[:,8]*X[:,9]

           ,c='red', alpha=0.2)

ax.set_xlim(- 5,20)

ax.set_ylim(-10,15)

plt.grid(color='gray')

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

plt.title('diabetes (duration<=1)')



ax = fig.add_subplot(223)

ax.scatter(X[:,0]*X[:,8]*X[:,12]

           ,X[:,1]*X[:,8]*X[:,12]

           ,c='red', alpha=0.2)

ax.set_xlim(- 5,20)

ax.set_ylim(-10,15)

plt.grid(color='gray')

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

plt.title('diabetes (duration>10)')



plt.show()
#PCA performance

print(

    np.cumsum(

        pca.fit(

            df_ex_na_std_dummies_PCA[[ '_BMI5','_RFSMOK3_Smorker','DROCDY3_','_DRNKWEK',

                                           '_FRUTSUM','_VEGESUM','STRFREQ_','PA1MIN_','PA1VIGM_']]).explained_variance_ratio_))
#PCA eigen value

pd.DataFrame(

    pca.fit(df_ex_na_std_dummies_PCA[[ '_BMI5','_RFSMOK3_Smorker','DROCDY3_','_DRNKWEK',

                                           '_FRUTSUM','_VEGESUM','STRFREQ_','PA1MIN_','PA1VIGM_']]).components_

    , columns=df_ex_na_std_dummies_PCA[[ '_BMI5','_RFSMOK3_Smorker','DROCDY3_','_DRNKWEK',

                                           '_FRUTSUM','_VEGESUM','STRFREQ_','PA1MIN_','PA1VIGM_']].columns[:]

    , index=["PC{}".format(x + 1) for x in range(len(df_ex_na_std_dummies_PCA[[ '_BMI5','_RFSMOK3_Smorker','DROCDY3_','_DRNKWEK',

                                           '_FRUTSUM','_VEGESUM','STRFREQ_','PA1MIN_','PA1VIGM_']].columns))])
#Data

# print(df_ex_na_std_dummies.dtypes)

del_col = ['DIABDUR_G_1','DIABDUR_G_5','DIABDUR_G_10','DIABDUR_G_99','DIABDUR_G_10>','DIABETE3_Yes','DIABETE3_No','DIABAGE2','DIABDUR',

           'HLTHPLN1_dont have health care coverage','SEX_Female','VETERAN3_Never','FLUSHOT6_No','PNEUVAC3_No','HIVTST6_No','_RFSMOK3_Not Smorker',

           'DRNKANY5_No','_RFBING5_NOT BINGE DRINKER','_RFDRHV5_NOT HEAVY DRINKER','_PAINDX1_Not Meet Aerobic Recom.','_PASTRNG_Not Meet muscle strengthening Recom.']

X = df_ex_na_std_dummies.drop(del_col,axis=1)

Y = df_ex_na_std_dummies[['DIABETE3_Yes','DIABDUR','CHECKUP1_0 ~ 1']]
X_2 = X[X['CHECKUP1_0 ~ 1']==1]

Y.loc[Y['DIABDUR'] > 1 , 'DIABETE3_Yes'] = 0

Y_2 = Y[Y['CHECKUP1_0 ~ 1']==1].drop(['DIABDUR','CHECKUP1_0 ~ 1'],axis=1)

print(X_2.shape)

print(Y_2.shape)
#Data partition - 2

X_train, X_test, Y_train, Y_test = train_test_split(X_2, Y_2, train_size=0.7, random_state=1234)
#RandomForest

model_01 = RandomForestClassifier(random_state=123,n_estimators=300,min_samples_split=3,

                                    oob_score=True,class_weight="balanced",

                                   max_depth=10)

model_01 = model_01.fit(X_train, Y_train)

plt.rcParams["figure.figsize"] = (4,4)



Y_pred = model_01.predict_proba(X_test)[:,1]

precision, recall, _ = precision_recall_curve(Y_test,Y_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve(Random Forest): AP={0:0.4f}'.format(average_precision_score(Y_test,Y_pred)))
#ROC curve,AUC

fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label=1)

plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (5,4)

plt.plot(fpr, tpr, label='ROC curve (area = %.4f)'%auc(fpr, tpr))

plt.legend()

plt.title('ROC curve(Random Forest)')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
#Importance of variables

features = X_train.columns

importances = model_01.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize=(8,18))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.title('Importance of variables(Random Forest)')

plt.show()
use_col = X_train.columns[indices[(indices.shape[0]-15):indices.shape[0]]]
#Importance of variables

features = X_train[use_col].columns

importances = model_01.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize=(10,6))

plt.barh(range(15), importances[indices][(indices.shape[0]-15):indices.shape[0]], color='b', align='center')

plt.yticks(range(15), features)

plt.title('Importance of variables(Random Forest top 15)')

plt.show()
#Neural Network

# 各層や活性関数に該当するレイヤを順に入れていく

# 作成したあとにmodel.add()で追加することも可能

model_02 = Sequential([

                            Dense(80, input_shape=(X_train[use_col].shape[1],)),

                            Activation('relu'),

                            Dropout(0.5),

                            Dense(20),

                            Activation('relu'),

                            Dropout(0.5),

                            Dense(1),

                            Activation('sigmoid')

                        ])

#compile

optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

model_02.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

num_epoch=200

mini_batch_size=64
#model fit

model_02_result = model_02.fit(X_train[use_col], Y_train, 

                                   batch_size=mini_batch_size, 

                                   verbose=1, epochs=num_epoch, 

                                   validation_data=(X_test[use_col], Y_test),

                                   class_weight = "balanced")
plt.rcParams["figure.figsize"] = (6,6)

plt.plot(range(1, num_epoch+1), model_02_result.history['loss'], label="train_loss")

plt.plot(range(1, num_epoch+1), model_02_result.history['val_loss'], label="test_loss")

plt.xlabel('Epochs')

plt.ylabel('loss')

plt.legend()

plt.title('Learning process')

plt.show()
plt.rcParams["figure.figsize"] = (4,4)

Y_pred = model_02.predict_proba(X_test[use_col])

precision, recall, _ = precision_recall_curve(Y_test,Y_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('Neural Network: AP={0:0.4f}'.format(average_precision_score(Y_test,Y_pred)))
#ROC curve,AUC

fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label=1)

plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (5,4)

plt.plot(fpr, tpr, label='ROC curve (area = %.4f)'%auc(fpr, tpr))

plt.legend()

plt.title('ROC curve(Neural Network)')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
model_03 = LogisticRegression(class_weight='balanced')

model_03 = model_03.fit(X_train[use_col], Y_train)
plt.rcParams["figure.figsize"] = (4,4)

Y_pred = model_03.predict_proba(X_test[use_col])[:,1]

precision, recall, _ = precision_recall_curve(Y_test,Y_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve(Neural Network): AP={0:0.4f}'.format(average_precision_score(Y_test,Y_pred)))
#ROC curve,AUC

fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label=1)

plt.rcParams["font.size"] = 12

plt.rcParams["figure.figsize"] = (5,4)

plt.plot(fpr, tpr, label='ROC curve (area = %.4f)'%auc(fpr, tpr))

plt.legend()

plt.title('ROC curve(Logistic Regression)')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
#parameter

coeff_df = DataFrame([X_train[use_col].columns, model_03.coef_[0], np.exp(model_03.coef_[0])]).T

coeff_df.columns = ['variables','coeff','odds_ratio']

coeff_df = coeff_df.sort_values('coeff', ascending=False)

print(coeff_df)