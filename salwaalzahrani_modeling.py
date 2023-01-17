# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import string

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer

import re

pd.options.mode.chained_assignment = None 

# from sklearn.cross_validation import train_test_split

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV,ElasticNet

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,BaggingClassifier

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import make_pipeline

from xgboost import XGBClassifier

from sklearn.impute import KNNImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score,accuracy_score, roc_auc_score

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from numpy.random import seed

import tensorflow 
health = pd.read_csv('../input/capsto/health_FNN.csv',dtype='str')
num_cols = ['IDENTIFICATION CODE', 'YEAR', 'HEALTH CONDITION/ 12245',

       'HEALTH CONDITION/ HEALTHY LIVEBORN INFANTS ACCORDING TO BIRTH TYPE',

       'HEALTH CONDITION/ INININDISEASES OF THE MUSCULOSKELETAL SYSTEM AND CONNECTIVE TISSUE',

       'HEALTH CONDITION/ INININPERSONS WITH A CONDITION INFLUENCING THEIR HEALTH STATUS',

       'HEALTH CONDITION/ INININSYMPTOMS, SIGNS AND ILL-DEFINED CONDITIONS',

       'HEALTH CONDITION/ VALID SKIP', 'HEALTH CONDITION/ 12197',

       'HEALTH CONDITION/ 12199', 'HEALTH CONDITION/ 12555',

       'HEALTH CONDITION/ INOPEN WOUND OF HEAD, NECK, TRUNK, AND LIMBS',

       'HEALTH CONDITION/ INPERSONS WITH A CONDITION INFLUENCING THEIR HEALTH STATUS',

       'HEALTH CONDITION/ PAPILLARY AND SQUAMOUS CELL NEOPLASMS',

       'HEALTH CONDITION/ 10911',

       'HEALTH CONDITION/ INDISEASES AND DISORDERS OF THE NERVOUS SYSTEM',

       'HEALTH CONDITION/ INDISEASES OF THE DIGESTIVE SYSTEM',

       'HEALTH CONDITION/ ININDISEASES OF THE DIGESTIVE SYSTEM',

       'HEALTH CONDITION/ INININININDISEASES OF THE RESPIRATORY SYSTEM',

       'HEALTH CONDITION/ INJURY TO BLOOD VESSELS',

       'HEALTH CONDITION/ INOTHER ACCIDENTS AND LATE EFFECTS OF ACCIDENTAL INJURY',

       'HEALTH CONDITION/ INPERSONS ENCOUNTERING HEALTH SERVICES IN CIRCUMSTANCES RELATED TO REPRODUCTION AND DEVELOPMENT',

       'HEALTH CONDITION/ INSPRAINS AND STRAINS OF JOINTS AND ADJACENT MUSCLES',

       'HEALTH CONDITION/ INSUPERFICIAL INJURY AND CONTUSION WITH INTACT SKIN SURFACE',

       'HEALTH CONDITION/ 10845', 'HEALTH CONDITION/ 11420',

       'HEALTH CONDITION/ ACCIDENTAL POISONING BY OTHER SOLID AND LIQUID SUBSTANCES, GASES AND VAPOURS',

       'HEALTH CONDITION/ INFRACTURE OF SKULL, SPINE, TRUNK, AND LIMBS',

       'HEALTH CONDITION/ ININSPRAINS AND STRAINS OF JOINTS AND ADJACENT MUSCLES',

       'HEALTH CONDITION/ MISADVENTURES TO PATIENTS DURING SURGICAL AND MEDICAL CARE',

       'HEALTH CONDITION/ 1', 'HEALTH CONDITION/ 11282', 'HEALTH CONDITION/ 9',

       'HEALTH CONDITION/ INCONGENITAL ANOMALIES', 'HEALTH CONDITION/ 12362',

       'HEALTH CONDITION/ INDISEASES OF CIRCULATORY SYSTEM',

       'HEALTH CONDITION/ INININJURY UNDETERMINED WHETHER ACCIDENTALLY OR PURPOSELY INFLICTED',

       'HEALTH CONDITION/ ININSYMPTOMS, SIGNS AND ILL-DEFINED CONDITIONS',

       'HEALTH CONDITION/ INPERSONS ENCOUNTERING HEALTH SERVICES IN OTHER CIRCUMSTANCES',

       'HEALTH CONDITION/ ZOONOTIC BACTERIAL DISEASES',

       'HEALTH CONDITION/ ACCIDENTAL POISONING BY DRUGS, MEDICAMENTS AND BIOLOGICALS',

       'HEALTH CONDITION/ SURGICAL AND MEDICAL PROCEDURES AS THE CAUSE OF ABNORMAL REACTION OF PATIENT OR LATER COMPLICATION',

       'HEALTH CONDITION/ ININDISLOCATION WITHOUT FRACTURE',

       'HEALTH CONDITION/ ARTHROPOD-BORNE VIRAL DISEASES',

       'BODY PART/ "ENTIRE BODY"', 'BODY PART/ "Entire body"', 'BODY PART/ 97',

       'BODY PART/ 98', 'BODY PART/ 99', 'BODY PART/ ABDOMEN',

       'BODY PART/ ADRENAL GLAND', 'BODY PART/ ANKLE', 'BODY PART/ ARM',

       'BODY PART/ Abdomen', 'BODY PART/ Arm', 'BODY PART/ BACK, DORSAL SPINE',

       'BODY PART/ BLADDER', 'BODY PART/ BLOOD, SPLEEN',

       'BODY PART/ BONE(S) NOS', 'BODY PART/ BRAIN, CNS, SPINAL CORD',

         'BODY PART/ BREAST, NIPPLE', 'BODY PART/ BREATHING',

       'BODY PART/ BUTTOCKS', 'BODY PART/ Back, dorsal spine',

       'BODY PART/ Bladder', 'BODY PART/ Blood, spleen',

       'BODY PART/ Bone(s) NOS', 'BODY PART/ Brain, CNS, spinal cord',

       'BODY PART/ Breast, nipple', 'BODY PART/ Buttocks', 'BODY PART/ CHEST',

       'BODY PART/ CHEST WALL, EXTERNAL CHEST; AXILLA', 'BODY PART/ CHIN',

       'BODY PART/ COLLARBONE', 'BODY PART/ Chest', 'BODY PART/ Chin',

       'BODY PART/ DIGESTIVE SYSTEM', 'BODY PART/ EAR (INNER AND OUTER)',

       'BODY PART/ ELBOW', 'BODY PART/ EMOTIONS, "NERVES"',

       'BODY PART/ ESOPHAGUS', 'BODY PART/ EYE', 'BODY PART/ EYELID',

       'BODY PART/ Elbow', 'BODY PART/ Emotions, "nerves"', 'BODY PART/ Eye',

       'BODY PART/ FACE, FOREHEAD, LIPS', 'BODY PART/ FINGERS',

       'BODY PART/ FOOT', 'BODY PART/ Face, forehead, lips',

       'BODY PART/ Fingers', 'BODY PART/ Foot', 'BODY PART/ GALLBLADDER',

       'BODY PART/ GROIN', 'BODY PART/ Gallbladder', 'BODY PART/ Groin',

       'BODY PART/ HAIR', 'BODY PART/ HAND (PALM)', 'BODY PART/ HEAD, SKULL',

       'BODY PART/ HEART', 'BODY PART/ HIP', 'BODY PART/ Hand (palm)',

       'BODY PART/ Heart', 'BODY PART/ Hip', 'BODY PART/ IN"ENTIRE BODY"',

       'BODY PART/ INBACK, DORSAL SPINE', 'BODY PART/ INTESTINE AND COLON',

       'BODY PART/ JAW', 'BODY PART/ JOINTS NOS', 'BODY PART/ Joints NOS',

       'BODY PART/ KIDNEYS', 'BODY PART/ KNEE, KNEECAP', 'BODY PART/ Kidneys',

       'BODY PART/ Knee, kneecap', 'BODY PART/ LEG', 'BODY PART/ LIVER',

       'BODY PART/ LOW BACK, LUMBAR SPINE', 'BODY PART/ LOWER ABDOMEN',

       'BODY PART/ LOWER ARM', 'BODY PART/ LOWER DIGESTIVE TRACT',

         'BODY PART/ LOWER LEG', 'BODY PART/ LUNG, TRACHEA AND BRONCHI',

       'BODY PART/ LYMPHATIC SYSTEM, LYMPH GLANDS', 'BODY PART/ Leg',

       'BODY PART/ Liver', 'BODY PART/ Low back, lumbar spine',

       'BODY PART/ Lower arm', 'BODY PART/ Lower leg',

       'BODY PART/ Lung, trachea and bronchi',

       'BODY PART/ Lymphatic system, lymph glands',

       'BODY PART/ MOUTH AND TONGUE',

       'BODY PART/ MUSCLES, TENDONS, LIGAMENTS NOS *',

       'BODY PART/ Mouth and tongue',

       'BODY PART/ Muscles, tendons, ligaments NOS *',

       'BODY PART/ NECK, CERVICAL VERTEBRAE', 'BODY PART/ NOSE',

       'BODY PART/ Neck, cervical vertebrae',

       'BODY PART/ OTHER FEMALE REPRODUCTIVE SYSTEM; FALLOPIAN TUBES; OVARIES',

       'BODY PART/ OTHER GENITOURINARY TRACT; URETHRA; URETER',

       'BODY PART/ OTHER MALE REPRODUCTIVE SYSTEM; SCROTUM; VAS DEFERENS; TESTES',

       'BODY PART/ OTHER NOS',

       'BODY PART/ Other female reproductive system; fallopian tubes; ovaries',

       'BODY PART/ PELVIS', 'BODY PART/ PENIS',

       'BODY PART/ PERIPHERAL NERVOUS SYSTEM', 'BODY PART/ PITUITARY GLAND',

       'BODY PART/ Pancreas', 'BODY PART/ Pelvis',

       'BODY PART/ Pituitary gland', 'BODY PART/ Prostate',

       'BODY PART/ RECTUM', 'BODY PART/ REFUSAL', 'BODY PART/ RIBS',

       'BODY PART/ Rectum', 'BODY PART/ Ribs', 'BODY PART/ SCALP',

       'BODY PART/ SHOULDER', 'BODY PART/ SIDE, FLANK', 'BODY PART/ SINUS',

       'BODY PART/ SKIN', 'BODY PART/ SPEECH', 'BODY PART/ STOMACH',

       'BODY PART/ Shoulder', 'BODY PART/ Side, flank', 'BODY PART/ Skin',

       'BODY PART/ Stomach', 'BODY PART/ TEETH', 'BODY PART/ THROAT, PHARYNX',

       'BODY PART/ THYROID GLAND', 'BODY PART/ TOES',

       'BODY PART/ TONSILS AND ADENOIDS', 'BODY PART/ TRUNK',

       'BODY PART/ Throat, pharynx', 'BODY PART/ Thyroid gland',

       'BODY PART/ Toes', 'BODY PART/ UPPER ABDOMEN', 'BODY PART/ UPPER ARM',

       'BODY PART/ UPPER DIGESTIVE TRACT', 'BODY PART/ UPPER LEG, THIGH',

       'BODY PART/ Upper arm', 'BODY PART/ Upper leg, thigh',

       'BODY PART/ VAGINA, CERVIX, UTERUS', 'BODY PART/ VALID SKIP',

       'BODY PART/ VASCULAR SYSTEM', 'BODY PART/ VISION',

       'BODY PART/ VOCAL CORDS, LARYNX', 'BODY PART/ Vagina, cervix, uterus',

       'BODY PART/ Vascular system', 'BODY PART/ WRIST', 'BODY PART/ Wrist',

       'REASONS R NOT HAVING HEALTH INSURANCE/ COST IS TOO HIGH',

       'REASONS R NOT HAVING HEALTH INSURANCE/ DEATH OF SPOUSE OR PARTNER',

       'REASONS R NOT HAVING HEALTH INSURANCE/ EMPLOYER DOES NOT OFFER COVERAGE',

       'REASONS R NOT HAVING HEALTH INSURANCE/ GOT DIVORCED OR SEPARATED',

       'REASONS R NOT HAVING HEALTH INSURANCE/ LOST MEDICAID',

       'REASONS R NOT HAVING HEALTH INSURANCE/ LOST MEDICAID (OTHER)',

       'REASONS R NOT HAVING HEALTH INSURANCE/ MEDICAL PLAN BECAUSE OF NEW JOB OR INCREASE IN INCOME',

       'REASONS R NOT HAVING HEALTH INSURANCE/ MEDICAL PLAN STOPPED AFTER PREGNANCY',

       'REASONS R NOT HAVING HEALTH INSURANCE/ NOT SELECTED',

       'REASONS R NOT HAVING HEALTH INSURANCE/ OR NOT ELIGIBLE FOR COVERAGE',

         'REASONS R NOT HAVING HEALTH INSURANCE/ OTHER (SPECIFY)',

       'REASONS R NOT HAVING HEALTH INSURANCE/ PERSON IN FAMILY WITH HEALTH INSURANCE LOST JOB OR CHANGED EMPLOYERS',

       'REASONS R NOT HAVING HEALTH INSURANCE/ REFUSAL',

       'REASONS R NOT HAVING HEALTH INSURANCE/ VALID SKIP',

       'REASONS R NOT HAVING HEALTH INSURANCE/ [FEMALE ONLY] MEDICAID',

       'HEALTH CONDITION/ 0', 'HEALTH CONDITION/ 12547',

       'HEALTH CONDITION/ BENIGN NEOPLASMS', 'HEALTH CONDITION/ BURNS',

       'HEALTH CONDITION/ CERTAIN CONDITIONS ORIGINATING IN THE PERINATAL PERIOD',

       'HEALTH CONDITION/ CERTAIN TRAUMATIC COMPLICATIONS AND UNSPECIFIED INJURIES',

       'HEALTH CONDITION/ COMPLICATIONS OF PREGNANCY, CHILDBIRTH AND THE PUERPERIUM',

       'HEALTH CONDITION/ COMPLICATIONS OF SURGICAL AND MEDICAL CARE NOT ELSEWHERE CLASSIFIED',

       'HEALTH CONDITION/ CONGENITAL ANOMALIES',

       'HEALTH CONDITION/ CRUSHING INJURY',

       'HEALTH CONDITION/ DISEASES AND DISORDERS OF THE NERVOUS SYSTEM',

       'HEALTH CONDITION/ DISEASES OF BLOOD AND BLOOD-FORMING ORGANS',

       'HEALTH CONDITION/ DISEASES OF CIRCULATORY SYSTEM',

       'HEALTH CONDITION/ DISEASES OF EAR AND MASTOID PROCESS',

       'HEALTH CONDITION/ DISEASES OF GENITOURINARY SYSTEM',

       'HEALTH CONDITION/ DISEASES OF PERIPHERAL NERVOUS SYSTEM',

       'HEALTH CONDITION/ DISEASES OF THE DIGESTIVE SYSTEM',

       'HEALTH CONDITION/ DISEASES OF THE MUSCULOSKELETAL SYSTEM AND CONNECTIVE TISSUE',

       'HEALTH CONDITION/ DISEASES OF THE RESPIRATORY SYSTEM',

       'HEALTH CONDITION/ DISEASES OF THE SKIN AND SUBCUTANEOUS TISSUE',

       'HEALTH CONDITION/ DISLOCATION WITHOUT FRACTURE',

       'HEALTH CONDITION/ DISORDERS OF THE EYE AND ADNEXA',

       'HEALTH CONDITION/ ENDOCRINE, NUTRITIONAL AND METABOLIC DISEASES AND IMMUNITY DISORDERS',

       'HEALTH CONDITION/ FRACTURE OF SKULL, SPINE, TRUNK, AND LIMBS',

       'HEALTH CONDITION/ HELMINTHIASES',

       'HEALTH CONDITION/ INCERTAIN TRAUMATIC COMPLICATIONS AND UNSPECIFIED INJURIES',

       'HEALTH CONDITION/ ININPERSONS ENCOUNTERING HEALTH SERVICES IN CIRCUMSTANCES RELATED TO REPRODUCTION AND DEVELOPMENT',

       'HEALTH CONDITION/ INJURY TO NERVES AND SPINAL CORD',

       'HEALTH CONDITION/ INJURY UNDETERMINED WHETHER ACCIDENTALLY OR PURPOSELY INFLICTED',

       'HEALTH CONDITION/ INTERNAL INJURY OF CHEST, ABDOMEN AND PELVIS',

       'HEALTH CONDITION/ INTESTINAL INFECTIOUS DISEASES',

       'HEALTH CONDITION/ INTRACRANIAL INJURY, EXCLUDING THOSE WITH SKULL FRACTURE',

       'HEALTH CONDITION/ LATE EFFECTS OF INFECTIOUS AND PARASITIC DISEASES',

       'HEALTH CONDITION/ LATE EFFECTS OF INJURIES, POISONING, TOXIC EFFECTS AND OTHER EXTERNAL CAUSES',

       'HEALTH CONDITION/ MALIGNANT NEOPLASMS',

       'HEALTH CONDITION/ MENTAL DISORDERS, EXCEPT MENTAL RETARDATION',

       'HEALTH CONDITION/ MENTAL RETARDATION', 'HEALTH CONDITION/ MYCOSES',

       'HEALTH CONDITION/ NEOPLASMS OF UNCERTAIN BEHAVIOUR OR OF UNSPECIFIED NATURE',

       'HEALTH CONDITION/ OPEN WOUND OF HEAD, NECK, TRUNK, AND LIMBS',

       'HEALTH CONDITION/ OTHER AND UNSPECIFIED EFFECTS OF EXTERNAL CAUSES',

       'HEALTH CONDITION/ OTHER BACTERIAL DISEASES',

       'HEALTH CONDITION/ OTHER DISEASES DUE TO VIRUSES AND CHLAMYDIAE',

       'HEALTH CONDITION/ OTHER INFECTIOUS AND PARASITIC DISEASES',

       'HEALTH CONDITION/ PERSONS ENCOUNTERING HEALTH SERVICES FOR SPECIFIC PROCEDURES AND AFTERCARE',

       'HEALTH CONDITION/ PERSONS ENCOUNTERING HEALTH SERVICES IN CIRCUMSTANCES RELATED TO REPRODUCTION AND DEVELOPMENT',

       'HEALTH CONDITION/ PERSONS ENCOUNTERING HEALTH SERVICES IN OTHER CIRCUMSTANCES',

       'HEALTH CONDITION/ PERSONS WITH A CONDITION INFLUENCING THEIR HEALTH STATUS',

       'HEALTH CONDITION/ PERSONS WITH POTENTIAL HEALTH HAZARDS RELATED TO PERSONAL AND FAMILY HISTORY',

       'HEALTH CONDITION/ PERSONS WITHOUT REPORTED DIAGNOSIS ENCOUNTERED DURING EXAMINATION AND INVESTIGATION OF INDIVIDUALS AND POPULATIONS',

       'HEALTH CONDITION/ POISONING BY DRUGS, MEDICAMENTS AND BIOLOGICAL SUBSTANCES',

       'HEALTH CONDITION/ SPRAINS AND STRAINS OF JOINTS AND ADJACENT MUSCLES',

       'HEALTH CONDITION/ SUPERFICIAL INJURY AND CONTUSION WITH INTACT SKIN SURFACE',

       'HEALTH CONDITION/ SYMPTOMS, SIGNS AND ILL-DEFINED CONDITIONS',

       'HEALTH CONDITION/ TOXIC EFFECTS OF SUBSTANCES CHIEFLY NONMEDICINAL AS TO SOURCE',

       'HEALTH CONDITION/ TUBERCULOSIS',

       'HEALTH CONDITION/ VIRAL DISEASES ACCOMPANIED BY EXANTHEM',

       'HEALTH CONDITION/ ACCIDENTAL FALLS',

       'HEALTH CONDITION/ HOMICIDE AND INJURY PURPOSELY INFLICTED BY OTHER PERSONS',

       'HEALTH CONDITION/ INDISEASES OF THE MUSCULOSKELETAL SYSTEM AND CONNECTIVE TISSUE',

       'HEALTH CONDITION/ INSYMPTOMS, SIGNS AND ILL-DEFINED CONDITIONS',

       'HEALTH CONDITION/ MOTOR VEHICLE TRAFFIC ACCIDENTS',

       'HEALTH CONDITION/ OTHER ACCIDENTS AND LATE EFFECTS OF ACCIDENTAL INJURY',

       'HEALTH CONDITION/ 10988',

       'HEALTH CONDITION/ ACCIDENTS DUE TO NATURAL AND ENVIRONMENTAL FACTORS',

       'HEALTH CONDITION/ INDISEASES OF GENITOURINARY SYSTEM',

       'HEALTH CONDITION/ INDISEASES OF THE RESPIRATORY SYSTEM',

       'HEALTH CONDITION/ MOTOR VEHICLE NONTRAFFIC ACCIDENTS',

       'HEALTH CONDITION/ OTHER ROAD VEHICLE ACCIDENTS',

       'HEALTH CONDITION/ POLIOMYELITIS AND OTHER NON-ARTHROPOD-BORNE VIRAL DISEASES OF CENTRAL NERVOUS SYSTEM',

       'HEALTH CONDITION/ 8550',

       'HEALTH CONDITION/ ACCIDENTS CAUSED BY SUBMERSION, SUFFOCATION AND FOREIGN BODIES',

       'HEALTH CONDITION/ DRUGS, MEDICAMENTS AND BIOLOGICAL SUBSTANCES CAUSING ADVERSE EFFECTS IN THERAPEUTIC USE',

       'HEALTH CONDITION/ EFFECTS OF FOREIGN BODY ENTERING THROUGH ORIFICE',

       'HEALTH CONDITION/ ININDISEASES OF THE RESPIRATORY SYSTEM',

       'HEALTH CONDITION/ LEGAL INTERVENTION',

       'HEALTH CONDITION/ PERSONS WITH POTENTIAL HEALTH HAZARDS RELATED TO COMMUNICABLE DISEASES',

       'DEPRESSED']
health_dumm = pd.get_dummies(health,columns=health.drop(num_cols,1).columns)
hlt_tbd=[]

for i in health_dumm:

    if 'NON INTERVIEW' in i:

        hlt_tbd.append(i)

    elif 'INVALID SKIP' in i:

        hlt_tbd.append(i)

    elif 'DONT KNOW' in i:

        hlt_tbd.append(i)

    elif 'NON INTERVIEW' in i:

        hlt_tbd.append(i)
health_dumm.drop(hlt_tbd,1,inplace=True)
for i in health_dumm:

#     print(i)

    if ('IDENTIFICATION CODE'not in i) & ('YEAR' not in i[:5]) & (health_dumm[i].dtype!=np.uint8):

            print(health_dumm[i].dtype)

            health_dumm[i]=health_dumm[i].astype(float).fillna(0).astype(np.uint8)
health_dumm['IDENTIFICATION CODE'] = health_dumm['IDENTIFICATION CODE'].astype(int)

health_dumm['YEAR'] = health_dumm['YEAR'].astype(int)
clean_col=[]

for i in health_dumm:

    j=i.replace("-"," ").replace("'","").replace(',','').replace('(',' ').replace('*','').replace('"','').replace(';',' ').replace(')',' ').replace('?','').replace('>','').replace('<','').replace('[','').replace(']','').replace('/',' ').replace('  ',' ')

#     print(i.replace("'","").replace(',','').replace('(',' ').replace('*','').replace('"','').replace(';',' ').replace(')',' ').replace('?','').replace('>','').replace('<','').replace('[]','').replace(']','').replace('/',' ').replace('  ',' '))

    clean_col.append(j)
health_dumm.columns=clean_col
%matplotlib inline

sns.countplot(x='DEPRESSED', data=health_dumm)
no_dep = len(health_dumm[health_dumm['DEPRESSED'] == 1])

non_dep_indices = health_dumm[health_dumm.DEPRESSED == 0].index

random_indices = np.random.choice(non_dep_indices,no_dep, replace=False)

dep_indices = health_dumm[health_dumm.DEPRESSED == 1].index

under_sample_indices = np.concatenate([dep_indices,random_indices])

under_sample = health_dumm.loc[under_sample_indices]
%matplotlib inline

sns.countplot(x='DEPRESSED', data=under_sample)
X_under = under_sample.loc[:,under_sample.columns != 'DEPRESSED']

y_under = under_sample.loc[:,under_sample.columns == 'DEPRESSED']

X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(X_under,y_under,test_size = 0.2, random_state = 0)
lr_under = LogisticRegression()

lr_under.fit(X_under_train,y_under_train)

y_under_pred = lr_under.predict(X_under_test)

print("recall_score = ",recall_score(y_under_test,y_under_pred))

print("accuracy_score = ",accuracy_score(y_under_test,y_under_pred))

print("classification_report = ",classification_report(y_under_test, y_under_pred)) 
lg_confusion_matrix_value = confusion_matrix(y_under_test,y_under_pred)

lg_confusion_matrix_value
ax = sns.heatmap(lg_confusion_matrix_value, annot=True, fmt='g', cmap=plt.cm.Blues)
gbm = XGBClassifier(learning_rate=0.01, n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)

gbm.fit(X_under_train,y_under_train)
predictions_xgb = gbm.predict(X_under_test)

print("recall_score = ",recall_score(y_under_test,predictions_xgb))

print("accuracy_score = ",accuracy_score(y_under_test,predictions_xgb))

print("classification_report = ",classification_report(y_under_test, y_under_pred))
xgb_confusion_matrix_value = confusion_matrix(y_under_test,predictions_xgb)
ax = sns.heatmap(xgb_confusion_matrix_value, annot=True, fmt='g', cmap=plt.cm.Blues)