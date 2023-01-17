import pandas as pd

import numpy as ny
df = pd.read_csv("D:\AI\healthplan.csv")

df.count()

df.head()
benefits={"ambulance":"AB","child_eye_exam":"EY","child_eyewear":"EW","diagnostic_test":"DT","emergency_room":"DM","generic_drugs":"ER","habilitation_services":"GD","home_health_care":"HA","hospice_service":"HS","imaging":"IM","inpatient_birth":"IB","inpatient_facility":"IP","inpatient_mental_health":"IN","inpatient_physician":"IH","inpatient_substance":"IS","non_preferred_brand_drugs":"ND","outpatient_facility":"OP","outpatient_mental_health":"OM","outpatient_physician":"OH","outpatient_substance":"OS","preferred_brand_drugs":"PD","prenatal_postnatal_care":"PN","preventative_care":"PV","primary_care_physician":"PC","rehabilitation_services":"RH","skilled_nursing":"SN","specialist":"SP","specialty_drugs":"SD","urgent_care":"UC"}

print(len(benefits))
Userdata={"1":"PREMI27","2":"PREMI50","3":"PREMI2C30","4":"PREMC2C30"}
accumlators=['_CopayInnTier1A','_CopayInnTier2A','_CoinsInnTier1A','_CoinsInnTier2A','_CopayOutofNetA','_CoinsOutofNetA']

target=['PLANNAME']
age = input ("Enter the age and dependents details as : 1 <-- individual age 27 2 <-- individual age 50 3 <-- individuals age 30 and 2 children aged 0-14 4 <-- two individuals age 30 and 2 children aged 0-14 ") 

print(age)

premium=input("Enter the afforadble premium amount")

benefits_user=input("Enter your benefits required")

benefitslist=[]

benefitslist=benefits_user.split(",")

print(benefitslist)

feature=[Userdata.get(age)]

target.append(Userdata.get(age))

print((len(benefitslist)*21)+1)

for x in benefitslist:

    for y in accumlators:

        target.append(benefits.get(x)+y)

print(feature)    
print(feature)

print(target)
from sklearn.model_selection import train_test_split
X = df[feature]

y = df[target]
print('df_boston shape =', df.shape)

print('X shape =', X.shape)

print('y shape =', y.shape)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=.20, random_state=5)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(solver='lbfgs', multi_class='auto')
print('X_train shape =', X_train.shape)

print('y_train shape =', y_train.shape)

print('X_test shape =',  X_test.shape)

print('y_test shape =',  y_test.shape)
logreg.fit(X_train,y_train)

Y_pred=logreg.predict(X_test)
Y_pred=logreg.predict(X_test)
import sklearn

print('sklearn: %s' % sklearn.__version__)
sklearn.metrics.accuracy_score(y_test,Y_pred)
print(Y_pred)
print(X_test)
x1=[['338.64','577.11','861.13','1227.88','0','1','0','0','112.647059','0', '0', '0', '1', '0', '4','40','0','0',

         '0',

         '0',

         '0',

         '100',

         '0',

         '4',

         '40'

         ]]

import numpy as np

x2=np.asarray(x1)

x3=x2.reshape(1,25)

x4=x3.astype(np.float64)

    
Y_pred=logreg.predict(x)

print(Y_pred)