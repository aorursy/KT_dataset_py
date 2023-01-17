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
from tkinter import *

import numpy as np

import pandas as pd

#List of the symptoms is listed here in list l1.

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',

'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',

'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',

'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',

'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',

'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',

'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',

'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',

'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',

'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',

'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',

'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',

'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',

'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',

'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',

'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',

'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',

'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',

'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',

'yellow_crust_ooze']

#List of Diseases is listed in list disease.

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',

'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',

' Migraine','Cervical spondylosis',

'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',

'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',

'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',

'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',

'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',

'Impetigo']

l2=[]

for i in range(0,len(l1)):

    l2.append(0)

df=pd.read_csv("/kaggle/input/Training.csv")

#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,

'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,

'Migraine':11,'Cervical spondylosis':12,

'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,

'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,

'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,

'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,

'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,

'Impetigo':40}},inplace=True)

#check the df 

#print(df.head())

X= df[l1]

#print(X)

y = df[["prognosis"]]

np.ravel(y)

#print(y)

#Read a csv named Testing.csv

tr=pd.read_csv("/kaggle/input/Testing.csv")

#Use replace method in pandas.

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,

'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,

'Migraine':11,'Cervical spondylosis':12,

'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,

'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,

'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,

'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,

'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,

'Impetigo':40}},inplace=True)

X_test= tr[l1]

y_test = tr[["prognosis"]]

#print(y_test)

np.ravel(y_test)

X_test= tr[l1]

y_test = tr[["prognosis"]]

np.ravel(y_test)

psymptoms = input().split(",")

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier() 

    clf3 = clf3.fit(X,y)

    

    for k in range(0,len(l1)):

        for z in psymptoms:

            if(z==l1[k]):

                l2[k]=1

    inputtest = [l2]

    predict = clf3.predict(inputtest)

    predicted=predict[0]

    h=0

    for a in range(0,len(disease)):

        if(predicted == a):

            print(disease[a])

            print("Disease Prediction using Decision Tree")

            break

    """if (h==1):

        t3.delete("1.0", END)

        t3.insert(END, disease[a])

    else:

        t3.delete("1.0", END)

        t3.insert(END, "Not Found")"""

DecisionTree()

def randomforest():

    from sklearn.ensemble import RandomForestClassifier

    clf4 = RandomForestClassifier()

    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy 

    

    

    for k in range(0,len(l1)):

        for z in psymptoms:

            if(z==l1[k]):

                l2[k]=1

    inputtest = [l2]

    predict = clf4.predict(inputtest)

    predicted=predict[0]

    h=0

    for a in range(0,len(disease)):

        if(predicted == a):

            print(disease[a])

            print("Disease Prediction using Random Forest")

            break

    """if (h==1):

        t3.delete("1.0", END)

        t3.insert(END, disease[a])

    else:

        t3.delete("1.0", END)

        t3.insert(END, "Not Found")"""

randomforest()

def NaiveBayes():

    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()

    gnb=gnb.fit(X,np.ravel(y))

    

    

    for k in range(0,len(l1)):

        for z in psymptoms:

            if(z==l1[k]):

                l2[k]=1

    inputtest = [l2]

    predict = gnb.predict(inputtest)

    predicted=predict[0]

    h=0

    for a in range(0,len(disease)):

        if(predicted == a):

            print(disease[a])

            print("Disease Prediction using Naive Bayes")

            break

    """if (h==1):

        t3.delete("1.0", END)

        t3.insert(END, disease[a])

    else:

        t3.delete("1.0", END)

        t3.insert(END, "Not Found")"""

NaiveBayes()