import numpy as np 

import pandas as pd 
data_2017 = pd.read_csv('../input/open-medic-2017/OPEN_MEDIC_2017.CSV',sep=';',encoding='latin-1')

data_2018 = pd.read_csv('../input/open-medic-2018/OPEN_MEDIC_2018.CSV',sep=';',encoding='latin-1')

data_2019 = pd.read_csv('../input/open-medic-2019/OPEN_MEDIC_2019.CSV',sep=';',encoding='latin-1')
data_2017.shape
data_2018.shape
data_2019.shape
data_2017.head()
data_2018.head()
data_2019.head()
data_2017["Year"]="2017"

data_2018["Year"]="2018"

data_2019["Year"]="2019"
data_2017.columns
data_2018.columns
data_2019.columns
columns= ['ATC1', 'l_ATC1', 'ATC2', 'L_ATC2', 'ATC3', 'L_ATC3', 'ATC4', 'L_ATC4',

       'ATC5', 'L_ATC5', 'CIP13', 'L_CIP13', 'TOP_GEN', 'GEN_NUM', 'AGE',

       'SEXE', 'BEN_REG', 'PSP_SPE', 'BOITES', 'REM', 'BSE','YEAR']
data_2017.columns =  columns

data_2018.columns =  columns

data_2019.columns =  columns
data_concat = pd.concat([data_2017,data_2018,data_2019])
data_concat.shape
data_concat
#Percentage of NAN Values 

NAN = [(c, data_concat[c].isna().mean()*100) for c in data_concat]

NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])

NAN
data_concat =data_concat.drop(['ATC1','ATC2','ATC3','ATC4','ATC5','CIP13','L_ATC3','L_ATC4','L_ATC5','TOP_GEN','GEN_NUM'],axis=1)

data_concat.shape
sex_map  = {1:'Masculin',2:'Feminin',9:'Inconnu'}

data_concat['SEXE'] = data_concat['SEXE'].map(sex_map)
data_concat['SEXE']
region_map = {5:'''Outre-mer ''',11:'Ile-de-France', 24:'Centre-Val de Loire', 27:'Bourgogne-Franche-Comté',

            28:'Normandie',32:'Nord-Pas-de-Calais-Picardie', 44:'Alsace-Champagne-Ardenne-Lorraine',

             52:'Pays de la Loire', 53:'Bretagne',75:'Aquitaine-Limousin-Poitou-Charentes',

             76:'Languedoc-Roussillon-Midi-Pyrénées',84:'Auvergne-Rhône-Alpes',93:'''Provence-Alpes-Côte d'Azur et Corse''',

            0:'Inconnue',9:'Inconnue',99:'Inconnue'}

data_concat['BEN_REG'] = data_concat['BEN_REG'].map(region_map)
data_concat['BEN_REG']
Prescriber_map = {1:'Médecine generale liberale',2:'Anesthésiste-réanimateur libéral', 

             3:'Pathologie cardio-vasculaire liberale', 4:'Chirurgie liberale',

            5:'Dermatologie et de vénéréologie liberale',6:'Radiologie liberale',

             7:'Gynecologie obstetrique liberale',

             8:'Gastro-entérologue et hepatologie liberale', 9:'Médecine interne libéral',

             11:'Oto rhino-laryngologie liberale',12:'Pédiatrie liberale',

             13:'Pneumologie liberale',

            14:'Rhumatologie liberale',15:'Ophtalmologie liberale',17:'Psychiatrie liberale' ,            

             18:'Stomatologie liberale',31:'Médecine physique et de réadaptation libérale',

             32:'Neurologie libérale',

             35:'Nephrologie libérale',37:'Anatomie-cytologie-pathologique libérale',

             38:'Directeur laboratoire médecin libéral',

             42:'Endocrinologie et metabolismes libéral',90:'Prescripteurs salaries',

             98:'Dentistes, Auxiliaires médicaux, Laboratoires, Sages-femmes',99:'Valeur inconnue',

}

data_concat['PSP_SPE'] = data_concat['PSP_SPE'].map(Prescriber_map)

data_concat['PSP_SPE'] 
age_map = {0:'0 to 19 years',20:'20 to 59 years', 60:'Over 60 years', 99:'Age Inconnu'}

data_concat['AGE'] = data_concat['AGE'].map(age_map)

data_concat['AGE']
data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('''MEDICAMENTS POUR LES TROUBLES DE L'ACIDITE''','''TROUBLES DE L'ACIDITE''') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('''MEDICAMENTS POUR LES TROUBLES FONCTIONNELS GASTROINTESTINAUX''','''TROUBLES FONCTIONNELS GASTROINTESTINAUX''') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS DE LA DIGESTION, ENZYMES INCLUSES','DIGESTION, ENZYMES INCLUSES') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS DU DIABETE','DIABETE') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('AUTRES MEDICAMENTS DES VOIES DIGESTIVES ET DU METABOLISME','AUTRES DES VOIES DIGESTIVES ET DU METABOLISME') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS AGISSANT SUR LE SYSTEME RENINE-ANGIOTENSINE','SYSTEME RENINE-ANGIOTENSINE') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS CONTRE LE PSORIASIS','CONTRE LE PSORIASIS') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS UROLOGIQUES','UROLOGIQUES') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('TOPIQUES POUR DOULEURS ARTICULAIRES OU MUSCULAIRES','DOULEURS ARTICULAIRES OU MUSCULAIRES') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS POUR LE TRAITEMENT DES DESORDRES OSSEUX','TRAITEMENT DES DESORDRES OSSEUX') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('AUTRES MEDICAMENTS DES DESORDRES MUSCULOSQUELETTIQUES','AUTRES DES DESORDRES MUSCULOSQUELETTIQUES') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('AUTRES MEDICAMENTS DU SYSTEME NERVEUX','AUTRES DU SYSTEME NERVEUX') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS POUR LES SYNDROMES OBSTRUCTIFS DES VOIES AERIENNES','SYNDROMES OBSTRUCTIFS DES VOIES AERIENNES') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS DU RHUME ET DE LA TOUX','RHUME ET TOUX') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('''AUTRES MEDICAMENTS DE L'APPAREIL RESPIRATOIRE''','''AUTRES DE L'APPAREIL RESPIRATOIRE''') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS OPHTALMOLOGIQUES','OPHTALMOLOGIQUES') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS OTOLOGIQUES','OTOLOGIQUES') 

data_concat["L_ATC2"]= data_concat["L_ATC2"].replace('MEDICAMENTS POUR DIAGNOSTIC','DIAGNOSTIC') 
data_concat["L_ATC2"]
new_cip=[]

for cip in data_concat["L_CIP13"]:

    i = 0

    while ((i < len(cip)) and (not(cip[i].isdigit()))):

        i= i+1

    new_cip.append(cip[:i-1].capitalize() )
data_concat['Drug_Name']=new_cip
data_concat =data_concat.drop(['L_CIP13'],axis=1)
data_concat['Drug_Name']
data_concat.head()