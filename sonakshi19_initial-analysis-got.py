import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
battles = pd.read_csv('../input/battles.csv')
battles.head(2)
battles.info()
battles.describe()
battles.isnull()
#Missing Columns shown in yellow

ax = plt.figure(figsize = (12,6))

sns.heatmap(battles.isnull(),yticklabels=False,cbar=False,cmap='viridis',linewidths = 1, linecolor = 'black')
battles.drop('attacker_2',axis=1,inplace=True)

battles.drop('attacker_3',axis=1,inplace=True)

battles.drop('attacker_4',axis=1,inplace=True)

battles.drop('defender_2',axis=1,inplace=True)

battles.drop('defender_3',axis=1,inplace=True)

battles.drop('defender_4',axis=1,inplace=True)

battles.drop('note',axis=1,inplace=True)
#Fill in the missing values with the mean value

battles["defender_size"].fillna(battles["defender_size"].mean(), inplace=True)

battles["attacker_size"].fillna(battles["attacker_size"].mean(), inplace=True)
ax = plt.figure(figsize = (12,6))

sns.heatmap(battles.isnull(),yticklabels=False,cbar=False,cmap='viridis',linewidths = 1, linecolor = 'black')
sns.countplot(x='attacker_outcome',data = battles,palette = 'viridis')
sns.countplot(x='attacker_outcome', hue= 'battle_type',data = battles, palette = 'viridis')
sns.countplot(x='year', data = battles, palette = 'viridis')
battles['attacker_king'].unique()
ax = plt.figure(figsize = (12,6))

sns.countplot(x='attacker_king', hue = 'attacker_outcome',data = battles, palette = 'viridis')
sns.barplot(x='attacker_outcome',y='attacker_size',data=battles,estimator = np.mean)
sns.barplot(x='attacker_outcome',y='defender_size',data=battles,estimator = np.mean)
#Now we need to change the categorical features into values which Machine Learning algorithm can predict

#Converting attacker_outcome column into dummy variables

winloss = pd.get_dummies(battles['attacker_outcome'],drop_first = 'True')
winloss.head()
battles = pd.concat([battles,winloss],axis=1)
#Let's use another dataset present here to get some interesting insights.

chardeath = pd.read_csv('../input/character-deaths.csv')
chardeath.head()
#Missing Columns shown in yellow

ax = plt.figure(figsize = (12,6))

sns.heatmap(chardeath.isnull(),yticklabels=False,cbar=False,cmap='viridis')
chardeath.drop('Book of Death',axis=1,inplace=True)

chardeath.drop('Death Chapter',axis=1,inplace=True)
chardeath['Allegiances'].unique()
sns.countplot(x='GoT',data = chardeath, palette = 'RdBu_r')
Lannister_Got = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'GoT'].sum()

Lannister_CoK = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'CoK'].sum()

Lannister_SoS = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'SoS'].sum()

Lannister_FfC = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'FfC'].sum()

Lannister_DwD = chardeath.loc[chardeath['Allegiances'] == "Lannister", 'DwD'].sum()



Lannister_death_data = np.array([Lannister_Got,Lannister_CoK,Lannister_SoS,Lannister_FfC,Lannister_DwD])

Lannister_death_data = pd.Series.from_array(Lannister_death_data)

Allegiance = range(len(Lannister_death_data))



#Create a Booklist array 

BookList = np.array(['Game of Thrones', 'Clash of Kings', 'Storm of Swords', 'Feast for Crows', 'Dance with Dragons'])



#Visualize the data 

plt.subplots_adjust(bottom=0.2)

font = {'family': 'serif',

        'color':  'black',

        'weight': 'normal',

        'size': 14,

        }

ax = Lannister_death_data.plot(kind='bar',figsize =(12,6),color = 'black')

ax.set_xlabel("Books by R R Martin",fontdict=font)

rects = ax.patches

for rect, label in zip(rects, Lannister_death_data):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize = 16)

plt.ylabel("Deaths in House Lannister in various books",fontdict = font)

plt.xticks(Allegiance,BookList, rotation = 45)
HouTargaryen_Got = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'GoT'].sum()

HouTargaryen_CoK = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'CoK'].sum()

HouTargaryen_SoS = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'SoS'].sum()

HouTargaryen_FfC = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'FfC'].sum()

HouTargaryen_DwD = chardeath.loc[chardeath['Allegiances'] == "House Targaryen", 'DwD'].sum()



Targaryen_Got = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'GoT'].sum()

Targaryen_CoK = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'CoK'].sum()

Targaryen_SoS = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'SoS'].sum()

Targaryen_FfC = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'FfC'].sum()

Targaryen_DwD = chardeath.loc[chardeath['Allegiances'] == "Targaryen", 'DwD'].sum()



HouTargaryen_Got = HouTargaryen_Got + Targaryen_Got

HouTargaryen_CoK = HouTargaryen_CoK + Targaryen_CoK

HouTargaryen_SoS = HouTargaryen_SoS + Targaryen_SoS

HouTargaryen_FfC = HouTargaryen_FfC + Targaryen_FfC

HouTargaryen_DwD = HouTargaryen_DwD + Targaryen_DwD



Targaryen_Death = np.array([HouTargaryen_Got,HouTargaryen_CoK,HouTargaryen_SoS,HouTargaryen_FfC,HouTargaryen_DwD])

Targaryen_Death = pd.Series.from_array(Targaryen_Death)



HouStark_GoT = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'GoT'].sum()

HouStark_CoK = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'CoK'].sum()

HouStark_SoS = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'SoS'].sum()

HouStark_FfC = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'FfC'].sum()

HouStark_DwD = chardeath.loc[chardeath['Allegiances'] == "House Stark", 'DwD'].sum()



Stark_GoT = chardeath.loc[chardeath['Allegiances'] == "Stark", 'GoT'].sum()

Stark_CoK = chardeath.loc[chardeath['Allegiances'] == "Stark", 'CoK'].sum()

Stark_SoS = chardeath.loc[chardeath['Allegiances'] == "Stark", 'SoS'].sum()

Stark_FfC = chardeath.loc[chardeath['Allegiances'] == "Stark", 'FfC'].sum()

Stark_DwD = chardeath.loc[chardeath['Allegiances'] == "Stark", 'DwD'].sum()



HouStark_GoT = HouStark_GoT + Stark_GoT

HouStark_CoK = HouStark_CoK + Stark_CoK

HouStark_SoS = HouStark_SoS + Stark_SoS

HouStark_FfC = HouStark_FfC + Stark_FfC

HouStark_DwD = HouStark_DwD + Stark_DwD



Stark_Death = np.array([HouStark_GoT,HouStark_CoK,HouStark_SoS,HouStark_FfC,HouStark_DwD])

Stark_Death = pd.Series.from_array(Stark_Death)



NightWatch_GoT = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'GoT'].sum()

NightWatch_CoK = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'CoK'].sum()

NightWatch_SoS = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'SoS'].sum()

NightWatch_FfC = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'FfC'].sum()

NightWatch_DwD = chardeath.loc[chardeath['Allegiances'] == "Night's Watch", 'DwD'].sum()





NightWatch_Death = np.array([NightWatch_GoT,NightWatch_CoK,NightWatch_SoS,NightWatch_FfC,NightWatch_DwD])

NightWatch_Death = pd.Series.from_array(NightWatch_Death)





Baratheon_GoT = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'GoT'].sum()

Baratheon_CoK = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'CoK'].sum()

Baratheon_SoS = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'SoS'].sum()

Baratheon_FfC = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'FfC'].sum()

Baratheon_DwD = chardeath.loc[chardeath['Allegiances'] == "Baratheon", 'DwD'].sum()



HouBaratheon_GoT = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'GoT'].sum()

HouBaratheon_CoK = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'CoK'].sum()

HouBaratheon_SoS = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'SoS'].sum()

HouBaratheon_FfC = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'FfC'].sum()

HouBaratheon_DwD = chardeath.loc[chardeath['Allegiances'] == "House Baratheon", 'DwD'].sum()



Baratheon_GoT = Baratheon_GoT + HouBaratheon_GoT

Baratheon_CoK = Baratheon_CoK + HouBaratheon_CoK

Baratheon_SoS = Baratheon_SoS + HouBaratheon_SoS

Baratheon_FfC = Baratheon_FfC + HouBaratheon_FfC

Baratheon_DwD = Baratheon_DwD + HouBaratheon_DwD





Baratheon_Death = np.array([Baratheon_GoT,Baratheon_CoK,Baratheon_SoS,Baratheon_FfC,Baratheon_DwD])

Baratheon_Death = pd.Series.from_array(Baratheon_Death)

Got = pd.Series([Baratheon_GoT,HouStark_GoT,HouTargaryen_Got,NightWatch_GoT,Lannister_Got],

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

Got_death = max(Got)



SoS = pd.Series([Baratheon_SoS,HouStark_SoS,HouTargaryen_SoS,NightWatch_SoS,Lannister_SoS],

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

SoS_death = max(SoS)



FfC = pd.Series([Baratheon_FfC,HouStark_FfC,HouTargaryen_FfC,NightWatch_FfC,Lannister_FfC],

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

FfC_death = max(FfC)



CoK = pd.Series([Baratheon_CoK,HouStark_CoK,HouTargaryen_CoK,NightWatch_CoK,Lannister_CoK], 

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

CoK_death = max(CoK)



DwD = pd.Series([Baratheon_DwD,HouStark_DwD,HouTargaryen_DwD,NightWatch_DwD,Lannister_DwD],

                index=["Baratheon", "Stark", "Targaryen", "Nightwatch", "Lannister"])

DwD_death = max(DwD)

Death = np.array([Got_death,SoS_death,FfC_death,CoK_death,DwD_death])

Death = pd.Series.from_array(Death)





Houses = pd.Series(['Lannister','NightWatch','Baratheon/Stark','Stark','Nightwatch'])



BookList1 = np.array(['Game of Thrones', 'Clash of Kings', 'Storm of Swords', 'Feast for Crows', 'Dance with Dragons'])

Allegiance1 = [0,1,2,3,4]



Death = Death.sort_values()

font = {'family': 'serif',

        'color':  'black',

        'weight': 'normal',

        'size': 14,

        }



sns.set_style("whitegrid")

ax3 = plt.figure(figsize = (12,6))

ax2 = sns.barplot(Allegiance1,Death,palette = "Blues_d")

ax2.set_xlabel("Maximum no. of Death in each book and respective house",fontdict = font)

rects = ax2.patches

for rect, label in zip(rects, Houses):

    height = rect.get_height()

    ax2.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize = 18)

plt.ylabel("Max death in various Allegiances",fontdict = font)

plt.xticks(Allegiance1,BookList1, rotation = 45)
chardeath.head(2)
charpred = pd.read_csv('../input/character-predictions.csv')
charpred.info()
charpred.head(3)
#Missing Columns shown in yellow

ax = plt.figure(figsize = (12,6))

sns.heatmap(charpred.isnull(),yticklabels=False,cbar=False,cmap='viridis')
charpred.drop('isAliveMother',axis=1,inplace=True)

charpred.drop('isAliveFather',axis=1,inplace=True)

charpred.drop('isAliveHeir',axis=1,inplace=True)

charpred.drop('isAliveSpouse',axis=1,inplace=True)

charpred.drop('mother',axis=1,inplace=True)

charpred.drop('father',axis=1,inplace=True)

charpred.drop('heir',axis=1,inplace=True)

charpred.drop('spouse',axis=1,inplace=True)
charpred.drop('dateOfBirth',axis=1,inplace=True)

charpred.drop('DateoFdeath',axis=1,inplace=True)

charpred.drop('age',axis=1,inplace=True)

charpred.drop('title',axis=1,inplace=True)
charpred.head(2)
sns.barplot(x='isAlive',y='popularity',data=charpred,estimator = np.mean)
Pop = charpred[charpred['popularity'] == 1]
pivot = Pop.pivot_table(values = 'numDeadRelations', index = 'culture',columns='name')

ax = plt.figure(figsize = (12,6))

font = {'family': 'serif',

        'color':  'red',

        'weight': 'normal',

        'size': 14,

        }

sns.heatmap(pivot, cmap= 'coolwarm')

plt.xlabel('Most popular characters in GOT',fontdict=font)

plt.ylabel('Different Culture',fontdict=font)

plt.title('Most popular characters in GOT and number of death associated with them(box and slide bar representing them)',fontdict

         =font)
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

charpred["house"]=charpred["house"].fillna("")

charpred["house_encode"] = lb_make.fit_transform(charpred["house"])

charpred[["house", "house_encode"]].head(3)

cult = {

    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],

    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],

    'Asshai': ["asshai'i", 'asshai'],

    'Lysene': ['lysene', 'lyseni'],

    'Andal': ['andal', 'andals'],

    'Braavosi': ['braavosi', 'braavos'],

    'Dornish': ['dornishmen', 'dorne', 'dornish'],

    'Myrish': ['myr', 'myrish', 'myrmen'],

    'Westermen': ['westermen', 'westerman', 'westerlands'],

    'Westerosi': ['westeros', 'westerosi'],

    'Stormlander': ['stormlands', 'stormlander'],

    'Norvoshi': ['norvos', 'norvoshi'],

    'Northmen': ['the north', 'northmen'],

    'Free Folk': ['wildling', 'first men', 'free folk'],

    'Qartheen': ['qartheen', 'qarth'],

    'Reach': ['the reach', 'reach', 'reachmen'],

}





for k, v in cult.items():

    charpred.loc[charpred.culture.str.upper().isin(v), 'culture'] = k

        

charpred['culture'].head(3)
lb_make = LabelEncoder()

charpred["culture"]=charpred["culture"].fillna("")

charpred["culture_encode"] = lb_make.fit_transform(charpred["culture"])

charpred[["culture", "culture_encode"]].head(3)
charpredn = charpred.copy()
charpredn.drop('culture',axis=1,inplace=True)

charpredn.drop('house',axis=1,inplace=True)

charpredn.drop('name',axis=1,inplace=True)

charpredn.drop('actual',axis=1,inplace=True)

charpredn.drop('pred',axis=1,inplace=True)

charpredn.drop('plod',axis=1,inplace=True)

charpredn.drop('alive',axis=1,inplace=True)

charpredn.drop('S.No',axis=1,inplace=True)
X = charpredn.drop('isAlive',axis=1)

y = charpredn['isAlive']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

pred = logmodel.predict_proba(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

print('\n')

print(confusion_matrix(y_test,predictions))
Coefficients = pd.DataFrame(list(zip(X_test.columns, logmodel.coef_[0])))

Coefficients
import numpy as np

from sklearn import linear_model, datasets, cross_validation, metrics



# import some data to play with

X = charpredn.drop('isAlive',axis=1)

y = charpredn['isAlive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



print("Accuracy: %2f" % metrics.accuracy_score(y_test, predictions))

print("Precision: %2f" % metrics.precision_score(y_test, predictions, average="binary"))

print("F1: %2f" % metrics.f1_score(y_test, predictions, average="binary"))
for label in np.arange(2):

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, predictions, pos_label=label)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    plt.plot(false_positive_rate, true_positive_rate, label='AUC(%d) = %0.2f' % (label, roc_auc))



    plt.title('Receiver Operating Characteristic')

plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.ylabel('True-Positive-Rate')

plt.xlabel('False-Positive-Rate')

plt.show()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
dec_predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,dec_predictions))

print('\n')

print(confusion_matrix(y_test,dec_predictions))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))