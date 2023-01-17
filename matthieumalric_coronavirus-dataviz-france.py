# library import



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime
# load of the data available on https://www.data.gouv.fr/fr/organizations/sante-publique-france/



national = pd.read_csv("../input/sursaud-covid19-quotidien-2020-04-07-19h00-france.csv")

national.head()
# We don't want to analyse by age, therefore we filter on 0 sursaud_cl_age



national.query('sursaud_cl_age_corona == "0"', inplace=True)

national.date_de_passage = pd.to_datetime(national.date_de_passage)

national.head()


plt.figure(figsize=(15,10))



plt.bar(national.date_de_passage, national.nbre_pass_corona, color="grey",alpha=0.5)

plt.plot(national.date_de_passage, national.nbre_hospit_corona, color="red")





plt.legend(["hospitalizations among emergency department visits for suspicion of COVID", 

            "Emergency room visits for suspicion of COVID ",

            

            ], fontsize=13, loc="upper left" )

plt.ylabel('nbre de patients', color="black", fontsize=14)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlim(("2020-03-01" , "2020-04-07" ))

plt.title('Emergency visit and hospitalizations  among them for Covid in France',fontsize=18)

plt.show()

plt.figure(figsize=(15,8))



plt.bar(national.date_de_passage, national.nbre_acte_corona, color="red")





plt.ylabel("number of medical acts", color="black", fontsize=14)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Daily medical acts (SOS Médecins) for suspicion of COVID-19 in France',fontsize=18)

# load of the data available on https://www.data.gouv.fr/fr/organizations/sante-publique-france/





hospit = pd.read_csv("../input/donnees-hospititalieres-covid19-incid.csv", sep = ";")

hospit.head()
# Computing a pourcentage of Rea



hospitclean=hospit.groupby("jour").sum()

hospitclean.reset_index(inplace=True)

hospitclean.jour = pd.to_datetime(hospitclean.jour)



hospitclean["pourcent_rea"] = hospitclean.incid_rea/hospitclean.incid_hosp*100

hospitclean.sort_values("jour",ascending=False)
plt.figure(figsize=(13,8))



plt.bar(hospitclean.jour, hospitclean.incid_rea, color="blue")





plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Daily number of new intensive care admissions in France',fontsize=18)
test = pd.read_csv("../input/donnees-tests-covid19-labo-quotidien-2020-04-07-19h00.csv", sep = ';')
test.head()
test.info()
test.jour = pd.to_datetime(test.jour)

testclean=test.groupby("jour").sum()

testclean.reset_index(inplace=True)

testclean["pourcentage"] = testclean.nb_pos/testclean.nb_test*100
testclean.sort_values("jour",ascending=False)
fig, ax1 = plt.subplots(figsize=(13, 8))





ax1.set_xlabel('day', fontsize=18)

ax1.set_ylabel("Porcentage of positive cases", color="red", fontsize=18)

ax1.plot(testclean.jour, testclean.pourcentage, color="red")

ax1.tick_params(axis='y', labelcolor="red")

plt.xticks(fontsize=14)





plt.yticks(fontsize=14)



ax2 = ax1.twinx()  # On initie un deuxième axe qui partage le même x-axis





ax2.set_ylabel('Number of test done in private labs', color="black",fontsize=18)  # Nous avons déjà le xlabel avec le ax1

ax2.bar(testclean.jour, testclean.nb_pos, color="grey",alpha=0.8)

ax2.tick_params(axis='y', labelcolor="black",labelsize=14)



ax1.legend(["Porcentage of positive cases"], fontsize=13, loc='best')

ax2.legend(["Number of test done in private labs"], fontsize=13, loc='best')



plt.title('numbers of detection test done in private labs and porcentage of positive cases',fontsize=18)



plt.show()