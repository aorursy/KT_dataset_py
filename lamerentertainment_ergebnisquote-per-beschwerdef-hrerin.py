import pandas as pd
import re
pd.set_option('precision', 3)

B2011 = pd.read_csv('/kaggle/input/bger6b2018/1B_2011.csv')
B2012 = pd.read_csv('/kaggle/input/bger6b2018/1B_2012.csv')
B2013 = pd.read_csv('/kaggle/input/bger6b2018/1B_2013.csv')
B2014 = pd.read_csv('/kaggle/input/bger6b2018/1B_2014.csv')
B2015 = pd.read_csv('/kaggle/input/bger6b2018/1B_2015.csv')
B2016 = pd.read_csv('/kaggle/input/bger6b2018/1B_2016.csv')
B2017 = pd.read_csv('/kaggle/input/bger6b2018/1B_2017.csv')
B2018 = pd.read_csv('/kaggle/input/bger6b2018/1B_2018.csv')
B2019 = pd.read_csv('/kaggle/input/bger6b2018/1B_2019.csv')
B2020 = pd.read_csv('/kaggle/input/bger6b2018/1B_2020.csv')
B2011b = pd.read_csv('/kaggle/input/bger6b2018/6B_2011.csv')
B2012b = pd.read_csv('/kaggle/input/bger6b2018/6B_2012.csv')
B2013b = pd.read_csv('/kaggle/input/bger6b2018/6B_2013.csv')
B2014b = pd.read_csv('/kaggle/input/bger6b2018/6B_2014.csv')
B2015b = pd.read_csv('/kaggle/input/bger6b2018/6B_2015.csv')
B2016b = pd.read_csv('/kaggle/input/bger6b2018/6B_2016.csv')
B2017b = pd.read_csv('/kaggle/input/bger6b2018/6B_2017.csv')
B2018b = pd.read_csv('/kaggle/input/bger6b2018/6B_2018.csv')
B2019b = pd.read_csv('/kaggle/input/bger6b2018/6B_2019.csv')
B2020b = pd.read_csv('/kaggle/input/bger6b2018/6B_2020.csv')



df = pd.concat([B2011, B2012, B2013, B2014, B2015, B2016, B2017, B2018, B2019, B2020, \
                      B2011b, B2012b, B2013b, B2014b, B2015b, B2016b, B2017b, B2018b, B2019b, B2020b])

df = df[['BeschwerdefuehrerIn', 'Rechtsvertreter', 'BeschwerdegegnerIn', 'Verfahrensergebnis']]


def staw_unifier(value):
    if re.search(r'[Ss]taatsanwalt|anwaltschaft|Ministère|Procureur|Ministero', value) is not None: 
        return 'Staatsanwaltschaft'
    else:
         return value
        
df['BeschwerdefuehrerIn'] = df['BeschwerdefuehrerIn'].apply(str).apply(staw_unifier)
df['BeschwerdegegnerIn'] = df['BeschwerdegegnerIn'].apply(str).apply(staw_unifier)

def anon_unifier(value):
    if re.search(r'[A-Z]\._', value) is not None:
        return 'anonym'
    else:
        return value

df['BeschwerdefuehrerIn'] = df['BeschwerdefuehrerIn'].apply(str).apply(anon_unifier)


haufige_beschwerdefuehrer = df.BeschwerdefuehrerIn.value_counts()
# print(haufige_beschwerdefuehrer.head(21))
anon = df['BeschwerdefuehrerIn'] == 'anonym'
df_anon = df[anon]

df_anon_rf = df_anon[df_anon.Rechtsvertreter != 'unvertreten']
total_beschwerden_anon_vertreten = df_anon_rf.Verfahrensergebnis.count()
df_anon_rf_gut = df_anon_rf[df_anon_rf.Verfahrensergebnis == 'Gutheissung']
total_gutheissungen_anon_vertreten = df_anon_rf_gut.Verfahrensergebnis.count()
prozent = round((total_gutheissungen_anon_vertreten*100)/total_beschwerden_anon_vertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_vertreten} Beschwerden von vertretenen anonymisierten privaten Parteien. \
#Davon wurden {total_gutheissungen_anon_vertreten} gutgeheissen. Was einem Prozensatz von {prozent}% entspricht.')

df_anon_rf_tlwgut = df_anon_rf[df_anon_rf.Verfahrensergebnis == 'teilweise Gutheissung']
total_tlwgutheissungen_anon_vertreten = df_anon_rf_tlwgut.Verfahrensergebnis.count()
prozenttlwgut = round((total_tlwgutheissungen_anon_vertreten*100)/total_beschwerden_anon_vertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_vertreten} Beschwerden von vertretenen anonymisierten privaten Parteien. \
#Davon wurden {total_tlwgutheissungen_anon_vertreten} teilweise gutgeheissen. Was einem Prozensatz von {prozenttlwgut}% entspricht.')

df_anon_rf_abw = df_anon_rf[df_anon_rf.Verfahrensergebnis == 'Abweisung']
total_abweisungen_anon_vertreten = df_anon_rf_abw.Verfahrensergebnis.count()
prozentabw = round((total_abweisungen_anon_vertreten*100)/total_beschwerden_anon_vertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_vertreten} Beschwerden von vertretenen anonymisierten privaten Parteien. \
#Davon wurden {total_abweisungen_anon_vertreten} abgewiesen. Was einem Prozensatz von {prozentabw}% entspricht.')

df_anon_rf_nicht = df_anon_rf[df_anon_rf.Verfahrensergebnis == 'Nichteintreten']
total_nichteintreten_anon_vertreten = df_anon_rf_nicht.Verfahrensergebnis.count()
prozentnicht = round((total_nichteintreten_anon_vertreten*100)/total_beschwerden_anon_vertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_vertreten} Beschwerden von vertretenen anonymisierten privaten Parteien. \
#Davon wurden auf {total_nichteintreten_anon_vertreten} nicht eingetreten. Was einem Prozensatz von {prozentnicht}% entspricht.')

df_anon_rf_gglos = df_anon_rf[df_anon_rf.Verfahrensergebnis == 'Gegenstandslosigkeit']
total_gegenstandslos_anon_vertreten = df_anon_rf_gglos.Verfahrensergebnis.count()
prozentgglos = round((total_gegenstandslos_anon_vertreten*100)/total_beschwerden_anon_vertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_vertreten} Beschwerden von vertretenen anonymisierten privaten Parteien. \
#Davon wurden  {total_gegenstandslos_anon_vertreten} gegenstandslos. Was einem Prozensatz von {prozentgglos}% entspricht.')
#print('---')

df_anon_norf = df_anon[df_anon.Rechtsvertreter == 'unvertreten']
total_beschwerden_anon_unvertreten = df_anon_norf.Verfahrensergebnis.count()
df_anon_norf_gut = df_anon_norf[df_anon_norf.Verfahrensergebnis == 'Gutheissung']
total_gutheissungen_anon_unvertreten = df_anon_norf_gut.Verfahrensergebnis.count()
prozent = round((total_gutheissungen_anon_unvertreten*100)/total_beschwerden_anon_unvertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_unvertreten} Beschwerden von unvertretenen anonymisierten privaten Parteien. \
#Davon wurden {total_gutheissungen_anon_unvertreten} gutgeheissen. Was einem Prozensatz von {prozent}% entspricht.')

df_anon_norf_tlwgut = df_anon_norf[df_anon_norf.Verfahrensergebnis == 'teilweise Gutheissung']
total_tlwgutheissungen_anon_unvertreten = df_anon_norf_tlwgut.Verfahrensergebnis.count()
prozenttlwgut = round((total_tlwgutheissungen_anon_unvertreten*100)/total_beschwerden_anon_unvertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_unvertreten} Beschwerden von unvertretenen anonymisierten privaten Parteien. \
#Davon wurden {total_tlwgutheissungen_anon_unvertreten} teilweise gutgeheissen. Was einem Prozensatz von {prozenttlwgut}% entspricht.')

df_anon_norf_abw = df_anon_norf[df_anon_norf.Verfahrensergebnis == 'Abweisung']
total_abweisungen_anon_unvertreten = df_anon_norf_abw.Verfahrensergebnis.count()
prozentabw = round((total_abweisungen_anon_unvertreten*100)/total_beschwerden_anon_unvertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_unvertreten} Beschwerden von unvertretenen anonymisierten privaten Parteien. \
#Davon wurden {total_abweisungen_anon_unvertreten} abgewiesen. Was einem Prozensatz von {prozentabw}% entspricht.')

df_anon_norf_nicht = df_anon_norf[df_anon_norf.Verfahrensergebnis == 'Nichteintreten']
total_nichteintreten_anon_unvertreten = df_anon_norf_nicht.Verfahrensergebnis.count()
prozentnicht = round((total_nichteintreten_anon_unvertreten*100)/total_beschwerden_anon_unvertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_unvertreten} Beschwerden von unvertretenen anonymisierten privaten Parteien. \
#Davon wurde auf {total_nichteintreten_anon_unvertreten} nicht eingetreten. Was einem Prozensatz von {prozentnicht}% entspricht.')

df_anon_norf_gglos = df_anon_norf[df_anon_norf.Verfahrensergebnis == 'Gegenstandslosigkeit']
total_gegenstandslos_anon_unvertreten = df_anon_norf_gglos.Verfahrensergebnis.count()
prozentgglos = round((total_gegenstandslos_anon_unvertreten*100)/total_beschwerden_anon_unvertreten, 2)
#print(f'Es gab total {total_beschwerden_anon_unvertreten} Beschwerden von unvertretenen anonymisierten privaten Parteien. \
#Davon wurden {total_gegenstandslos_anon_unvertreten} gegenstandslos. Was einem Prozensatz von {prozentgglos}% entspricht.')
#print('---')

staw = df['BeschwerdefuehrerIn'] == 'Staatsanwaltschaft'
df_staw = df[staw]
total_beschwerden_staw = df_staw.Verfahrensergebnis.count()
df_staw_gut = df_staw[df_staw.Verfahrensergebnis == 'Gutheissung']
total_gutheissungen_staw = df_staw_gut.Verfahrensergebnis.count()
prozent = round((total_gutheissungen_staw*100)/total_beschwerden_staw, 2)
#print(f'Es gab total {total_beschwerden_staw} Beschwerden von einer Staatsanwaltschaft. \
#Davon wurden {total_gutheissungen_staw} gutgeheissen. Was einem Prozensatz von {prozent}% entspricht.')

df_staw_tlwgut = df_staw[df_staw.Verfahrensergebnis == 'teilweise Gutheissung']
total_tlwgutheissungen_staw = df_staw_tlwgut.Verfahrensergebnis.count()
prozenttlwgut = round((total_tlwgutheissungen_staw*100)/total_beschwerden_staw, 2)
#print(f'Es gab total {total_beschwerden_staw} Beschwerden von einer Staatsanwaltschaft. \
#Davon wurden {total_tlwgutheissungen_staw} teilweise gutgeheissen. Was einem Prozensatz von {prozenttlwgut}% entspricht.')

df_staw_abw = df_staw[df_staw.Verfahrensergebnis == 'Abweisung']
total_abweisungen_staw = df_staw_abw.Verfahrensergebnis.count()
prozentabw = round((total_abweisungen_staw*100)/total_beschwerden_staw, 2)
#print(f'Es gab total {total_beschwerden_staw} Beschwerden von einer Staatsanwaltschaft. \
#Davon wurden {total_abweisungen_staw} abgewiesen. Was einem Prozensatz von {prozentabw}% entspricht.')

df_staw_nicht = df_staw[df_staw.Verfahrensergebnis == 'Nichteintreten']
total_nichteintreten_staw = df_staw_nicht.Verfahrensergebnis.count()
prozentnicht = round((total_nichteintreten_staw*100)/total_beschwerden_staw, 2)
#print(f'Es gab total {total_beschwerden_staw} Beschwerden von einer Staatsanwaltschaft. \
#Davon wurde auf {total_nichteintreten_staw} nicht eingetreten. Was einem Prozensatz von {prozentnicht}% entspricht.')

df_staw_gglos = df_staw[df_staw.Verfahrensergebnis == 'Gegenstandslosigkeit']
total_gegenstandslos_staw = df_staw_gglos.Verfahrensergebnis.count()
prozentgglos = round((total_gegenstandslos_staw*100)/total_beschwerden_staw, 2)
#print(f'Es gab total {total_beschwerden_staw} Beschwerden von einer Staatsanwaltschaft. \
#Davon wurden {total_gegenstandslos_staw} gegenstandslos. Was einem Prozensatz von {prozentnicht}% entspricht.')

from matplotlib import pyplot as plt

labels = 'Gut', 'tlw Gut', 'Abw', 'NichtE', 'GGlos'
colors = ['green', 'lightgreen', 'red', 'darkred', 'lightblue']
anon_vertreten = [total_gutheissungen_anon_vertreten, total_tlwgutheissungen_anon_vertreten, total_abweisungen_anon_vertreten, total_nichteintreten_anon_vertreten, total_gegenstandslos_anon_vertreten]
anon_unvertreten = [total_gutheissungen_anon_unvertreten, total_tlwgutheissungen_anon_unvertreten, total_abweisungen_anon_unvertreten, total_nichteintreten_anon_unvertreten, total_gegenstandslos_anon_unvertreten]
anon_staw = [total_gutheissungen_staw, total_tlwgutheissungen_staw, total_abweisungen_staw, total_nichteintreten_staw, total_gegenstandslos_staw]

fig, axs = plt.subplots(3, 1)
axs[0].pie(anon_vertreten, labels=labels, colors=colors, autopct='%1.1f%%', counterclock=False)
axs[0].set_title(f'vertretene private Beschwerdeführer (tot={total_beschwerden_anon_vertreten})')
axs[1].pie(anon_unvertreten, labels=labels, colors=colors, autopct='%1.1f%%', counterclock=False)
axs[1].set_title(f'unvertretene private Beschwerdeführer (tot={total_beschwerden_anon_unvertreten})')
axs[2].pie(anon_staw, labels=labels, colors=colors, autopct='%1.1f%%', counterclock=False)
axs[2].set_title(f'Staatsanwaltschaft als Beschwerdeführerin. (tot={total_beschwerden_staw})')


fig.set_figheight(25)
fig.set_figwidth(25)

