import pandas as pd

B2020 = pd.read_csv('/kaggle/input/bger6b2018/1B_2020.csv')
B2020b = pd.read_csv('/kaggle/input/bger6b2018/6B_2020.csv')

super_db = pd.concat([B2020, B2020b])
print(f'Auswertung von ' + str(super_db.shape[0]) + ' veröffentlichten Urteilen mit Jahrgangsnummer 2020.')
 
## data cleaning
# unnötige Zeile löschen
super_db = super_db.drop(columns=['Unnamed: 0'])
# Reihen mit get-Vorinstanz-Fehler löschen
err_rows = super_db[super_db.Vorinstanz == '# Fehler bei get_vorinstanz'].index
super_db.drop(err_rows, inplace=True)

# Reihen mit zweideutiger Vorinstanz 'Cour de cassation pénal' VD/NE löschen bis Regex optimiert
err_rows = super_db[super_db.Vorinstanz == 'Cour de cassation pénale'].index
super_db.drop(err_rows, inplace=True)


## Verfahrensergebnisse pro Vorinstanz zählen
tot_count = super_db.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'tot'})
df_gut = super_db[super_db.Verfahrensergebnis == 'Gutheissung'].reset_index()
gut_count = df_gut.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'gut'})
df_tlwgut = super_db[super_db.Verfahrensergebnis == 'teilweise Gutheissung'].reset_index()
tlwgut_count = df_tlwgut.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'tlwgut'})
df_abw = super_db[super_db.Verfahrensergebnis == 'Abweisung'].reset_index()
abw_count = df_abw.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'abw'})
df_ne = super_db[super_db.Verfahrensergebnis == 'Nichteintreten'].reset_index()
ne_count = df_ne.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'ne'})
df_gglos = super_db[super_db.Verfahrensergebnis == 'Gegenstandslosigkeit'].reset_index()
gglos_count = df_gglos.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'gglos'})
# print(gglos_count.head(10))

## pivot mit Verfahrensergebnissen pro Vorinstanz erstellen
df1 = pd.merge(tot_count, gut_count, how='outer')
df2 = pd.merge(df1, tlwgut_count, how='outer')
df3 = pd.merge(df2, abw_count, how='outer')
df4 = pd.merge(df3, ne_count, how='outer')
pivot_ergebnisse = pd.merge(df4, gglos_count, how='outer')
pivot_ergebnisse.fillna(0, inplace=True)
pivot_ergebnisse['gut'] = pd.to_numeric(pivot_ergebnisse['gut'], downcast='signed')
pivot_ergebnisse['tlwgut'] = pd.to_numeric(pivot_ergebnisse['tlwgut'], downcast='signed')
pivot_ergebnisse['abw'] = pd.to_numeric(pivot_ergebnisse['abw'], downcast='signed')
pivot_ergebnisse['ne'] = pd.to_numeric(pivot_ergebnisse['ne'], downcast='signed')
pivot_ergebnisse['gglos'] = pd.to_numeric(pivot_ergebnisse['gglos'], downcast='signed')
# pivot_ergebnisse.style

## Gutheissungen und teilw. Gutheissungen zusammenzählen, neue Reihe "gut+tlwgut", nach tlwgut einsortieren
pivot_ergebnisse['gut+tlwgut'] = pivot_ergebnisse.gut + pivot_ergebnisse.tlwgut
reihenfolge = pivot_ergebnisse.columns.tolist()
neuereihenfolge = reihenfolge[:4] + [reihenfolge[-1]] + reihenfolge[4:-1]
pivot_ergebnisse = pivot_ergebnisse[neuereihenfolge]

## materielle Entscheide durch Subtrahieren der "ne" und "gglos" eruieren, neue Reihe "tot_mat", nach "tot" einsortieren
pivot_ergebnisse['tot_mat'] = pivot_ergebnisse['tot'] - pivot_ergebnisse['ne'] - pivot_ergebnisse['gglos']
reihenfolge = pivot_ergebnisse.columns.tolist()
neuereihenfolge = reihenfolge[:2] + [reihenfolge[-1]] + reihenfolge[2:-1]
pivot_ergebnisse = pivot_ergebnisse[neuereihenfolge]

## Prozentangabe Gutheissungen erstellen
pivot_ergebnisse['%gut'] = pivot_ergebnisse['gut']/pivot_ergebnisse['tot']*100

## Prozentangabe Gutheissungen + teilw. Gutheissungen erstellen
pivot_ergebnisse['%gut+tlwgut'] = pivot_ergebnisse['gut+tlwgut']/pivot_ergebnisse['tot']*100

## Prozentangabe Gutheissungen + teilw. Gutheissungen aller materiell beurteilten Fälle erstellen
pivot_ergebnisse['%gut+tlwgut/tot_mat'] = (pivot_ergebnisse['gut+tlwgut'])/pivot_ergebnisse['tot_mat']*100

## Prozentangabe Gutheissungen erstellen
pivot_ergebnisse['%gut+tlwgut'] = pivot_ergebnisse['gut+tlwgut']/pivot_ergebnisse['tot']*100

## Vorinstanz mit total-count unter 5 aussortieren
pivot_ergebnisse = pivot_ergebnisse[pivot_ergebnisse.tot >= 10].reset_index(drop=True)


#funktion um Spalten hervorzuheben:
def highlight_cols(x):
    #copy df to new - original data are not changed
    df = x.copy()
    #select all values to default value - none
    df.loc[:,:] = ''
    #overwrite values lightyellow color
    df[['tot']] = 'font-weight: bold'
    #overwrite values lightyellow color
    df[['gut', 'tlwgut', 'gut+tlwgut']] = 'background-color: lightgreen'
    #overwrite values lightyellow color
    df[['gut+tlwgut']] = 'font-weight: bold; background-color: lightgreen'
    #overwrite values lightred color
    df[['abw', 'ne', 'gglos']] = 'background-color: lightyellow'
    #return color df
    return df      

pivot_ergebnisse.style.background_gradient(subset=['%gut', '%gut+tlwgut', '%gut+tlwgut/tot_mat'], cmap='Blues')\
.apply(highlight_cols, axis=None)
from matplotlib import pyplot as plt

## Nach Gutheissungsquote sortieren
pivot_ergebnisse = pivot_ergebnisse.sort_values('%gut+tlwgut', ascending=False)


## Graph erstellen
plt.figure(figsize=[14, 4.8])
ax1 = plt.subplot()
ax1.set_xticks(range(len(pivot_ergebnisse['%gut+tlwgut'])))
ax1.set_xticklabels(pivot_ergebnisse.Vorinstanz.to_list(), rotation='vertical')
plt.bar(range(len(pivot_ergebnisse['%gut+tlwgut'])), pivot_ergebnisse['%gut+tlwgut'].to_list())

plt.title('Gutheissungsquote pro Vorinstanz in Prozent im Verhältnis zu allen beurteilten Fällen')
plt.ylabel('%')
plt.xlabel('Vorinstanz')

plt.show()
from matplotlib import pyplot as plt

## Nach Gutheissungsquote sortieren
pivot_ergebnisse = pivot_ergebnisse.sort_values('%gut+tlwgut/tot_mat', ascending=False)


## Graph erstellen
plt.figure(figsize=[14, 4.8])
ax1 = plt.subplot()
ax1.set_xticks(range(len(pivot_ergebnisse['%gut+tlwgut/tot_mat'])))
ax1.set_xticklabels(pivot_ergebnisse.Vorinstanz.to_list(), rotation='vertical')
plt.bar(range(len(pivot_ergebnisse['%gut+tlwgut/tot_mat'])), pivot_ergebnisse['%gut+tlwgut/tot_mat'].to_list())

plt.title('Gutheissungsquote pro Vorinstanz in Prozent am Anteil aller materiell entschiedenen Fälle (Abweisungen und Gutheissungen)')
plt.ylabel('%')
plt.xlabel('Vorinstanz')

plt.show()

B2020b = pd.read_csv('/kaggle/input/bger6b2018/6B_2020.csv')



super_db = pd.concat([B2020b])


## data cleaning
# unnötige Zeile löschen
super_db = super_db.drop(columns=['Unnamed: 0'])
# Reihen mit get-Vorinstanz-Fehler löschen
err_rows = super_db[super_db.Vorinstanz == '# Fehler bei get_vorinstanz'].index
super_db.drop(err_rows, inplace=True)

# Reihen mit zweideutiger Vorinstanz 'Cour de cassation pénal' VD/NE löschen bis Regex optimiert
err_rows = super_db[super_db.Vorinstanz == 'Cour de cassation pénale'].index
super_db.drop(err_rows, inplace=True)


## Verfahrensergebnisse pro Vorinstanz zählen
tot_count = super_db.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'tot'})
df_gut = super_db[super_db.Verfahrensergebnis == 'Gutheissung'].reset_index()
gut_count = df_gut.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'gut'})
df_tlwgut = super_db[super_db.Verfahrensergebnis == 'teilweise Gutheissung'].reset_index()
tlwgut_count = df_tlwgut.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'tlwgut'})
df_abw = super_db[super_db.Verfahrensergebnis == 'Abweisung'].reset_index()
abw_count = df_abw.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'abw'})
df_ne = super_db[super_db.Verfahrensergebnis == 'Nichteintreten'].reset_index()
ne_count = df_ne.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'ne'})
df_gglos = super_db[super_db.Verfahrensergebnis == 'Gegenstandslosigkeit'].reset_index()
gglos_count = df_gglos.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'gglos'})
# print(gglos_count.head(10))

## pivot mit Verfahrensergebnissen pro Vorinstanz erstellen
df1 = pd.merge(tot_count, gut_count, how='outer')
df2 = pd.merge(df1, tlwgut_count, how='outer')
df3 = pd.merge(df2, abw_count, how='outer')
df4 = pd.merge(df3, ne_count, how='outer')
pivot_ergebnisse = pd.merge(df4, gglos_count, how='outer')
pivot_ergebnisse.fillna(0, inplace=True)
pivot_ergebnisse['gut'] = pd.to_numeric(pivot_ergebnisse['gut'], downcast='signed')
pivot_ergebnisse['tlwgut'] = pd.to_numeric(pivot_ergebnisse['tlwgut'], downcast='signed')
pivot_ergebnisse['abw'] = pd.to_numeric(pivot_ergebnisse['abw'], downcast='signed')
pivot_ergebnisse['ne'] = pd.to_numeric(pivot_ergebnisse['ne'], downcast='signed')
pivot_ergebnisse['gglos'] = pd.to_numeric(pivot_ergebnisse['gglos'], downcast='signed')
# pivot_ergebnisse.style

## Gutheissungen und teilw. Gutheissungen zusammenzählen, neue Reihe "gut+tlwgut", nach tlwgut einsortieren
pivot_ergebnisse['gut+tlwgut'] = pivot_ergebnisse.gut + pivot_ergebnisse.tlwgut
reihenfolge = pivot_ergebnisse.columns.tolist()
neuereihenfolge = reihenfolge[:4] + [reihenfolge[-1]] + reihenfolge[4:-1]
pivot_ergebnisse = pivot_ergebnisse[neuereihenfolge]

## materielle Entscheide durch Subtrahieren der "ne" und "gglos" eruieren, neue Reihe "tot_mat", nach "tot" einsortieren
pivot_ergebnisse['tot_mat'] = pivot_ergebnisse['tot'] - pivot_ergebnisse['ne'] - pivot_ergebnisse['gglos']
reihenfolge = pivot_ergebnisse.columns.tolist()
neuereihenfolge = reihenfolge[:2] + [reihenfolge[-1]] + reihenfolge[2:-1]
pivot_ergebnisse = pivot_ergebnisse[neuereihenfolge]

## Prozentangabe Gutheissungen erstellen
pivot_ergebnisse['%gut'] = pivot_ergebnisse['gut']/pivot_ergebnisse['tot']*100

## Prozentangabe Gutheissungen + teilw. Gutheissungen erstellen
pivot_ergebnisse['%gut+tlwgut'] = pivot_ergebnisse['gut+tlwgut']/pivot_ergebnisse['tot']*100

## Prozentangabe Gutheissungen + teilw. Gutheissungen aller materiell beurteilten Fälle erstellen
pivot_ergebnisse['%gut+tlwgut/tot_mat'] = (pivot_ergebnisse['gut+tlwgut'])/pivot_ergebnisse['tot_mat']*100

## Prozentangabe Gutheissungen erstellen
pivot_ergebnisse['%gut+tlwgut'] = pivot_ergebnisse['gut+tlwgut']/pivot_ergebnisse['tot']*100

## Vorinstanz mit total-count unter 5 aussortieren
pivot_ergebnisse = pivot_ergebnisse[pivot_ergebnisse.tot >= 10].reset_index(drop=True)


#funktion um Spalten hervorzuheben:
def highlight_cols(x):
    #copy df to new - original data are not changed
    df = x.copy()
    #select all values to default value - none
    df.loc[:,:] = ''
    #overwrite values lightyellow color
    df[['tot']] = 'font-weight: bold'
    #overwrite values lightyellow color
    df[['gut', 'tlwgut', 'gut+tlwgut']] = 'background-color: lightgreen'
    #overwrite values lightyellow color
    df[['gut+tlwgut']] = 'font-weight: bold; background-color: lightgreen'
    #overwrite values lightred color
    df[['abw', 'ne', 'gglos']] = 'background-color: lightyellow'
    #return color df
    return df      

pivot_ergebnisse.style.background_gradient(subset=['%gut', '%gut+tlwgut', '%gut+tlwgut/tot_mat'], cmap='Blues')\
.apply(highlight_cols, axis=None)
## Nach Gutheissungsquote sortieren
pivot_ergebnisse = pivot_ergebnisse.sort_values('%gut+tlwgut', ascending=False)


## Graph erstellen
plt.figure(figsize=[14, 4.8])
ax1 = plt.subplot()
ax1.set_xticks(range(len(pivot_ergebnisse['%gut+tlwgut'])))
ax1.set_xticklabels(pivot_ergebnisse.Vorinstanz.to_list(), rotation='vertical')
plt.bar(range(len(pivot_ergebnisse['%gut+tlwgut'])), pivot_ergebnisse['%gut+tlwgut'].to_list())

plt.title('Gutheissungsquote pro Vorinstanz in Prozent im Verhältnis zu allen beurteilten Fällen')
plt.ylabel('%')
plt.xlabel('Vorinstanz')

plt.show()
## Nach Gutheissungsquote sortieren
pivot_ergebnisse = pivot_ergebnisse.sort_values('%gut+tlwgut/tot_mat', ascending=False)


## Graph erstellen
plt.figure(figsize=[14, 4.8])
ax1 = plt.subplot()
ax1.set_xticks(range(len(pivot_ergebnisse['%gut+tlwgut/tot_mat'])))
ax1.set_xticklabels(pivot_ergebnisse.Vorinstanz.to_list(), rotation='vertical')
plt.bar(range(len(pivot_ergebnisse['%gut+tlwgut/tot_mat'])), pivot_ergebnisse['%gut+tlwgut/tot_mat'].to_list())

plt.title('Gutheissungsquote pro Vorinstanz in Prozent am Anteil aller materiell entschiedenen Fälle (Abweisungen und Gutheissungen)')
plt.ylabel('%')
plt.xlabel('Vorinstanz')

plt.show()

B2020 = pd.read_csv('/kaggle/input/bger6b2018/1B_2020.csv')



super_db = pd.concat([B2020])


## data cleaning
# unnötige Zeile löschen
super_db = super_db.drop(columns=['Unnamed: 0'])
# Reihen mit get-Vorinstanz-Fehler löschen
err_rows = super_db[super_db.Vorinstanz == '# Fehler bei get_vorinstanz'].index
super_db.drop(err_rows, inplace=True)

# Reihen mit zweideutiger Vorinstanz 'Cour de cassation pénal' VD/NE löschen bis Regex optimiert
err_rows = super_db[super_db.Vorinstanz == 'Cour de cassation pénale'].index
super_db.drop(err_rows, inplace=True)


## Verfahrensergebnisse pro Vorinstanz zählen
tot_count = super_db.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'tot'})
df_gut = super_db[super_db.Verfahrensergebnis == 'Gutheissung'].reset_index()
gut_count = df_gut.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'gut'})
df_tlwgut = super_db[super_db.Verfahrensergebnis == 'teilweise Gutheissung'].reset_index()
tlwgut_count = df_tlwgut.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'tlwgut'})
df_abw = super_db[super_db.Verfahrensergebnis == 'Abweisung'].reset_index()
abw_count = df_abw.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'abw'})
df_ne = super_db[super_db.Verfahrensergebnis == 'Nichteintreten'].reset_index()
ne_count = df_ne.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'ne'})
df_gglos = super_db[super_db.Verfahrensergebnis == 'Gegenstandslosigkeit'].reset_index()
gglos_count = df_gglos.groupby(['Vorinstanz']).Verfahrensnummer.count().reset_index().rename(columns={'Verfahrensnummer': 'gglos'})
# print(gglos_count.head(10))

## pivot mit Verfahrensergebnissen pro Vorinstanz erstellen
df1 = pd.merge(tot_count, gut_count, how='outer')
df2 = pd.merge(df1, tlwgut_count, how='outer')
df3 = pd.merge(df2, abw_count, how='outer')
df4 = pd.merge(df3, ne_count, how='outer')
pivot_ergebnisse = pd.merge(df4, gglos_count, how='outer')
pivot_ergebnisse.fillna(0, inplace=True)
pivot_ergebnisse['gut'] = pd.to_numeric(pivot_ergebnisse['gut'], downcast='signed')
pivot_ergebnisse['tlwgut'] = pd.to_numeric(pivot_ergebnisse['tlwgut'], downcast='signed')
pivot_ergebnisse['abw'] = pd.to_numeric(pivot_ergebnisse['abw'], downcast='signed')
pivot_ergebnisse['ne'] = pd.to_numeric(pivot_ergebnisse['ne'], downcast='signed')
pivot_ergebnisse['gglos'] = pd.to_numeric(pivot_ergebnisse['gglos'], downcast='signed')
# pivot_ergebnisse.style

## Gutheissungen und teilw. Gutheissungen zusammenzählen, neue Reihe "gut+tlwgut", nach tlwgut einsortieren
pivot_ergebnisse['gut+tlwgut'] = pivot_ergebnisse.gut + pivot_ergebnisse.tlwgut
reihenfolge = pivot_ergebnisse.columns.tolist()
neuereihenfolge = reihenfolge[:4] + [reihenfolge[-1]] + reihenfolge[4:-1]
pivot_ergebnisse = pivot_ergebnisse[neuereihenfolge]

## materielle Entscheide durch Subtrahieren der "ne" und "gglos" eruieren, neue Reihe "tot_mat", nach "tot" einsortieren
pivot_ergebnisse['tot_mat'] = pivot_ergebnisse['tot'] - pivot_ergebnisse['ne'] - pivot_ergebnisse['gglos']
reihenfolge = pivot_ergebnisse.columns.tolist()
neuereihenfolge = reihenfolge[:2] + [reihenfolge[-1]] + reihenfolge[2:-1]
pivot_ergebnisse = pivot_ergebnisse[neuereihenfolge]

## Prozentangabe Gutheissungen erstellen
pivot_ergebnisse['%gut'] = pivot_ergebnisse['gut']/pivot_ergebnisse['tot']*100

## Prozentangabe Gutheissungen + teilw. Gutheissungen erstellen
pivot_ergebnisse['%gut+tlwgut'] = pivot_ergebnisse['gut+tlwgut']/pivot_ergebnisse['tot']*100

## Prozentangabe Gutheissungen + teilw. Gutheissungen aller materiell beurteilten Fälle erstellen
pivot_ergebnisse['%gut+tlwgut/tot_mat'] = (pivot_ergebnisse['gut+tlwgut'])/pivot_ergebnisse['tot_mat']*100

## Prozentangabe Gutheissungen erstellen
pivot_ergebnisse['%gut+tlwgut'] = pivot_ergebnisse['gut+tlwgut']/pivot_ergebnisse['tot']*100

## Vorinstanz mit total-count unter 5 aussortieren
pivot_ergebnisse = pivot_ergebnisse[pivot_ergebnisse.tot >= 10].reset_index(drop=True)


#funktion um Spalten hervorzuheben:
def highlight_cols(x):
    #copy df to new - original data are not changed
    df = x.copy()
    #select all values to default value - none
    df.loc[:,:] = ''
    #overwrite values lightyellow color
    df[['tot']] = 'font-weight: bold'
    #overwrite values lightyellow color
    df[['gut', 'tlwgut', 'gut+tlwgut']] = 'background-color: lightgreen'
    #overwrite values lightyellow color
    df[['gut+tlwgut']] = 'font-weight: bold; background-color: lightgreen'
    #overwrite values lightred color
    df[['abw', 'ne', 'gglos']] = 'background-color: lightyellow'
    #return color df
    return df      

pivot_ergebnisse.style.background_gradient(subset=['%gut', '%gut+tlwgut', '%gut+tlwgut/tot_mat'], cmap='Blues')\
.apply(highlight_cols, axis=None)
## Nach Gutheissungsquote sortieren
pivot_ergebnisse = pivot_ergebnisse.sort_values('%gut+tlwgut', ascending=False)


## Graph erstellen
plt.figure(figsize=[14, 4.8])
ax1 = plt.subplot()
ax1.set_xticks(range(len(pivot_ergebnisse['%gut+tlwgut'])))
ax1.set_xticklabels(pivot_ergebnisse.Vorinstanz.to_list(), rotation='vertical')
plt.bar(range(len(pivot_ergebnisse['%gut+tlwgut'])), pivot_ergebnisse['%gut+tlwgut'].to_list())

plt.title('Gutheissungsquote pro Vorinstanz in Prozent im Verhältnis zu allen beurteilten Fällen')
plt.ylabel('%')
plt.xlabel('Vorinstanz')

plt.show()
## Nach Gutheissungsquote sortieren
pivot_ergebnisse = pivot_ergebnisse.sort_values('%gut+tlwgut/tot_mat', ascending=False)


## Graph erstellen
plt.figure(figsize=[14, 4.8])
ax1 = plt.subplot()
ax1.set_xticks(range(len(pivot_ergebnisse['%gut+tlwgut/tot_mat'])))
ax1.set_xticklabels(pivot_ergebnisse.Vorinstanz.to_list(), rotation='vertical')
plt.bar(range(len(pivot_ergebnisse['%gut+tlwgut/tot_mat'])), pivot_ergebnisse['%gut+tlwgut/tot_mat'].to_list())

plt.title('Gutheissungsquote pro Vorinstanz in Prozent am Anteil aller materiell entschiedenen Fälle (Abweisungen und Gutheissungen)')
plt.ylabel('%')
plt.xlabel('Vorinstanz')

plt.show()