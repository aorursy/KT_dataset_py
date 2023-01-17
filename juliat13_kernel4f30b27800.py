# Pandas & numpy fuer Datenframes und -manipulationen
import pandas as pd
import numpy as np
# Schönes plotten
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Path Modul um Pfade in Windows vernuenftig zu integrieren
from pathlib import Path

# Feature Selection, Performance und Klassifikationsalgorithmen
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
 
# Yellowbrick visualisiert den Feature Selection Algorithmus RFE schöner
from yellowbrick.features import RFECV as RFECV_yb
# Zum Ausgleich von Unbalancen in der Verteilung der Zielvariablen
from imblearn.over_sampling import SMOTE
# Einlesen der Daten mit pandas
# data_folder = Path("D:/Bankanalyse/")
# file_to_open = data_folder / "TrainData.csv"
# file_test_to_open = data_folder / "TestData.csv"

# Trainingsdatensatz
Bankdaten = pd.read_csv("../input/bankdatenfinal/TrainData.csv", sep=',')

# Testdatensatz
Bankdaten_klass = pd.read_csv("../input/bankdatenfinal/TestData.csv", sep=',')
#Die ersten 10 Zeilen des Datensatzes
Bankdaten.head(10)
#Statistische Beschreibung jedes Attributes des Datensatzes
Bankdaten.describe()
# Informationen über die abgespeicherten Datentypen
Bankdaten.info()
# Konvertierung von numerischen Features als float Datentypen
# Schleife durch jede Spalte des Trainigsdatensatzes
for col in list(Bankdaten.columns):
    # Wähle die Spalten bzw. Features aus, die konvertiert werden sollen:
    if ('Tag' in col or 'Dauer' in col or 'Alter' in col or 'Kontostand' in 
        col or 'Anzahl der Ansprachen' in col or 'Anzahl KOntakte letzte Kampagne' in col):
        # Konvertiere in float
        Bankdaten[col] = Bankdaten[col].astype(float)
        
# Schleife durch jede Spalte des Testdatensatzes
for col in list(Bankdaten_klass.columns):
    # Wähle die Spalten bzw. Features aus, die konvertiert werden sollen:
    if ('Tag' in col or 'Dauer' in col or 'Alter' in col or 'Kontostand' in 
        col or 'Anzahl der Ansprachen' in col or 'Anzahl KOntakte letzte Kampagne' in col):
        # Konvertiere in float
        Bankdaten_klass[col] = Bankdaten_klass[col].astype(float)
# Identifiziere Missing Values und Nullen in dem Datensatz 

def missing_values_tabelle(data):
    
        
        # Gibt die Summe aller NaNs pro Spalte aus:
        nr_miss_val = data.apply(lambda x: sum(x.isnull()),axis=0)
        
        # Prozentsatz der fehlenden Werte
        miss_val_prozent = ((nr_miss_val / len(data))* 100).round(2)
        
        # Zähle die Nullen in jeder Spalte bzw. Attribut
        nr_zeros = data.apply(lambda x: sum(x==0.0),axis=0)
        
        #Zähle den Text 'Unbekannt' oder 'Unknown'
        #nr_unknown = data.apply(lambda x: sum(x=='unknown'), axis=0)
        
        # Übersichtstabelle zur besseren Darstellung mit pandas
        miss_val_tabelle = pd.concat([nr_miss_val, miss_val_prozent, nr_zeros], axis=1)
        
        # Tabellenbeschriftung
        miss_val_tabelle_name_columns = miss_val_tabelle.rename(
        columns = {0 : 'Missing Values', 1 : '% Missing Values', 2 : 'Nullen'})
        
        
        # Sortiere die Zeilen nach Prozentsatz der Missing Values um:
        miss_val_tabelle_name_columns = miss_val_tabelle_name_columns.sort_values('% Missing Values', ascending=False)
        
#         # Print some summary information
#         print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
#             "There are " + str(mis_val_table_ren_columns.shape[0]) +
#               " columns that have missing values.")
        
        
        return miss_val_tabelle_name_columns
print("Tabelle für Trainingsdatensatz")
missing_values_tabelle(Bankdaten)
print("Tabelle für Testdatensatz")
missing_values_tabelle(Bankdaten_klass)
# Entferne die Features 'Stammnummer', 'Anruf-ID' und 'Tage seit letzter Kampagne' aus Trainingsdatensatz
Bankdaten = Bankdaten.drop(columns = ['Stammnummer', 'Anruf-ID', 'Tage seit letzter Kampagne'])

# Entferne die Features 'Zielvariable', 'Anruf-ID' und 'Tage seit letzter Kampagne' aus Testdatensatz
Bankdaten_klass = Bankdaten_klass.drop(columns = ['Zielvariable', 'Anruf-ID', 'Tage seit letzter Kampagne'])
# Umwandlung der kategorischen Zielvariablen in einen numerischen Datentyp. 1 für Ja und 0 für Nein.
Bankdaten['Zielvariable'] = Bankdaten['Zielvariable'].apply(lambda x: 1 if x == 'ja' else 0)
# Funktion die alle Histogramme der numerischen Attribute automatisch plottet

def plot_numeric_histos(data,Subplot_Spalten):
    
    '''Der Funktion wird übergeben:
    - data: Datensatz der geplottet werden soll
    - Subplot_Spalten: wieviele Plots nebeneinander geplottet werden sollen
    '''
    
    
    # Auswählen aller numerischer Attribute
    numeric_subset = data.select_dtypes('number')

    # Anzahl numerischer Attribute
    nr_att = len(numeric_subset.columns)
    
    # Indexzähler
    i=1
    
    # Berechnung der nötigen Subplotdimension über die Anzahl der zu plottenden Attribute und der User definierten Spaltenanzahl
    Subplot_Zeilen = (int(nr_att/Subplot_Spalten)) + (nr_att % Subplot_Spalten)
    
    # Anpassung der Bildgröße anhand der Anzahl der zu plottenden Attribute
    x = 10*Subplot_Spalten
    y = 8*Subplot_Zeilen
    fig = plt.figure(figsize=(x,y))
    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Schleife durch alle Spalten und plotten aller Histogramme
    for col in numeric_subset.columns:
    
        ax = fig.add_subplot(Subplot_Zeilen, Subplot_Spalten, i)
        ax.grid(True)
        ax.hist(Bankdaten[col], color='#0504aa',alpha=0.6, edgecolor = 'k')
        ax.set_xlabel(col, fontsize = 20.0)
        ax.set_ylabel('Anzahl Kunden', fontsize = 20.0)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        i+=1
        
    #return 1
        

plot_numeric_histos(Bankdaten,2)
# Funktion zum Plotten der Boxplots aller numerischer Attribute

def plot_numeric_box (data, Subplot_Spalten):
    
    '''Der Funktion wird übergeben:
    - data: Datensatz der geplottet werden soll
    - Subplot_Spalten: wieviele Plots nebeneinander geplottet werden sollen
    '''
    
    # Auswählen aller numerischer Attribute
    numeric_subset = data.select_dtypes('number')

    # Anzahl numerischer Attribute
    nr_att = len(numeric_subset.columns)
    
    # Indexzähler
    i=1
    
    # Berechnung der nötigen Subplotdimension über die Anzahl der zu plottenden Attribute und der User definierten Spaltenanzahl
    Subplot_Zeilen = (int(nr_att/Subplot_Spalten)) + (nr_att % Subplot_Spalten)
    
    # Anpassung der Bildgröße anhand der Anzahl der zu plottenden Attribute
    x = 10*Subplot_Spalten
    y = 8*Subplot_Zeilen
    fig = plt.figure(figsize=(x,y))
    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    

    for col in numeric_subset.columns:
        
        if col == 'Zielvariable':
            continue
    
        ax = fig.add_subplot(Subplot_Zeilen, Subplot_Spalten, i)
        ax.grid(True)
        ax.boxplot(data[col].dropna(), whis=3.0, patch_artist=True)
        ax.set_xlabel(col, fontsize = 20.0)
        ax.set_ylabel('Häufigkeit', fontsize = 20.0)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        i+=1
        
plot_numeric_box(Bankdaten, 2)
# Funktion welche alle numerischen Attribute transformiert und plottet

def plot_trafo_cols (data, list_trafo_cols, Subplot_Spalten):
    
    '''Der Funktion wird übergeben:
    - data: Datensatz der geplottet werden soll
    - list_trafo_cols: Eine Liste der transformierten Features
    - Subplot_Spalten: wieviele Plots nebeneinander geplottet werden sollen
    '''
     
    nr_att = len(list_trafo_cols)
    
    # Indexzähler
    i=1
    
    # Berechnung der nötigen Subplotdimension über die Anzahl der zu plottenden Attribute und der User definierten Spaltenanzahl
    Subplot_Zeilen = (int((2*nr_att)/Subplot_Spalten)) + ((2*nr_att) % Subplot_Spalten)
    
    # Anpassung der Bildgröße anhand der Anzahl der zu plottenden Attribute
    x = 10*Subplot_Spalten
    y = 8*Subplot_Zeilen
    fig = plt.figure(figsize=(x,y))
    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Plotten der Histogramme
    for col in list_trafo_cols:
    
        ax = fig.add_subplot(Subplot_Zeilen, Subplot_Spalten, i)
        ax.grid(True)
        ax.hist(data[col].dropna(), color='#0504aa',alpha=0.6, edgecolor = 'k')
        ax.set_xlabel(col, fontsize = 20.0)
        ax.set_ylabel('Anzahl Kunden', fontsize = 20.0)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        i+=1
    
    # Plotten der Boxplots
    for col in list_trafo_cols:
        ax = fig.add_subplot(Subplot_Zeilen, Subplot_Spalten, i)
        ax.grid(True)
        ax.boxplot(data[col].dropna(), whis=3.0, patch_artist=True)
        ax.set_xlabel(col, fontsize = 20.0)
        ax.set_ylabel('Anzahl Kunden', fontsize = 20.0)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        i+=1
        
# für den Trainingsdatensatz:
#Logarithmieren
Bankdaten['Alter_log'] = np.log10(Bankdaten['Alter'])
Bankdaten['Tag_log'] = np.log10(Bankdaten['Tag'])
Bankdaten['Anzahl der Ansprachen_log'] = np.log10(Bankdaten['Anzahl der Ansprachen'])

# Wurzel ziehen
Bankdaten['Alter_sqrt'] = np.sqrt(Bankdaten['Alter'])
Bankdaten['Tag_sqrt'] = np.sqrt(Bankdaten['Tag'])
Bankdaten['Anzahl der Ansprachen_sqrt'] = np.sqrt(Bankdaten['Anzahl der Ansprachen'])

# für den Testdatensatz:
#Logarithmieren
Bankdaten_klass['Alter_log'] = np.log10(Bankdaten_klass['Alter'])
Bankdaten_klass['Tag_log'] = np.log10(Bankdaten_klass['Tag'])
Bankdaten_klass['Anzahl der Ansprachen_log'] = np.log10(Bankdaten_klass['Anzahl der Ansprachen'])

# Wurzel ziehen
Bankdaten_klass['Alter_sqrt'] = np.sqrt(Bankdaten_klass['Alter'])
Bankdaten_klass['Tag_sqrt'] = np.sqrt(Bankdaten_klass['Tag'])
Bankdaten_klass['Anzahl der Ansprachen_sqrt'] = np.sqrt(Bankdaten_klass['Anzahl der Ansprachen'])


# Liste mit den transformierten Features
trafo_cols = ['Alter_log','Tag_log', 'Anzahl der Ansprachen_log', 'Alter_sqrt', 'Tag_sqrt', 'Anzahl der Ansprachen_sqrt']

# Funktion welche die Histogramme und Boxplots der transformierten Features plottet
plot_trafo_cols(Bankdaten, trafo_cols, 2)

# Funktion die alle numerischen Attribute in Abhängigkeit von der Zielvariablen plottet

def plot_numeric_dep_target(data,Subplot_Spalten):
    
    '''Der Funktion wird übergeben:
    - data: Datensatz der geplottet werden soll
    - Subplot_Spalten: wieviele Plots nebeneinander geplottet werden sollen
    '''
    
    # Auswählen aller numerischer Attribute
    numeric_subset = data.select_dtypes('number')

    # Anzahl numerischer Attribute
    nr_att = len(numeric_subset.columns)
    
    # Indexzähler
    i=1
    
    # Berechnung der nötigen Subplotdimension über die Anzahl der zu plottenden Attribute und der User definierten Spaltenanzahl
    Subplot_Zeilen = (int(nr_att/Subplot_Spalten)) + (nr_att % Subplot_Spalten)
    
    # Anpassung der Bildgröße anhand der Anzahl der zu plottenden Attribute
    x = 10*Subplot_Spalten
    y = 8*Subplot_Zeilen
    fig = plt.figure(figsize=(x,y))
    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Liste der Zielvariable 
    target_types = data.dropna(subset=['Zielvariable'])
    target_types = target_types['Zielvariable'].value_counts()
    
    target_types = list(target_types.index)
    
    # Geht durch alle numerischen Features
    for col in numeric_subset.columns:
        
        if col == 'Zielvariable':
            continue
    
        ax = fig.add_subplot(Subplot_Zeilen, Subplot_Spalten, i)
        
        # Für jedes numerische Feature wird ein Plot für die Zielvariable 1 und 0 erstellt, welche
        # in target_types stehen
        for target in target_types:
            
            subset = data[data['Zielvariable'] == target]
            
            # Density plot des Features
            sns.set_style("whitegrid")
            sns.kdeplot(subset[col].dropna(),label = target, shade = False, alpha = 0.8)
            
            
            ax.grid(True)
            ax.set_xlabel(col, fontsize = 20.0)
            ax.set_ylabel('Density', fontsize = 20.0)
            ax.tick_params(axis='both', which='major', labelsize=15)
            
        i+=1
    
plot_numeric_dep_target(Bankdaten,2)
# Funktion um alle kategorischen Attribute in Abhängigkeit von der Zielvariablen zu plotten

def plot_categ_dep_target (data, Subplot_Spalten):
    
    '''Der Funktion wird übergeben:
    - data: Datensatz der geplottet werden soll
    - Subplot_Spalten: wieviele Plots nebeneinander geplottet werden sollen
    '''
    
    # Auswählen aller kategorischen Attribute
    categ_subset = data.select_dtypes('object')
    
    # Anzahl der Attribute
    nr_att = len(categ_subset.columns)
    #print(nr_att)
    
    # Indexzähler
    i=1
    
    # Berechnung der nötigen Subplotdimension über die Anzahl der zu plottenden Attribute und der User definierten Spaltenanzahl
    Subplot_Zeilen = (int(nr_att/Subplot_Spalten)) + (nr_att % Subplot_Spalten)
    
    # Anpassung der Bildgröße anhand der Anzahl der zu plottenden Attribute
    x = 10*Subplot_Spalten
    y = 8*Subplot_Zeilen
    fig = plt.figure(figsize=(x,y))
    
    
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    

    for col in categ_subset:
        
        #print(col)
    
        ax = fig.add_subplot(Subplot_Zeilen, Subplot_Spalten, i)
        pd.crosstab(data[col].dropna(), data['Zielvariable']).plot(kind='bar',ax=ax)
        ax.grid(axis=True)
        ax.set_xlabel(col, fontsize= 20.0)
        ax.set_ylabel('Häufigkeit', fontsize= 20.0)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        i+=1
plot_categ_dep_target(Bankdaten, 2)
# Wandle kategorische Attribute in numerische um

# für den Trainingsdatensatz:

# Numerisches Subset
numeric_subset = Bankdaten.select_dtypes('number')

# Kategorisches Subset
categ_subset = Bankdaten.select_dtypes('object')

# Wende Dummies auf das kategorische Subset an, 
# d.h. es wird nun für jede Unterkategorie ein neues Feature erzeugt
categ_subset = pd.get_dummies(categ_subset)

# Füge die beiden Subsets wieder zusammen
features = pd.concat([numeric_subset, categ_subset], axis = 1)

#---------------------------------------------------------------------

# das gleiche nochmal für den Testdatensatz: 
numeric_subset_klass = Bankdaten_klass.select_dtypes('number')
categ_subset_klass = Bankdaten_klass.select_dtypes('object')
categ_subset_klass = pd.get_dummies(categ_subset_klass)
Datensatz_klass = pd.concat([numeric_subset_klass, categ_subset_klass], axis = 1)


# Aufsplitten und Ausbalancieren der Daten 
# Unterteile den ursprünglichen Trainingsdatensatz (jetzt features genannt) in einen Datensatz ohne Zielvariable
# und einen Datensatz der nur die Zielvariable enthält:
X = features.drop(columns = ['Zielvariable'])
y = features['Zielvariable']


# Initialisierung von Oversampling
over_sampling = SMOTE(random_state=0)

# Aufsplitten des ursprünglichen Trainingsdatensatzes in ein Trainings- und Testdatensatz.
# Der Testdatensatz wird 25% der ursprünglichen Menge enthalten, bzw. der Trainingsdatensatz 75%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
columns = X_train.columns

# Oversampling AUSSCHLIEßLICH auf dem neuen Trainingsdatensatz
over_sampling_data_X, over_sampling_data_y = over_sampling.fit_sample(X_train, y_train)
# Fügt die neu entstandenen Datensätze wieder als pandas DataFrame ein
over_sampling_data_X = pd.DataFrame(data = over_sampling_data_X, columns=columns )
over_sampling_data_y = pd.DataFrame(data = over_sampling_data_y, columns=['Zielvariable'])


# Überblick über die Aktion
print("Größe der Daten nach dem Oversampling: ",len(over_sampling_data_X))
print("")
print("Anzahl der 'Nein' Entscheidungen nach dem Oversampling:",
      len(over_sampling_data_y[over_sampling_data_y['Zielvariable']==0]))
print("")
print("Anzahl der 'Ja' Entscheidungen nach dem Oversampling:",
      len(over_sampling_data_y[over_sampling_data_y['Zielvariable']==1]))

# Lineare Korrelationen zur Zielvariablen 
correlations = features.corr()['Zielvariable'].dropna().sort_values(ascending=False)
print(correlations)
# Fügt die Korrelationen zur Zielvariablen in ein pandas DataFrame
correlations_df = pd.DataFrame({'Attributes':correlations.index, 'Korrelation_Zielvariable':correlations.values})

# Funktion, welche die Korrelation unter den Features berechnet und Features entfernt, die eine Korrelation
# über einem bestimmten Schwellwert aufweisen. Dasjenige Feature, welches die höhere Korrelation zur Zielvariablen
# aufweist, wird behalten

def del_collinear_features(X_train_data, y_train_data, threshold):
    '''
    Der Funktion wird übergeben:
    
    - X_train_data: Trainingsdatensatz ohne Zielvariable
    - y_train_data: Zielvariable des Trainigsdatensatzes
    - threshold: Schwellwert der Korrelation über welchen Features entfernt werden
    '''

    # Korrelation zur Zielvariablen soll nicht entfernt werden
    y = y_train_data
    
    # Bennene das Trainingsdatensatz aus Übersichtsgründen um
    features_list_ohne_Zielvariable = X_train_data
    
    # Berechnung der Korrelationsmatrix
    corr_matrix_ohne_Zielvariable = features_list_ohne_Zielvariable.corr()
    

    '''Berechnung wie oft durch die Matrix durchiteriert werden muss.
    len(corr_matrix.columns) gibt die Anzahl der Spalten in der Matrix an.
    Konvertiere Anzahl der Spalten in der Matrix in ein iterierbares Objekt durch range.
    ACHTUNG: range(0,i-1) bedeutet dass der Interpreter von Index 1 bis i geht.
    ABER: range(i) startet bei Index 0'''
    iters = range(len(corr_matrix_ohne_Zielvariable.columns)-1)
    
    drop_cols = []

    # Iteriere durch die [Anzahl] der Spalten der Matrix
    for i in iters:
        '''Man braucht nicht die ganze Korrelationsmatrix, da sie spiegelsymmetrisch an der Diagonalen ist.
        Das heißt dass die Diagonalmatrix reicht;
        Die Schleife muss nach folgendem Schema durch die Indices durchgehen (am Beispiel der oberen Dreiecksmatrix):
        0 1 2 3 4
          1 2 3 4
            2 3 4
              3 4
                4
        etc.'''
        # Iteriere durch Zeilen der Matrix
        for j in range(i):
            '''durchlaufe die obere Dreiecksmatrix: greife auf ein Wert der Matrix mit 
            iloc(Zeilenauswahl, Spaltenauswahl, geht bis spalten- und zeilenauswahl-1) zu: 
            z.B. für die 2. Spalte(mit i=1 -> j=(0,1)) -> holt das Matrixelement von Zeile 0 
            und von Spaltenindex 1 bis (2-1) 
            Der erste Eintrag bei Zeile 0 und Spalte 0 gehört zur Diagonalen und ist immer 1, den braucht man nicht.
            i=2(3. Spalte) -> j=(0,2) -> 1. Wert: Zeilenindex 0, Spaltenindex 2; 2. Wert: Zeilenindex 1, Spaltenindex 2
            usw.'''
            
            item = corr_matrix_ohne_Zielvariable.iloc[j:(j+1), i:(i+1)]
            col = item.columns
            row = item.index
            
            # Absoluter Wert der Korrelation, weil man auch hohe negative Korrelationen nicht im Datenset haben möchte
            val = abs(item.values)
            
            # Wenn der Korrelationswert über einem user-definierten Schwellwert liegt
            if val >= threshold:
                
                # Gibt die Spaltennamen und Zeilennamen des ersten Werts des herausgefilterten Matrixelements an 
                # (es gibt nur eins)
                # d.h. beide Attributnamen die miteinander korreliert sind
                correlation_col = col.values[0]
                correlation_row = row.values[0]
                
                
                # Zeile in dataframe der Korrelationen zur Zielvariablen für korreliertes Attribut:
                target_corr_1 = correlations_df.loc[correlations_df['Attributes'] == correlation_col].reset_index(drop=True)["Korrelation_Zielvariable"]
                target_corr_2 = correlations_df.loc[correlations_df['Attributes'] == correlation_row].reset_index(drop=True)["Korrelation_Zielvariable"]

                # Vergleicht welches Attribut die niedrigere bzw. höhere Korrelation zur Zielvariablen besitzt
                # und erstellt eine Liste mit Attributen die verworfen werden sollen
                
                if any(abs(target_corr_1) > abs(target_corr_2)):
                    # Falls die Korrelation zur Zielvariablen 1 größer ist als Korrelation 2,
                    # setze das Feature mit Korrelation 2 auf die Liste der zu löschenden Features
                    drop_cols.append(correlation_row)
                    
                elif any(abs(target_corr_1) < abs(target_corr_2)):
                    drop_cols.append(correlation_col)
                    
               

    # Liste mit Features, die verworfen werden sollen
    drops = set(drop_cols)
    print("Folgende Features werden entfernt:")
    print("")
    print(drops)
    print("")
    print("Anzahl der entfernten Features:", len(drops))
    print("")
    
    # Erstelle Trainingsdatensatz ohne die verworfenen Features
    features_list_ohne_Zielvariable = features_list_ohne_Zielvariable.drop(columns = drops)
    
    
    # Füge die Zielvariable wieder in den Trainingsdatensatz
    features_list_ohne_Zielvariable['Zielvariable'] = y
    features_list_mit_Zielvariable_Corrdel = features_list_ohne_Zielvariable
    print("Anzahl der Features im neuen Datenset:", features_list_mit_Zielvariable_Corrdel.shape[1])
             
    return features_list_mit_Zielvariable_Corrdel
# Berechnung und Entfernung der Features mit zu hoher Korrelation untereinander auf dem Trainingsdatensatz:
without_coll_features_df = del_collinear_features(over_sampling_data_X, over_sampling_data_y, 0.6)
# Entfernung der gleichen Features auf dem Testsatz und dem zu klassifizierenden Datensatz
# Features, welche entfernt werden
drop_cols = ['Kontaktart_Handy', 'Anzahl der Ansprachen', 'Anzahl der Ansprachen_sqrt', 'Familienstand_single', 'Ausfall Kredit_nein', 'Alter_sqrt', 'Schulabschluß_Abitur', 'Alter_log', 'Tag', 'Tag_sqrt']

# Testdatensatz
X_test = X_test.drop(columns = drop_cols)

# 
Datensatz_klass = Datensatz_klass.drop(columns = drop_cols)

# auf dem Trainingsdatensatz aus dem Split-----------------------------------------------------------------

# Varianz der Dauer 


Dauer_mean_training = without_coll_features_df["Dauer"].mean()
without_coll_features_df['Dauer_var'] = (without_coll_features_df['Dauer']- Dauer_mean_training)**2


# Dauer * Ergebnis letzte Kampagne_Erfolg -> Zeigt die Dauer der Gespräche aller Kunden, die auch beim 
# letzten Mal ein Produkt abgeschlossen haben
without_coll_features_df['Dauer_x_Ergebnis_letzte_Kampagne'] = without_coll_features_df['Dauer'] * without_coll_features_df['Ergebnis letzte Kampagne_Erfolg']

# auf dem Testdatensatz aus dem Split----------------------------------------------------------------------

Dauer_mean_test = X_test["Dauer"].mean()
X_test['Dauer_var'] = (X_test['Dauer']- Dauer_mean_test)**2
X_test['Dauer_x_Ergebnis_letzte_Kampagne'] = X_test['Dauer'] * X_test['Ergebnis letzte Kampagne_Erfolg']

# auf dem Datensatz, der klassifiziert werden soll---------------------------------------------------------

Dauer_mean_klass = Datensatz_klass["Dauer"].mean()
Datensatz_klass['Dauer_var'] = (Datensatz_klass['Dauer']- Dauer_mean_klass)**2
Datensatz_klass['Dauer_x_Ergebnis_letzte_Kampagne'] = Datensatz_klass['Dauer'] * Datensatz_klass['Ergebnis letzte Kampagne_Erfolg']



Y = without_coll_features_df['Zielvariable']
X = without_coll_features_df.drop(columns = ['Zielvariable'])


# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=6)
# Decision Trees
dec_tree = DecisionTreeClassifier(max_depth=None)
# Logistische Regression
log_reg = LogisticRegression(solver = 'liblinear')

# Scikit-Yellowbrick Visualisierung um die Kreuzvalidierungswerte von RFECV besser darstellen zu können:
# 4-fache Kreuzvalidierung mit dem RFE Feature Selection Algoritmus
# Es kann sein, dass Kaggle hier ein Problem hat und den Code nicht richtig ausführt und an dieser Stelle hängt
# Die Plots sind für den reinen Code nicht entscheidend und werden im Abschlussbericht auftauchen
# viz_rf = RFECV_yb(rf, step=1, cv=StratifiedKFold(2), scoring='roc_auc', njobs=-1)
# viz_rf.fit(X, Y)
# viz_rf.poof()

# viz_dec_tree = RFECV_yb(dec_tree, step=1, cv=StratifiedKFold(2), scoring='roc_auc', njobs=-1)
# viz_dec_tree.fit(X, Y)
# viz_dec_tree.poof()

# viz_log_reg = RFECV_yb(log_reg, step=1, cv=StratifiedKFold(2), scoring='roc_auc', njobs=-1)
# viz_log_reg.fit(X, Y)
# viz_log_reg.poof()




rf = RandomForestClassifier(n_estimators=100, max_depth=6)
rf.fit(X, Y) 
rf.score(X, Y)

# Tabelle mit Feature Importance
feature_importances = pd.DataFrame({'Features':X.columns, 'Feature Importance':rf.feature_importances_}).sort_values('Feature Importance', ascending=False)

# Die 24 wichtigsten Features
top_feature_importances = feature_importances.nlargest(24, "Feature Importance").round(3)

# Die 26 unwichtigsten Features
feature_importances_to_drop = feature_importances.nsmallest(26, "Feature Importance").round(3)

# Liste mit Features die entfern werden
feature_drop_list = feature_importances_to_drop['Features'].tolist()


# Neues Train Datenset in dem nur die 24 besten Features enthalten sind:
X_train_features_importances_selected = X.drop(columns = feature_drop_list)

print("Die 24 wichtigsten Features")
top_feature_importances
print("Die 26 unwichtigsten Features")
feature_importances_to_drop

# Scikit-Yellowbrick um die Kreuzvalidierungswerte besser darstellen zu können

rf = RandomForestClassifier(n_estimators=100, max_depth=6)
dec_tree = DecisionTreeClassifier(max_depth=10)
log_reg = LogisticRegression(solver = 'liblinear')

# auch hier wurde aus Testzwecken der Code von yellowbrick erstmal auskommentiert, da Kaggle hier eventuell ein Problem hat
# # Erstelle die yellowbrick RFECV Visualisierungen :
# # Random Forest
# viz_rf = RFECV_yb(rf, step=1, cv=StratifiedKFold(2), scoring='roc_auc', njobs=-1)
# viz_rf.fit(X_train_features_importances_selected, Y)
# viz_rf.poof()

# # Decision Trees
# viz_dec_tree = RFECV_yb(dec_tree, step=1, cv=StratifiedKFold(2), scoring='roc_auc', njobs=-1)
# viz_dec_tree.fit(X_train_features_importances_selected, Y)
# viz_dec_tree.poof()

# # Logistische Regression
# viz_log_reg = RFECV_yb(log_reg, step=1, cv=StratifiedKFold(2), scoring='roc_auc', njobs=-1)
# viz_log_reg.fit(X_train_features_importances_selected, Y)
# viz_log_reg.poof()

# Erstelle die inneren und äußeren Kreuzvalidierungen mit der 
# Feature Selection Methode RFE und einer 3-fachen eingenestetetn Kreuzvalidierung:

# Random Forest
rfecv_rf = RFECV(estimator=rf, step=1, cv=StratifiedKFold(3),scoring='roc_auc')
rfecv_rf.fit(X_train_features_importances_selected, Y)
# äußere Kreuzvalidierung:
nested_scores_rf = cross_val_score(rfecv_rf, X_train_features_importances_selected, Y, cv=StratifiedKFold(3))

# Decision Trees
rfecv_dec_tree = RFECV(estimator=dec_tree, step=1, cv=StratifiedKFold(3),scoring='roc_auc')
rfecv_dec_tree.fit(X_train_features_importances_selected, Y)
# äußere Kreuzvalidierung:
nested_scores_dec_tree = cross_val_score(rfecv_dec_tree, X_train_features_importances_selected, Y, cv=StratifiedKFold(3))

# Logistische Regression
rfecv_log_reg = RFECV(estimator=log_reg, step=1, cv=StratifiedKFold(3),scoring='roc_auc')
rfecv_log_reg.fit(X_train_features_importances_selected, Y)
# äußere Kreuzvalidierung:
nested_scores_log_reg = cross_val_score(rfecv_log_reg, X_train_features_importances_selected, Y, cv=StratifiedKFold(3))



# Angabe der optimalen Anzahl von Features sowie die Modellstabilität über die Resultate der inneren und
# äußeren Kreuzvalidierung:

# für den Random Forest
best_number_features_index_rf = [viz_rf.n_features_-1]
Mean_cross_val_score_rf = viz_rf.cv_scores_[best_number_features_index_rf, :].mean(axis=1).round(3)
Std_cross_val_score_rf = viz_rf.cv_scores_[best_number_features_index_rf, :].std(axis=1).round(3)
nested_scores_rf_mean = nested_scores_rf.mean().round(3)
nested_scores_rf_std = nested_scores_rf.std().round(3)

# für Decision Trees
best_number_features_index_dectree = [viz_dec_tree.n_features_-1]
Mean_cross_val_score_dectree = viz_dec_tree.cv_scores_[best_number_features_index_dectree, :].mean(axis=1).round(3)
Std_cross_val_score_dectree = viz_dec_tree.cv_scores_[best_number_features_index_dectree, :].std(axis=1).round(3)
nested_scores_dectree_mean = nested_scores_dec_tree.mean().round(3)
nested_scores_dectree_std = nested_scores_dec_tree.std().round(3)

# für die logistische Regression
best_number_features_index_logReg = [viz_log_reg.n_features_-1]
Mean_cross_val_score_logReg = viz_log_reg.cv_scores_[best_number_features_index_logReg, :].mean(axis=1).round(3)
Std_cross_val_score_logReg = viz_log_reg.cv_scores_[best_number_features_index_logReg, :].std(axis=1).round(3)
nested_scores_logreg_mean = nested_scores_log_reg.mean().round(3)
nested_scores_logreg_std = nested_scores_log_reg.std().round(3)



print("Optimale Anzahl an Features für Random Forest Classifier: ", viz_rf.n_features_)
print("Mittlere AUC für Random Forest: ", Mean_cross_val_score_rf, "+-", Std_cross_val_score_rf)
print("Mittlere AUC der Feature Selection für Random Forest: ", nested_scores_rf_mean, "+-", nested_scores_rf_std)
print("")

print("Optimale Anzahl an Features für Decision Trees: ", viz_dec_tree.n_features_)
print("Mittlere AUC für Decision Trees: ", Mean_cross_val_score_dectree, "+-", Std_cross_val_score_dectree)
print("Mittlere AUC der Feature Selection für Decision Trees: ", nested_scores_dectree_mean, "+-", nested_scores_dectree_std)
print("")

print("Optimale Anzahl an Features für Logistische Regression: ", viz_log_reg.n_features_)
print("Mittlere AUC für Logistische Regression: ", Mean_cross_val_score_logReg, "+-", Std_cross_val_score_logReg)
print("Mittlere AUC der Feature Selection für Logistische Regression: ", nested_scores_logreg_mean, "+-", nested_scores_logreg_std)


# Funktion zur Erstellung von Datensätzen für die jeweiligen maschinellen Lerner

def select_features(X_train_data, Y_train_data, X_test_data, X_class_data, classifier):
    
    '''
    Der Funktion wird übergeben:
    
    - X_train_data: Trainingsdatensatz ohne Zielvariable
    - Y_train_data: Zielvariable des Trainigsdatensatzes
    - X_test_data: Der Testdatensatz aus dem aufgesplitteten Trainingsdatensatz 
      (wird zur Evaluierung im weiteren Verlauf benötigt)
    - X_class_data: Der zu klassifizierende Datensatz
    - classifier: maschineller Lerner mit dem trainiert werden soll;
      "rf" für Random Forest, "dectree" für Entscheidungsbäume und "logreg" für Logistische 
    
    '''
    
    
    # Wähle die Features abhängig vom Classifier aus:
    # Random Forest
    if classifier == "rf":
        rfe_train = RFE(estimator=rf, n_features_to_select=n_features_rf)
        rfe_train.fit(X_train_data, Y_train_data)
        print("Random Forest ausgewählt")
    
    # Decision Trees
    elif classifier == "dectree":
        rfe_train = RFE(estimator=dec_tree, n_features_to_select=n_features_dec_tree)
        rfe_train.fit(X_train_data, Y_train_data)
        print("Decision Trees ausgewählt")
    
    # Logistische Regression
    elif classifier == "logreg":
        rfe_train = RFE(estimator=log_reg, n_features_to_select=n_features_log_reg)
        rfe_train.fit(X_train_data, Y_train_data)
        print("Logistische Regression ausgewählt")
        
    #rfe_train.fit(X_train_data, Y_train_data)
    
    # Boolsche Maske mit Auswahl der Feature Selection RFE
    support = rfe_train.support_
    # Namen der finalen Features:
    feature_names = np.array(X_train_data.columns)[support]
    

    return X_train_data[feature_names],X_test_data[feature_names], X_class_data[feature_names]


# Aus dem zu klassifizierenden Datensatz muss noch die Variable 'Stammnummer' aussortiert werden
# Diese wird nach der Klassifizierung wieder hinzugefügt
Stammnummer = Datensatz_klass["Stammnummer"]
Datensatz_klass = Datensatz_klass.drop(columns = 'Stammnummer')
# Wende Feature Selection auf Trainings- und Testdatensets für die jeweiligen Classifier an:

# Anzahl Features welche die Features Selection auswählen soll
n_features_rf = 24
n_features_dec_tree = 15
n_features_log_reg = 17

# Random Forest
X_train_after_FS_rf, X_test_after_FS_rf, Datensatz_klass_after_FS_rf = select_features(X_train_features_importances_selected, Y, X_test, Datensatz_klass, "rf")

# Decision Trees
X_train_after_FS_dectree, X_test_after_FS_dectree, Datensatz_klass_after_FS_dectree = select_features(X_train_features_importances_selected, Y, X_test, Datensatz_klass, "dectree")

# Logistische Regression
X_train_after_FS_logreg, X_test_after_FS_logreg, Datensatz_klass_after_FS_logreg = select_features(X_train_features_importances_selected, Y, X_test, Datensatz_klass, "logreg")
#print(X_train_after_FS_dectree)
# Grid Search in eingenesteter dreifachen Kreuzvalidierung
# Random Forest-------------------------------------------------------------------------------------------------------
rf = RandomForestClassifier()

rf_p_grid = {"n_estimators": [20, 50, 100, 200, 400, 600, 800],
          "max_depth": [3, 6, 8, 10, 30, 50, 70, 100, None],
          "max_features": [2, 3, 5, 7, 9, 11, 13, 17]
         }

clf_rf = GridSearchCV(estimator=rf, param_grid=rf_p_grid, cv=StratifiedKFold(3), scoring='roc_auc', n_jobs=-1)
clf_rf.fit(X_train_after_FS_rf, Y)
# äußere Kreuzvalidierung:
nested_scores_hypertuning_rf = cross_val_score(clf_rf, X_train_after_FS_rf, Y, cv=StratifiedKFold(3), n_jobs=-1)

nested_scores_hypertuning_mean_rf = nested_scores_hypertuning_rf.mean().round(3)
nested_scores_hypertuning_std_rf = nested_scores_hypertuning_rf.std().round(3)


print("Random Forest: Innere Kreuzvalidierung AUC und beste Parameter: ", clf_rf.best_score_, clf_rf.best_params_)
print("Random Forest: Äußere Kreuzvalidierung AUC (Stabilität der Methode): ", nested_scores_hypertuning_mean_rf, "+-", nested_scores_hypertuning_std_rf)
print("")


# Decision Trees----------------------------------------------------------------------------------------------------

dec_tree = DecisionTreeClassifier()

dec_tree_p_grid = {"criterion": ['gini', 'entropy'],
          "max_depth": [3, 6, 8, 10, 30, 50, 70, 100, None],
          "max_features": [2, 5, 7, 9, 10, 12, 15]
         }

clf_dec_tree = GridSearchCV(estimator=dec_tree, param_grid=dec_tree_p_grid, cv=StratifiedKFold(3), scoring='roc_auc', n_jobs=-1)
clf_dec_tree.fit(X_train_after_FS_dectree, Y)
# äußere Kreuzvalidierung:
nested_scores_hypertuning_dec_tree = cross_val_score(clf_dec_tree, X_train_after_FS_dectree, Y, cv=StratifiedKFold(3), n_jobs=-1)

nested_scores_hypertuning_mean_dec_tree = nested_scores_hypertuning_dec_tree.mean().round(3)
nested_scores_hypertuning_std_dec_tree = nested_scores_hypertuning_dec_tree.std().round(3)


print("Decision Trees: Innere Kreuzvalidierung AUC und beste Parameter: ", clf_dec_tree.best_score_, clf_dec_tree.best_params_)
print("Decision Trees: Äußere Kreuzvalidierung AUC (Stabilität der Methode): ", nested_scores_hypertuning_mean_dec_tree, "+-", nested_scores_hypertuning_std_dec_tree)
print("")

# Logistische Regression--------------------------------------------------------------------------------------------

log_reg = LogisticRegression()

log_reg_p_grid = {#"penalty": ['l1', 'l2'],
          "C": [0.001, 0.01, 0.1, 1, 10, 100],
          "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag']
         }

clf_log_reg = GridSearchCV(estimator=log_reg, param_grid=log_reg_p_grid, cv=StratifiedKFold(3), scoring='roc_auc', n_jobs=-1)
clf_log_reg.fit(X_train_after_FS_logreg, Y)
# äußere Kreuzvalidierung:
nested_scores_hypertuning_log_reg = cross_val_score(clf_log_reg, X_train_after_FS_logreg, Y, cv=StratifiedKFold(3), n_jobs=-1)

nested_scores_hypertuning_mean_log_reg = nested_scores_hypertuning_log_reg.mean().round(3)
nested_scores_hypertuning_std_log_reg = nested_scores_hypertuning_log_reg.std().round(3)


print("Logistische Regression: Innere Kreuzvalidierung AUC und beste Parameter: ", clf_log_reg.best_score_, clf_log_reg.best_params_)
print("Logistische Regression: Äußere Kreuzvalidierung AUC (Stabilität der Methode): ", nested_scores_hypertuning_mean_log_reg, "+-", nested_scores_hypertuning_std_log_reg)

# Speicherung der kreuzvalidierten Resultate aus der Grid Search in pandas Dataframe

results_rf = pd.DataFrame(clf_rf.cv_results_).sort_values('mean_test_score', ascending = False)
results_dec_tree = pd.DataFrame(clf_dec_tree.cv_results_).sort_values('mean_test_score', ascending = False)
results_log_reg = pd.DataFrame(clf_log_reg.cv_results_).sort_values('mean_test_score', ascending = False)



# Plots
#--------------------------------------------------------------------------------
# Die CV Ergebnisse werden jeweils nach den verschiedenen Hyperparametern der Modelle gruppiert. 
# Dann wird die Parameterkombination ausgesucht, welche
# den maximalen Score besitzen, und dieser wird dann zum Plotten benutzt. In diesem Fall kann man in jedem Plot sehr
# übersichtlich sehen, welche Einstellungen man lieber wählen möchte und bekommt gleichzeitg den maximalen Score 
# für diese Einstellung

# -------------------------------------------------------------------------------------------------------------------
# Random Forest Plots------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

# Suche für alle Parameter die Kombinationen der restlichen Parametern welche die maximale AUC aufweisen
plt.figure(figsize=(10,8))
# Abhängigkeit der AUC von n_estimators
best_clfs_nestimators = results_rf.groupby('param_n_estimators').apply(
    lambda g: g.nlargest(1, 'mean_test_score'))


best_clfs_nestimators.plot(x='param_n_estimators', y='mean_test_score', yerr='std_test_score',legend=False)

plt.xlabel("Anzahl Bäume")
plt.ylabel("Area under ROC Curve (AUC)")
plt.title("Random Forest n_estimator")

#plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------
# Abhängigkeit der AUC von max_features
best_clfs_max_features_rf = results_rf.groupby('param_max_features').apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs_max_features_rf.plot(x='param_max_features', y='mean_test_score', yerr='std_test_score',
               legend=False)

plt.xlabel("Anzahl zufällig gezogener Features")
plt.ylabel("Area under ROC Curve (AUC)")
plt.title("Random Forest max_features")


plt.show()

#----------------------------------------------------------------------------------
# Abhängigkeit der AUC von max_depth
best_clfs_max_depth_rf = results_rf.groupby('param_max_depth').apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs_max_depth_rf.plot(x='param_max_depth', y='mean_test_score', yerr='std_test_score',
               legend=False)

plt.xlabel("Maximale Tiefe der Bäume")
plt.ylabel("Area under ROC Curve (AUC)")
plt.title("Random Forest max_depth")

#plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------------
# Decision Trees-------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# Abhängigkeit der AUC von criterion
best_clfs_criterion = results_dec_tree.groupby('param_criterion').apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs_criterion.plot(x='param_criterion', y='mean_test_score', yerr='std_test_score',legend=False)


plt.xlabel("Kriterium")
plt.ylabel("Area under ROC Curve (AUC)")
plt.title("Entscheidungsbäume criterion")

#plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------
# Abhängigkeit der AUC von max_features
best_clfs_max_features_dec_tree = results_dec_tree.groupby('param_max_features').apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs_max_features_dec_tree.plot(x='param_max_features', y='mean_test_score', yerr='std_test_score',
               legend=False)

plt.xlabel("Anzahl zufällig gezogener Features")
plt.ylabel("Area under ROC Curve (AUC)")
plt.title("Entscheidungsbäume max_features")

#plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------
# Abhängigkeit der AUC von max_depth
best_clfs_max_depth_dec_tree = results_dec_tree.groupby('param_max_depth').apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs_max_depth_dec_tree.plot(x='param_max_depth', y='mean_test_score', yerr='std_test_score',
               legend=False)

plt.xlabel("Maximale Tiefe der Bäume")
plt.ylabel("Area under ROC Curve (AUC)")
plt.title("Entscheidungsbäume max_depth")

#plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------------------------------------
# Logistische Regression-----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# Abhängigkeit der AUC von solver
best_clfs_solver = results_log_reg.groupby('param_solver').apply(
    lambda g: g.nlargest(1, 'mean_test_score'))


best_clfs_solver.plot(x='param_solver', y='mean_test_score', yerr='std_test_score',legend=False)


plt.xlabel("Solvermethode")
plt.ylabel("Area under ROC Curve (AUC)")
plt.title("Logistische Regression solver")

#plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------
# Abhängigkeit der AUC von C
best_clfs_C = results_log_reg.groupby('param_C').apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs_C.plot(x='param_C', y='mean_test_score', yerr='std_test_score',
               legend=False)

plt.xlabel("Inverse Regularisierungsstärke C")
plt.ylabel("Area under ROC Curve (AUC)")
plt.title("Logistische Regression C")

#plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------

# Ausgabe der Parametereinstellungen für den maximalen Score für einen manuell eingegebenen Parameter.
# Für den Fall der User möchte wissen, für z.B. n_estimators=100, welche Parameterwerte die maximale AUC erzeugt haben

# Funktion zur Ausgabe für den Random Forest
def Ausgabe_Parametereinstellungen_RF(parameter, wert):
    
    '''
    Funktion nimmt die Parameternamen max_depth, n_estimators und max_features an. Der Wert entspricht dem 
    Wert des Parameters, für den die Einstellungen der anderen Parameter für den maximalen Score gesucht werden 
    sollen
    '''

    if parameter == "max_depth":
        Best_score_n_estimators = best_clfs_max_depth_rf.loc[best_clfs_max_depth_rf['param_max_depth'] == wert].reset_index(drop=True)["param_n_estimators"]
        Best_score_max_features = best_clfs_max_depth_rf.loc[best_clfs_max_depth_rf['param_max_depth'] == wert].reset_index(drop=True)["param_max_features"]
        
        print("Für den Parameter max_depth=",wert)
        print("n_estimators für maximalen Score: ",Best_score_n_estimators)
        print("max_features für maximalen Score: ",Best_score_max_features)
        
    if parameter == "n_estimators":
        Best_score_max_depth = best_clfs_nestimators.loc[best_clfs_nestimators['param_n_estimators'] == wert].reset_index(drop=True)["param_max_depth"]
        Best_score_max_features = best_clfs_nestimators.loc[best_clfs_nestimators['param_n_estimators'] == wert].reset_index(drop=True)["param_max_features"]
        
        print("Für den Parameter n_estimators=",wert)
        print("max_depth für maximalen Score: ",Best_score_max_depth)
        print("max_features für maximalen Score: ",Best_score_max_features)
        
    if parameter == "max_features":
        Best_score_n_estimators = best_clfs_max_features_rf.loc[best_clfs_max_features_rf['param_max_features'] == wert].reset_index(drop=True)["param_n_estimators"]
        Best_score_max_depth = best_clfs_max_features_rf.loc[best_clfs_max_features_rf['param_max_features'] == wert].reset_index(drop=True)["param_max_depth"]
        
        print("Für den Parameter max_features=",wert)
        print("n_estimators für maximalen Score: ",Best_score_n_estimators)
        print("max_depth für maximalen Score: ",Best_score_max_depth)
#-----------------------------------------------------------------------------------------------

# Funktion zur Ausgabe für Entscheidungsbäume
def Ausgabe_Parametereinstellungen_DecTree(parameter, wert):
    
    '''
    Funktion nimmt die Parameternamen max_depth, n_estimators und max_features an. Der Wert entspricht dem 
    Wert des Parameters, für den die Einstellungen der anderen Parameter für den maximalen Score gesucht werden 
    sollen
    '''

    if parameter == "max_depth":
        Best_score_criterion = best_clfs_max_depth_dec_tree.loc[best_clfs_max_depth_dec_tree['param_max_depth'] == wert].reset_index(drop=True)["param_criterion"]
        Best_score_max_features = best_clfs_max_depth_dec_tree.loc[best_clfs_max_depth_dec_tree['param_max_depth'] == wert].reset_index(drop=True)["param_max_features"]
        
        print("Für den Parameter max_depth=",wert)
        print("Criterion für maximalen Score: ",Best_score_criterion)
        print("max_features für maximalen Score: ",Best_score_max_features)
        
    if parameter == "criterion":
        Best_score_max_depth = best_clfs_criterion.loc[best_clfs_criterion['param_criterion'] == wert].reset_index(drop=True)["param_max_depth"]
        Best_score_max_features = best_clfs_criterion.loc[best_clfs_criterion['param_criterion'] == wert].reset_index(drop=True)["param_max_features"]
        
        print("Für den Parameter Criterion=",wert)
        print("max_depth für maximalen Score: ",Best_score_max_depth)
        print("max_features für maximalen Score: ",Best_score_max_features)
        
    if parameter == "max_features":
        Best_score_criterion = best_clfs_max_features_dec_tree.loc[best_clfs_max_features_dec_tree['param_max_features'] == wert].reset_index(drop=True)["param_criterion"]
        Best_score_max_depth = best_clfs_max_features_dec_tree.loc[best_clfs_max_features_dec_tree['param_max_features'] == wert].reset_index(drop=True)["param_max_depth"]
        
        print("Für den Parameter max_features=",wert)
        print("Criterion für maximalen Score: ",Best_score_criterion)
        print("max_depth für maximalen Score: ",Best_score_max_depth)
#-----------------------------------------------------------------------------------------------

# Funktion zur Ausgabe für die Logistische Regression
def Ausgabe_Parametereinstellungen_LogReg(parameter, wert):
    
    '''
    Funktion nimmt die Parameternamen max_depth, n_estimators und max_features an. Der Wert entspricht dem 
    Wert des Parameters, für den die Einstellungen der anderen Parameter für den maximalen Score gesucht werden 
    sollen
    '''

    if parameter == "solver":
        
        Best_score_C = best_clfs_solver.loc[best_clfs_solver['param_solver'] == wert].reset_index(drop=True)["param_C"]
        
        print("Für den Parameter max_depth=",wert)
        print("C für maximalen Score: ",Best_score_C)
        
        
    if parameter == "C":
        Best_score_solver = best_clfs_C.loc[best_clfs_C['param_C'] == wert].reset_index(drop=True)["param_solver"]
        
        print("Für den Parameter C=",wert)
        print("Solver für maximalen Score: ",Best_score_solver)
        
        
    

Ausgabe_Parametereinstellungen_RF("max_depth", 30)
print("")
Ausgabe_Parametereinstellungen_RF("n_estimators", 400)
print("")
Ausgabe_Parametereinstellungen_RF("max_features", 5)
print("")
Ausgabe_Parametereinstellungen_DecTree("max_depth", 8)
print("")
Ausgabe_Parametereinstellungen_DecTree("criterion", 'gini')
print("")
Ausgabe_Parametereinstellungen_LogReg("C", 10)
# Funktion zum plotten der ROC Kurve
def plot_roc(name, probs):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    #plt.clf()
    plt.plot(fpr, tpr, label=str(name)+'(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Kurven für verschiedene Modelle")
    plt.legend(loc="lower right")
#     plt.show()


#Dictionaries mit verschiedenen Einstellungen für jeden maschinellen Lerner

dictionary_rf = {"RF_best": {
                                 'n_estimators': 800, 
                                 'max_features': 2, 
                                 'max_depth': 30
                             },
                 "RF_estimators": {
                                 'n_estimators': 400,
                                 'max_features': 2,
                                 'max_depth': 50
                             },
                 "RF_features": {
                                 'n_estimators': 800,
                                 'max_features': 5,
                                 'max_depth': None
                             }
}

dictionary_dec_tree = {"DT_best": {
                                 'criterion': 'entropy', 
                                 'max_features': 7, 
                                 'max_depth': 8
                             },
                       "DT_criterion": {
                                 'criterion': 'gini', 
                                 'max_features': 5, 
                                 'max_depth': 8
                             }
}

dictionary_log_reg = {"LR_best": {
                                 'solver': 'newton-cg', 
                                 'C': 100     
                             },
                      "LR_C": {
                                 'solver': 'newton-cg', 
                                 'C': 10     
                             }
}

# Anlegen einer Übersichtstabelle über die Performance und die Parametereinstellungen
Overview = pd.DataFrame(columns=['Modellname','AUC','Precision','Recall','Accuracy' , 'n_estimators', 'max_features', 'max_depth', 'criterion', 'C', 'Solver'])


# Schleife durch das jeweilige Dictionary
for key in dictionary_rf:
    
    # Zählindex zum einfügen in die Tabelle. i gibt die Zeile an in die eingefügt wird
    i = len(Overview)
    
    # Training und Anwendung des Modells
    rf_final_model = RandomForestClassifier().set_params(**dictionary_rf[key])
    rf_final_model.fit(X_train_after_FS_rf, Y)
    rf_final_pred = rf_final_model.predict(X_test_after_FS_rf)


    # Konfidenzwerte der Baumentscheidungen
    probs = rf_final_model.predict_proba(X_test_after_FS_rf)
    
   
    # Berechnung AUC und andere Performancewerte
    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    roc_auc = auc(fpr, tpr)
    precision = precision_score(y_test, rf_final_pred)
    recall = recall_score(y_test, rf_final_pred)
    accuracy = accuracy_score(y_test, rf_final_pred)
    
    plot_roc(str(key), probs[:,1])
    

    
    # Fügt Performanceergebnisse in dataframe ein
    Overview.loc[i, 'Modellname'] = key
    Overview.loc[i,'AUC'] = roc_auc.round(3)
    Overview.loc[i, 'n_estimators'] = dictionary_rf[key]['n_estimators']
    Overview.loc[i, 'max_features'] = dictionary_rf[key]['max_features']
    Overview.loc[i, 'max_depth'] = dictionary_rf[key]['max_depth']
    Overview.loc[i, 'Precision'] = precision.round(3)
    Overview.loc[i, 'Recall'] = recall.round(3)
    Overview.loc[i, 'Accuracy'] = accuracy.round(3)


#     # Plotten der Konfidenzverteilungen für die tatsächlichen Zustimmungen zum Produkt
#     y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
#     hist_tabelle_df = pd.DataFrame(probs)
#     hist_tabelle_df['y_test'] = y_test_df
#     y_actual_true = hist_tabelle_df[hist_tabelle_df['y_test']==1]

#     plt.figure()
#     y_actual_true.iloc[:,1].hist()
#     plt.xlabel('Mittlere Konfidenzwerte')
#     plt.ylabel('Anzahl')
#     plt.title("Konfidenzverteilung die tatsächlichen Zustimmungen zum Produkt für " + str(key))

#--------------------------------------------------------------------------------------------------------------------

for key in dictionary_dec_tree:
    
    i = len(Overview)
    
    # rf_final_model = RandomForestClassifier(n_estimators=800, max_features=2, max_depth=30)
    dec_tree_model = DecisionTreeClassifier().set_params(**dictionary_dec_tree[key])
    dec_tree_model.fit(X_train_after_FS_dectree, Y)
    dec_tree_pred = dec_tree_model.predict(X_test_after_FS_dectree)


    # Konfidenzwerte der Baumentscheidungen
    probs = dec_tree_model.predict_proba(X_test_after_FS_dectree)
    
   
    # Berechnung AUC
    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    roc_auc = auc(fpr, tpr)
    precision = precision_score(y_test, dec_tree_pred)
    recall = recall_score(y_test, dec_tree_pred)
    accuracy = accuracy_score(y_test, dec_tree_pred)
    
    plot_roc(str(key), probs[:,1])
    
    
    # Fügt Performanceergebnisse in dataframe ein
    Overview.loc[i, 'Modellname'] = key
    Overview.loc[i,'AUC'] = roc_auc.round(3)
    Overview.loc[i, 'criterion'] = dictionary_dec_tree[key]['criterion']
    Overview.loc[i, 'max_features'] = dictionary_dec_tree[key]['max_features']
    Overview.loc[i, 'max_depth'] = dictionary_dec_tree[key]['max_depth']
    Overview.loc[i, 'Precision'] = precision.round(3)
    Overview.loc[i, 'Recall'] = recall.round(3)
    Overview.loc[i, 'Accuracy'] = accuracy.round(3)


#     # Plotten der Konfidenzverteilungen für die tatsächlichen Zustimmungen zum Produkt
#     y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
#     hist_tabelle_df = pd.DataFrame(probs)
#     hist_tabelle_df['y_test'] = y_test_df
#     y_actual_true = hist_tabelle_df[hist_tabelle_df['y_test']==1]

#     plt.figure()
#     y_actual_true.iloc[:,1].hist()
#     plt.xlabel('Mittlere Konfidenzwerte')
#     plt.ylabel('Anzahl')
#     plt.title("Konfidenzverteilung die tatsächlichen Zustimmungen zum Produkt für " + str(key))
    
    
#-------------------------------------------------------------------------------------------------------------------

for key in dictionary_log_reg:
    

    i = len(Overview)
    
    # rf_final_model = RandomForestClassifier(n_estimators=800, max_features=2, max_depth=30)
    log_reg_model = LogisticRegression().set_params(**dictionary_log_reg[key])
    log_reg_model.fit(X_train_after_FS_logreg, Y)
    log_reg_pred = log_reg_model.predict(X_test_after_FS_logreg)


    # Konfidenzwerte der Baumentscheidungen
    probs = log_reg_model.predict_proba(X_test_after_FS_logreg)
    
   
    # Berechnung AUC
    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    roc_auc = auc(fpr, tpr)
    precision = precision_score(y_test, log_reg_pred)
    recall = recall_score(y_test, log_reg_pred)
    accuracy = accuracy_score(y_test, log_reg_pred)
    
    plot_roc(str(key), probs[:,1])
    
    # Fügt Performanceergebnisse in dataframe ein
    Overview.loc[i, 'Modellname'] = key
    Overview.loc[i,'AUC'] = roc_auc.round(3)
    Overview.loc[i, 'C'] = dictionary_log_reg[key]['C']
    Overview.loc[i, 'Solver'] = dictionary_log_reg[key]['solver']
    Overview.loc[i, 'Precision'] = precision.round(3)
    Overview.loc[i, 'Recall'] = recall.round(3)
    Overview.loc[i, 'Accuracy'] = accuracy.round(3)


#     # Plotten der Konfidenzverteilungen für die tatsächlichen Zustimmungen zum Produkt
#     y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
#     hist_tabelle_df = pd.DataFrame(probs)
#     hist_tabelle_df['y_test'] = y_test_df
#     y_actual_true = hist_tabelle_df[hist_tabelle_df['y_test']==1]

#     plt.figure()
#     y_actual_true.iloc[:,1].hist()
#     plt.xlabel('Mittlere Konfidenzwerte')
#     plt.ylabel('Anzahl')
#     plt.title("Konfidenzverteilung die tatsächlichen Zustimmungen zum Produkt für " + str(key))


#     # Konfusionsmatrix
#     pd.crosstab(y_test, rf_final_pred, rownames=["Actual"], colnames=["Predicted"])
plt.show()

Overview = Overview.replace(np.nan, '/', regex=True)    
 




Overview.sort_values('AUC', ascending=False)
# Training und Anwendung des Modells
rf_model_features = RandomForestClassifier(n_estimators=800, max_features=5, max_depth=None)
rf_model_features.fit(X_train_after_FS_rf, Y)
rf_model_features_pred = rf_model_features.predict_proba(Datensatz_klass_after_FS_rf)


Data_class_df = pd.DataFrame()
Data_class_df["ID"] = Stammnummer
Data_class_df["Expected"] = pd.DataFrame(rf_model_features_pred[:,1])





# Schreibt das Dataframe in eine csv Datei
Data_class_df.to_csv('Loesung.csv', sep=',', index=False)