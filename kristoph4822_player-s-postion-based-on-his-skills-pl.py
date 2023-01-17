import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import tree

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import NearestCentroid

from sklearn.naive_bayes import GaussianNB
# Wczytanie i transformacja danych

df = pd.read_csv('../input/fifa19/data.csv', index_col = 0)

df.head()
df.columns
# Pozbycie się zbędnych kolumn

df = df.drop(columns=['ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag', 'Overall', 'Potential', 'Club', 'Club Logo'])

df = df.drop(columns=['Value', 'Wage', 'Special', 'International Reputation', 'Work Rate', 'Body Type', 'Real Face'])

df = df.drop(columns=['Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until', 'LS', 'ST', 'RS', 'LW'])

df = df.drop(columns=['LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM'])

df = df.drop(columns=['CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause'])
df.isnull().sum() # liczba wartości NaN dla każdej kolumny
df.dropna(inplace = True)
df.describe().T # opis danych numerycznych
norm_list = ["Weak Foot", "Skill Moves"]



for n in norm_list:

    df[n] = 100 * df[n]/df[n].max() #normalizacja 0-100



df[norm_list].head()
df.drop(columns = ["Position"]).describe(include=['object']).T
def ft_to_inch(ft): # funkcja do zamiany jednostki [stopy+cale] na [cale]

    x = [int(i) for i in ft.split("'")] 

    x[1] += x[0] * 12

    return x[1]



df["Height"] = df["Height"].map(lambda x: ft_to_inch(x))

df["Height"] = round(100*df["Height"]/df["Height"].max(), 0) # normalizacja 0-100

df["Height"].describe()
df["Weight"] = df["Weight"].map(lambda x:  int(round(int(x[:-3]), 0)))

df["Weight"] = round(100*df["Weight"]/df["Weight"].max(), 0)

df["Weight"].describe()
df['isLeft'] = df["Preferred Foot"].map(lambda x: 0 if x == "Right" else 1)

df.drop(columns=["Preferred Foot"], inplace = True)

df['isLeft'].describe()
len(df.Position.unique())
df.groupby("Position")["Position"].count().sort_values(ascending = False)
df.groupby("Position")["Position"].count().sort_values(ascending = False).plot(kind = 'bar', figsize=(20,10))
df = df[(df.Position != 'RAM') & (df.Position != 'LAM') & (df.Position != 'RF') & (df.Position != 'LF')]
defe = ['RWB', 'RCB', 'RB', 'LWB', 'LCB', 'LB', 'CB'] # obrońcy

fwd = ['ST', 'RW', 'RS', 'LW', 'LS', 'CF'] # napastnicy

mid = ['RM', 'RDM', 'RCM', 'LDM', 'LCM', 'CM', 'CDM', 'CAM', 'LM'] # pomocnicy

gk = ['GK'] # bramkarz

all_pos = fwd + mid + defe + gk # wszystkie pozycje
# wykres dla preferowanej nogi

dict1 = {}

for p in all_pos:

    dict1[p] = df[df.Position == p].groupby('isLeft')['isLeft'].count()[1] / df[df.Position == p].count()["isLeft"]



lfoot = pd.DataFrame.from_dict(dict1, orient='index').sort_values(by = 0, ascending = False)

lfoot.plot.bar(legend = False)
rl = {}

rl['R'] = df[df.Position.str.startswith('R')].groupby('isLeft')['isLeft'].count()[1] / df[df.Position.str.startswith('R')].count()["Position"]

rl['L'] = df[df.Position.str.startswith('L')].groupby('isLeft')['isLeft'].count()[1] / df[df.Position.str.startswith('L')].count()["Position"]

plt.bar(range(len(rl)), list(rl.values()), align='center')

plt.xticks(range(len(rl)), list(rl.keys()))
isLeft = df["isLeft"]

df.drop(columns = ["isLeft"], inplace = True)
for p in [fwd, mid, defe, gk]: 

    dff = df.loc[df['Position'].isin(p)].groupby('Position', as_index=True).mean()

    plt.figure(figsize=(20,len(p)/2),dpi = 80)

    sns.heatmap(dff, cmap="bone", annot=True, vmin = 10, vmax = 90)
plt.figure(figsize=(20,16),dpi = 80)

sns.heatmap(df.corr(),annot = df.corr()) #mapa ciepła z korelacją
"Liczba atrybutów przed scalaniem: " + str(len(list(df.drop(columns=['Position']).select_dtypes(exclude=["object"]).columns)))
thr = 0.9 #próg współczynnika korelacji, atrybuty skorelowane ponad próg zostaną scalone



dff = df.select_dtypes(['number']).copy()

corr = dff.corr()

continueLoop = True

attr_id = 0;

while(continueLoop):

    continueLoop = False

    for i in range(0,len(corr)):

        attrs = [i]

        for j in range(0,len(corr)):

            if(i != j and corr.iloc[i,j] > thr):

                attrs.append(j)

        if (len(attrs) > 1):

            attr_name = 'attr_' + str(attr_id)

            attr_names = []

            for k in attrs:

                attr_names.append(dff.columns[k])

            print('Combining ' + str(attr_names) + ' into ' + attr_name)

            attr_id = attr_id + 1

            dff[attr_name] = dff.iloc[:,attrs].mean(axis=1)

            dff = dff.drop(dff.columns[attrs], axis=1)

            continueLoop = True

            corr = dff.corr()

            break;     

plt.figure(figsize=(14,10),dpi = 80)

sns.heatmap(dff.corr(),annot = dff.corr())

"Liczba atrybutów po scalaniu: " + str(len(dff.columns))
# SelectKBest

y = df['Position'] #target

X = dff #nasz zbiór



n = len(X.columns) - 2 #K - ile najlepszych atrybutów uwzględniamy w modelu



bestfeatures = SelectKBest(score_func=chi2, k=n)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

featureScores.nlargest(n,'Score').set_index('Specs').plot(kind='bar', figsize=(20,10))



X = X[featureScores.nlargest(n,'Score').iloc[:,0].values].join(isLeft)
#parametry modeli sprawdzane przez GridSearch

dt = {KNeighborsClassifier() : {'n_neighbors': np.arange(57, 63, 2)},

      NearestCentroid() : {'metric': ['euclidean', 'manhattan']},

      GaussianNB() : {},

      DecisionTreeClassifier() : {'max_depth': np.arange(5, 12)}}



best_estimator = None #zmienna best_estimator zachowa najlepszy model (z 4 wybranych) z najlepszymi parametrami 

tmp = 0



for key in dt:

    m = GridSearchCV(key, dt[key]) #przekazujemy model wraz z listą parametrów do GridSearch

    m.fit(X, y)

    

    if m.best_score_ > tmp:

        best_estimator = m.best_estimator_

        tmp = m.best_score_

    

    print ('Model {}'.format(type(key).__name__))

    print ('Najlepszy wynik : {}'.format(m.best_score_))

    print ('Najlepsze parametry: {}'.format(m.best_params_))

    print('\n\n')
model = best_estimator



#podział danych na treningowe i testowe

train_x, val_x, train_y, val_y = train_test_split(X, y, random_state = 0, test_size = 0.2) 



model.fit(train_x, train_y)

wynik_test = model.predict(val_x)



print("Wykres kolumnowy - do jakiej pozycji model przypisał piłkarzy względem ich prawdziwej pozycji:")

mp_test = pd.crosstab(val_y, wynik_test)



f = plt.figure(figsize=(20, 20))



ax = []

rate_df = pd.Series({})



j = 0

    

for i in range(0, len(mp_test.index)):

    r = mp_test.iloc[i]

    r = r[r != 0].sort_values(ascending=False)

    cname = mp_test.index[i]

    

    f.tight_layout()

    

    ax.append(f.add_subplot(6, 4,i+1))

    ax[i].title.set_text(cname)

    ax[i].bar(r.index, r, width = 0.8)

    

    if cname in mp_test.columns:

        rate_df[cname] = mp_test.iloc[i][j] / mp_test.iloc[i].sum()

        j += 1

    else:

        rate_df[cname] = 0

    

plt.show()



print("Procent poprawnego dopasowania piłkarzy na konkretnej pozycji:")

print(rate_df.sort_values(ascending=False))

        

rate_df.sort_values(ascending=False).plot(kind = 'bar', figsize=(20,10))
#Podział na GK, DEF, MID i FWD

results = pd.DataFrame({})

results['real'] = val_y

results['pred'] = wynik_test

results['pred'] = results['pred'].map(lambda x: 'GK' if x in gk else ('DF' if x in defe else ('MF' if x in mid else 'FW')))

results['real'] = results['real'].map(lambda x: 'GK' if x in gk else ('DF' if x in defe else ('MF' if x in mid else 'FW')))

a = (len(results[results['pred'] == results['real']]) / len(results)) * 100

print('Skuteczność modelu wynosi', a, '%')
cross_val_score(estimator=best_estimator, X=X, y=y, cv=5).mean()