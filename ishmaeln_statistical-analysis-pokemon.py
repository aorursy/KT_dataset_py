import pandas as pd # To read csv

import re # Regex

import seaborn as sb # Statistical data visualization

import matplotlib.pyplot as plt
frame = pd.read_csv('../input/Pokemon.csv') # Read csv into dataframe 

frame.head(1)
frame.columns = ['#','Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed', 'Generation', 'Legendary']

frame = frame.set_index('#')
frame.head(10)
frame.tail(10)
frame.Name = frame.Name.apply(lambda x: re.sub(r'(.+)(Mega.+)',r'\2',x))

frame.Name = frame.Name.apply(lambda x: re.sub(r'(.+)(Primal.+)',r'\2',x))

frame.Name = frame.Name.apply(lambda x: re.sub(r'(HoopaHoopa)(.+)','Hoopa'+r'\2',x))

frame.head(10)
frame.tail(10)
All = frame.loc[(frame['Name'].str.contains('Mega ')==False) & (frame['Name'].str.contains('Primal')==False)] # filter all pokemon without Mega and Primal in name.

All.head(10)
Poke = All.loc[(All['Legendary']==False)]

PokeL = All.loc[(All['Legendary']==True)]
#Pie chart of legendary split

LSplit = [Poke['Name'].count(),PokeL['Name'].count()]

LegendPie = plt.pie(LSplit,labels= ['Not Legendary', 'Legendary'], autopct ='%1.1f%%', shadow = True, startangle = 90,explode=(0, 0.1))

plt.title('Legendary Split',fontsize = 12)

fig = plt.gcf()

fig.set_size_inches(11.7,8.27)

plt.savefig("LegendPie.png")
Gen1 = Poke.loc[Poke['Generation'] == 1]

Gen2 = Poke.loc[Poke['Generation'] == 2]

Gen3 = Poke.loc[Poke['Generation'] == 3]

Gen4 = Poke.loc[Poke['Generation'] == 4]

Gen5 = Poke.loc[Poke['Generation'] == 5]

Gen6 = Poke.loc[Poke['Generation'] == 6]
#Pie chart of type 1 vs type 2

TySplit = [Poke['Type1'].count() - Poke['Type2'].count(),Poke['Type2'].count()]

TypePie = plt.pie(TySplit,labels= ['Primary only', 'Pri and Sec'], autopct ='%1.1f%%', shadow = True, startangle = 90,explode=(0, 0.1))

plt.title('Single Type vs Dual Type Pokemon',fontsize = 12)

fig = plt.gcf()

fig.set_size_inches(11.7,8.27)

plt.savefig("TypePie.png")
type1_colours= ['#6890F0',  # Water

                    '#A8A878',  # Normal

                    '#A8B820',  # Bug

                    '#78C850',  # Grass

                    '#F08030',  # Fire

                    '#F85888',  # Psychic

                    '#F8D030',  # Electric

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#A040A0',  # Poison

                    '#E0C068',  # Ground

                    '#705848',  # Dark

                    '#C03028',  # Fighting

                    '#98D8D8',  # Ice

                    '#B8B8D0',  # Steel

                    '#7038F8',  # Dragon

                    '#EE99AC',  # Fairy

                    '#A890F0',  # Flying

                   ]



type2_colours= ['#78C850',  #None

                '#A890F0',  # Flying

                '#A040A0',  # Poison

                '#E0C068',  # Ground

                '#78C850',  # Grass

                '#F85888',  # Psychic

                '#B8B8D0',  # Steel

                '#C03028',  # Fighting

                '#EE99AC',  # Fairy

                '#705848',  # Dark

                '#B8A038',  # Rock

                '#6890F0',  # Water

                '#705898',  # Ghost

                '#7038F8',  # Dragon

                '#98D8D8',  # Ice

                '#F08030',  # Fire

                '#F8D030',  # Electric

                '#A8A878',  # Normal

                '#A8B820',  # Bug  

                ]

Type1 = pd.value_counts(Poke['Type1'])

sb.set()

dims = (11.7,8.27) #A4 dimensions

fig, ax = plt.subplots(figsize=dims)

BarT = sb.barplot(x=Type1.index,y=Type1,data=Poke, palette = type1_colours, ax=ax)

BarT.set_xticklabels(BarT.get_xticklabels(), rotation = 75, fontsize = 12)

BarT.set(xlabel ='Pokemon Primary Types', ylabel='Freq')

BarT.set_title('Dist. of Primary Pokemon Types')

FigBar = BarT.get_figure()

FigBar.savefig("BarPlot_PrimaryType.png")
for row in Poke.loc[Poke.Type2.isnull(), 'Type2'].index:

    Poke.at[row, 'Type2'] = 'None'

    

Type2 = pd.value_counts(Poke['Type2'])

sb.set()

dims = (11.7,8.27) #A4 dimensions

fig, ax = plt.subplots(figsize=dims)

BarT = sb.barplot(x=Type2.index,y=Type2,data=Poke, palette = type2_colours, ax=ax)

BarT.set_xticklabels(BarT.get_xticklabels(), rotation = 75, fontsize = 12)

BarT.set(xlabel ='Pokemon Secondary Types', ylabel='Freq')

BarT.set_title('Dist. of Secondary Pokemon Types')

FigBar = BarT.get_figure()

FigBar.savefig("BarPlot_SecondaryType.png")
Corr = Poke[['Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

sb.set()

dims = (11.7,8.27) #A4 dimensions

fig, ax = plt.subplots(figsize=dims)

CorrelationMap = sb.heatmap(Corr.corr(),annot = True, ax = ax)

CorrelationMap.set(title = 'HeatMap to show Correlation between Base Stats')

FigMap = CorrelationMap.get_figure()

FigMap.savefig("HeatMapCorr.png")
sb.set()

AttvTot = sb.lmplot(x='Sp.Def', y='Total',data=Poke,

                   fit_reg = False, size = 9, aspect = 1.2) #Can Add Hue to distinguish types

plt.ylim(150,700)

plt.xlim(0,175)

plt.title('Sp.Def vs. Total',fontsize = 25)

plt.xlabel('Sp.Def',fontsize = 18)

plt.ylabel('Total',fontsize = 18)

AttvTot.savefig("SP_SpDefvsTot.png")
dims = (11.7,8.27) #A4 dimensions

fig, ax = plt.subplots(figsize=dims)

SpDhist = sb.distplot(Poke['Sp.Def'],color='y',hist=True,ax=ax)

SpAhist = sb.distplot(Poke['Sp.Atk'],color = 'b', hist = True,ax=ax)

SpAhist.set(title = 'Distribution of Sp.Def and Sp.Atk', xlabel = 'Sp.Def:y , Sp.Atk:b')

FigHist=SpAhist.get_figure()

FigHist.savefig("HistSpDvSpAt.png")
dims = (11.7,8.27) #A4 dimensions

fig, ax = plt.subplots(figsize=dims)

Defhist = sb.distplot(Poke['Defense'],color='g',hist=True,ax=ax)

Atthist = sb.distplot(Poke['Attack'],color = 'r', hist = True,ax=ax)

Atthist.set(title = 'Distribution of Defense and Attack', xlabel = 'Defense:g , Attack:r')

FigHist=Atthist.get_figure()

FigHist.savefig("HistDvAtt.png")
DS = Corr.describe() #Summary Statistics

print(DS)
stats = ['Total','HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']
def maxStat(Poke, column):

    statement = ''

    for col in column:

        stat = Poke[col].max()

        name = Poke[Poke[col]==Poke[col].max()]['Name'].values[0] #Find the names linked with the max stat

        gen =  Poke[Poke[col]==Poke[col].max()]['Generation'].values[0] #Find the Gen linked with the max stat

        statement += name+' of Generation '+str(gen)+' has the best '+col+' stat of '+str(stat)+'.\n'

    return statement



# print(maxStat('name of dataframe','array of stats'))



def minStat(Poke, column):

    statement = ''

    for col in column:

        stat = Poke[col].min()

        name = Poke[Poke[col]==Poke[col].min()]['Name'].values[0] #Find the names linked with the min stat

        gen = Poke[Poke[col]==Poke[col].min()]['Generation'].values[0] #Find the Gen linked with the min stat

        statement += name+' of Generation '+str(gen)+' has the worst '+col+' stat of '+str(stat)+'.\n'

    return statement



# print(minStat('name of dataframe','array of stats'))

print(maxStat(Poke,stats))
print(maxStat(Gen1,stats))
print(minStat(Gen2,stats))
print(minStat(Gen3,stats))
sb.set()

dims = (11.7,8.27) #A4 dimensions

fig, ax = plt.subplots(figsize=dims)

bp = sb.boxplot(x='Generation',y='Total', data=Poke, ax=ax)

plt.title('Box plot of Generation base stat total',fontsize = 18)

plt.xlabel('Generation',fontsize = 12)

plt.ylabel('Total',fontsize = 12)

figBP = bp.get_figure()

figBP.savefig("Box_Gen.png")
def Top10(Gen1,Bstat):

        subGen = Gen1[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

        Strong = subGen.sort_values([Bstat], ascending = False)

        print(Strong.head(10))

        return
Top10(Gen1,'Attack')
def Weakness(Gen1,Bstat,arg2):

    if arg2 == 'Water':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Grass') ^ (Gen1['Type1']=='Electric') ^ 

                                (Gen1['Type2'] == 'Grass') ^ (Gen1['Type2']=='Electric')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Normal':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Fighting') ^ 

                                (Gen1['Type2'] == 'Fighting')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Fire':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Water') ^ (Gen1['Type1']=='Ground') ^ (Gen1['Type1']=='Rock') ^ 

                                (Gen1['Type2'] == 'Water') ^ (Gen1['Type2']=='Ground') ^ (Gen1['Type2']=='Rock')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Electric':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Ground') ^ 

                                (Gen1['Type2'] == 'Ground')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Grass':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Fire') ^ (Gen1['Type1']=='Ice') ^ (Gen1['Type1']=='Poison') ^ (Gen1['Type1'] == 'Flying') ^ (Gen1['Type1']=='Bug') ^ 

                                (Gen1['Type2'] == 'Fire') ^ (Gen1['Type2']=='Ice') ^ (Gen1['Type2']=='Poison') ^ (Gen1['Type2'] == 'Flying') ^ (Gen1['Type2']=='Bug')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Ice':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Fire') ^ (Gen1['Type1']=='Fighting') ^ (Gen1['Type1']=='Rock') ^ (Gen1['Type1'] == 'Steel') ^ 

                                (Gen1['Type2'] == 'Fire') ^ (Gen1['Type2']=='Fighting') ^ (Gen1['Type2']=='Rock') ^ (Gen1['Type2'] == 'Steel')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Fighting':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Flying') ^ (Gen1['Type1']=='Psychic') ^ (Gen1['Type1']=='Fairy') ^ 

                                (Gen1['Type2'] == 'Flying') ^ (Gen1['Type2']=='Psychic') ^ (Gen1['Type2']=='Fairy')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Poison':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Ground') ^ (Gen1['Type1']=='Psychic') ^ 

                                (Gen1['Type2'] == 'Ground') ^ (Gen1['Type2']=='Psychic')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Ground':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Water') ^ (Gen1['Type1']=='Grass') ^ (Gen1['Type1']=='Ice') ^ 

                                (Gen1['Type2'] == 'Water') ^ (Gen1['Type2']=='Grass') ^ (Gen1['Type2']=='Ice')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Flying':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Electric') ^ (Gen1['Type1']=='Ice') ^ (Gen1['Type1']=='Rock') ^ 

                                (Gen1['Type2'] == 'Electric') ^ (Gen1['Type2']=='Ice') ^ (Gen1['Type2']=='Rock')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Psychic':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Bug') ^ (Gen1['Type1']=='Ghost') ^ (Gen1['Type1']=='Dark') ^ 

                                (Gen1['Type2'] == 'Bug') ^ (Gen1['Type2']=='Ghost') ^ (Gen1['Type2']=='Dark')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Bug':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Fire') ^ (Gen1['Type1']=='Flying') ^ (Gen1['Type1']=='Rock') ^ 

                                (Gen1['Type2'] == 'Fire') ^ (Gen1['Type2']=='Flying') ^ (Gen1['Type2']=='Rock')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Rock':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Water') ^ (Gen1['Type1']=='Grass') ^ (Gen1['Type1']=='Fighting') ^ (Gen1['Type1'] == 'Ground') ^ (Gen1['Type1']=='Steel') ^ 

                                (Gen1['Type2'] == 'Water') ^ (Gen1['Type2']=='Grass') ^ (Gen1['Type2']=='Fighting') ^ (Gen1['Type2'] == 'Ground') ^ (Gen1['Type2']=='Steel')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Ghost':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Ghost') ^ (Gen1['Type1']=='Dark') ^ 

                                (Gen1['Type2'] == 'Ghost') ^ (Gen1['Type2']=='Dark')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Dragon':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Ice') ^ (Gen1['Type1']=='Dragon') ^ (Gen1['Type1']=='Fairy') ^ 

                                (Gen1['Type2'] == 'Ice') ^ (Gen1['Type2']=='Dragon') ^ (Gen1['Type2']=='Fairy')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Dark':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Figthing') ^ (Gen1['Type1']=='Bug') ^ (Gen1['Type1']=='Fairy') ^ 

                                (Gen1['Type2'] == 'Fighting') ^ (Gen1['Type2']=='Bug') ^ (Gen1['Type2']=='Fairy')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Steel':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Fire') ^ (Gen1['Type1']=='Fighting') ^ (Gen1['Type1']=='Ground') ^ 

                                (Gen1['Type2'] == 'Fire') ^ (Gen1['Type2']=='Fighting') ^ (Gen1['Type2']=='Ground')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    elif arg2 == 'Fairy':

            Weakness = Gen1.loc[(Gen1['Type1'] == 'Poison') ^ (Gen1['Type1']=='Steel') ^ 

                                (Gen1['Type2'] == 'Poison') ^ (Gen1['Type2']=='Steel')]

            subGen = Weakness[['Name','Type1', 'Type2','Total' ,'HP', 'Attack','Defense','Sp.Atk','Sp.Def','Speed']]

            Strong = subGen.sort_values([Bstat], ascending = False)

            print(Strong.head(6))

    return
Weakness(Gen1,'Attack','Water')