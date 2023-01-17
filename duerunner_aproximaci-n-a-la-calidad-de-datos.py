!pip install pywaffle
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pywaffle import Waffle
df_presupuesto = pd.read_csv("../input/presupuesto-ejecutado-caba-2018-3er-trimestre/presupuesto-ejecutado-2018-tercer-trimestre.csv")

print('El dataset tiene ' + str(df_presupuesto.shape[0]) + ' filas, y ' + str(df_presupuesto.shape[1]) + ' columnas')

df_presupuesto.head()
df_presupuesto.isnull().sum() == 0
#Revisamos los tipos de cada columna

df_presupuesto.dtypes
#Convertimos las variebles numéricas que son códigos a categóricas

df_presupuesto['car'] = df_presupuesto.car.astype('category')

df_presupuesto['jur'] = df_presupuesto.jur.astype('category')

df_presupuesto['sjur'] = df_presupuesto.sjur.astype('category')

df_presupuesto['og'] = df_presupuesto.og.astype('category')

df_presupuesto['ue'] = df_presupuesto.ue.astype('category')

df_presupuesto['prog'] = df_presupuesto.prog.astype('category')

df_presupuesto['sprog'] = df_presupuesto.sprog.astype('category')

df_presupuesto['proy'] = df_presupuesto.proy.astype('category')

df_presupuesto['actividad'] = df_presupuesto.actividad.astype('category')

df_presupuesto['ob'] = df_presupuesto.ob.astype('category')

df_presupuesto['fin'] = df_presupuesto.fin.astype('category')

df_presupuesto['fun'] = df_presupuesto.fun.astype('category')

df_presupuesto['inc'] = df_presupuesto.inc.astype('category')

df_presupuesto['ppal'] = df_presupuesto.ppal.astype('category')

df_presupuesto['par'] = df_presupuesto.par.astype('category')

df_presupuesto['spar'] = df_presupuesto.spar.astype('category')

df_presupuesto['eco'] = df_presupuesto.eco.astype('category')

df_presupuesto['fte'] = df_presupuesto.fte.astype('category')

df_presupuesto['geo'] = df_presupuesto.geo.astype('category')

df_presupuesto['fin'] = df_presupuesto.fin.astype('category')

df_presupuesto['ent'] = df_presupuesto.ent.astype('category')
df_presupuesto.describe()
df_unique = df_presupuesto.select_dtypes(['object']).nunique().to_frame().reset_index()

df_unique.columns = ['Variable','Cantidad de Valores Unicos']
df_unique
df_presupuesto.describe(include=object)
df = df_presupuesto

calculated = df.car_desc.value_counts()

data = calculated.to_dict()



fig = plt.figure(

    FigureClass=Waffle,

    figsize = [7,5],

    rows=5,

    columns=20,

    values=data,

    plot_anchor='N',

    tight_layout= False,

    title={

        'label': 'Frecuencia Car_desc',

        'loc': 'center',

        'fontdict': {

            'fontsize': 20

        }

    },

    labels=[f"{k} ({int(v / sum(data.values()) * 100)}%)" for k, v in data.items()],

        legend={

        'loc': 'lower right',

        'bbox_to_anchor': (0.787, -0.4),

        'framealpha': 0,

        'fontsize': 12

    }



)

fig.savefig('car.jpeg', bbox_inches='tight', pad_inches=0)
#var_categoricas = list(df_presupuesto.select_dtypes(object))

var_categoricas = ['jur_desc','sjur_desc', 'geo_desc',"fin_desc","fun_desc", "inc_desc"]

for variable in var_categoricas:

    sns.countplot(y=variable, data=df_presupuesto)

    plt.title(F'Frecuencia de {variable}')

    plt.xlabel('Apariciones')

    plt.ylabel(variable)

    plt.tight_layout()

    plt.setp(plt.xticks()[1], rotation=30, ha='right')

    if (variable in ['jur_desc','sjur_desc'] ):

        plt.gcf().set_size_inches(14,6 )

    elif (variable in ['fin_desc', "fun_desc","inc_desc"] ):

        plt.gcf().set_size_inches(14,4 )

    plt.tick_params(axis='y', labelsize=10)

    plt.savefig(f"{variable}.jpeg", bbox_inches='tight')

    plt.show()
df_presupuesto['jur_desc'].value_counts(normalize=True)
df_presupuesto['sjur_desc'].value_counts(normalize=True)
df_presupuesto['ent_desc'].value_counts().head(10)
df_presupuesto['eg_desc'].value_counts(normalize=True).head(10)
# Es raro que spar se parezca tanto a par así que revisamos si hay diferencias 

df_presupuesto['eg_desc'].equals(df_presupuesto['ent_desc'])

#Y comprobamos que no son iguales siempre, solo en sus principales items
df = df_presupuesto

df1 = df_presupuesto['eg_desc']

df2 = df_presupuesto['ent_desc']

df["Eq"] = df1.where(df1.values!=df2.values).notna()

new = df.loc[df['eg_desc'] != df['ent_desc']]

new[['eg_desc','ent_desc']].to_csv("comparacion-eg-vs-ent.csv")
df_presupuesto['ue_desc'].value_counts().head(10)
df_presupuesto['prog_desc'].value_counts(normalize=True).head(10)
df_presupuesto['sprog_desc'].value_counts(normalize=True).head(10)
df_presupuesto['proy_desc'].value_counts(normalize=True).head(10)
df_presupuesto['act_desc'].value_counts().head(10)
df_presupuesto['ob_desc'].value_counts().head(10)
# Es raro que spar se parezca tanto a par así que revisamos si hay diferencias 

df['ob_desc'].equals(df['act_desc'])

#Y comprobamos que no son iguales siempre, solo en sus principales items

df = df_presupuesto

df1 = df_presupuesto['ob_desc']

df2 = df_presupuesto['act_desc']

df["Eq"] = df1.where(df1.values!=df2.values).notna()

new = df.loc[df['ob_desc'] != df['act_desc']]

new[['ob_desc','act_desc']]

new[['ob_desc','act_desc']].to_csv("comparacion.csv")
df_presupuesto['fin'].value_counts()
df_presupuesto['fun'].value_counts()
df_presupuesto['ppal_desc'].value_counts(normalize=True).head(10)
df_presupuesto['par_desc'].value_counts().head(10)
df_presupuesto['spar_desc'].value_counts(normalize=True).head(10)
# Es raro que spar se parezca tanto a par así que revisamos si hay diferencias 

df['par_desc'].equals(df['spar_desc'])

#Y comprobamos que no son iguales siempre, solo en sus principales items

df = df_presupuesto

df1 = df_presupuesto['par_desc']

df2 = df_presupuesto['spar_desc']

df["Eq"] = df1.where(df1.values!=df2.values).notna()

new = df.loc[df['par_desc'] != df['spar_desc']]

new[['par_desc','spar_desc']]

new[['par_desc','spar_desc']].to_csv("comparacion par vs spar.csv")
df_presupuesto['eco_desc'].value_counts()
df = df_presupuesto

calculated = df.eco_desc.value_counts()

data = calculated.to_list()

otros = sum(filter(lambda x: (x/52053)*100<=1, data))

data = filter(lambda x: (x/52053)*100>3, data) 

data = [x/52053*100 for x in data] + [otros/52053*100]

data
labels = ['Remuneraciones Al Personal', 'Bienes De Consumo ', 'Servicios No Personales', 'Maquinaria Y Equipo ', 'Otros']

sizes = data

labels = [f"{k} ({round(v,2)}%)" for k, v in zip(labels, data)]

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'grey']

patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)

plt.legend(patches, labels,  bbox_to_anchor=(0.7, 0.85), prop={'size': 16})

plt.axis('equal')

plt.gcf().set_size_inches(12,4 )

plt.savefig("eco_desc.jpeg", bbox_inches='tight')

plt.show()
df_presupuesto['fte_desc'].value_counts().head(10)
df = df_presupuesto

calculated = df.fte_desc.value_counts()

data = calculated.to_list()

otros = sum(filter(lambda x: (x/52053)*100<=1, data))

data = filter(lambda x: (x/52053)*100>3, data) 

data = [x/52053*100 for x in data] + [otros/52053*100]

labels = df.fte_desc.unique()

sizes = data

labels = [f"{k} ({round(v,2)}%)" for k, v in zip(labels, data)]

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'grey']

patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)

plt.legend(patches, labels,  bbox_to_anchor=(0.7, 0.85), prop={'size': 16})

plt.axis('equal')

plt.gcf().set_size_inches(12,4 )

plt.savefig("fte_desc.jpeg", bbox_inches='tight')

plt.show()
df_S = df_presupuesto.loc[(df_presupuesto["sancion"] > 10000000) & (df_presupuesto["sancion"] != 0)]

df_s = df_presupuesto.loc[(df_presupuesto["sancion"] <= 10000000) & (df_presupuesto["sancion"] != 0)]

df_presupuesto[df_presupuesto.sancion == 0].count()["sancion"] 
sns.set(style="whitegrid")

ax = sns.boxplot(x=df_S["sancion"])

ax.set_title('Frecuencia de sancion > 10 M')

fig = ax.get_figure()

fig.savefig('S.png')
sns.set(style="whitegrid")

ax = sns.boxplot(x=df_s["sancion"])

ax.set_title('Frecuencia de sancion <= 10 M')

fig = ax.get_figure()

fig.savefig('s.png')
print(df_s["sancion"].mean(),"\n", df_s["sancion"].median(), "\n", df_s["sancion"].std())

print("\n\n\n")

print(df_S["sancion"].mean(),"\n", df_S["sancion"].median(), "\n", df_S["sancion"].std())

df_V = df_presupuesto.loc[(df_presupuesto["vigente"] > 10000000) & (df_presupuesto["vigente"] != 0)]

df_v = df_presupuesto.loc[(df_presupuesto["vigente"] <= 10000000) & (df_presupuesto["vigente"] != 0)]

print(df_presupuesto[df_presupuesto.vigente == 0].count()["vigente"])



print(df_v["vigente"].mean(),"\n", df_v["vigente"].median(), "\n", df_v["vigente"].std())

print("\n\n\n")

print(df_V["vigente"].mean(),"\n", df_V["vigente"].median(), "\n", df_V["vigente"].std())

sns.set(style="whitegrid")

ax = sns.boxplot(x=df_V["vigente"],palette="Reds")

ax.set_title('Frecuencia de vigente > 10 M')

fig = ax.get_figure()

fig.savefig('V.png')

sns.set(style="whitegrid")

ax = sns.boxplot(x=df_v["vigente"],palette="Reds")

ax.set_title('Frecuencia de vigente <= 10 M')

fig = ax.get_figure()

fig.savefig('v.png')
df_D = df_presupuesto.loc[(df_presupuesto["definitivo"] > 10000000) & (df_presupuesto["definitivo"] != 0)]

df_d = df_presupuesto.loc[(df_presupuesto["definitivo"] <= 10000000) & (df_presupuesto["definitivo"] != 0)]

print(df_presupuesto[df_presupuesto.definitivo == 0].count()["definitivo"])



print(df_d["definitivo"].mean(),"\n", df_d["definitivo"].median(), "\n", df_d["definitivo"].std())

print("\n\n\n")

print(df_D["definitivo"].mean(),"\n", df_D["definitivo"].median(), "\n", df_D["definitivo"].std())
sns.set(style="whitegrid")

ax = sns.boxplot(x=df_D["definitivo"], palette="Greens")

ax.set_title('Frecuencia de definitivo > 10 M')

fig = ax.get_figure()

fig.savefig('D.png')
sns.set(style="whitegrid")

ax = sns.boxplot(x=df_d["definitivo"], palette="Greens")

ax.set_title('Frecuencia de definitivo <= 10 M')

fig = ax.get_figure()

fig.savefig('d.png')
df_Dv = df_presupuesto.loc[(df_presupuesto["devengado"] > 10000000) & (df_presupuesto["devengado"] != 0)]

df_dv = df_presupuesto.loc[(df_presupuesto["devengado"] <= 10000000) & (df_presupuesto["devengado"] != 0)]

print(df_presupuesto[df_presupuesto.devengado == 0].count()["devengado"])



print(df_dv["devengado"].mean(),"\n", df_dv["devengado"].median(), "\n", df_dv["devengado"].std())

print("\n\n\n")

print(df_Dv["devengado"].mean(),"\n", df_Dv["devengado"].median(), "\n", df_Dv["devengado"].std())
sns.set(style="whitegrid")

ax = sns.boxplot(x=df_Dv["devengado"], palette="Purples")

ax.set_title('Frecuencia de devengado > 10 M')

fig = ax.get_figure()

fig.savefig('Dv.png')



sns.set(style="whitegrid")

ax = sns.boxplot(x=df_dv["devengado"], palette="Purples")

ax.set_title('Frecuencia de devengado <= 10 M')

fig = ax.get_figure()

fig.savefig('dv.png')
df = df_presupuesto[["sancion","vigente","definitivo","devengado"]]

ax = sns.boxplot(x="variables", y="valores", data=pd.melt(df, var_name='variables', value_name='valores'))

ax.set_title('Comparacion Variables Numericas')

fig = ax.get_figure()

plt.show()

fig.savefig('CVM.png')

df = df.loc[(df["devengado"] <= 10000000) & (df["devengado"] != 0) &  (df["vigente"] <= 10000000) & (df["vigente"] != 0) &  (df["sancion"] <= 10000000) & (df["sancion"] != 0) &  (df["definitivo"] <= 10000000) & (df["definitivo"] != 0)]

ax = sns.boxplot(x="variables", y="valores", data=pd.melt(df, var_name='variables', value_name='valores'))

ax.set_title('Variables Numericas <= 10 Millones')

fig = ax.get_figure()

plt.show()

fig.savefig('CVMchicas.png')
df = df_presupuesto[["sancion","vigente","definitivo","devengado"]]

df = df.loc[(df["devengado"] > 10000000)  |  (df["vigente"] > 10000000) | (df["sancion"] > 10000000) |  (df["definitivo"] > 10000000)]

ax = sns.boxplot(x="variables", y="valores", data=pd.melt(df, var_name='variables', value_name='valores'))

ax.set_title('Boxplots Variables Numericas > 10 Millones')

fig = ax.get_figure()

plt.show()

fig.savefig('CVMgrandes.png')

df = df_presupuesto[["sancion","vigente","definitivo","devengado"]]

ax = df.plot.scatter(x='sancion',

                      y='devengado',

                      c='DarkBlue')



ax.set_title('Dispersion Devengado vs Sancionado')

fig = ax.get_figure()

plt.show()

fig.savefig('sancion-vs-devengado.png')
df.corr()
import matplotlib.pyplot as plt

import matplotlib as mpl 

colors = sns.cubehelix_palette(8, start=.5, rot=-.75)



cmap = mpl.colors.ListedColormap(colors)



plt.figure(figsize=(7,5))

plt.matshow(df.corr(), cmap=cmap, fignum=1)

plt.xticks(range(len(df.columns)), df.columns , rotation = 30,)

plt.yticks(range(len(df.columns)), df.columns)

plt.colorbar()

plt.tight_layout()



fig = plt.gcf()

fig.suptitle("Correlacion entre variables numericas", fontsize=16, y=1.1)

plt.savefig('matrix.png')
df = df_presupuesto[["fin_desc","car_desc"]]

df = df.melt(var_name='Type', value_name='M')

df
df = df_presupuesto[["fin_desc","car_desc"]]

df = pd.crosstab(df['fin_desc'], df['car_desc'])



df.sort_values("Administracion Central").plot(kind='bar', stacked=True, figsize = (10,4))

plt.xticks(rotation=15)

plt.title("Comparacion fin_desc vs car_desc")

plt.tight_layout()

plt.savefig("Comparacion categoricas")

              

df
df = df_presupuesto[["geo_desc","devengado"]]

colors=sns.color_palette("Set2")

df.groupby('geo_desc', sort=True)['devengado'].sum().plot.bar(color = colors, figsize=(14,6))

plt.xticks(rotation=30)

plt.tight_layout()

plt.title("Total devengado por comuna")

plt.savefig("Comparacion comunas")  