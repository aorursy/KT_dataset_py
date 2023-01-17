import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
np.random.seed(42)
import os
print(os.listdir("../input"))
df_tr = pd.read_csv('../input/application_train.csv')
print (df_tr.info())
sns.set(style='whitegrid')
ax = sns.factorplot(
    x='TARGET',
    y='AMT_CREDIT',
    kind='bar',
    estimator=len,
    data=df_tr
)
ax.set_ylabels('Credit Count')
media = float(len(df_tr[df_tr['TARGET'] == 0]))/len(df_tr)
print ("Proportion of 0s in the train dataset %.3f" %media)
columns = df_tr.columns
col_int = [x for x in range(2, len(columns)) if df_tr[df_tr.columns[x]].dtypes == 'int64']
col_int2 = [x for x in col_int if len(df_tr[df_tr.columns[x]].unique()) <= 3]
col_obj = [x for x in range(0, len(columns)) if df_tr[df_tr.columns[x]].dtypes == object]
columns_rem = col_int2 + col_obj
columns_rem.sort()
indicadores = pd.DataFrame(columns=['Values', 'PROP', 'Attribute'])
for i in columns_rem :
    dfs = df_tr.loc[:, ['SK_ID_CURR', 'TARGET', df_tr.columns[i]]]
    dfg = dfs.groupby([df_tr.columns[i], 'TARGET']).count().reset_index()
    dfg['TOTAL'] = dfg.groupby(df_tr.columns[i])['SK_ID_CURR'].transform(sum)
    dfg['PROP'] = dfg['SK_ID_CURR'] / dfg ['TOTAL']
    dfg['Attribute'] = df_tr.columns[i]
    dfg.rename(columns={df_tr.columns[i]: 'Values'}, inplace=True)
    dfg = dfg[dfg['TARGET'] == 0]
    dfg.drop(['TARGET', 'SK_ID_CURR', 'TOTAL'], axis=1, inplace=True)
    indicadores = indicadores.append(dfg)
graficas = indicadores['Attribute'].unique()
columnas = 3
i = 0
filas = len(graficas)//3 + 1
fig, ax = plt.subplots(filas, columnas, figsize=(20,100), sharey=True)
fig.subplots_adjust(hspace= 0.5)
plt.setp(fig.axes, ylim=(0.8, 1))
for row in range(0, filas):
    for col in range(0, columnas):
        sns.factorplot(
            y='PROP',
            x='Values',
            kind='bar',
            data=indicadores[indicadores['Attribute'] == graficas[i]],
            legend=False,
            palette='muted',
            ax=ax[row, col]
        )
        ax[row, col].set_ylabel('Proportion of 0s')
        ax[row, col].set_xlabel('')
        ax[row, col].hlines(y=media, xmin=-10, xmax=100, color='r')
        ax[row, col].set_title(graficas[i])
        if len(indicadores['Values'][indicadores['Attribute'] == graficas[i]].unique()) >= 5:
            ax[row, col].set_xticklabels(ax[row,col].get_xticklabels(), rotation=45, ha='right')
        if len(indicadores['Values'][indicadores['Attribute'] == graficas[i]].unique()) > 20:
            ax[row, col].set_xticklabels('')
        plt.clf()
        i += 1
        if i >= len(graficas): break
plt.show()