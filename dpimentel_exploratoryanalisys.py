import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.listdir("../input/b3.csv/")
data_path = "../input/b3.csv/b3.csv"
b3 = pd.read_csv(data_path)#, compression="gzip")
b3['datahora'] = pd.to_datetime(b3['datahora'], format="%Y%m%d%H%M")
b3.set_index(['datahora'], inplace=True)
b3.info(null_counts=True)
values = b3['fechamento_atual'].shift(1)
condition = b3['codigo'] == b3['codigo'].shift(1)
b3['fechamento_anterior'] = [value if truth else np.nan for value, truth in zip(values, condition)]
b3['log_retorno'] = np.log(b3["fechamento_atual"]) - np.log(b3["fechamento_anterior"])

b3.fillna(method='backfill', inplace=True)
b3.head()
b3.index.hour.value_counts()
b3[b3.index.hour > 16]['fechamento_atual'].count()
assets = b3.codigo.unique()
print("número de ativos na B3: {}\nativos:".format(len(assets)))
print(assets)
grouped_b3 = b3.groupby("codigo", as_index=False)
grouped_b3.get_group('GOLL4').head()
plt.close()
rows = len(grouped_b3)//3+1
plt.figure(figsize=(60, 200))
for asset, i in zip(assets, range(1,len(grouped_b3))):
    ax = plt.subplot(rows,3,i)
    data = grouped_b3.get_group(asset)['log_retorno']
    ax.hist(data, log=True, bins=np.arange(-.2, .2, 0.01))
    plt.vlines(x=data.mean(), color='r', linestyle='-', ymin=0, ymax=100000, linewidth=3)
    ax.set_ylim([1,100000])
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.set_title(asset, fontsize=35)
#plt.savefig('../graphics/fech_diff_all.jpg')
plt.show()
corr_pearson = [[grouped_b3.get_group(asset_1)['log_retorno'].corr(grouped_b3.get_group(asset_2)['log_retorno'])
                if asset_1 != asset_2
                else np.nan
                for asset_2 in assets] 
                for asset_1 in assets]
corr_df = pd.DataFrame(corr_pearson, index=assets, columns=assets)
cm = sns.light_palette("green", as_cmap=True)
threshold = 0.60
corr_above_threshold = [col for col in assets if corr_df[col].max() > threshold]
corr_df.loc[corr_above_threshold,corr_above_threshold].fillna(1).style.background_gradient(cmap=cm, axis=1)
def summary(data, assets_list, attribute):
    info = [data.get_group(x)[attribute].describe() for x in assets_list]
    info = pd.DataFrame(info, index=assets_list)
    #info['sum'] = [data.get_group(x)[attribute].sum() for x in assets_list]
    print(info)
to_analize = ['BBDC4','ITUB4','GGBR4','CSNA3']
summary(grouped_b3, to_analize, 'log_retorno')
def plot_fechamento(data, assets_list):
    
    k = len(assets_list)
    
    plt.close()
    colors = ['r','b','g','c','m','y','k']

    plt.close()
    plt.figure(figsize=(40,10))
    for x, color, i in zip(assets_list,colors[:k], range(0,k)):
        data.get_group(x)['fechamento_atual'].plot(kind='line', style=color, fontsize=24)
    plt.title("Valor de fechamento (15min)", fontsize=40)
    plt.legend(assets_list, fontsize=40)
    plt.xlabel("")
    plt.show()
to_analize = ['BBDC4','ITUB4']
plot_fechamento(grouped_b3, to_analize)
to_analize = ['GGBR4','CSNA3','VALE3']
plot_fechamento(grouped_b3, to_analize)
def plot_log_retorno(data, assets_list):

    k = len(assets_list)
    
    plt.close()
    colors = ['r','b','g','c','m','y','k']

    fig, ax = plt.subplots(nrows=k, sharex=True, figsize=(40,20))
    for x, color, i in zip(assets_list,colors[:k], range(0,k)):
        data.get_group(x)['log_retorno'].plot(kind='line', style=color, ax=ax[i])
        ax[i].set_ylim([-0.1,0.1])
        ax[i].legend([x], fontsize=30)
        ax[i].xaxis.set_tick_params(labelsize=24)
        ax[i].yaxis.set_tick_params(labelsize=24)
    plt.show()
to_analize = ['BBDC4','ITUB4','GGBR4','CSNA3','VALE3']

plot_log_retorno(grouped_b3, to_analize)
b3['ano_mes_dia'] = b3.index.strftime("%Y-%m-%d")

b3_daily = b3.sort_index(ascending=True).groupby(['ano_mes_dia','codigo']).agg({'fechamento_atual' : 'last', 
                                                                                'fechamento_anterior' : 'first'})

b3_daily['log_retorno'] = np.log(b3_daily["fechamento_atual"]) - np.log(b3_daily["fechamento_anterior"])

threshold = 0.015
b3_daily['bin_log_retorno'] = [1 if x > threshold 
                               else -1 if x < -threshold 
                               else 0 
                               for x in b3_daily['log_retorno']]

grouped_b3_daily = b3_daily.groupby("codigo", as_index=False)

b3_daily.info()
def plot_stacked_bin(data, assets_list):

    plt.close()

    assets_list = to_analize
    k = len(assets_list)

    plt.figure(figsize=(160,5))
    colors = ['r','b','g','c','m','y','k']

    x = np.arange(len(grouped_b3_daily.get_group(assets_list[0])['bin_log_retorno'].values))
    data = grouped_b3_daily.get_group(assets_list[0])['bin_log_retorno'].values
    p1 = plt.bar(x, data, color=colors[0])

    bottom = 0

    for asset, color, i in zip(assets_list[1:], colors[1:k], range(0,k)): 

        bottom = bottom + data
        data = grouped_b3_daily.get_group(asset)['bin_log_retorno'].values
        p1 = plt.bar(x, data, color=color, bottom = bottom)

    plt.legend(assets_list, fontsize=30)

    plt.title("Concordância entre ativos (diário)", fontsize=40)
    plt.xticks([])
    plt.yticks([])

    plt.show()
to_analize = ['GGBR4','CSNA3','BRAP4']

plot_stacked_bin(grouped_b3_daily, to_analize)
print("Concordância entre Ativos (diária)")
def concordance_accuracy(grouped_b3_daily, to_analize):

    assets_list = to_analize

    sum_ = grouped_b3_daily.get_group(assets_list[0])['bin_log_retorno'].values + \
           grouped_b3_daily.get_group(assets_list[1])['bin_log_retorno'].values + \
           grouped_b3_daily.get_group(assets_list[2])['bin_log_retorno'].values

    prev_item = (x for x in sum_[:-2])
    item = (x for x in sum_[1:-1])
    next_item = (x for x in sum_[2:])
    a = [1 if (abs(w) <= abs(x) <= abs(y)) 
         else -1 if (abs(w) >= abs(x) >= abs(y)) 
         else 0 for w,x,y in zip(prev_item,item,next_item)]
    plt.close()
    plt.hist(a)
    plt.show()
to_analize = ['GGBR4','CSNA3','BRAP4']
concordance_accuracy(grouped_b3_daily, to_analize)