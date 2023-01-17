## Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
path = 'https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7'

# load covid19 dat
data = pd.read_csv(path, sep=';',parse_dates=True, dayfirst=True)
data['jour'] = pd.to_datetime(data.jour)
data['jour'] = data['jour'].dt.strftime('%Y-%m-%d')

latest_update=data.jour.max()
print('latest update: ' + str(latest_update))

# load departments data
deps_path = 'https://www.data.gouv.fr/en/datasets/r/987227fb-dcb2-429e-96af-8979f97c9c84'
deps = pd.read_csv(deps_path)

m= pd.merge(data, deps, how='inner', left_on='dep', right_on = 'num_dep' )

m.head()
# ITALIAN DATA
ita_path = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
#ita_path = 'https://github.com/pcm-dpc/COVID-19/blob/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
ita_data = pd.read_csv(ita_path,parse_dates=True, dayfirst=True)
#ita_data.data = pd.to_datetime(ita_data.data)
ita_data.index = pd.to_datetime(ita_data.data)
def pie_chart(values,labels,n_regions, title):
    colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'orange', 'pink', 'purple', 'navy']
    fig_pie, ax_pie = plt.subplots()
    size = 1
    props_names = dict(boxstyle='round', facecolor='white', alpha=1)
    compressed_values = np.append(values[:n_regions], np.sum(values[n_regions:]))
    compressed_labes = np.append(np.array(labels[:n_regions]), 'Autres regions')

    ax_pie.pie(compressed_values, 
               radius=1, 
               wedgeprops=dict(width=size, edgecolor='w'), 
               labels=compressed_labes, 
               colors=colors,
               pctdistance=0.7,
               autopct=lambda p : '{:.0f}'.format(p * sum(compressed_values)/100)
                              )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_pie.text(1.2, 0.8, 'Source : \nwww.data.gouv.fr', fontsize=10,verticalalignment='top', bbox=props)   
    plt.title(title, size=15)
    plt.tight_layout()
# group by department
regions_tot = m.groupby('dep').max().groupby('region_name').sum().sort_values(by='dc', ascending=False)
print(regions_tot.dc.sum())
pie_chart(regions_tot.dc, regions_tot.index, 7, 'COVID19 : Déces par region [' + latest_update + ']' )

def plot_data(data, label, plot_diff=False):
    fig1, ax1 = plt.subplots(figsize=(20,10))
    ax1.set_title('COVID19 en ' + label)
    if(plot_diff==True):
        plt.plot(data.index, data.hosp_diff,color='b', linestyle='--', label='hosp_diff')
        plt.plot(data.index, data.rea_diff,color='orange', linestyle='--', label='rea_diff')
        plt.plot(data.index, data.dc_diff,color='r', linestyle='--',label='dc_diff')
    else:
        plt.plot(data.index, data.hosp,color='b', linestyle='-', label='hosp')
        plt.plot(data.index, data.rea,color='orange', linestyle='-', label='rea')
        plt.plot(data.index, data.dc,color='r', linestyle='-',label='dc')    
    
    plt.legend()
    ax1.set_xticklabels(list(data.index),rotation=90)
    axes = plt.gca()
    axes.yaxis.grid()   
    plt.show()
    
def shift_data(data, min_shift, max_shift, col_names):
    for c in col_names:
        for d in np.arange(min_shift,max_shift):
            data[c + '_sh_' + str(d)] = data[c].shift(d)        
    return data

from datetime import datetime

# prep data to predict future dates
def add_dates_to_predict(data, predicted_days):
    print(data)

    mindate = min(data.index)
    maxdate = max(data.index)
    ixs = pd.date_range(start=mindate,end=maxdate+pd.DateOffset(predicted_days))
    data = data.reindex(ixs)
    return data
# aggregated on the country
france_data = m[(m.sexe == 0)].groupby('jour').sum().sort_values(by='jour')
france_data.index = pd.to_datetime(france_data.index)


# add diff columns to have daily 
france_data['dc_diff'] = france_data.dc.diff()
france_data['hosp_diff'] = france_data.hosp.diff()
france_data['rea_diff'] = france_data.rea.diff()
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(france_data.index, france_data['hosp_diff'], color='r', label = 'Hospitalisations diff') # prediction on train
plt.plot(france_data.index, france_data['rea_diff'], color='b', label = 'IC diff ') # prediction on train
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.legend(fontsize=15)
# prediction on test
plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(france_data.index, france_data['hosp'], color='b', linestyle='--', label = 'FRA Hospitalisations') # 
plt.plot(ita_data.index, ita_data.ricoverati_con_sintomi, color='b', label = 'ITA Hospitalisations') # 
plt.plot(france_data.index, france_data['rea'], color='r', linestyle='--', label = 'FRA Intensive Care') # 
plt.plot(ita_data.index, ita_data.terapia_intensiva, color='r', label = 'ITA Intensive Care') # 
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.legend(fontsize = 20)
plt.title('COVID19 Hospital data [France, Italy]')
# prediction on test
plt.xticks(rotation=90)
from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots(figsize=(20,10))
plt.plot(france_data.index, 100 * france_data['rea'] / (france_data['hosp'] + france_data['rea']), color='b', label = 'FRA') # 
plt.plot(ita_data.index, 100 * ita_data.terapia_intensiva / ita_data.totale_ospedalizzati, color='r', label = 'ITA') # 
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.legend(fontsize = 20)
plt.title('[Covid19] Intensive care vs Hospitalisations ratio', fontsize = 30)
ax.yaxis.set_major_formatter(PercentFormatter())
# prediction on test
plt.xticks(rotation=90)

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.8, 0.75, 'FRA source: data.gouv.fr  \nITA source:  Protezione Civile', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)



# simple Random Forest model to predict future deaths
def rf_predict_series(train_feature_set, train_targets_set, test_feature_set):

    from sklearn.ensemble import RandomForestRegressor
    random_forest = RandomForestRegressor(n_estimators=2000,
    max_depth=6,
    max_features=10,
    random_state=42)

    random_forest.fit(train_feature_set,train_targets_set) 
    train_preds = random_forest.predict(train_feature_set)
    test_preds = random_forest.predict(test_feature_set)
    
    return test_preds, train_preds
  
# prep data
target_label = 'dc_diff'
non_feature_columns = ['sexe','hosp','rea','rad','dc', 'dc_diff', 'hosp_diff','rea_diff']

# add dates to predict
days_to_predict = 15 # how many days do we want to predict?
aug_france_data = add_dates_to_predict(france_data, days_to_predict)

#add lagged columns
history_length = 30 # how long we want to look back
shifted = shift_data(aug_france_data,days_to_predict, history_length,('hosp_diff', 'rea_diff','dc_diff'))

# remove first days_to_predict rows
shifted = shifted[history_length:]

# define feature series (including rows for the future)
feature_series = shifted.drop(non_feature_columns, axis=1)

train_ft_set = feature_series[:-days_to_predict]
targets_series = shifted[target_label]
train_tg_set = targets_series[:-days_to_predict]
test_ft_set = feature_series[-days_to_predict:]        
    
test_preds, train_preds = rf_predict_series(train_ft_set, train_tg_set, test_ft_set)

fig, ax = plt.subplots(figsize=(20,10))
plt.title("France COVID19 deaths prediction based on hospitalization, IC and death data [update: " + str(latest_update) + "]",fontsize=20)
#plt.plot(train_tg_set.index, train_tg_set, color='b', label ='Deaths: train ground truth') # ground truth on train
plt.plot(train_tg_set.index, train_preds, color='r', label = 'Train model predictions') # prediction on train
plt.plot(test_ft_set.index, test_preds, color = 'r', linestyle='--', label ='Model predictions over ' + str(days_to_predict) + ' days')
plt.plot(france_data.index, france_data['dc_diff'], 'b', linestyle='-', label ='Ground truth')
plt.legend(fontsize=15)
plt.grid(color='black', linestyle='-', linewidth=0.5)
# prediction on test
plt.xticks(rotation=90)
