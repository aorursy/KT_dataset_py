# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import re
%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Indicators.csv")
df.shape
df.head()
indicators = df['IndicatorName'].unique().tolist()
indicators.sort()
#print(indicators)
indicators_list = df[['IndicatorName','IndicatorCode']].drop_duplicates().values
indicators_list
new_indicators =[]
indicators_code =[]

for ind in indicators_list:
    indicator = ind[0]
    code = ind[1].strip()
    if code not in indicators_code:
        #Delete ,() from indicators and convert all characters to lower case
        modified_indicator = re.sub('[,()]',"",indicator).lower()
        #Replace - with "to"
        modified_indicator = re.sub('-'," to ",modified_indicator).lower()
        new_indicators.append([modified_indicator,code])
        indicators_code.append(code)
new_indicators[:5]
indicators_code[:5]
Indicators = pd.DataFrame(new_indicators, columns=['IndicatorName','IndicatorCode'])
Indicators = Indicators.drop_duplicates()
print(Indicators.shape)
Indicators.head()
key_word_dict = {}

key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']
key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']
key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']
key_word_dict['Economy'] = ['income','gdp','gini','deficit','budget','market','stock','bond','infrastructure','debt']
key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']
key_word_dict['Education'] = ['education','literacy','youth']
key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']
def pick_indicator(feature):
    
    for indicator_ele in Indicators.values:
        
        if feature not in key_word_dict.keys():
            print("Choose the right feature!")
            break
        
        for word in key_word_dict[feature]:
            
            word_list = indicator_ele[0].split() # it would split from ','
            
            if word in word_list or word+'s' in word_list:
                
                print(indicator_ele)
                
                break
pick_indicator('Health')
df_CRU = df[(df['CountryCode'] == 'CHN') | (df['CountryCode'] == 'RUS') | (df['CountryCode'] == 'USA')]
df_CRU.head()
chosen_indicators = [#Health
                    'SP.DYN.IMRT.IN','SH.STA.BRTC.ZS','SH.XPD.TOTL.ZS','SL.TLF.0714.ZS', # Employment
                    'SL.UEM.TOTL.ZS','SL.UEM.TOTL.MA.ZS','SL.UEM.TOTL.FE.ZS','SL.GDP.PCAP.EM.KD',
                    'SL.EMP.1524.SP.NE.ZS','SL.UEM.1524.NE.ZS', # Economy
                    'NY.GDP.PCAP.CD','NY.GDP.PCAP.KD','NY.GDP.PCAP.KD.ZG','SL.GDP.PCAP.EM.KD',
                    'SI.POV.GINI','SI.DST.10TH.10','SI.DST.FRST.10','GC.DOD.TOTL.GD.ZS','SH.XPD.TOTL.ZS',
                    # Energy
                    'EN.ATM.CO2E.PC','EG.USE.COMM.CL.ZS','EG.IMP.CONS.ZS','EG.ELC.RNWX.KH',
                    'EG.USE.ELEC.KH.PC','EG.ELC.NUCL.ZS','EG.ELC.ACCS.ZS','EG.ELC.ACCS.RU.ZS',
                    'EG.ELC.ACCS.UR.ZS','EG.FEC.RNEW.ZS', # Demography
                    'SP.DYN.CBRT.IN','SP.DYN.CDRT.IN','SP.DYN.LE00.IN','SP.POP.65UP.TO.ZS',
                    'SP.POP.1564.TO.ZS','SP.POP.TOTL.FE.ZS','SP.POP.TOTL','SH.DTH.IMRT','SH.DTH.MORT',
                    'SP.POP.GROW','SE.ADT.LITR.ZS','SI.POV.NAHC','SH.CON.1524.MA.ZS','SH.STA.DIAB.ZS', #Trade
                    'NE.IMP.GNFS.ZS','NE.EXP.GNFS.CD','NE.IMP.GNFS.CD','NE.TRD.GNFS.ZS']
df_CRU_subset = df_CRU[df_CRU['IndicatorCode'].isin(chosen_indicators)]
print(df_CRU_subset.shape)
df_CRU_subset.head()
def stage_prep(indicator):
    
        df_stage_china = df_CRU_subset[(df_CRU_subset['CountryCode'] == 'CHN') &
                                       (df_CRU_subset['IndicatorCode'] == indicator)]
        
        df_stage_russia = df_CRU_subset[(df_CRU_subset['CountryCode'] == 'RUS') &
                                        (df_CRU_subset['IndicatorCode'] == indicator)]
        
        df_stage_usa = df_CRU_subset[(df_CRU_subset['CountryCode'] == 'USA') &
                                     (df_CRU_subset['IndicatorCode'] == indicator)]
        
        if((df_stage_china.empty) | (df_stage_russia.empty) | (df_stage_usa.empty)):

            print("This indicator is not present in all three countries. Please choose another indicator.")
        
        else:
            
            min_year_c = df_stage_china.Year.min()
            max_year_c = df_stage_china.Year.max()
            
            min_year_r = df_stage_russia.Year.min()
            max_year_r = df_stage_russia.Year.max()
            
            min_year_us = df_stage_usa.Year.min()
            max_year_us = df_stage_usa.Year.max()
            
            min_list = [min_year_c, min_year_r,min_year_us]
            max_among_all_min_years = max(min_list)
            
            max_list = [max_year_c,max_year_r,max_year_us]
            min_among_all_max_years = min(max_list)
        
            if( (min_year_c == min_year_r== min_year_us) & (max_year_c == max_year_r == max_year_us)):
                
                df_stage = df_CRU_subset[df_CRU_subset['IndicatorCode'] == indicator]
                
                return df_stage
            
            else:
                year_and_indicator_filter = ((df_CRU_subset['Year'] >= max_among_all_min_years) & 
                                             (df_CRU_subset['Year'] <= min_among_all_max_years) &
                                             (df_CRU_subset['IndicatorCode'] == indicator))
                        
                df_stage = df_CRU_subset[year_and_indicator_filter] 
                
                return df_stage
import plotly 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)
def plot_barchart(df_stages):
    
    figure ={
    'data':[],
    'layout':{}
    }
    
    # Creating stage for each country
    df_stages_china = df_stages[df_stages['CountryCode'] == 'CHN']
    
    trace_1 = go.Bar({
        'y': list(df_stages_china['Year']),
        'x': list(df_stages_china['Value']),
        'text': list(df_stages_china['Value']),
        'name': 'China',
        'orientation': 'h'
    })
    
    figure['data'].append(trace_1)
    
    df_stages_russia = df_stages[df_stages['CountryCode'] == 'RUS']
   
    trace_2 = go.Bar({
        'y': list(df_stages_russia['Year']),
        'x': list(df_stages_russia['Value']),
        'text': list(df_stages_russia['Value']),
        'name': 'Russia',
        'orientation': 'h'
    })
    
    figure['data'].append(trace_2)
    
    df_stages_usa = df_stages[df_stages['CountryCode'] == 'USA']
          
    trace_3= go.Bar({
        'y': list(df_stages_usa['Year']),
        'x': list(df_stages_usa['Value']),
        'text': list(df_stages_usa['Value']),
        'name': 'USA',
        'orientation': 'h'
    })
    
    figure['data'].append(trace_3)
    
    title = df_stages['IndicatorName'].iloc[0]
    
    figure['layout']['title'] = title
    figure['layout']['xaxis'] = {'title': 'Value'}
    figure['layout']['yaxis'] = {'title': 'Years'}
    figure['layout']['hovermode'] = 'compare'
    
    iplot(figure)
    
def plot_line(df_stages):
    
    # Initializing figure. If we initialize it outside of function then new data will get appended with old data and 
    # plot will show all the data including new and old. To avoid that repetation of data in figure, we initialize it inside.
    figure ={
    'data':[],
    'layout':{}
    }
    
    # Creating stage for each country
    df_stages_china = df_stages[df_stages['CountryCode'] == 'CHN']
    
    trace_1 = go.Scatter({
        'x': list(df_stages_china['Year']),
        'y': list(df_stages_china['Value']),
        'connectgaps': True,
        'text': list(df_stages_china['Value']),
        'name': 'China'
    })
    
    figure['data'].append(trace_1)
    
    df_stages_russia = df_stages[df_stages['CountryCode'] == 'RUS']
   
    trace_2 = go.Scatter({
        'x': list(df_stages_russia['Year']),
        'y': list(df_stages_russia['Value']),
        'connectgaps': True,
        'text': list(df_stages_russia['Value']),
        'name': 'Russia'
    })
    
    figure['data'].append(trace_2)
    
    df_stages_usa = df_stages[df_stages['CountryCode'] == 'USA']
          
    trace_3= go.Scatter({
        'x': list(df_stages_usa['Year']),
        'y': list(df_stages_usa['Value']),
        'connectgaps': True,
        'text': list(df_stages_usa['Value']),
        'name': 'USA'
    })
    
    figure['data'].append(trace_3)
    
    title = df_stages['IndicatorName'].iloc[0]
    
    figure['layout']['title'] = title
    figure['layout']['xaxis'] = {'title': 'Years'}
    figure['layout']['yaxis'] = {'title': 'Value'}
    figure['layout']['hovermode'] = 'compare'
    
    iplot(figure, validate =False)
df_stage_health_1 = stage_prep(chosen_indicators[0])
print(df_stage_health_1.shape)
print("Min year: ",df_stage_health_1.Year.min()," Max year: ", df_stage_health_1.Year.max())
df_stage_health_1.head()
plot_barchart(df_stage_health_1)
plot_line(df_stage_health_1)
df_stage_health_2 = stage_prep(chosen_indicators[1])
print(df_stage_health_2.shape)
print("Min year: ",df_stage_health_2.Year.min()," Max year: ", df_stage_health_2.Year.max())
df_stage_health_2.head()
plot_line(df_stage_health_2)
df_stage_health_3 = stage_prep(chosen_indicators[18])
print(df_stage_health_3.shape)
print("Min year: ",df_stage_health_3.Year.min()," Max year: ", df_stage_health_3.Year.max())
df_stage_health_3.head()
plot_barchart(df_stage_health_3)
plot_line(df_stage_health_3)
df_stage_emp_1 = stage_prep(chosen_indicators[4])
print(df_stage_emp_1.shape)
print("Min year: ",df_stage_emp_1.Year.min()," Max year: ", df_stage_emp_1.Year.max())
df_stage_emp_1.head()
plot_barchart(df_stage_emp_1)
plot_line(df_stage_emp_1)
df_stage_ec_1 = stage_prep(chosen_indicators[10])
print(df_stage_ec_1.shape)
print("Min year: ",df_stage_ec_1.Year.min()," Max year: ", df_stage_ec_1.Year.max())
df_stage_ec_1.head()
plot_barchart(df_stage_ec_1)
plot_line(df_stage_ec_1)
df_stage_ec_2 = stage_prep(chosen_indicators[12])
print(df_stage_ec_2.shape)
print("Min year: ",df_stage_ec_2.Year.min()," Max year: ", df_stage_ec_2.Year.max())
df_stage_ec_2.head()
plot_barchart(df_stage_ec_2)
plot_line(df_stage_ec_2)
df_stage_ec_3 = stage_prep(chosen_indicators[14])
print(df_stage_ec_3.shape)
print("Min year: ",df_stage_ec_3.Year.min()," Max year: ", df_stage_ec_3.Year.max())
df_stage_ec_3.head()
plot_line(df_stage_ec_3)
df_stage_ec_4 = stage_prep(chosen_indicators[15])
print(df_stage_ec_4.shape)
print("Min year: ",df_stage_ec_4.Year.min()," Max year: ", df_stage_ec_4.Year.max())
df_stage_ec_4.head()
plot_line(df_stage_ec_4)
df_stage_energy_1 = stage_prep(chosen_indicators[19])
print(df_stage_energy_1.shape)
print("Min year: ",df_stage_energy_1.Year.min()," Max year: ", df_stage_energy_1.Year.max())
df_stage_energy_1.head()
plot_barchart(df_stage_energy_1)
plot_line(df_stage_energy_1)
df_stage_energy_2 = stage_prep(chosen_indicators[20])
print(df_stage_energy_2.shape)
print("Min year: ",df_stage_energy_2.Year.min()," Max year: ", df_stage_energy_2.Year.max())
df_stage_energy_2.head()
plot_barchart(df_stage_energy_2)
plot_line(df_stage_energy_2)
df_stage_energy_3 = stage_prep(chosen_indicators[22])
print(df_stage_energy_3.shape)
print("Min year: ",df_stage_energy_3.Year.min()," Max year: ", df_stage_energy_3.Year.max())
df_stage_energy_3.tail()
plot_barchart(df_stage_energy_3)
plot_line(df_stage_energy_3)
df_stage_energy_4 = stage_prep(chosen_indicators[25])
print(df_stage_energy_4.shape)
print("Min year: ",df_stage_energy_4.Year.min()," Max year: ", df_stage_energy_4.Year.max())
df_stage_energy_4.head()
plot_line(df_stage_energy_4)
df_stage_dg_1 = stage_prep(chosen_indicators[31])
print(df_stage_dg_1.shape)
print("Min year: ",df_stage_dg_1.Year.min()," Max year: ", df_stage_dg_1.Year.max())
df_stage_dg_1.head()
plot_line(df_stage_dg_1)
df_stage_trd_1 = stage_prep(chosen_indicators[45])
print(df_stage_trd_1.shape)
print("Min year: ",df_stage_trd_1.Year.min()," Max year: ", df_stage_trd_1.Year.max())
df_stage_trd_1.head()
plot_line(df_stage_trd_1)
plot_barchart(df_stage_trd_1)
df_stage_trd_2 = stage_prep(chosen_indicators[44])
print(df_stage_trd_2.shape)
print("Min year: ",df_stage_trd_2.Year.min()," Max year: ", df_stage_trd_2.Year.max())
df_stage_trd_2.head()
plot_line(df_stage_trd_2)
plot_barchart(df_stage_trd_2)
