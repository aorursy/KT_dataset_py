# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#renoamendo colunas
data = pd.read_csv("/kaggle/input/car-data/car.csv")
data.columns = ['buying', 'maint','doors','persons','lug_boot','safety','class']
data.head()
# Filtrando combinação buying e maint

vhigh_med=data[(data['buying']=='vhigh') & (data['maint']=='med')]
vhigh_low=data[(data['buying']=='vhigh') & (data['maint']=='low')]
high_high=data[(data['buying']=='high') & (data['maint']=='high')]
high_med=data[(data['buying']=='high') & (data['maint']=='med')]
high_low=data[(data['buying']=='high') & (data['maint']=='low')]
med_high=data[(data['buying']=='med') & (data['maint']=='high')]
med_med=data[(data['buying']=='med') & (data['maint']=='med')]
med_low=data[(data['buying']=='med') & (data['maint']=='low')]
low_vhigh=data[(data['buying']=='low') & (data['maint']=='vhigh')]
low_high=data[(data['buying']=='low') & (data['maint']=='high')]
low_med=data[(data['buying']=='low') & (data['maint']=='med')]
low_low=data[(data['buying']=='low') & (data['maint']=='low')]
#Selecionar "people" quando for igual á 2,ou quando "safety" for igual a low,
#ou quando "buying" for igual vhigh e "maint" for igual vhigh,
#ou "buying" igual vhigh e "maint" igual high, 
#ou "buiying" igual high e "maint" igual vhigh.

result1=data[(data['persons']=='2')|(data['safety']=='low')| 
                       ((data['buying']=='vhigh') & (data['maint']=='vhigh'))| 
                       ((data['buying']=='vhigh') & (data['maint']=='high')) |
                       ((data['buying']=='high') & (data['maint']=='vhigh'))]
result1=result1.drop_duplicates()
#total classificados como unacc
print(len(result1[(result1['class']=='unacc')]))
#total da classificação geral
print(len(result1))
#Selecionar quando "people" for igual á 4 e "doors" for igual á 2 ou 3 
#e "lug_boot" for igual a small e "safety" igual a med,
#ou"lug_boot" for igual a med e "safety" igual a med

u1=vhigh_med[(vhigh_med['persons']=='4') & ((vhigh_med['doors']=='2')|(vhigh_med['doors']=='3')) &
           (((vhigh_med['lug_boot']=='small') & (vhigh_med['safety']=='med'))|
           ((vhigh_med['lug_boot']=='med') & (vhigh_med['safety']=='med')))]

u2=vhigh_low[(vhigh_low['persons']=='4') & ((vhigh_low['doors']=='2')|(vhigh_low['doors']=='3')) & 
            ((vhigh_low['lug_boot']=='small') & (vhigh_low['safety']=='med')) |
            ((vhigh_low['lug_boot']=='med') & (vhigh_low['safety']=='med'))]

u3=high_high[(high_high['persons']=='4') & ((high_high['doors']=='2')|(high_high['doors']=='3')) & 
            ((high_high['lug_boot']=='small') & (high_high['safety']=='med')) |
            ((high_high['lug_boot']=='med') & (high_high['safety']=='med'))]

u4=high_med[(high_med['persons']=='4') & ((high_med['doors']=='2')|(high_med['doors']=='3')) &
           (((high_med['lug_boot']=='small') & (high_med['safety']=='med'))|
           ((high_med['lug_boot']=='med') & (high_med['safety']=='med')))]

u5=med_high[(med_high['persons']=='4') & ((med_high['doors']=='2')|(med_high['doors']=='3')) &
           (((med_high['lug_boot']=='small') & (med_high['safety']=='med'))|
           ((med_high['lug_boot']=='med') & (med_high['safety']=='med')))]

u6=high_low[(high_low['persons']=='4') & ((high_low['doors']=='2')|(high_low['doors']=='3')) &
           (((high_low['lug_boot']=='small') & (high_low['safety']=='med'))|
           ((high_low['lug_boot']=='med') & (high_low['safety']=='med')))]


u7=low_vhigh[(low_vhigh['persons']=='4') & ((low_vhigh['doors']=='2')|(low_vhigh['doors']=='3')) &
           (((low_vhigh['lug_boot']=='small') & (low_vhigh['safety']=='med'))|
           ((low_vhigh['lug_boot']=='med') & (low_vhigh['safety']=='med')))]

#Juntar sub-tabelas

frames = [u1, u2, u3,u4,u5,u6,u7]
result2 = pd.concat(frames).drop_duplicates()
#total classificados como unacc
print(len(result2[(result2['class']=='unacc')]))
#total da classificação geral, 10 foram classificados como acc
print(len(result2))
#Selecionar quando "people" for igual á more e "doors" for igual á 2 
#e "lug_boot" for igual a small,
#ou"lug_boot" for igual a med e "safety" igual a med

b1=vhigh_med[(vhigh_med['persons']=='more') & (vhigh_med['doors']=='2') & 
            ((vhigh_med['lug_boot']=='small') | ((vhigh_med['lug_boot']=='med') & 
            (vhigh_med['safety']=='med')))]

b2=vhigh_low[(vhigh_low['persons']=='more') & (vhigh_low['doors']=='2') & 
            ((vhigh_low['lug_boot']=='small') | ((vhigh_low['lug_boot']=='med') & 
            (vhigh_low['safety']=='med')))]


b3=high_high[(high_high['persons']=='more') & (high_high['doors']=='2') & 
            ((high_high['lug_boot']=='small') | ((high_high['lug_boot']=='med') & 
            (high_high['safety']=='med')))]


b4=high_med[(high_med['persons']=='more') & (high_med['doors']=='2') & 
            ((high_med['lug_boot']=='small') | ((high_med['lug_boot']=='med') & 
            (high_med['safety']=='med')))]


b5=high_low[(high_low['persons']=='more') & (high_low['doors']=='2') & 
            ((high_low['lug_boot']=='small') | ((high_low['lug_boot']=='med') & 
            (high_low['safety']=='med')))]


b6=med_high[(med_high['persons']=='more') & (med_high['doors']=='2') & 
            ((med_high['lug_boot']=='small') | ((med_high['lug_boot']=='med') & 
            (med_high['safety']=='med')))]


b7=low_vhigh[(low_vhigh['persons']=='more') & (low_vhigh['doors']=='2') & 
            ((low_vhigh['lug_boot']=='small') | ((low_vhigh['lug_boot']=='med') & 
            (low_vhigh['safety']=='med')))]

frames = [b1,b2, b3,b4,b5,b6,b7]
result3 = pd.concat(frames).drop_duplicates()
#total classificados como unacc
print(len(result3[(result3['class']=='unacc')]))
#total da classificação geral
print(len(result3))

#Selecionar quando "people" for igual á more ou 4 e "doors" for igual á 3,4 ou 5-more
#e "lug_boot" for igual a small e "safety" igual a med

c1=vhigh_med[((vhigh_med['persons']=='more') | (vhigh_med['persons']=='4')) & 
            ((vhigh_med['doors']=='3')|(vhigh_med['doors']=='4') | (vhigh_med['doors']=='5more')) &
            ((vhigh_med['lug_boot']=='small') & (vhigh_med['safety']=='med'))]

c2=vhigh_low[((vhigh_low['persons']=='more')| (vhigh_low['persons']=='4')) &
             ((vhigh_low['doors']=='3') | (vhigh_low['doors']=='4') |
             (vhigh_low['doors']=='5more')) & ((vhigh_low['lug_boot']=='small') &
             (vhigh_low['safety']=='med'))]

c3=high_high[((high_high['persons']=='more')| (high_high['persons']=='4')) &
             ((high_high['doors']=='3') | (high_high['doors']=='4') |
             (high_high['doors']=='5more')) & ((high_high['lug_boot']=='small') &
             (high_high['safety']=='med'))]

c4=high_med[((high_med['persons']=='more') | (high_med['persons']=='4')) & 
            ((high_med['doors']=='3')|(high_med['doors']=='4') | (high_med['doors']=='5more')) &
            ((high_med['lug_boot']=='small') & (high_med['safety']=='med'))]

c5=high_low[((high_low['persons']=='more') | (high_low['persons']=='4')) & 
            ((high_low['doors']=='3')|(high_low['doors']=='4') | (high_low['doors']=='5more')) &
            ((high_low['lug_boot']=='small') & (high_low['safety']=='med'))]

c6=med_high[((med_high['persons']=='more') | (med_high['persons']=='4')) & 
            ((med_high['doors']=='3')|(med_high['doors']=='4') | (med_high['doors']=='5more')) &
            ((med_high['lug_boot']=='small') & (med_high['safety']=='med'))]

c7=low_vhigh[((low_vhigh['persons']=='more') | (low_vhigh['persons']=='4')) & 
            ((low_vhigh['doors']=='3')|(low_vhigh['doors']=='4') | (low_vhigh['doors']=='5more')) &
            ((low_vhigh['lug_boot']=='small') & (low_vhigh['safety']=='med'))]

frames = [c1,c2, c3,c4,c5,c6,c7]
result4 = pd.concat(frames).drop_duplicates()
#total classificado como unacc
print(len(result4[(result4['class']=='unacc')]))
#total da classificação geral
print(len(result4))
#Selecionar quando "people" for igual á more e "doors" for igual á 2 
#e "lug_boot" for igual a small.

d1=med_med[(med_med['persons']=='more') & (med_med['doors']=='2') & (med_med['lug_boot']=='small')]

d2=med_low[(med_low['persons']=='more') & (med_low['doors']=='2') & (med_low['lug_boot']=='small')]

d3=low_high[(low_high['persons']=='more') & (low_high['doors']=='2') & 
             (low_high['lug_boot']=='small')]

d4=low_med[(low_med['persons']=='more') & (low_med['doors']=='2') & 
             (low_med['lug_boot']=='small')]

d5=low_low[(low_low['persons']=='more') & (low_low['doors']=='2') & 
             (low_low['lug_boot']=='small')]

frames = [d1,d2, d3,d4,d5]
result5 = pd.concat(frames).drop_duplicates()
#total classificado como unacc
print(len(result5[(result5['class']=='unacc')]))
#total da classificação geral
print(len(result5))
#jutando sub_tabelas e eleminando repetições
frames = [result1,result2, result3,result4,result5]
unacc_rule= pd.concat(frames)
unacc_rule=unacc_rule.drop_duplicates()



#visualização
unacc_record = { 'true_total': "1209", 'hits': len(unacc_rule)}
unacc_rule_visualization = 'Total verdadeiro UNACC é: {} e o número de acertos foi: {}'

print(unacc_rule_visualization.format(unacc_record['true_total'], unacc_record['hits']))



#Filtrando com "people" for diferente de 2 e "safety" for diferente de low
drop_person_2=data[(data['persons']!='2')]
drop_safety_low=drop_person_2[(drop_person_2['safety'] !='low')]#utilizado em códigos posteriores

# Filtrando combinação buying e maint

vhigh_med=drop_safety_low[(drop_safety_low['buying']=='vhigh') & (drop_safety_low['maint']=='med')]
vhigh_low=drop_safety_low[(drop_safety_low['buying']=='vhigh') & (drop_safety_low['maint']=='low')]
high_high=drop_safety_low[(drop_safety_low['buying']=='high') & (drop_safety_low['maint']=='high')]
high_med=drop_safety_low[(drop_safety_low['buying']=='high') & (drop_safety_low['maint']=='med')]
high_low=drop_safety_low[(drop_safety_low['buying']=='high') & (drop_safety_low['maint']=='low')]
med_high=drop_safety_low[(drop_safety_low['buying']=='med') & (drop_safety_low['maint']=='high')]
med_med=drop_safety_low[(drop_safety_low['buying']=='med') & (drop_safety_low['maint']=='med')]
med_low=drop_safety_low[(drop_safety_low['buying']=='med') & (drop_safety_low['maint']=='low')]
low_vhigh=drop_safety_low[(drop_safety_low['buying']=='low') & (drop_safety_low['maint']=='vhigh')]
low_high=drop_safety_low[(drop_safety_low['buying']=='low') & (drop_safety_low['maint']=='high')]
low_med=drop_safety_low[(drop_safety_low['buying']=='low') & (drop_safety_low['maint']=='med')]
low_low=drop_safety_low[(drop_safety_low['buying']=='low') & (drop_safety_low['maint']=='low')]

#Selecionar quando "people" for igual á 4 e "doors" for igual á 2 ou 3 
#e "lug_boot" for igual a small e "safety" igual a high,
#ou"lug_boot" for igual a med e "safety" igual a high,
#ou "lug_boot" igual a big


u1=vhigh_med[(vhigh_med['persons']=='4') & ((vhigh_med['doors']=='2')|(vhigh_med['doors']=='3')) &
           (((vhigh_med['lug_boot']=='small') & (vhigh_med['safety']=='high'))|
           ((vhigh_med['lug_boot']=='med') & (vhigh_med['safety']=='high'))|
           (vhigh_med['lug_boot']=='big'))]
#fazer big!low
u2=vhigh_low[(vhigh_low['persons']=='4') & ((vhigh_low['doors']=='2')|(vhigh_low['doors']=='3')) & 
            (((vhigh_low['lug_boot']=='small') & (vhigh_low['safety']=='high')) |
            ((vhigh_low['lug_boot']=='med') & (vhigh_low['safety']=='high'))|
            (vhigh_low['lug_boot']=='big'))]

u3=high_high[(high_high['persons']=='4') & ((high_high['doors']=='2')|(high_high['doors']=='3')) & 
            (((high_high['lug_boot']=='small') & (high_high['safety']=='high')) |
            ((high_high['lug_boot']=='med') & (high_high['safety']=='high'))|
            (high_high['lug_boot']=='big'))]

u4=high_med[(high_med['persons']=='4') & ((high_med['doors']=='2')|(high_med['doors']=='3')) &
           (((high_med['lug_boot']=='small') & (high_med['safety']=='high'))|
           ((high_med['lug_boot']=='med') & (high_med['safety']=='high'))|
            (high_med['lug_boot']=='big'))]

u5=med_high[(med_high['persons']=='4') & ((med_high['doors']=='2')|(med_high['doors']=='3')) &
           (((med_high['lug_boot']=='small') & (med_high['safety']=='high'))|
           ((med_high['lug_boot']=='med') & (med_high['safety']=='high'))|
           (med_high['lug_boot']=='big'))]

u6=high_low[(high_low['persons']=='4') & ((high_low['doors']=='2')|(high_low['doors']=='3')) &
           (((high_low['lug_boot']=='small') & (high_low['safety']=='high'))|
           ((high_low['lug_boot']=='med') & (high_low['safety']=='high'))|
           (high_low['lug_boot']=='big'))]


u7=low_vhigh[(low_vhigh['persons']=='4') & ((low_vhigh['doors']=='2')|(low_vhigh['doors']=='3')) &
           (((low_vhigh['lug_boot']=='small') & (low_vhigh['safety']=='high'))|
           ((low_vhigh['lug_boot']=='med') & (low_vhigh['safety']=='high'))|
           (low_vhigh['lug_boot']=='big'))]
#Juntar sub-tabelas

frames = [u1, u2, u3,u4,u5,u6,u7]
result2 = pd.concat(frames).drop_duplicates()
#total classificados como unacc
print(len(result2[(result2['class']=='acc')]))
#total da classificação geral
print(len(result2))
#Selecionar quando "people" for igual á more e "doors" for igual á 2 
#e "lug_boot" for igual a big e "safety" igual a med,
#ou "safety" igual a high

b1=vhigh_med[(vhigh_med['persons']=='more') & (vhigh_med['doors']=='2') & 
            ((vhigh_med['lug_boot']=='big')| ((vhigh_med['lug_boot']=='med') &
            (vhigh_med['safety']=='high')))]

b2=vhigh_low[(vhigh_low['persons']=='more') & (vhigh_low['doors']=='2') & 
            ((vhigh_low['lug_boot']=='big')|((vhigh_low['lug_boot']=='med') &
            (vhigh_low['safety']=='high')))]


b3=high_high[(high_high['persons']=='more') & (high_high['doors']=='2') & 
            ((high_high['lug_boot']=='big')|((high_high['lug_boot']=='med') &
            (high_high['safety']=='high')))]


b4=high_med[(high_med['persons']=='more') & (high_med['doors']=='2') & 
            ((high_med['lug_boot']=='big')|((high_med['lug_boot']=='med') &
            (high_med['safety']=='high')))]


b5=high_low[(high_low['persons']=='more') & (high_low['doors']=='2') & 
            ((high_low['lug_boot']=='big')|((high_low['lug_boot']=='med') & 
            (high_low['safety']=='high')))]

b6=med_high[(med_high['persons']=='more') & (med_high['doors']=='2') & 
            ((med_high['lug_boot']=='big')|((med_high['lug_boot']=='med') & 
            (med_high['safety']=='high')))]


b7=low_vhigh[(low_vhigh['persons']=='more') & (low_vhigh['doors']=='2') & 
            ((low_vhigh['lug_boot']=='big')|((low_vhigh['lug_boot']=='med') & 
            (low_vhigh['safety']=='high')))]

frames = [b1,b2, b3,b4,b5,b6,b7]
result3 = pd.concat(frames).drop_duplicates()
#total classificados como unacc
print(len(result3[(result3['class']=='acc')]))
#total da classificação geral
print(len(result3))

#Selecionar quando "people" for igual á more ou 4 e "doors" for igual á 4 ou 5-more
#e "lug_boot" for igual a small e "safety" igual a high,
#ou lug_boot igual a small

c1=vhigh_med[((vhigh_med['persons']=='more') | (vhigh_med['persons']=='4')) & 
            ((vhigh_med['doors']=='4') | (vhigh_med['doors']=='5more')) &
            (((vhigh_med['lug_boot']=='small') & (vhigh_med['safety']=='high'))|
            (vhigh_med['lug_boot']!='small'))]

c2=vhigh_low[((vhigh_low['persons']=='more')| (vhigh_low['persons']=='4')) &
             ( (vhigh_low['doors']=='4') |(vhigh_low['doors']=='5more')) &
             (((vhigh_low['lug_boot']=='small') &(vhigh_low['safety']=='high'))|
             (vhigh_low['lug_boot']!='small'))]

c3=high_high[((high_high['persons']=='more')| (high_high['persons']=='4')) &
             ((high_high['doors']=='4') |(high_high['doors']=='5more')) &
             (((high_high['lug_boot']=='small') &(high_high['safety']=='high'))|
             (high_high['lug_boot']!='small'))]

c4=high_med[((high_med['persons']=='more') | (high_med['persons']=='4')) & 
            ((high_med['doors']=='4') | (high_med['doors']=='5more')) &
            (((high_med['lug_boot']=='small') & (high_med['safety']=='high'))|
            (high_med['lug_boot']!='small'))]

c5=high_low[((high_low['persons']=='more') | (high_low['persons']=='4')) & 
            ((high_low['doors']=='4') | (high_low['doors']=='5more')) &
            (((high_low['lug_boot']=='small') & (high_low['safety']=='high'))|
            (high_low['lug_boot']!='small'))]

c6=med_high[((med_high['persons']=='more') | (med_high['persons']=='4')) & 
            ((med_high['doors']=='4') | (med_high['doors']=='5more')) &
            (((med_high['lug_boot']=='small') & (med_high['safety']=='high'))|
            (med_high['lug_boot']!='small'))]

c7=low_vhigh[((low_vhigh['persons']=='more') | (low_vhigh['persons']=='4')) & 
            ((low_vhigh['doors']=='4') |(low_vhigh['doors']=='5more')) &
            (((low_vhigh['lug_boot']=='small') & (low_vhigh['safety']=='high'))|
            (low_vhigh['lug_boot']!='small'))]

frames = [c1,c2, c3,c4,c5,c6,c7]
result4 = pd.concat(frames).drop_duplicates()
#total classificado como unacc
print(len(result4[(result4['class']=='acc')]))
#total da classificação geral
print(len(result4))
#Selecionar quando "people" for igual á more e "doors" for igual á 2 
#ou people igual a 4 e doors igual a 2

d1=med_med[(med_med['persons']=='more') & (med_med['doors']=='2') &
           (med_med['lug_boot']!='small')]

d2=med_low[(med_low['persons']=='more') & (med_low['doors']=='2') &
           ((med_low['lug_boot']=='med')& (med_med['safety']=='med'))]

d3=low_high[(low_high['persons']=='more') & (low_high['doors']=='2') & 
             (low_high['lug_boot']!='small')]

d4=low_med[(low_med['persons']=='more') & (low_med['doors']=='2') & 
             ((low_med['lug_boot']=='small') & (low_med['safety']=='med'))]

d5=low_low[(low_low['persons']=='more') & (low_low['doors']=='2') & 
             ((low_low['lug_boot']=='small')& (low_low['safety']=='med'))]

e1=med_med[(med_med['persons']=='4') & (med_med['doors']=='2') & ((med_med['lug_boot']!='big') |
         ((med_med['lug_boot']=='big')&(med_med['safety']=='med')))]

e2=med_med[(med_med['persons']=='4') & ((med_med['doors']=='2')|(med_med['doors']=='3')) & 
         ((med_med['lug_boot']!='big') & (med_med['safety']=='med'))]
              
frames = [d1,d2, d3,d4,d5,e1,e2]
result5 = pd.concat(frames).drop_duplicates()
#total classificado como unacc
print(len(result5[(result5['class']=='acc')]))
#total da classificação geral
print(len(result5))

#jutando sub_tabelas e eleminando repetições
frames = [result2, result3,result4,result5]
acc_rule= pd.concat(frames)
acc_rule=acc_rule.drop_duplicates()



#visualização
acc_record = { 'true_total': "384", 'hits': len(acc_rule)}
acc_rule_visualization = 'Total verdadeiro ACC é: {} e o número de acertos foi: {}'

print(acc_rule_visualization.format(acc_record['true_total'], acc_record['hits']))
#Filtrando apenas quando "buying" for igual a med ou low"
filter_good_buying=drop_safety_low[(drop_safety_low['buying']=='med')|
                                   (drop_safety_low['buying']=='low')]#utulizado posteriomente
#Filtrando apenas quando "maint" for igual a med ou low"
good_rule=filter_good_buying[(filter_good_buying['maint']=='med')|
          (filter_good_buying['maint']=='low')]

#eliminando duplicações
only_good=good_rule[(good_rule['class']=='good')].drop_duplicates()


#visualization
good_record = {
'true_total': "69",
'hits': len(only_good)}

good_rule_visualization = 'Total verdadeiro de good é: {} e o número de acertos foi: {}'
                            

print(good_rule_visualization.format(good_record ['true_total'],
                                        good_record['hits']))
#total de classificações
print(len(good_rule))
 #filtrando "maint" diferente de vhigh
filter_vgood_maint=drop_safety_low[(drop_safety_low['maint']!='vhigh')]

#filtrando "safety" igual a high
filter_safety=filter_vgood_maint[(filter_vgood_maint['safety']=='high')]

#filtrando "lug_boot" diferente de small
vgood_rule=filter_safety[(filter_safety['lug_boot']!='small')]

#eliminando duplicações
only_vgood=vgood_rule[(vgood_rule['class']=='vgood')].drop_duplicates()

#visualization
vgood_record = {
'true_total': "65",
'hits': len(only_vgood)}

vgood_rule_visualization = 'Total verdadeiro de vgood é: {} e o número de acertos foi: {}'
                            

print(vgood_rule_visualization.format(vgood_record ['true_total'],
                                        vgood_record['hits']))
#total de classificações
print(len(vgood_rule))