import os
import pprint 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import OrderedDict 
from ipywidgets import interact
import ipywidgets as widgets

from matplotlib import pyplot as plt
from scipy import stats
dataset_dir = '../input/'
class FeatureDescriptions:
    def __init__(self):
        self.descriptions = OrderedDict()
        self.subdescriptions = OrderedDict()
        
    def __repr__(self):
        return self.descriptions.__repr__()
    
    def add_description(self, key, dvalue):
        self.descriptions[key] = dvalue
        
    def add_subdescription(self, d_key, subd_key, subd_value):
        d = self.subdescriptions[d_key]
        d[subd_key] = subd_value
        
    def generate_subd_keys(self):
        for key in self.descriptions:
            self.subdescriptions[key] = OrderedDict()
def parse_data_subdescription(line):
    l = line.split()
    if len(l) != 0:
        key = l[0]
        val = l[1]
        for w in l[2:]:
            val += ' ' + w
        return (key,val)
    return (None, None)

def parse_data_description(abs_file_path):
    listOfFeatures = FeatureDescriptions()
    # Some set of the features are not in the data description file
    listOfFeatures.descriptions['Id'] = 'House identification number'
    
    file = open(abs_file_path, 'r')
    for line in file:
        if (':' in line) and ('2nd level' not in line):
            key, d_value = line.split(':')
            listOfFeatures.add_description(key,d_value.strip())
    file.close()
    
    # Some set of the features are not in the data description file
    listOfFeatures.descriptions['SalePrice'] = 'Market value of house'
    
    listOfFeatures.generate_subd_keys()
    
    file = open(abs_file_path, 'r')
    
    description_key = None
    
    for line in file:
        if (':' in line) and ('2nd level' not in line):
            description_key, d_value = line.split(':')
        else:
            subd_key, subd_val = parse_data_subdescription(line)
            if (subd_key != None) and (subd_val != None):
                listOfFeatures.add_subdescription(description_key, subd_key,subd_val)
    file.close()
                
    return listOfFeatures
features = parse_data_description(dataset_dir+'data_description.txt')
features_dtype = {}
features_na_values = {}
for feature in features.descriptions:
    if len(features.subdescriptions[feature]) > 0:
        features_dtype[feature] = object
        features_na_values[feature] = ['', '#N/A', '#N/A','N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null']
    else:
        features_na_values[feature] = ['NA', 'NaN']
data = pd.read_csv(dataset_dir+'train.csv', keep_default_na=False, 
                   na_values=features_na_values,
                   dtype=features_dtype)
def featureFrequencies(feature,data_values):
    keys, freqs = np.unique(data_values,return_counts=True)

    dict_freqs = dict(zip(keys,freqs))
    sorted_freqs = []
    for key in features.subdescriptions[feature]:
        try:
            sorted_freqs.append(dict_freqs[key])
        except KeyError:
            sorted_freqs.append(0)
    return features.subdescriptions[feature].keys(), sorted_freqs
style = {'description_width': 'initial'}

list_of_features = widgets.Dropdown(
    options=data.columns,
    value=data.columns[1],
    description='Feature:',
    disabled=False
)


price_split_bool = widgets.Checkbox(
    value=True,
    description='Split Data at Price Point ($)?',
    disabled=False
)

price_split = widgets.BoundedFloatText(
    value=250000,
    min=min(data['SalePrice']),
    max=max(data['SalePrice']),
    description='Split SalePrice at ($):',
    disabled=False,
    style=style
)
    
def plot_against_price(feature, split_bool, split):
    sale_price = data['SalePrice']

    plt.close('all')
    
    if (len(features.subdescriptions[feature]) > 0):
        plt.figure(figsize=(18,9))
        sp = []
        for key in features.subdescriptions[feature].keys():
            d = data[feature]
            if type(key) != type(d[0]):
                key = np.int64(key)

            d_fil = d[d==key]
            sp_fil = sale_price[d==key]
            sp.append(sp_fil)

        plt.subplot(1,2,1)
        plt.title('Box Plot of Feature: ' + feature)
        plt.boxplot(sp,labels=features.subdescriptions[feature].keys())
        plt.xlabel('Labels')
        plt.ylabel('Sale Price')
        
        
        plt.subplot(1,2,2)
        plt.title('Frequency Plot of Feature: ' + feature)
        plt.ylabel('Frequencies')
        plt.xlabel('Labels')
        
        
        if not split_bool:
            sorted_keys, sorted_frequencies = featureFrequencies(feature,data[feature])
            N=np.arange(len(sorted_keys))
            plt.bar(N,sorted_frequencies)
            plt.xticks(N, sorted_keys, rotation=90)
        else:
            sorted_k_lt, sorted_f_lt  = featureFrequencies(feature,data[feature][data['SalePrice']<=split])
            sorted_k_gt, sorted_f_gt = featureFrequencies(feature,data[feature][data['SalePrice']>split])

            width=0.3
            N=np.arange(len(sorted_k_lt))
            
            p1 = plt.bar(N+width,sorted_f_lt,width,label='<=' + str(split))
            p2 = plt.bar(N,sorted_f_gt,width,label='>' + str(split))
            plt.xticks(N+width/2, sorted_k_lt, rotation=90)
            plt.legend()
    else:
        plt.figure(figsize=(9,9))
        if not split_bool:
            plt.plot(sale_price,data[feature],'o')
        else:
            plt.plot(sale_price[data['SalePrice']>split],data[feature][data['SalePrice']>split],'o',label='>'+str(split)+'$')
            plt.plot(sale_price[data['SalePrice']<=split],data[feature][data['SalePrice']<=split],'o',label='<=' + str(split)+'$')
            plt.legend()
            
        plt.xlabel('Sale Price ($)')
        plt.ylabel(feature)
        
    plt.show()
    
    print(feature, ':', features.descriptions[feature])
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(features.subdescriptions[feature])
    
interact(plot_against_price, 
         feature=list_of_features,
         split_bool=price_split_bool,
         split=price_split
        )
