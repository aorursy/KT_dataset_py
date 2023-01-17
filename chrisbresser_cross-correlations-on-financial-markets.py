!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import collections
import os
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")
def fincorr(train_date,valid_date,test_date,train_threshold,valid_threshold,lag):
    
    # DATA
    nasdaq = pd.read_csv("../input/markets/nasdaq2.csv", index_col=0)
    clean = nasdaq.copy()
    train = clean[test_date-(3+valid_date+train_date):test_date-(2+valid_date)]
    valid = clean[test_date-(3+valid_date):test_date-2]
    test = clean[test_date-2:test_date]
    full = clean[test_date-(3+valid_date+train_date):test_date]

    # TRAINING
    train_results = {}
    for symbol in tqdm(train.columns): 
        df = train.copy()
        df[symbol] = df[symbol].shift(lag)
        corr_matrix = np.corrcoef(df[lag:], rowvar=False)[df.columns.get_loc(symbol)]
        corr_series = pd.Series(corr_matrix, index =df.columns) 
        high_corr = corr_series[(corr_series>train_threshold)&(corr_series<0.99)].to_dict() 
        train_results[symbol] = high_corr  
    train_results = {k: v for k, v in train_results.items() if v}

    # VALIDATION
    valid_symbols = []
    valid_results = {}
    for k, v in train_results.items():
        x = []
        x.append(k)
        for k1, v1 in v.items():
            x.append(k1)
        valid_symbols.append(x)
    for s in valid_symbols:
        symbol = s[0]
        df = valid[s]
        df[symbol] = df[symbol].shift(lag)
        df = df.iloc[lag:]
        x = df.corr()[symbol]
        y = x[(x>valid_threshold)&(x<1)].to_dict()
        valid_results[symbol] = y
    valid_results = {k: v for k, v in valid_results.items() if v}

    # RESULTS
    test_symbols = []
    tp,tn,fp,fn = 0,0,0,0
    for k, v in valid_results.items():
        x = []
        x.append(k)
        for k1, v1 in v.items():
            x.append(k1)
        test_symbols.append(x)
    for s in test_symbols:
        symbol = s[0]
        df = test[s]
        df[symbol] = df[symbol].shift(lag)
        df = df.iloc[lag]
        for sym in s[1:]:
            if (df[0] > 0) & (df[sym] > 0):
                tp += 1
            if (df[0] < 0) & (df[sym] < 0):
                tn += 1
            if (df[0] > 0) & (df[sym] < 0):
                fp += 1
            if (df[0] < 0) & (df[sym] > 0):
                fn +=1            

    def accuracy(tp,tn,fp,fn):
        try:
            return (tp+tn)/(tp+tn+fp+fn) 
        except ZeroDivisionError:
            return 0                           
    accuracy = round(accuracy(tp,tn,fp,fn),2)
    
    def precision(tp,fp):
        try:
            return tp/(tp+fp) 
        except ZeroDivisionError:
            return 0
    precision = round(precision(tp,fp),2)
    
    def recall(tp,tn):
        try:
            return tp/(tp+tn)
        except ZeroDivisionError:
            return 0
    recall = round(recall(tp,tn),2)
    
    def f1(precision,recall):
        try:
            return 2 * (precision*recall)/(precision+recall)
        except ZeroDivisionError:
            return 0
    F1 = round(f1(precision,recall),2)
    
    print('Trained pairs:', len(train_results))
    print('Validation pairs:', len(valid_results), '\n')
    print('TP:',tp, '|', 'TN:',tn, '|', 'FP:', fp, '|', 'FN:', fn, '\n')            
    print('Accuracy:', accuracy) 
    print('Precision:', precision) 
    print('Recall:', recall) 
    print('F1:', F1) 

#     # GRAPHS
#     with PdfPages('Graphs.pdf') as export_pdf:
#         for k, v in valid_results.items():
#             for k1, v1 in v.items():        
#                 df = full[[k, k1]]
#                 df[k] = df[k].shift(1)
#                 ax = df.plot(figsize=(10,4), grid=True, color=['blue', 'red','black'])
#                 ymin, ymax = ax.get_ylim()
#                 xmin, xmax = ax.get_xlim()
#                 ax.vlines([test_date-1,test_date-(2+valid_date)], ymin=ymin, ymax=ymax, color='black')
#                 ax.hlines(0, xmin=xmin, xmax=xmax, color='black')
#                 export_pdf.savefig()
    return len(train_results), len(valid_results), tp, tn, fp, fn, accuracy, precision, recall, F1
# fincorr(8,16,24,0.7,0.7,1)
# HYPERPARAMETERS OPTIMIZATION
train_range = np.arange(8,21)
valid_range = np.arange(2,9)
d = {}

for x in tqdm(train_range):
    temp = {}
    for i in valid_range:
        temp[i] = fincorr(30-i-x,30-i,30,0.7,0.7,1)
    d[x] = temp
    
# metrics = pd.DataFrame.from_dict(d, orient='index', columns=['Trained Pairs','Validation Pairs','TP','TN','FP','FN','Accuracy','Precision','Recall','F1'])
# METRICS DICTIONARY TO DATAFRAME
metrics = pd.DataFrame.from_dict({(i,j): d[i][j] for i in d.keys() for j in d[i].keys()}, orient='index')
metrics.columns = ['Trained Pairs','Validation Pairs','TP','TN','FP','FN','Accuracy','Precision','Recall','F1']
#metrics.sort_values('Accuracy', ascending=False)
#metrics.to_csv('metrics.csv')
metrics
# METRICS TO GRAPH
with PdfPages('Graphs.pdf') as export_pdf:
    metrics.iloc[:,6:7].plot()
    #plt.suptitle("Metrics Training Days 7-20 nasdaq[30:]")
    export_pdf.savefig()
    metrics.iloc[:,:6].plot()
    #plt.suptitle("Metrics Training Days 7-20 nasdaq[30:]")
    export_pdf.savefig()
fincorr(8,3,45,0.7,0.7,1)
nasdaq = pd.read_csv("../input/markets/nasdaq2.csv", index_col=0)
clean = nasdaq.copy()
train = clean[0:10]
train_results = {}
d = (train >= 0)
for symbol in tqdm(train.columns):
    df = d.copy()
    df[symbol] = df[symbol].shift(1).astype('bool')
    s = df[1:].corr()[symbol]
    y = s[(s>0.7)][1:].to_dict()
    train_results[symbol] = y
train_results = {k: v for k, v in train_results.items() if v}
def finposneg(train_date,valid_date, test_date,train_threshold,valid_threshold,lag):
    
    # DATA
    nasdaq = pd.read_csv("../input/markets/nasdaq2.csv", index_col=0)
    clean = nasdaq.copy()
    train = clean[0:train_date]
    valid = clean[train_date-1:valid_date]
    test = clean[valid_date-1:test_date]
    full = clean[:test_date]

    train_results = {}
    d = (train >= 0)
    for symbol in tqdm(train.columns):
        df = d.copy()
        df[symbol] = df[symbol].shift(1).astype('bool')
        s = df[1:].corr()[symbol]
        y = s[(s>0.7)][1:].to_dict()
        train_results[symbol] = y
    train_results = {k: v for k, v in train_results.items() if v}
    
    valid_symbols = []
    for k, v in x.items():
        x = []
        x.append(k)
        for k1, v1 in v.items():
            x.append(k1)
        valid_symbols.append(x)

    valid_results = {}
    for s in tqdm(valid_symbols[0]):
        symbol = s[0]
        df = valid[s]
        d = (df >= 0)
        d[symbol] = d[symbol].shift(1).astype('bool')
        s = d.corr()[symbol]
        y = s[(s>0.9)].drop(symbol).to_dict()
        train_results[symbol] = y
    valid_results = {k: v for k, v in valid_results.items() if v}
    valid_results

    return train_results
x=finposneg(10,18,24,0.7,0.7,1)
valid_symbols = []
for k, v in valid.items():
    x = []
    x.append(k)
    for k1, v1 in v.items():
        x.append(k1)
    valid_symbols.append(x)
    
valid_results = {}
for s in tqdm(valid_symbols[0]):
    symbol = s[0]
    df = valid[s]
    d = (df >= 0)
    d[symbol] = d[symbol].shift(1).astype('bool')
    s = d.corr()[symbol]
    y = s[(s>0.9)].drop(symbol).to_dict()
    train_results[symbol] = y
valid_results = {k: v for k, v in valid_results.items() if v}
valid_results