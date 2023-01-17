import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn

import warnings
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sns.set(font_scale=1.0)


raw_data = pd.read_csv('../input/complete-pokemon-dataset-updated-090420/pokedex_(Update.04.20).csv')

!pip install pydot
!pip install pydotplus
!pip install pydot-ng 
raw_data.head()
raw_data.info()
raw_data.egg_type_1.value_counts()
raw_data.describe().T
nulls_summary = pd.DataFrame(raw_data.isnull().any(), columns=['Nulls'])   
nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(raw_data.isnull().sum())   
nulls_summary['Num_of_nulls [%]'] = round((raw_data.isnull().mean()*100),2)   
print(nulls_summary) 
raw_data.describe()
raw_data.dropna(axis=0, subset=['egg_type_1'], inplace=True)
raw_data.dropna(axis=0, subset=['percentage_male'], inplace=True)
raw_data.dropna(axis=0, subset=['egg_cycles'], inplace=True)
raw_data.dropna(axis=0, subset=['growth_rate'], inplace=True)
raw_data.reset_index()
data=raw_data[[ 'egg_type_1','percentage_male','egg_cycles' ,'growth_rate' ,'type_number']]
data.head()
data['percentage_male'].value_counts()
def create_np_array_from_input_list(input_list,output_type):
    np_target = []
    
    entries = []
    entries_idx = []
    for entry in input_list:
        duplicate = 0
        for active_entry in entries:
            if entry == active_entry:
                duplicate = 1
        
        if duplicate == 0:
            entries.append(entry)
        
        no_entries = len(entries)
        
    for i in range(0,no_entries):
        entries_idx.append(i)
        
    for entry in input_list:
        for i in range(0,no_entries):
            if entry == entries[i]:
                np_target.append(entries_idx[i])
                
    if output_type == 'numpy':
        return(np_target)
    elif output_type == 'categories':
        return(entries)
    else:
        raise ValueError('output_type must be \'numpy\' or \'categories\'')
np_data = create_np_array_from_input_list(data['egg_type_1'],'numpy')
cats = create_np_array_from_input_list(data['egg_type_1'],'categories')
data = data.reset_index()
data_copy = data.copy()

for i in range(0,len(np_data)):
    data_copy.at[i,'egg_type_1'] = np_data[i]


data_copy
for i in range(0,len(np_data)):
    data_copy.at[i,'growth_rate'] = np_data[i]
data_copy
data_copy.info()
for  col in ['egg_type_1', 'growth_rate']:
  data_copy[col]=data_copy[col].astype('int')
data_copy.corr()
target = data_copy['egg_type_1']
data_copyb = data_copy[['percentage_male','egg_cycles']].to_numpy()
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_copy, target)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.svm import SVC

classifier = SVC(C=1.0, kernel='linear')

classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)
classifier = SVC(C=1.0, kernel='rbf')

classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=1, random_state=42)
classifier.fit(data_copy, target)
from sklearn.tree import DecisionTreeClassifier, export_graphviz


from io import StringIO
#from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
def make_decision_tree(max_depth, data_copy,target):
    # trenowanie modelu
    classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    classifier.fit(data_copyb, target)

    # eksport grafu drzewa
    dot_data = StringIO()
    export_graphviz(classifier,
                   out_file=dot_data,
                   
                   
                   special_characters=True,
                   rounded=True,
                   filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('graph.png')
    
    # obliczenie dokładności
    acc = classifier.score(data_copy, target) 

    # wykreślenie granic decyzyjnych
    colors='#f1865b,#31c30f,#64647F,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf'
    plt.figure(figsize=(12, 8))
    ax = plot_decision_regions(data_copyb, target, classifier, legend=0, colors=colors)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, cats, framealpha=0.3)
    plt.xlabel('percentage_male')
    plt.ylabel('egg_cycles')
    plt.title(f'Drzewo decyzyjne: max_depth={max_depth}, accuracy={acc * 100:.2f}')

    return Image(graph.create_png(), width=200 + max_depth * 120)

from mlxtend.plotting import plot_decision_regions
from IPython.display import Image
max_depth=2
targetb=target.to_numpy()
import pydotplus
make_decision_tree(max_depth, data_copyb,targetb)
max_depth=3
make_decision_tree(max_depth, data_copyb,targetb)
max_depth=4
make_decision_tree(max_depth, data_copyb,targetb)
max_depth=7
make_decision_tree(max_depth, data_copyb,targetb)