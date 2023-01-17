# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_dictionary_loc = '../input/CAB_data_dictionary.xlsx'
data_dic = pd.read_excel(data_dictionary_loc, dtype = object)
data_dic['File Content Description'] #well, how to import the correct column width? can be viewed using other programs
data_dic
data_uttarakhand = pd.read_csv('../input/CAB_05_UT.csv', low_memory = False) 
#needed to specify low_memory because columns (14, 43 had mixed types)
data_uttarakhand.head()
data_rajasthan = pd.read_csv('../input/CAB_08_RJ.csv', low_memory = False) 
data_u_pradesh = pd.read_csv('../input/CAB_09_UP.csv', low_memory = False) 
data_bihar = pd.read_csv('../input/CAB_10_BH.csv', low_memory = False) 
data_assam = pd.read_csv('../input/CAB_18_AS.csv', low_memory = False) 
data_jharkhand = pd.read_csv('../input/CAB_20_JH.csv', low_memory = False) 
data_odisha = pd.read_csv('../input/CAB_21_OR.csv', low_memory = False) 
data_chhattisgarh = pd.read_csv('../input/CAB_22_CT.csv', low_memory = False) 
data_m_pradesh = pd.read_csv('../input/CAB_23_MP.csv', low_memory = False)
data_u_pradesh.head()
states_var = pd.Series([data_uttarakhand, data_rajasthan, data_u_pradesh, data_bihar, data_assam, data_jharkhand, data_odisha, data_chhattisgarh, data_m_pradesh])
states_str = ["Uttarakhand", "Rajasthan", "Uttar Pradesh", "Bihar","Assam", "Jharkhand", "Odisha", "Chhattisgarh", "Madhya Pradesh"]
summary = pd.DataFrame(columns = ["state", "men", "women", "avg_age"])
summary.state = states_str
#code 1 for male, 2 for female
summary.men = states_var.apply(lambda state : pd.value_counts(state.sex)[1])
summary.women = states_var.apply(lambda state : pd.value_counts(state.sex)[2])
summary.avg_age = states_var.apply(lambda state : 2018 - state.year_of_birth.mean())
summary
from matplotlib import pyplot as plt
# with %matplotlib inline you turn on the immediate display.
%matplotlib inline
heme_dict = {-1: "NA", 1: "Measured", 2: "Not measured", 3: "Not measured", 4:"Not measured"}
ill_dict = {-1: "NA", 0:"No illness", 1: "Diarrhoea/dysentery", 2: "Respiratory", 3:"Fever", 4:"Other"}
sugar_dict = {-1: "NA", 1: "Measures", 2: "Not measured", 3:"Not measured", 4:"Not measured"}
def plot_illness(state, name):
    # Figure can contain multiple plots and you can also set params
    fig = plt.figure(figsize=[6, 16])

    # Multiple plots on one figure - add_subplot(nrows, ncols, index)
    df = pd.value_counts(state['haemoglobin_test']).to_frame()
    df.index = pd.Series(df.index).map(heme_dict).tolist()
    fig.add_subplot(311).bar(df.index.tolist() ,df['haemoglobin_test'].values.tolist())
    plt.title(name+" diabetes")
    
    df = pd.value_counts(state['illness_type']).to_frame()
    df.index = pd.Series(df.index).map(ill_dict).tolist()
    #fig.add_subplot(312).bar(df.index.tolist(), df["illness_type"].tolist())
    #plt.title(name+" acute illness in children")
    
    df = pd.value_counts(state['haemoglobin_test']).to_frame()
    df.index = pd.Series(df.index).map(heme_dict).tolist()
    fig.add_subplot(312).bar(df.index.tolist() ,df['haemoglobin_test'].tolist())
    plt.title(name+" haemoglobin")
    plt.show()
   
for i in range(len(states_var)):
    plot_illness(states_var[i], states_str[i])