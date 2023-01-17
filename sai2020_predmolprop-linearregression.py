# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('/kaggle/input'))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/predmolprop-featureengineering-finaltrain/train_extend.csv")

test = pd.read_csv("../input/predmolprop-featureengineering-finaltest/test_extend.csv")

train.columns
# Encode atoms

def encode_atoms(df):

    type2encoding = {'H': 1,'C': 2,'N': 3,'O': 4,'F': 5,'': -1}

    df['atom_end_type']=df.atom_end_type.map(type2encoding)

    df['atom_2_type']=df.atom_2_type.map(type2encoding)

    df['atom_3_type']=df.atom_3_type.map(type2encoding)

    return df



train = encode_atoms(train)

test = encode_atoms(test)



pd.set_option('display.max_columns', None)

train.head()
train_target = train['scalar_coupling_constant']



# Remove some features

train_pop_list = ['id','molecule_name', 'atom_index_0','atom_index_1', 'num_bonds','bond_1', 'atom_0_type2', 'atom_end_type2', 'scalar_coupling_constant']

test_pop_list = ['molecule_name', 'atom_index_0','atom_index_1', 'num_bonds','bond_1', 'atom_0_type2', 'atom_end_type2'] # Keep ID as this is needed for submission



train_features = train.drop(columns=train_pop_list)

test_features = test.drop(columns=test_pop_list)



# Replace NaNs with -1

train_features.fillna(value =-1,inplace= True)

test_features.fillna(value =-1,inplace= True)

test.fillna(value=-1,inplace=True)



pd.set_option('display.max_columns', None)

train_features.head()
# First train a regressor on 75% of the data and use the rest for validation



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sns



mae_dict = {}; train_mae_dict = {}; mean_dict = {}; regr_dict = {};

mae_total = 0;



types = train.type.unique()

# plt.figure(); [fig, axs] = plt.subplots(2,4,figsize=(12, 10));

for mol_type in types:

    print('Now training type: '+str(mol_type))

    

    # Prepare data for the model

    train_by_type = train_features[train_features['type']==mol_type]

    train_by_type.pop('type')

    # train_by_type.drop(columns=['mu','spin','path_count']) # Remove some additional features for now

    target_by_type = train[train['type']==mol_type].scalar_coupling_constant

    

    train_X, val_X, train_Y, val_Y = train_test_split(train_by_type, target_by_type,test_size=0.2,random_state=42)

    

    mae_dict[mol_type] = []; train_mae_dict[mol_type] = []; mean_dict[mol_type] = []; regr_dict[mol_type] = [];

    

    # Make model

    regr_model = LinearRegression(copy_X=True, n_jobs=-1)

    fit = regr_model.fit(train_X, train_Y)

    

    # Predict

    pred_Y = regr_model.predict(val_X)

    

    # Calculate MSE

    mae = mean_absolute_error(val_Y, pred_Y) # MSE for validation data

    train_mae = mean_absolute_error(train_Y, regr_model.predict(train_X)) # MSE for the training data

    mean = sum(val_Y)/len(val_Y) # Mean of the validation data

    print('val mae: '+str(mae)+'train mae: '+str(train_mae)+'mean: '+str(mean))

    

    # Save all MSEs

    mae_dict[mol_type].append(mae)

    train_mae_dict[mol_type].append(train_mae)

    mean_dict[mol_type].append(mean)

    regr_dict[mol_type].append(fit)



    # Calculate total MSE

    min_mae = min(mae_dict[mol_type])

    # num_val = len(val_Y.index)

    mae_total += np.log(min_mae)

    print('mae_total: '+str(mae_total))

    

    # axs[j//4, ((j+1)%4)-1].plot(val_Y, pred_Y, 'o')

    # axs[j//4, ((j+1)%4)-1].set_title(mol_type)

    # g = sns.FacetGrid(pd.DataFrame({'type':mol_type,'scalar_coupling_constant': val_Y,'predictions':pred_Y}), 

    #                      col="type", col_order = types,sharex=False,sharey=False)

    # g.map(sns.scatterplot, "scalar_coupling_constant","predictions")

    

    plt.figure(figsize=[6,6])

    sns.scatterplot(x=val_Y, y=pred_Y)

    plt.plot(val_Y,val_Y,color='black')

    plt.title(mol_type)

    plt.xlabel('scalar coupling constant')

    plt.ylabel('predicted value')

    

    lims = plt.xlim()

    width=lims[1]-lims[0]

    lims=lims[0]-0.1*width,lims[1]+0.1*width

    plt.xlim(lims)

    plt.ylim(lims)

#    xlim=plt.xlim()

#    ylim=plt.ylim()

#    plt.xlim(min(xlim[0],ylim[0]),max(xlim[1],ylim[1]))

#    plt.ylim(min(xlim[0],ylim[0]),max(xlim[1],ylim[1]))

    plt.show()



mae_total = mae_total/8

print('Log MAE using a different linear regressor for each type: '+str(mae_total))

# plt.show()
