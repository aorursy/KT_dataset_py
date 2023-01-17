# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
train.columns.values
train[['Product_Info_1','Product_Info_2', 'Product_Info_3','Product_Info_4',
       'Product_Info_5','Product_Info_6','Product_Info_7' ]].head()
train[['Employment_Info_1','Employment_Info_2', 
       'Employment_Info_3', 'Employment_Info_4',
       'Employment_Info_5', 'Employment_Info_6']].head()
train[['InsuredInfo_1',
       'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5',
       'InsuredInfo_6', 'InsuredInfo_7']].head()
train[['Insurance_History_1',
       'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4',
       'Insurance_History_5', 'Insurance_History_7', 'Insurance_History_8',
       'Insurance_History_9']].head()
train[['Family_Hist_1', 'Family_Hist_2',
       'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']].head()
train[['Medical_History_1', 'Medical_History_2', 'Medical_History_3',
       'Medical_History_4', 'Medical_History_5', 'Medical_History_6',
       'Medical_History_7', 'Medical_History_8', 'Medical_History_9',
       'Medical_History_10', 'Medical_History_11', 'Medical_History_12',
       'Medical_History_13', 'Medical_History_14', 'Medical_History_15',
       'Medical_History_16', 'Medical_History_17', 'Medical_History_18',
       'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
       'Medical_History_22', 'Medical_History_23', 'Medical_History_24',
       'Medical_History_25', 'Medical_History_26', 'Medical_History_27',
       'Medical_History_28', 'Medical_History_29', 'Medical_History_30',
       'Medical_History_31', 'Medical_History_32', 'Medical_History_33',
       'Medical_History_34', 'Medical_History_35', 'Medical_History_36',
       'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
       'Medical_History_40', 'Medical_History_41']].head()
from sklearn.preprocessing import Imputer

def parse_data(X):
    
    X['BMI_Ins_age'] = X.BMI*X.Ins_Age
    
    X['Product_Info2_let'] =X.Product_Info_2.str[0]
    X['Product_Info2_num'] = X.Product_Info_2.str[1]
    
    X['Product_Info2_let'] = pd.factorize(X.Product_Info2_let)[0]+1
    X['Product_Info_2'] = pd.factorize(X.Product_Info_2)[0]+1
    
    X['Medical_KW'] = X[['Medical_Keyword_1',
       'Medical_Keyword_2', 'Medical_Keyword_3', 'Medical_Keyword_4',
       'Medical_Keyword_5', 'Medical_Keyword_6', 'Medical_Keyword_7',
       'Medical_Keyword_8', 'Medical_Keyword_9', 'Medical_Keyword_10',
       'Medical_Keyword_11', 'Medical_Keyword_12', 'Medical_Keyword_13',
       'Medical_Keyword_14', 'Medical_Keyword_15', 'Medical_Keyword_16',
       'Medical_Keyword_17', 'Medical_Keyword_18', 'Medical_Keyword_19',
       'Medical_Keyword_20', 'Medical_Keyword_21', 'Medical_Keyword_22',
       'Medical_Keyword_23', 'Medical_Keyword_24', 'Medical_Keyword_25',
       'Medical_Keyword_26', 'Medical_Keyword_27', 'Medical_Keyword_28',
       'Medical_Keyword_29', 'Medical_Keyword_30', 'Medical_Keyword_31',
       'Medical_Keyword_32', 'Medical_Keyword_33', 'Medical_Keyword_34',
       'Medical_Keyword_35', 'Medical_Keyword_36', 'Medical_Keyword_37',
       'Medical_Keyword_38', 'Medical_Keyword_39', 'Medical_Keyword_40',
       'Medical_Keyword_41', 'Medical_Keyword_42', 'Medical_Keyword_43',
       'Medical_Keyword_44', 'Medical_Keyword_45', 'Medical_Keyword_46',
       'Medical_Keyword_47', 'Medical_Keyword_48']].sum(axis = 1)
    
    X['Na_Num'] = X.isnull().sum(axis = 1)



    
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    col = X.columns.values    
    X = pd.DataFrame(imp.fit_transform(X))
    X.columns = col
    

    return X
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
X = parse_data(train)
y = X.Response

columns_to_drop = ['Id', 'Response']

from sklearn.ensemble import ExtraTreesClassifier
rf = ExtraTreesClassifier(n_estimators=300,
                              random_state=0)
rf.fit(X.drop(columns_to_drop, axis = 1), y)
importances =pd.DataFrame({'features' :X.drop(columns_to_drop, axis = 1).columns,
                           'importances' : rf.feature_importances_})
importances.sort_values(by = 'importances', ascending = False).head(20)

importances.sort_values(by = 'importances', ascending = False).tail(20)

#plot importances

import matplotlib.pyplot as plt
%matplotlib inline

importances.sort_values(by = 'importances', ascending = True, inplace = True)
val = importances.importances*100    # the bar lengths
pos = np.arange(importances.shape[0])+.5 

plt.figure(figsize = (13,28))
plt.barh(pos,val, align='center')
plt.yticks(pos, importances.features.values)
plt.xlabel('Importances')
plt.title('Features importances')
plt.grid(True)
#cumsum of importances
importances.sort_values(by = 'importances', ascending = False, inplace = True)

importances['cumul'] = np.cumsum(importances.importances, axis = 0)
importances.sort_values(by = 'importances', ascending = True, inplace = True)

val = importances.cumul*100    # the bar lengths
pos = np.arange(importances.shape[0])+.5 

plt.figure(figsize = (13,28))
plt.barh(pos,val, align='center')
plt.yticks(pos, importances.features.values)
plt.xlabel('Importances')
plt.title('Features importances')
plt.grid(True)
for i in np.arange(50,100,5):
    print('Nombre de variables pour avoir {0} % d\' \"importance\" des variables  : {1} sur {2}'.format(i,importances.features[importances.cumul<i/100].shape[0],
                                                                                                    importances.features.shape[0]))
#Variables ro remove to get X % of importances 

X = 90

importances.features[importances.cumul>X/100].values
