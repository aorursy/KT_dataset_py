import seaborn as sns

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt

from pandas import DataFrame,read_csv,isnull,notnull

from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential

from keras.layers import Dense,Activation

from keras.wrappers.scikit_learn import KerasRegressor

from numpy import log# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd

from sklearn.model_selection import KFold

import numpy as np

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.preprocessing import Normalizer

# Any results you write to the current directory are saved as output.
train=read_csv('../input/train.csv',header='infer')
def find_category_cols(data):

    uniques={}

    for col in data:

            uniques[col]={'Unique_Values':len(pd.unique(data[col])),'Max':np.max(data[col]),'Min':np.min(data[col]),'Missing':len(data[pd.isnull(data[col])])}

    return uniques

def cat_to_num(data):

    cat_strings=data.dtypes[data.dtypes=='object'].index

    cat_strings=cat_strings[cat_strings!='timestamp']

    for col in cat_strings:

        data[col]=pd.Categorical.from_array(data[col]).labels

    return data
#Unique_Info=pd.DataFrame(find_category_cols(train)).T

#Unique_Info.sort_values(by='Unique_Values',ascending=True)

train=cat_to_num(train)

train.head()
house_features=train.iloc[:,:13]

house_features['price_doc']=train['price_doc']
house_features=train.iloc[:,:13]

house_features['price_doc']=train['price_doc']
features=train.columns[(train.columns!='id')&(train.columns!='price_doc')&(train.columns!='timestamp')]
nn = MLPRegressor(

    hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',

    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,

    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



#n = nn.fit(train[features], train['price_doc'])
np.array(train)
def adjust_sq_ft(house_feat):

    no_sq=(house_feat.full_sq==0)|(isnull(house_feat['full_sq']))

    #missing_full=house_feat[no_sq]

    no_kitch=(house_feat.kitch_sq==0)|(isnull(house_feat['kitch_sq']))

    #missing_kitch=house_feat[no_kitch]

    no_l_sq=(house_feat.life_sq==0)|(isnull(house_feat['life_sq']))

    #missing_life=house_feat[no_l_sq]

    has_all_sq_ft=(house_feat.full_sq!=0)&(notnull(house_feat['full_sq']))&(house_feat.kitch_sq!=0)&(notnull(house_feat['kitch_sq']))&(house_feat.life_sq!=0)&(notnull(house_feat['life_sq']))&((house_feat.full_sq-house_feat.kitch_sq)>0)&((house_feat.full_sq-house_feat.life_sq)>0)&((house_feat.life_sq-house_feat.kitch_sq)>0)

    all_sq_ft=house_feat[has_all_sq_ft]

    all_sq_ft.loc[:,'% Kitch/Full']=100*all_sq_ft.loc[:,'kitch_sq']/all_sq_ft.loc[:,'full_sq']

    all_sq_ft.loc[:,'% Kitch/Life']=100*all_sq_ft.loc[:,'kitch_sq']/all_sq_ft.loc[:,'life_sq']

    all_sq_ft.loc[:,'% Life/Full']=100*all_sq_ft.loc[:,'life_sq']/all_sq_ft.loc[:,'full_sq']

    final_sq_ft=all_sq_ft[all_sq_ft['% Kitch/Life']<=60]

    sq_ft_breakage=final_sq_ft.describe()

    #Adjusting the missing full sq

    house_feat.loc[no_sq,'full_sq']=house_feat.loc[no_sq,'life_sq']

    #life_big_than_full=(house_feat.full_sq-house_feat.life_sq)<0

    #house_feat.loc[life_big_than_full,'full_sq']=house_feat.loc[life_big_than_full,'life_sq']

    # Adjusting kitch_sq using full_sq

    too_kitch_in_full=(house_feat.full_sq-house_feat.kitch_sq)<0

    house_feat.loc[(no_kitch)|(too_kitch_in_full),'kitch_sq']=house_feat.loc[(no_kitch)|(too_kitch_in_full),'full_sq']*sq_ft_breakage.loc['50%','% Kitch/Full']/100

    #Adjusting the life_sq using full sq

    too_life_in_full=(house_feat.full_sq-house_feat.life_sq)<0

    house_feat.loc[(no_l_sq)|(too_life_in_full),'life_sq']=house_feat.loc[(no_l_sq)|(too_life_in_full),'full_sq']*sq_ft_breakage.loc['50%','% Life/Full']/100

    #too_much=house_feat[(too_kitch_in_life)|(too_life_in_full)|(too_kitch_in_full)]

    #Adjusting kitch_sq using life_sq

    too_kitch_in_life=(house_feat.life_sq/house_feat.kitch_sq)>=0.6

    house_feat.loc[(no_kitch)|(too_kitch_in_life),'kitch_sq']=house_feat.loc[(no_kitch)|(too_kitch_in_life),'life_sq']*sq_ft_breakage.loc['50%','% Kitch/Life']/100

    return house_feat
def adjust_floor(house_feat):

    too_high_floor=house_feat.floor>house_feat.max_floor

    house_feat.loc[too_high_floor,'floor']=house_feat.loc[too_high_floor,'max_floor']

    return house_feat
new_house_feat=adjust_floor(adjust_sq_ft(house_features))
new_house_feat.head()
class neuralnet:        

    def initialize(self,inp_nodes,hid_nodes,out_nodes,l_rate):

        self.inodes=inp_nodes

        self.hnodes=hid_nodes

        self.onodes=out_nodes

        self.lr=l_rate

        self.wih=np.random.rand(hid_nodes,inp_nodes)-0.5

        self.woh=np.random.rand(out_nodes,hid_nodes)-0.5

        self.activation_function=lambda x:expit(x)

        pass

    def train(self):

        pass

    def query(self,input_list):

        inputs=np.array(input_list,ndmin=2).T

        hidden_inputs=np.dot(self.wih,inputs)

        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs=np.dot(self.woh,hidden_outputs)

        final_outputs=self.activation_function(final_inputs)

        return final_outputs

        

        

        pass

    def soph_weights(self):

        self.wih=np.random.normal(0.0,np.power(self.hnodes,-0.5),(self.hnodes,self.inodes))

        self.woh=np.random.normal(0.0,np.power(self.onodes,-0.5),(self.onodes,self.hnodes))

    pass
n1=neuralnet()

n1.initialize(3,5,3,0.5)

n1.query([1,2,1])