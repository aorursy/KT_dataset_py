# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn.model_selection import cross_val_score

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import graphviz

import matplotlib.pyplot as plt



%matplotlib inline



import os

print(os.listdir("../input"))

pd.options.mode.chained_assignment = None

# Any results you write to the current directory are saved as output.
comb_dat = pd.read_csv("../input/combine_data_since_2000_PROCESSED_2018-04-26.csv")

comb_dat.head()
comb_dat['Round'].fillna(8.0, inplace = True)

comb_dat['Pick'].fillna(300.0, inplace = True)
comb_dat.groupby('Pos').count()
comb_dat.mean()
comb_dat.loc[:,'Pos']=comb_dat.Pos.replace({'C':'IOL','G':'IOL','OG':'IOL','OL':'IOL'

                                            ,'NT':'DT',

                                            'EDGE':'OLB',

                                            'DB':'S','SS':'S','FS':'S',

                                            'LB':'ILB'})
comb_dat=comb_dat.loc[~comb_dat.Pos.isin(['K','P','FB','LS']),:]

comb_dat=comb_dat.loc[comb_dat.Year<2018,:]
comb_dat.groupby('Pos').count()
i=0

pos=comb_dat.Pos.unique()



for posi in pos:



    row_dict={'Pos':[np.nan],



              'Count':[np.nan],

              

              'mean_mae':[np.nan],

              'med_mae':[np.nan],

              'max_mae':[np.nan],

              'min_mae':[],



              'Ht_imp':[np.nan],

              'Wt_imp':[np.nan],

              'Forty_imp':[np.nan],

              'Vertical_imp':[np.nan],

              'BenchReps_imp':[np.nan],

              'BroadJump_imp':[np.nan],

              'Cone_imp':[np.nan],

              'Shuttle_imp':[np.nan],



              'Ht_med':[np.nan],

              'Wt_med':[np.nan],

              'Forty_med':[np.nan],

              'Vertical_med':[np.nan],

              'BenchReps_med':[np.nan],

              'BroadJump_med':[np.nan],

              'Cone_med':[np.nan],

              'Shuttle_med':[np.nan],



              'Round1':[np.nan],

              'Round2':[np.nan],

              'Round3':[np.nan],

              'Round4':[np.nan],

              'Round5':[np.nan],

              'Round6':[np.nan],

              'Round7':[np.nan],

              'Undrafted':[np.nan]

             }







    #posi='WR'

    mod_vars=['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps',

           'BroadJump', 'Cone', 'Shuttle']

    pos_dat=comb_dat[comb_dat.Pos==posi]

    na_sum=pos_dat.loc[:,mod_vars].isna().sum()



    droppers=na_sum[na_sum>pos_dat.shape[0]/2].index

    #print(droppers)

    for drop in droppers:

        #print(drop)

        mod_vars.remove(drop)



    na_fill=pos_dat[mod_vars].quantile(.5).to_dict()

    pos_dat=pos_dat.fillna(na_fill)

    #print(pos_dat[mod_vars].isna().sum())

    pos_tree=tree.DecisionTreeRegressor(criterion='mae',max_depth=5,min_weight_fraction_leaf=.04,random_state=214)

    cv_scores=abs(cross_val_score(pos_tree,pos_dat[mod_vars],pos_dat['Round'],cv=10,scoring='neg_median_absolute_error'))



    round_count=pos_dat.groupby('Round').Player.count()

    round_count=round_count/pos_dat.shape[0]

    pos_tree=pos_tree.fit(pos_dat[mod_vars],pos_dat['Round'])



    row_dict['Pos']=[posi]

    row_dict['Count']=[pos_dat.shape[0]]



    row_dict['mean_mae']=[np.mean(cv_scores)]

    row_dict['med_mae']=[np.median(cv_scores)]

    row_dict['min_mae']=[np.min(cv_scores)]

    row_dict['max_mae']=[np.max(cv_scores)]



    row_dict.update(dict(zip([i+'_imp' for i in mod_vars],[[i] for i in pos_tree.feature_importances_])))

    row_dict.update(dict(zip([i+'_med' for i in mod_vars],[[i] for i in list(na_fill.values())])))

    row_dict.update(dict(zip(['Round1','Round2','Round3','Round4','Round5','Round6','Round7','Undrafted'],[[i] for i in round_count.values])))



    if i==0:

        row_frame=pd.DataFrame.from_dict(row_dict)

        i+=1

    else:

        row_frame=row_frame.append(pd.DataFrame.from_dict(row_dict),ignore_index=True)
row_frame.sort_values('mean_mae')
pos=row_frame.sort_values('mean_mae',ascending=False).Pos.values

def render_sum(posi):

    print('{} Model Summary'.format(posi))

    display(row_frame.loc[row_frame['Pos']==posi,:])

    

    mod_vars=['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps',

               'BroadJump', 'Cone', 'Shuttle']

    pos_dat_raw=comb_dat.loc[comb_dat.Pos==posi,:]

    pos_dat=pos_dat_raw.copy()

    na_sum=pos_dat.loc[:,mod_vars].isna().sum()



    droppers=na_sum[na_sum>pos_dat.shape[0]/2].index

        #print(droppers)

    for drop in droppers:

            #print(drop)

        mod_vars.remove(drop)



    na_fill=pos_dat.loc[:,mod_vars].quantile(.5).to_dict()

    pos_dat=pos_dat.fillna(na_fill)





    pos_tree=tree.DecisionTreeRegressor(criterion='mae',max_depth=5,min_weight_fraction_leaf=.04,random_state=214)

    pos_tree.fit(pos_dat.loc[:,mod_vars],pos_dat.loc[:,'Round'])

    preds=pos_tree.predict(pos_dat.loc[:,mod_vars])

    #print(preds)

    pos_dat_raw.loc[:,'pred_Round']=preds

    pos_dat_raw.loc[:,'res']=pos_dat_raw.loc[:,'Round']-pos_dat_raw.loc[:,'pred_Round']



    showvars=['Pos','Player','Year','Team']+mod_vars+['Round','pred_Round']

    print('Top 10 Underdrafted {}:'.format(posi))

    display(pos_dat_raw.sort_values(['res','Year'],ascending=[False,False])[showvars].head(10))

    print('Top 10 Overdrafted {}:'.format(posi))

    display(pos_dat_raw.sort_values(['res','Year'],ascending=[True,False])[showvars].head(10))





    dot_data = tree.export_graphviz(pos_tree, out_file=None, 

                         feature_names=mod_vars,  

                         class_names=['1','2','3','4','5','6','7','8'],

                         filled=True, rounded=True,  

                         special_characters=True)  

    graph = graphviz.Source(dot_data)  

    return(graph)
g=render_sum(pos[0])

g
g=render_sum(pos[1])

g
g=render_sum(pos[2])

g




g=render_sum(pos[3])

g
g=render_sum(pos[4])

g
g=render_sum(pos[5])

g
g=render_sum(pos[6])

g
g=render_sum(pos[7])

g
g=render_sum(pos[8])

g
g=render_sum(pos[9])

g
g=render_sum(pos[10])

g
g=render_sum(pos[11])

g