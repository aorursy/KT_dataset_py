import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import colors as mcolors,cm,colorbar

import scipy

import re

from IPython.display import HTML

from sklearn import model_selection
def cleanup(df_bur):

    # Clean the dataset: drop some columns we won't need, drop leading zeros from the

    # codes, transform from long to wide format

    df_bur=df_bur.drop(['First name','Sex'],axis=1)

    df_bur['Polling station']=df_bur['Polling station'].apply(lambda x:str(x)[2:] if (str(x)[:2]=='BV') else x).apply(lambda x: re.sub("^[0]+","",str(x)))

    df_bur['Commune code']=df_bur['Commune code'].apply(lambda x: re.sub("^[0]+","",str(x)))

    df_bur['Department code']=df_bur['Department code'].apply(lambda x: re.sub("^[0]+","",str(x)))

    df_bur['Constituency code']=df_bur['Constituency code'].apply(lambda x: re.sub("^[0]+","",str(x)))

    e1=df_bur[[u'Department',

       u'Constituency', u'Commune','Polling station']+[u'Registered',u'Abstentions',                  u'% Abs/Reg',

                          u'Voters',                  u'% Vot/Reg',

                           u'None of the above(NOTA)',               u'% NOTA/Reg',

                     u'% NOTA/Vot',                       u'Nulls',

                       u'% Nulls/Reg',                 u'% Nulls/Vot',

                         u'Expressed']].set_index([u'Department',

       u'Constituency', u'Commune', 'Polling station']).drop_duplicates()

    

    df_b1=df_bur.pivot_table(index=[u'Department',

       u'Constituency', u'Commune', 'Polling station'],columns = u'Surname',values='Voted')

    

    tab=pd.merge(df_b1,e1,left_index=True,right_index=True)

    

    if ('MÉLENCHON' in tab.columns):

        tab=tab.rename(columns={'MÉLENCHON':'MELENCHON'})

    

    return tab
df_bur = pd.read_csv("../input/French_Presidential_Election_2017_First_Round.csv",sep=',',

                    dtype={'Department code':'object','Polling station':'object',29:'object'})

df_bur2 = pd.read_csv("../input/French_Presidential_Election_2017_Second_Round.csv",sep=',',

                     dtype={'Polling station':'object'})
tab=cleanup(df_bur)
tab2=cleanup(df_bur2)
print(np.all(tab['Voters']+tab['Abstentions']-tab['Registered']==0))

print((tab[[u'ARTHAUD', u'ASSELINEAU', u'CHEMINADE', u'DUPONT-AIGNAN', u'FILLON',

       u'HAMON', u'LASSALLE', u'LE PEN', u'MACRON', u'POUTOU',u'MELENCHON']].sum(axis=1)-tab[u'Expressed']!=0).sum())

print((tab[[u'ARTHAUD', u'ASSELINEAU', u'CHEMINADE', u'DUPONT-AIGNAN', u'FILLON',

       u'HAMON', u'LASSALLE', u'LE PEN', u'MACRON', u'POUTOU',u'MELENCHON', u'None of the above(NOTA)',u'Nulls']].sum(axis=1)-tab[u'Voters']!=0).sum())
merged=pd.merge(tab,tab2,left_index=True,right_index=True,how='inner')
merged['Abstentions, NOTA, null_y']=merged[['None of the above(NOTA)_y','Nulls_y','Abstentions_y']].sum(axis=1)

merged['Abstentions, NOTA, null_x']=merged[['None of the above(NOTA)_x','Nulls_x','Abstentions_x']].sum(axis=1)

merged['Other candidates']=merged[['ARTHAUD','ASSELINEAU','CHEMINADE','LASSALLE','POUTOU']].sum(axis=1)
nms_compressed=['Other candidates',

 u'DUPONT-AIGNAN',

 u'FILLON',

 u'HAMON',

 u'LE PEN_x',

 u'MELENCHON',

 'MACRON_x','Abstentions, NOTA, null_x']
options_2iem_compressed=['LE PEN_y', 'MACRON_y', 'Abstentions, NOTA, null_y']
def optimize(merged):

    print('opimizing, input table has '+str(merged.shape[0])+" rows")

    

    y1=(merged[options_2iem_compressed].T/merged['Registered_y']).T[merged['Registered_x']!=0]

    X1=(merged[nms_compressed].T/merged['Registered_x']).T[merged['Registered_x']!=0]

    n_2iem=len(options_2iem_compressed)

    

    # Probabilities are naturally organized as a table, but scipy.optimize.minimize works with 

    # a list of parameters. So this functions reshapes this list to a table

    def rshp(prob):

        tmp1=np.reshape(prob,(len(options_2iem_compressed)-1,X1.shape[1])).T

        tmp2=np.concatenate((tmp1,np.array([1-tmp1.sum(axis=1)]).T),axis=1)

        return(tmp2)



    # This is a loss function, with takes as an input probabilities and outputs the quadratic

    # deviation of the computed results of the second round from the actual ones

    def loss_func(prob):

        y1=(merged[options_2iem_compressed].T/merged['Registered_y']).T[merged['Registered_x']!=0]

        tmp2=rshp(prob)

        ret=np.sum((np.dot(X1,tmp2)-y1)**2).sum()

        return ret

    

    

    # Constraint for the probabilities table: sum of values in each row should be equalt to 1.

    def fun_constr(prob):

        return 1-np.reshape(prob,(n_2iem-1,X1.shape[1])).sum(axis=0)

    

    bs=[(0,1)]*(X1.shape[1])*(n_2iem-1) # bounds for probabilities: bewteen 0 and 1

    x0=np.array(X1.shape[1]*(n_2iem-1)*[1/float(n_2iem)]) # starting point for the optimization procedure

    constr={'type':'ineq','fun':fun_constr} # impose the constraint: sum inside each row omitting the last element should be smaller than 1

    print(x0)

    print(bs)

    # run the quadratic optimization

    opt=scipy.optimize.minimize(loss_func,

                   x0,#jac=jac,

                   method = 'SLSQP',

                   bounds = bs,

                   constraints = constr



        )

    print(opt)

    print('error: '+str(np.sqrt(opt.fun/len(y1))))

    res=pd.DataFrame(rshp(opt.x),columns=options_2iem_compressed,index=nms_compressed).round(3)

    return(res)
# run the optimization procedure on the whole dataset

res_c=optimize(merged)
# take a random subset of the original table droping 1/10 of the data

splitter = model_selection.ShuffleSplit(1,0.1)

train = merged.iloc[[x for x in splitter.split(merged)][0][0]]
#run the optimization procedure on the first random subset

res_train=optimize(train)
# take another random subset

splitter = model_selection.ShuffleSplit(1,0.1)

train = merged.iloc[[x for x in splitter.split(merged)][0][0]]
#run the optimization procedure on the second random subset

res_train1=optimize(train)
res_train1-res_train
# Votes for particular candidates as a share of all registered voters (first round)

res_premier=merged[nms_compressed].sum()/merged['Registered_x'].sum()
res_new=((res_premier*res_c.T).T).round(3)
def form(x): 

    return x if ((len(x)<2) or ((x[len(x)-2:]!='_x') and  (x[len(x)-2:]!='_y') )) else x[:len(x)-2]

chart1=res_new.copy()

chart1['total']=chart1.sum(axis=1)

chart1=chart1.sort_values(['total'])*100

bar_h=10

x_init=0.14*100

fig=plt.figure(facecolor='white')

ax=fig.add_subplot(111)

ax.set_facecolor('white')

ax.text(0,bar_h*len(chart1.index)+bar_h*0.2,'First round',fontsize=13)

for i in range(len(chart1.index)):

    ax.bar(left=x_init+chart1.iloc[i,0],height=bar_h,width=chart1.iloc[i,1],

           bottom=bar_h*i,color='red',align='edge',edgecolor='black')

    ax.bar(left=x_init,height=bar_h,width=chart1.iloc[i,0],

           bottom=bar_h*i,color='blue',align='edge',edgecolor='black')

    ax.bar(left=x_init+chart1.iloc[i,0]+chart1.iloc[i,1],height=bar_h,width=chart1.iloc[i,2],

           bottom=bar_h*i,color='grey',align='edge',edgecolor='black')

    ax.text(0,bar_h*i+bar_h*0.5,form(chart1.index[i]),fontweight='bold',fontsize=13)

lg=ax.legend(['Macron','Le Pen','Abstentions, NOTA, null'],loc=(0.65,0.12),title='Second round',fontsize=13)

plt.setp(lg.get_title(),fontsize=14)

ax.set_yticks([])

ticks=np.arange(0,40,5)

ax.set_xticks(ticks+x_init)

ax.set_xticklabels([str(x) for x in ticks],fontsize=13)

ax.text(0,bar_h*(len(chart1.index)+1),'Transfer of votes between the two rounds',

             fontweight='bold',

             fontsize=17

             )

ax.set_xlabel('% of all votes',fontsize=13)

plt.show()
table_html=(pd.DataFrame(np.array(res_c),columns=['Le Pen','Macron','Abstentions, NOTA, null'],

             index=['Other candidates','Dupont-Aignan','Fillon','Hamon',

                    'Le Pen','Melenchon','Macron','Abstentions, NOTA, null']).sort_values('Macron',ascending=False)*100).round(1)

table_html
res_new.columns=['Le Pen','Macron','Abstentions, NOTA, null']

res_new.index=['Other candidates','Dupont-Aignan','Fillon','Hamon',

                    'Le Pen','Melenchon','Macron','Abstentions, NOTA, null']

res_new