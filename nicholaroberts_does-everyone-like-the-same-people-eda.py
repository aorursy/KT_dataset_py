#import widgets

from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

from IPython.display import HTML

#from IPython.core.interactiveshell import InteractiveShell

#InteractiveShell.ast_node_interactivity = "all"

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings

warnings.filterwarnings("ignore")
#toggle code off and on (taken stackoverflow)

HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

The raw code for this IPython notebook is by default hidden for easier reading.

To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from matplotlib.colors import ListedColormap

%matplotlib inline



#show more columns and rows by default

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



#import data

data = pd.read_csv('../input/speed-dating-experiment/Speed Dating Data.csv',encoding="ISO-8859-1")

data.set_index('iid',inplace=True)

#data.head(1)
#create a table for each idd. which unchanging features (i.e. drop information related to individual dates)

iid_lookup = data[['gender','age','field_cd','race','imprace','imprelig','goal','date','go_out','sports','tvsports','exercise','dining','museums','art','hiking','gaming','clubbing','reading','tv','theater','movies','concerts','music','shopping','yoga','exphappy','expnum','attr3_1','sinc3_1','fun3_1','intel3_1','amb3_1']]

iid_lookup = iid_lookup.groupby('iid').mean() #have to group on some criteria, just taken mean because it's easy

#iid_lookup.head(1)
#create a cut down table for each invididual date

list_of_dates = data[['pid','match','dec','dec_o','wave','int_corr','samerace','attr','sinc','intel','fun','amb','shar','like','prob','met','attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','like_o','prob_o','met_o']].copy()

#list_of_dates.head(1)

#list_of_dates
#add a new column summerising the sucess of each date

list_of_dates['decs'] = 2*list_of_dates['dec_o']+1*list_of_dates['dec']

#create colormap (1=neither positive,2=they positive only, 3=partner postive only, 4=match)

cust = ["#E0E0E0","#E0E0E0","#404040", "#FF0000"]

my_cmap = ListedColormap(sns.color_palette(cust).as_hex())
#create a heatmap summerising how each date went

waves=[]

for i in range(21):

    #create a cutdown table for each wave

    date_decision = list_of_dates[list_of_dates['wave']==(i+1)][['pid','decs']]

    #rearrange the table to the right format

    wave = pd.pivot_table(date_decision,values='decs',index=['iid'],columns=['pid'])

    waves.append(wave)
#create a slider running through the different waves, display as heatmap.

def wave_display(i):

    sns.heatmap(waves[(i-1)],cmap=my_cmap,cbar=False, square=True)

    plt.xlabel('IID decision making partner')

    plt.ylabel('IID decision recieving partner')

    plt.title('wave {}'.format(i))

interact(wave_display,i=(widgets.IntSlider(value=9,description='Wave Num.',min=1,max=21,step=1,continuous_update=False)));
#find the average of how each data did on their dates.

daters_means = list_of_dates.groupby('iid').mean()

daters_means.drop(['pid','samerace','int_corr','decs'],axis=1,inplace=True) #meaningless here

#daters_means.head(1)
#create a table for followup info

follow_up = data[['you_call','them_cal','date_3','numdat_3','num_in_3']]

follow_up = follow_up.groupby('iid').mean()

#follow_up.head(1)
#join all the tables together again. This is so all the information on each dater is together

joined = iid_lookup.join(daters_means)

#joined.head(20)
#plot figures showing how daters did overall and seperated by sex. This is for the dec_o column, which is decision made about person.

plt.figure(figsize=(15,5))



#total

plt.subplot(1,3,1)

sns.distplot(joined['dec_o'],kde=False,bins=10)

plt.xlabel('fraction of positive decisions recieved total')

plt.ylabel('number of people total')

plt.tight_layout()

plt.ylim(0,110)



#women

plt.subplot(1,3,2)

sns.distplot(joined[joined['gender']==0]['dec_o'],label='women',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='red')

plt.xlabel('fraction of positive decisions recieved by women ')

plt.ylabel('number of women')

plt.tight_layout()

plt.ylim(0,110)



plt.subplot(1,3,3)

sns.distplot(joined[joined['gender']==1]['dec_o'],label='men',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='green')

#plt.legend()

plt.xlabel('fraction of positive decisions recieved by men')

plt.ylabel('number of men')

plt.ylim(0,110);
#plot figures showing how daters did overall and seperated by sex. This is for the dec column, which is decision made by the person.

plt.figure(figsize=(15,5))



#total

plt.subplot(1,3,1)

sns.distplot(joined['dec'],kde=False,bins=10)

plt.xlabel('fraction of positive decicions given')

plt.ylabel('number of people total')

plt.tight_layout()

plt.ylim(0,100)



#women

plt.subplot(1,3,2)

sns.distplot(joined[joined['gender']==0]['dec'],label='women',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='red')

plt.xlabel('fraction of positive decisions given by women')

plt.ylabel('number of women')

plt.tight_layout()

plt.ylim(0,100)



#men

plt.subplot(1,3,3)

sns.distplot(joined[joined['gender']==1]['dec'],label='men',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='green')

#plt.legend()

plt.xlabel('fraction of positive decisions given by men')

plt.ylabel('number of women')

plt.ylim(0,100);
list_of_dates.groupby('dec').mean()['dec_o'];
#graphs showing distribution of matches

plt.figure(figsize=(15,5))



#total

plt.subplot(1,3,1)

sns.distplot(joined['match'],kde=False,bins=10)

plt.xlabel('fraction of matches total')

plt.ylabel('number of people total')

plt.tight_layout()

plt.ylim(0,175)





plt.subplot(1,3,2)

sns.distplot(joined[joined['gender']==0]['match'],label='women',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='red')

plt.xlabel('fraction of matches for women ')

plt.ylabel('number of women')

plt.tight_layout()

plt.ylim(0,175)



plt.subplot(1,3,3)

sns.distplot(joined[joined['gender']==1]['match'],label='men',kde=False,bins=10,hist_kws=dict(alpha=0.3),color='green')

#plt.legend()

plt.xlabel('fraction of matches for men')

plt.ylabel('number of men')

plt.ylim(0,175);
corr = joined.corr()
plt.figure(figsize=(20,5));

plt.subplot(1,5,1);

sns.scatterplot(joined['attr3_1'],joined['attr_o'],hue=joined['dec_o'],cmap='coolwarm',s=50,legend=False);

plt.xlabel('self rated');

plt.ylabel('average rating by partner');

plt.title('attraction corr= {:0.2f}'.format(corr['attr3_1']['attr_o']));

plt.tight_layout();

plt.subplot(1,5,2);

sns.scatterplot(joined['sinc3_1'],joined['sinc_o'],hue=joined['dec_o'],cmap='coolwarm',s=50,legend=False);

plt.xlabel('self rated');

plt.ylabel('average rating by partner');

plt.title('sinserity corr= {:0.2f}'.format(corr['sinc3_1']['sinc_o']));

plt.tight_layout();

plt.subplot(1,5,3);

sns.scatterplot(joined['fun3_1'],joined['fun_o'],hue=joined['dec_o'],cmap='coolwarm',s=50,legend=False);

plt.xlabel('self rated');

plt.ylabel('average rating by partner');

plt.title('fun corr= {:0.2f}'.format(corr['fun3_1']['fun_o']));

plt.tight_layout();

plt.subplot(1,5,4);

sns.scatterplot(joined['intel3_1'],joined['intel_o'],hue=joined['dec_o'],cmap='coolwarm',s=50,legend=False);

plt.xlabel('self rated');

plt.ylabel('average rating by partner');

plt.title('intellegence corr= {:0.2f}'.format(corr['intel3_1']['intel_o']));

plt.tight_layout();

plt.subplot(1,5,5);

sns.scatterplot(joined['amb3_1'],joined['amb_o'],hue=joined['dec_o'],cmap='Wistia',s=50,legend=False);

plt.xlabel('self rated');

plt.ylabel('average rating by partner');

plt.title('ambition corr= {:0.2f}'.format(corr['amb3_1']['amb_o']));

plt.tight_layout();
predictive_atts = corr['dec_o'].sort_values(ascending=False)

pd.DataFrame(predictive_atts[abs(predictive_atts)>0.1]).T.round(2)
plt.figure(figsize=(20,5));

plt.subplot(1,4,1);

sns.scatterplot(joined['attr_o'],joined['dec_o'],hue=joined['gender']);

plt.legend(['female','male']);

plt.title('average attraction given by partners = {:0.2f}'.format(corr['attr_o']['dec_o']));

plt.tight_layout();

plt.subplot(1,4,2);

sns.scatterplot(joined['attr3_1'],joined['dec_o'],hue=joined['gender']);

plt.legend(['female','male']);

plt.title('self rated attraction corrolation = {:0.2f}'.format(corr['attr3_1']['dec_o']));

plt.tight_layout();

plt.subplot(1,4,3);

sns.scatterplot(joined['expnum'],joined['dec_o'],hue=joined['gender']);

plt.legend(['female','male']);

plt.title('expected number of matches corrolation = {:0.2f}'.format(corr['expnum']['dec_o']));

plt.tight_layout();

plt.subplot(1,4,4);

sns.scatterplot(joined['exercise'],joined['dec_o'],hue=joined['gender']);

plt.legend(['female','male']);

plt.title('exercise interest corrolation = {:0.2f}'.format(corr['expnum']['dec_o']));

plt.tight_layout();
predictive_atts = corr['dec'].sort_values(ascending=False)

pd.DataFrame(predictive_atts[abs(predictive_atts)>0.1]).T.round(2)
predictive_atts = corr['gender'].sort_values(ascending=False)

pd.DataFrame(predictive_atts[abs(predictive_atts)>0.1]).T.round(2)
#find the attributes that are most corrolated 



best_corrs=[]

#run through attributes in corrolation matrix

for ind in list(corr.index.values):

    for i in range(20):

    #sort the column for this atribute by the corrolation and take the top value (0 position always==1)

        att, cor =corr.sort_values(by=ind,ascending=False)[ind].iloc[[(i+1)]].to_string().split()

        best_corrs.append([ind,att,cor])



    

#sort final list by corolation 

best_corrs.sort(key=lambda x:x[2],reverse=True)

#take top 40 values (note each row is dublicated so only take every other row)

best_corrs = pd.DataFrame(best_corrs).round(2)

(best_corrs[0:600:2]).T.round(2)

plt.figure(figsize=(20,5));

plt.subplot(1,4,1);

sns.scatterplot(joined['attr_o'],joined['like_o'],hue=joined['gender']);

plt.legend(['female','male']);

plt.title((('average attraction and like scores given by partners \n, {:0.2f}')).format(corr['attr_o']['like_o']));

plt.tight_layout();

plt.subplot(1,4,2);

sns.scatterplot(joined['attr_o'],joined['fun_o'],hue=joined['gender']);

plt.legend(['female','male']);

plt.title('average attraction and fun scores given by partners,\n {:0.2f}'.format(corr['attr_o']['fun_o']));

plt.tight_layout();

plt.subplot(1,4,3);

sns.scatterplot(joined['fun_o'],joined['shar_o'],hue=joined['gender']);

plt.legend(['female','male']);

plt.title('average fun and shared interest scores given \n by partners, {:0.2f}'.format(corr['fun_o']['shar_o']));

plt.tight_layout();

plt.subplot(1,4,4);

sns.scatterplot(joined['expnum'],joined['attr3_1'],hue=joined['gender']);

plt.legend(['female','male']);

plt.title('expected number of matches and self rated \n attractiveness, {:0.2f}'.format(corr['expnum']['attr3_1']));

plt.tight_layout();
#lose nas. unfortunetly have to lose expnum column as a lot left this blank

joined_na=joined.drop(['expnum'],axis=1)

joined_na.dropna(inplace=True)
#set up data

#inputs

X = joined_na[['gender','age','field_cd','race','imprace','imprelig','goal','date','go_out','sports','tvsports','exercise','dining','museums','art','hiking','gaming','clubbing','reading','tv','theater','movies','concerts','music','shopping','yoga','exphappy','wave','attr3_1','sinc3_1','intel3_1','fun3_1','amb3_1']]

#outputs

Y = joined_na['dec_o']

#split data 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,shuffle=False)
#scale data, so no one variable skews fitting

#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()

#scaler.fit(X_train)

#X_train = pd.DataFrame(data=scaler.transform(X_train))

#X_test = pd.DataFrame(data=scaler.transform(X_test),columns=X_test.columns)
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

#create feature columns loop taken from stackover flow

my_columns=[]

import pandas.api.types as ptypes



feat_cols = []



for col in X_train.columns:

  if ptypes.is_string_dtype(X_train[col]): #is_string_dtype is pandas function

    feat_cols.append(tf.feature_column.categorical_column_with_hash_bucket(col,                            

       hash_bucket_size= len(X_train[col].unique())))

  elif ptypes.is_numeric_dtype(X_train[col]): #is_numeric_dtype is pandas function



    feat_cols.append(tf.feature_column.numeric_column(col))
#set up tensorflow

#define input, and output

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=Y_train,batch_size=10,num_epochs=100,shuffle=False)

#set up estimator

model = tf.estimator.DNNRegressor(hidden_units=[10,10,10],feature_columns=feat_cols)

#train

model.train(input_fn=input_func,steps=2500)

#predcit function

predict_input_func = tf.estimator.inputs.pandas_input_fn(

x=X_test,batch_size=10,num_epochs=1,shuffle=False)

#get outputs

pred_gen= model.predict(predict_input_func)

predictions = list(pred_gen)

final_preds =[]

for pred in predictions:

    final_preds.append(pred['predictions'])



#measure error

from sklearn.metrics import mean_squared_error 

#mean_squared_error(Y_test,final_preds)**0.5
#plt.plot(range(0,len(Y_test)),Y_test)

#plt.plot(final_preds)
#set up data

#inputs

X = joined_na.drop(['match','dec','dec_o'],axis=1)

#outputs

Y = joined_na['dec_o']

#split data 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,shuffle=False)
import tensorflow as tf

#create feature columns loop taken from stackover flow

my_columns=[]

import pandas.api.types as ptypes



feat_cols = []



for col in X_train.columns:

  if ptypes.is_string_dtype(X_train[col]): #is_string_dtype is pandas function

    feat_cols.append(tf.feature_column.categorical_column_with_hash_bucket(col,                            

       hash_bucket_size= len(X_train[col].unique())))

  elif ptypes.is_numeric_dtype(X_train[col]): #is_numeric_dtype is pandas function

    feat_cols.append(tf.feature_column.numeric_column(col))
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=Y_train,batch_size=10,num_epochs=100,shuffle=False)

model = tf.estimator.DNNRegressor(hidden_units=[10,10,10,10],feature_columns=feat_cols)

model.train(input_fn=input_func,steps=2500)

predict_input_func = tf.estimator.inputs.pandas_input_fn(

x=X_test,batch_size=10,num_epochs=1,shuffle=False)

pred_gen= model.predict(predict_input_func)

predictions = list(pred_gen)



final_preds_2 =[]

for pred in predictions:

    final_preds_2.append(pred['predictions'])



from sklearn.metrics import mean_squared_error 

#mean_squared_error(Y_test,final_preds_2)**0.5
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#init_notebook_mode(connected=True) 

plt.subplot(2,1,1)

plt.plot(range(0,len(Y_test)),Y_test,label='actual success',c='black')

plt.plot(final_preds_2,label='including partner rating')

plt.plot(range(0,len(Y_test)),X_test['attr_o']/10,ls='--',lw=0.4,label='partner rated attraction')

plt.xlabel('dater')

plt.legend(bbox_to_anchor=(1,1))



plt.subplot(2,1,2)

plt.plot(range(0,len(Y_test)),Y_test,label='actual success',c='black')

plt.plot(final_preds,label='excluding partner rating')

plt.plot(range(0,len(Y_test)),X_test['attr3_1']/10,ls='--',lw=0.4,label='self rated attraction')

plt.ylabel('fraction of partners who want to see them again')

plt.legend(bbox_to_anchor=(1,1))

plt.xlabel('dater');

wave_sort=[]

for i in range(21):

    temp_wave_sort = (waves[i].join(joined['dec_o']).sort_values(by='dec_o',ascending=False).T.join(joined['dec_o']).sort_values(by='dec_o',ascending=False).T)

    wave_sort.append(temp_wave_sort)

#wave_sort[0]
def wave_display_2(i):

    sns.heatmap(wave_sort[(i-1)],cmap=my_cmap,cbar=False, square=True)

    plt.xlabel('IID decision making partner')

    plt.ylabel('IID decision recieving partner')

    plt.title('wave {}'.format(i))

interact(wave_display_2,i=(widgets.IntSlider(value=9,description='Wave Num.',min=1,max=21,step=1,continuous_update=False)));
#add the overall sucess of each dater to the dates list

success_table=daters_means[['dec_o']]

success_table.rename({'dec_o':'success'},axis=1,inplace=True)

dates=list_of_dates[['pid','match','dec']]

dates=dates.join(success_table)

dates.rename({'success':'suc'},axis=1,inplace=True)

dates=dates.join(success_table,on='pid')

dates.rename({'success':'suc_o'},axis=1,inplace=True)
plt.figure(figsize=(20,5));

plt.subplot(1,3,1)

sns.scatterplot(dates['suc'],dates['suc_o'],hue=dates['dec'])

plt.title('Daters decision, shown for overall successfulness of each partner');

plt.ylabel('Partners overall successfulness')

plt.xlabel('Daters overall successfulness')

plt.tight_layout();

plt.legend(title='Dater wants to see again',title_fontsize='large',frameon=True,facecolor='white',framealpha=1.0,loc=5,ncol=4,borderpad=0.1,shadow=True)

plt.subplot(1,3,2)

sns.scatterplot(dates['suc'],dates['suc_o'],hue=dates['match'])

plt.legend(['no match','match']);

plt.title('Match, shown for overall successfulness of each partner');

plt.ylabel('Partners overall successfulness')

plt.xlabel('Daters overall successfulness')

plt.tight_layout();

plt.legend(title='Match',frameon=True,title_fontsize='large',facecolor='white',framealpha=1.0,loc=5,ncol=4,borderpad=0.1,shadow=True);

#plt.subplot(1,3,3)

#sns.scatterplot(dates[dates['dec']==1]['suc'],dates[dates['dec']==1]['suc_o'],hue=dates['gender'])

#plt.legend(['no match','match']);

#plt.title('Match, shown for overall successfulness of each partner');

#plt.ylabel('Partners overall successfulness')

#plt.xlabel('Daters overall successfulness')

#plt.tight_layout();

#plt.legend(title='Match',frameon=True,title_fontsize='large',facecolor='white',framealpha=1.0,loc=5,ncol=4,borderpad=0.1,shadow=True);

plt.subplot(1,3,3)

sns.scatterplot(dates[dates['match']==1]['suc'],dates[dates['match']==1]['suc_o'])

#b= sns.scatterplot(dates[dates['dec']==1]['suc'],dates[dates['dec']==1]['suc_o'])

#plt.legend(['not see again','see again']);

#plt.title('Matches');

#plt.ylabel('Partners overall successfulness')

#plt.xlabel('Daters overall successfulness')

#plt.tight_layout();

#plt.legend(title='Partner wants to see again',title_fontsize='x-large',frameon=True,facecolor='white',framealpha=1.0,loc=5,ncol=4,borderpad=0.1,shadow=True);
sns.heatmap(dates.corr(),annot=True,linecolor='white',linewidths=2.0,square=True,mask=np.triu(np.ones(5)).astype(np.bool),cmap='Reds');