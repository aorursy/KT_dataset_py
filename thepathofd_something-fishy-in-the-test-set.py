import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()
import plotly.offline as py
py.init_notebook_mode(connected=True)

# set-up
local_path='../input/8th-as-train-data-9th-as-test-data/'
kaggle_path='../input/talkingdata-adtracking-fraud-detection/'

def load_data(name,rows=None):
    ''' Load the csv files into a TimeSeries dataframe with minimal data types to reduce the used RAM space.
    Arg:
        -name (str): train, train_sample or test
    Returns:
        pd.DataFrame
    '''
    
    # Defining dtypes
    types = {'ip':np.uint32,'app': np.uint16,'os': np.uint16,'device': np.uint16,'channel':np.uint16,'click_time': object}
    
    if name=='test':
        types['click_id']= np.uint32
    else:
        types['is_attributed']='bool'
    
    # Defining reading arguments
    read_args={'nrows':rows,'parse_dates':['click_time'],'infer_datetime_format':True,'index_col':'click_time','usecols':list(types.keys()),'dtype':types,
        'engine':'c',
        'sep':','
        }
    
    # Setting file path and compression type
    if name in['train','test']:
        file_path='{}{}.csv'.format(kaggle_path,name)
    else:
        file_path='{}{}.csv.zip'.format(local_path,name)
        read_args['compression']='gzip'
    
    # Reading file and setting the timezone 
    with open(file_path,'rb') as File:
        data=(pd
            .read_csv(File,**read_args)
            .tz_localize('UTC')
            .tz_convert('Asia/Shanghai')
            .reset_index()
        )

    return data
class test_val_compare(object):
    def __init__(self,validation,test):
        self.val=load_data(validation)
        self.val.name='validation'
        self.test=load_data(test)
        self.test.name='test'
        self.actors=[]
        self.val_cap=[]
        self.capped=False
    
    def info(self,**kwargs):
        print('Validation data:')
        print(self.val.info(**kwargs))
        print('Test data:')
        print(self.test.info(**kwargs))
        
    def def_actors(self,cols,merge=False):
        self.actors=(pd.concat([self.val[cols].drop_duplicates().assign(left_data=True,right_data=False).astype({'left_data':'bool'}),
                         self.test[cols].drop_duplicates().assign(left_data=False,right_data=True).astype({'left_data':'bool'})])
                 .reset_index(drop=True)
                 .assign(actor=lambda x: x.index.values.astype(np.uint32))
                )
        ##if Merge:
            ## add the actors columns to the frames
            
    def cap(self):
        if not self.capped:
            real_max = self.test.describe().loc['max',:]
            self.val_cap=self.val.loc[(self.val['ip']<real_max['ip'])&(self.val['app']<real_max['app'])&(self.val['device']<real_max['device'])&(self.val['os']<real_max['os'])&(self.val['channel']<real_max['channel']),:]
            self.val_cap.name='capped_validation'
            self.capped=True
        else:
            print('capped validation set already created, use .val_cap')

    def overlap(self,var,cap=False):
        if not cap:
            frames=[self.val,self.test]
        else:
            frames=[self.val_cap,self.test]
        intersection=list(set.intersection(*[set(x[var]) for x in frames]))
        overlap=pd.DataFrame({'val':list(set.union(*[set(x[var]) for x in frames]))})
        for frame in frames:
            overlap[frame.name]=overlap.val.isin(frame[var])
        overlap=(overlap
                 .set_index(['val'])
                 .stack()
                 .reset_index()
                 .rename(columns={'level_1':'set',0:'keep'})
                 .loc[lambda x:x.keep,['val','set']]
                 .assign(is_shared=lambda x:x.val.isin(intersection))
                )
        return overlap

    def plot_overlap(self,var,with_cap=False,dual=False):
        if dual:
            if not self.capped:
                self.cap()
            frame_list=[frame.rename(columns={var:frame.name})[frame.name] for frame in [self.val,self.val_cap]]
            frame2=self.test.rename(columns={var:self.test.name})[self.test.name]
            fig, axs = plt.subplots(ncols=2,figsize=(30,6))
            for i in [False,True]:
                (sns.stripplot(data=self.overlap(var,i),y='set',x='val',hue='is_shared',jitter=True,dodge=True,palette='Set1',ax=axs[i])
                .set(title='Difference in {} values'.format(var),xlabel='',ylabel=''));
        else:
            plt.figure(figsize=(15,6))
            if with_cap:
                if not self.capped:
                    self.cap()
                (sns.stripplot(data=self.overlap(var,True),y='set',x='val',hue='is_shared',jitter=True,dodge=True,palette='Set1')
                    .set(title='Difference in {} values'.format(var),xlabel='',ylabel=''));
            else:
                (sns.stripplot(data=self.overlap(var),y='set',x='val',hue='is_shared',jitter=True,dodge=True,palette='Set1')
                    .set(title='Difference in {} values'.format(var),xlabel='',ylabel=''));
            
    def heat_compare(self,var1,var2,with_cap=False,combine=False,respace=False,black=False):
        data1=self.val
        data2=self.test
        if with_cap:
            data1=self.val_cap
        if combine & respace:
            min_var1=min([min(data1[var1]),min(data2[var1])])
            max_var1=max([max(data1[var1]),max(data2[var1])])
            min_var2=min([min(data1[var2]),min(data2[var2])])
            max_var2=max([max(data1[var2]),max(data2[var2])])
            new_range=[range(min_var1,max_var1),range(min_var2,max_var2)]
            new_index=pd.MultiIndex.from_product(new_range,names=[var1,var2])

            fig,ax=plt.subplots(figsize=(30,30))
            sns.heatmap(data1.groupby([var1,var2])['click_time'].count().pipe(np.log1p).reindex(index=new_index).unstack(),
                cbar=False,
                ax=ax)
            sns.heatmap(data2.groupby([var1,var2])['click_time'].count().pipe(np.log1p).reindex(index=new_index).unstack(),
                cmap='YlGnBu_r',
                cbar=False,
                ax=ax)
        elif combine:
            var1_union=np.union1d(data1[var1],data2[var1])
            var2_union=np.union1d(data1[var2],data2[var2])
            new_range=[var1_union,var2_union]
            new_index=pd.MultiIndex.from_product(new_range,names=[var1,var2])
            fig,ax=plt.subplots(figsize=(30,30))
            sns.heatmap(data1.groupby([var1,var2])['click_time'].count().pipe(np.log1p).reindex(index=new_index).unstack(),
                cbar=False,
                ax=ax)
            sns.heatmap(data2.groupby([var1,var2])['click_time'].count().pipe(np.log1p).reindex(index=new_index).unstack(),
                cmap='YlGnBu_r',
                cbar=False,
                ax=ax)
        else:
            if black:
                fig,axes=plt.subplots(ncols=2,figsize=(30,20))
                sns.heatmap(data1.groupby([var1,var2])['click_time'].count().pipe(np.log1p).unstack().fillna(0),
                            cbar=False,
                            ax=axes[0]).set(title=data1.name)
                sns.heatmap(data2.groupby([var1,var2])['click_time'].count().pipe(np.log1p).unstack().fillna(0),
                            cbar=False,
                            ax=axes[1]).set(title=data2.name)
            else:
                fig,axes=plt.subplots(ncols=2,figsize=(30,20))
                sns.heatmap(data1.groupby([var1,var2])['click_time'].count().pipe(np.log1p).unstack(),
                            cbar=False,
                            ax=axes[0]).set(title=data1.name)
                sns.heatmap(data2.groupby([var1,var2])['click_time'].count().pipe(np.log1p).unstack(),
                            cbar=False,
                            ax=axes[1]).set(title=data2.name)
%%time
analysis=test_val_compare('last_day','test')
analysis.info()
analysis.plot_overlap('ip')
analysis.plot_overlap('app')
analysis.plot_overlap('channel')
analysis.plot_overlap('device')
analysis.plot_overlap('os')
analysis.cap()
analysis.plot_overlap('ip',with_cap=True)
analysis.plot_overlap('app',with_cap=True)
analysis.plot_overlap('channel',with_cap=True)
analysis.plot_overlap('device',with_cap=True)
analysis.plot_overlap('os',with_cap=True,)
analysis.heat_compare('ip','app',True)
# the kernel dies with this one, it takes too much memory (exit code 137)
#heat_compare(validation_data_cap,test_data,'ip','device')
analysis.heat_compare('ip','os',True)
analysis.heat_compare('ip','channel',True)
analysis.heat_compare('app','device',True)
analysis.heat_compare('app','os',True)
analysis.heat_compare('app','channel',True)
analysis.heat_compare('device','os',True)
analysis.heat_compare('device','channel',True)
analysis.heat_compare('os','channel',True)
# defining our actors
actors_factors=['ip','app','device','os','channel']
analysis.def_actors(actors_factors)

# defining the nodes of our graphs
times=analysis.val.click_time.dt.hour.unique()
states=[0,1,2]
color_dict={0 : 'red', 1 : 'grey', 2 : 'green'}
state_dict={0 : 'no activity', 1 : 'click only', 2 : 'download'}
nodes=(pd.DataFrame(index=pd.MultiIndex.from_product([times,states],names=['hour','state']))
       .reset_index()
       .assign(code=lambda x:10*x.hour+x.state,
               label=lambda x: x.state.replace(state_dict).str.cat(x.hour.astype(str),sep='_'),
              color=lambda x: x.state.replace(color_dict))
      )
node_codes=nodes.reset_index().set_index(['code'])['index'].to_dict()
nodes.head()
# defining the links
links=(analysis.val.assign(hour=lambda x: x.click_time.dt.hour.astype(np.uint8))
         .groupby(actors_factors+['hour'])
         .max()
         .reset_index()
         .merge(analysis.actors[actors_factors+['actor']],on=actors_factors,how='left')
         .reset_index(drop=True)
         .loc[:,['hour','actor','is_attributed']]
         .assign(code=lambda x: 10*x.hour+x.is_attributed+1)
         .pivot(index='actor',columns='hour',values='code')
         .fillna({x:10*x for x in times})
)
links=pd.concat([links.iloc[:,k].T.reset_index(drop=True).T for k in [[i,i+1]for i in range(len(links.columns)-1)]]).reset_index()
links.columns=['actor','source','target']
links.head()
links_grouped=links.groupby(['source','target']).count().reset_index()
data = {
    'type': 'sankey',
     'domain' : {'x': [0,1], 'y': [0,1]},
    'orientation': 'h',
    'node' : {
      'pad':  15,
      'thickness' : 20,
      'line' : {
            'color' : "black",
            'width' : 0.5
          },
      'label':  nodes.label,
      'color' : nodes.color
    },
    'link' : {
          'source' : links_grouped.loc[:,'source'].replace(node_codes),
          'target': links_grouped.loc[:,'target'].replace(node_codes),
          'value' : links_grouped.loc[:,'actor'],
          'label' : links_grouped.loc[:,'actor']
            }
}

layout =  {
    'title' : "Flow of activity between hour slices for the validation set",
    'font' : dict(size = 10)
}

fig = dict(data=[data], layout=layout)
py.iplot(fig, validate=False)
links=(analysis.test.assign(hour=lambda x: x.click_time.dt.hour.astype(np.uint8))
         .groupby(actors_factors+['hour'])
         .max()
         .reset_index()
         .merge(analysis.actors[actors_factors+['actor']],on=actors_factors,how='left')
         .reset_index(drop=True)
         .loc[:,['hour','actor']]
         .assign(code=lambda x: 10*x.hour+1)
         .pivot(index='actor',columns='hour',values='code')
         .fillna({x:10*x for x in times})
)
links=pd.concat([links.iloc[:,k].T.reset_index(drop=True).T for k in [[i,i+1]for i in range(len(links.columns)-1)]]).reset_index()
links.columns=['actor','source','target']
links_grouped=links.groupby(['source','target']).count().reset_index()
data = {
    'type': 'sankey',
     'domain' : {'x': [0,1], 'y': [0,1]},
    'orientation': 'h',
    'node' : {
      'pad':  15,
      'thickness' : 20,
      'line' : {
            'color' : "black",
            'width' : 0.5
          },
      'label':  nodes.label,
      'color' : nodes.color
    },
    'link' : {
          'source' : links_grouped.loc[:,'source'].replace(node_codes),
          'target': links_grouped.loc[:,'target'].replace(node_codes),
          'value' : links_grouped.loc[:,'actor'],
          'label' : links_grouped.loc[:,'actor']
            }
}

layout =  {
    'title' : "Flow of activity between hour slices for the ",
    'font' : dict(size = 10)
}

fig = dict(data=[data], layout=layout)
py.iplot(fig, validate=False)