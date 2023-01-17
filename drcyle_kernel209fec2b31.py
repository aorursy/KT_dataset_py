# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





import multiprocessing,os,datetime,time,pickle,csv,numpy,pandas,urllib

from keras.preprocessing.text import Tokenizer

from scipy.stats.mstats import gmean

from functools import reduce

#

#attempt at custom loss

from keras import backend as K



def mse_0(y_true, y_pred):

    """

    #original vision, but numpy methods dont quite work on tensors

    q5h=numpy.where(y_true!=0)

    return keras.losses.mean_squared_error(y_true[q5h], y_pred[q5h])

    """

    """

    q5f=K.eval(y_true)

    q5h=numpy.where(q5f!=0)

    q5i=max(q5h0[-1])+1

    print(y_true)

    q5f=y_true!=0

    q5h=y_true[q5f]

    q5i=y_pred[q5f]

    i=1

    while sum(y_true[0,0,-i:])==0:

     i+=1

    i-=1

    """

    y0=K.relu(y_true[:,:,:], alpha=0.0, max_value=0.00001, threshold=0.000001)/.00001

    #yt=K.dot(y_true[0,:,:],K.transpose(y0))

    #yp=K.dot(y_pred[0,:,:],K.transpose(y0))

    yp=y_pred*y0

    #print(K.int_shape(y0))

    #print(K.int_shape(y_true))

    #print(K.int_shape(y_true))

    #print(K.int_shape(yp))

    #return K.mean(K.square(y_true - y_pred))

    #return K.mean(K.square(y_true - y_pred),axis=-1)

    #return keras.losses.mean_squared_error(y_true,yp)

    return keras.losses.categorical_crossentropy(y_true,yp)





"""

#illustration of keras backend functions

true = K.variable(np.array([[1, 1, 0, 0, 0, 0, 2.0, 3.0]]), dtype='float32')

pred = K.variable(np.array([[0.6, 0.1, 0.2, 0.05, 0.05, 0.0]]), dtype='float32')



K.eval(odds_loss(true, pred))

"""



#exec(open('fc3b','r').read())

#exec(open('fc5b','r').read())

#exec(open('fc5c','r').read())

#exec(open('fc5d','r').read())



class preproc:

  def __init__(self, csv):

    df = pandas.read_csv(csv)

    self.df = df

    df.Date=df.Date.apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    df0= df.iloc[:,2]+';'+df.iloc[:,1].fillna(value='')

    df['region']=df0

    #self.datepen=df['Date'].unique()[-2]

    #self.datefin=df['Date'].unique()[-1]

    #label encode via tokenizer

    #from keras.preprocessing.text import Tokenizer

    #from keras.utils import to_categorical

    t0 = Tokenizer(filters='',split='|')

    self.t0=t0

    t0.fit_on_texts(df.region.unique())

    dfi=t0.texts_to_sequences(df.region)

    df['hot_region']=numpy.array(dfi)



    #def func0(self):

    #clean up if same data for casea nd fatilites repeated; motivated from italy

    #so this is necessary before operation with daily differences in cases, where else it would produe zeros and numpy.nan upon division 

    #df=self.df

    df4=df.iloc[:,4]

    df5=df.iloc[:,5]

    #dh=df[((df4==df4.shift(1)) & (df5==df5.shift(1)))]

    #df=df[~((df4==df4.shift(1)) & (df5==df5.shift(1)))]

    #dg=df[((df4==df4.shift(1)))]

    df=df[~((df4==df4.shift(1)))]

    #df=df[~((df4==df4.shift(1)) & df.iloc[:,4]>100)] #exclude cardinality <100 for first condition

    df=df.reset_index().drop('index',axis=1)

    #self.df=df

    #resetting index is crucial to further shifting operations

    #

    #difference

    for j in df.columns[[4,5]]:

     dfi=df[j] - df[j].shift(1)

     dfi[df.iloc[:,3]=='2020-01-22']=0

     df['diff'+j[:1]]=dfi

    #the purpose of next line is data correction

    df=df[df.diffC>0]

    df=df.reset_index().drop('index',axis=1)

    self.df=df

    

    #Fatality ratio

    dfi=df.Fatalities/df.ConfirmedCases

    dfi[dfi==numpy.inf]=numpy.nan

    df['F_C']=dfi

    dfi=df.diffF/df.diffC

    dfi[dfi==numpy.inf]=numpy.nan

    df['diffF_C']=dfi

    df['diffF_C_']=dfi.rolling(window=k,center=False).mean()

    df['logdiffF_C_']=numpy.log(df['diffF_C_'])

    #self.df=df



  def func0(self,k):

    df=self.df

    self.k=k

    #ratio to prior confirmed

    #initially this was done on total confirmed; however this masks the underlying numbers via ever incrasing tally; so it is done on daily difference 

    #for j in df.columns[-2:]:

    #for i in [4,5]:

    for j in ['diffC',]:

     #for k in [1,3,5]:

     #for k in [1,3]:

     #for k in [1,self.k]:

     for k in [self.k]:

      datek=sorted(df['Date'].unique())[k]

      dfi=df[j] / df[j].shift(k)

      #this sets first entry for each region to nan insetad of ratio with last entry of prior region

      dfi[dfi==numpy.inf]=numpy.nan

      dfi[dfi==0]=numpy.nan

      dfi[df.Date<datek]=numpy.nan

      #if k>1:

      # df['diffC_{}'.format(k)]=dfi

      dfj=(df.Date-df.Date.shift(k)).apply(lambda x: x.days)

      #dfi= dfi**(k/dfj)

      dfi= dfi**(1/dfj)

      df['exp{}'.format(k)+j]=dfi

      dfi = dfi.rolling(window=k,center=False).apply(gmean)

      df['exp{}_'.format(k)+j]=dfi

      #dfj=1.0/(df.Date-df.Date.shift(k)).apply(lambda x: x.days)

      #dfi=dfi.apply(lambda x: x**(1.0/k))

      #dfi= dfi**dfj

      df['logexp{}_'.format(k)+j]=numpy.log(dfi)

      

    

    #computing exp5_diffC.cummax()

    ##dgi=dg[[*dg.columns[2:4]]+[*dg.columns[-2:-1]]]

    ##dgi=dgi[[*dgi.columns[[*[0,-1]]]]]

    ##dgi=dg[[*dg.columns[2:4]]+[*dg.columns[-2:-1]]].groupby(dg.columns[2]).max()

    ##dgi=dgi.groupby(dg.columns[2]).max()

    ##dgi=dg[[*dg.columns[[*[2,3,-2]]]]]

    #dgi=dg[[*dg.columns[[*[3,4,-2]]]]]

    dfi=df[[*df.columns[[*[6,7,-2]]]]]

    ##dgi=pandas.merge(dg,dgi,how='left',on='hot_region')

    ##dgi=pandas.merge(dg,dgi.groupby(dg.columns[2]).max(),how='left',on='hot_region')

    ##dg['expmaxratio']=dg.iloc[:,-3]/dg.iloc[:,-1]

    ##dg['expmaxratio']=dgi.iloc[:,-2]/dgi.iloc[:,-1]

    ##dgi=dgi.groupby(dg.columns[2]).cummax()

    ##dg['cummax{}_ratio'.format(self.k)]=dg.iloc[:,-2]/dgi.iloc[:,-1]

    ##dg['cummmax']=dgi.groupby(dg.columns[2]).cummax().iloc[:,-1]

    #dg['cummmax']=dgi.groupby(dg.columns[3]).cummax().iloc[:,-1]

    #dg['cummax{}_ratio'.format(self.k)]=dg['exp{}_'.format(k)+j]/dg.cummmax

    df['cummmax']=dfi.groupby(df.columns[6]).cummax().iloc[:,-1]

    df['cummax{}_ratio'.format(self.k)]=df['exp{}_'.format(k)+j]/df.cummmax



    #k=5

    #k=3

    #k is carried over from last logexpk column

    #nonnull logexp rows and index reset timed after cummax computation and before time series



    #def func2(self):

    #df=self.df

    #k=self.k

    #model fitting columns

    #dg=df[[*df.columns[4:10]]+[*df.columns[12:]]]

    #dg=df[[*df.columns[4:]]]

    #dg=df[[*df.columns[[*[3,4,5]]]]+[*df.columns[7:]]]

    dg=df[[*df.columns[3:]]]

   

    #nonnull logexp5_diffC rows

    #dg=dg.dropna(axis=0)

    #the above only drops if row isnull

    dg=dg[~dg['logexp{}_diffC'.format(k)].isnull()]

    dg=dg.reset_index() #.drop('index',axis=1)

    

    #specify number of time series columns

    self.tscol=3



    #diff_f/c arranged as timeseries

    for i in range(k+0):

     #dfi[di['bool']==2]= dg['logexp5_diffC'].shift(6-j)[di['bool']==2]

     dg['fc'+str(i+1)]= dg['diffF_C_'].shift(k-i)

     

    """

    #logexp1_diffC arranged as timeseries

    for i in range(k+0):

     #dfi[di['bool']==2]= dg['logexp1_diffC'].shift(6-j)[di['bool']==2]

     dg['l'+str(i+1)]= dg['logexp1_diffC'].shift(k-i)

    """

    

    #exp5.cummax arranged as timeseries

    for i in range(k+0):

     #dfi[di['bool']==2]= dg['logexp5_diffC'].shift(6-j)[di['bool']==2]

     #dh['m'+str(i+1)]= dh['cummaxk_ratio'].shift(k-i)

     dg['m'+str(i+1)]= dg['cummax{}_ratio'.format(k)].shift(k-i)

     

    #logexp5 arranged as timeseries

    for i in range(k+0):

     #dfi[di['bool']==2]= dg['logexp5_diffC'].shift(6-j)[di['bool']==2]

     dg['k'+str(i+1)]= dg['logexp{}_diffC'.format(k)].shift(k-i)

    

    #dgi=pandas.merge(dg,dgi,how='right')

    

    #dbi=(dh['index'].rolling(window=6).min()+5==dh['index'])

    #dgi=di[dbi==True] #this doesn't work

    dbi=(dg['index'].rolling(window=k+1).min()+k==dg['index'])

    dh=dg[dbi]

    #self.dh=dh

    self.dh=dh

    self.dg=dg

    #exec("self.%s = %d" % ('dh{}'.format(k),dh))



  #def func1(self,modelcl=cl0,col=3,pat0=6,datefin0=None):

  def func1(self,modelcl=None,col=3,pat0=6,datefin0=None):

    self.col=col

    x000=self.dh.iloc[:,5] #country

    #y0=d.dh.iloc[:,15].values.reshape(-1,1)

    #y0=d.dh.iloc[:,[15,17]].values.reshape(-1,2)

    #y0=d.dh.iloc[:,[14,16]].values.reshape(-1,2)

    y0=self.dh.loc[:,['logexp{}_diffC'.format(k),'cummax{}_ratio'.format(k)]].values.reshape(-1,2)

    

    #in this first attempt we are clustering on raw training data itself just to see 

    sc_X = StandardScaler()

    #y3 = numpy.concatenate([dh.iloc[:,:3],y2],axis=1)

    x2=x0.reshape(-1,col*self.k)

    x2 = sc_X.fit_transform(x2)

    self.sc_X=sc_X

    

    km = KMeans(n_clusters=k0)

    #km.fit(v3.reshape(-1,4))

    #km.predict(v3.reshape(-1,4))

    km.fit(x2) #.reshape(-1,1))

    km.predict(x2) #.reshape(-1,1))

    #for i in range(k0):

    #print(labels.tolist().count(i))

    labels = km.labels_ +1

    self.km=km

    

    #x001=d.dh.labels

    #x00=to_categorical(x000)

    x00=to_categorical(x000,num_classes=self.t0.document_count+1)

    x01=to_categorical(labels,num_classes=self.km.n_clusters+1)

    #x01=to_categorical(x001)

    #x00=numpy.concatenate([x00,x01],axis=1)

    

    outputs=y0

    #cl1=cl0(inputs,outputs)

    cl1=modelcl(inputs,outputs)

    self.cl1=cl1

    cl1.fit0(pat=pat0)

    cl1.compile()

    #cl1.fit()

    cl1.xgbpipe()



  def func2(self,model=None,datefin0=None,rounds=30):

    k=self.k

    dg=self.dg

    #dh=self.dh

    if not hasattr(self,'dj'):

     self.dj=self.df.iloc[:,3:]

     self.dj=self.dj.reset_index() #.drop('index',axis=1)

    dj=self.dj

    if datefin0==None:

      #datefin=dj['Date'].unique().max()

      datenew=dj['Date'].unique().max()

    else:

      #datefin=datefin0

      datenew=datefin0

    #di=dg[dg.Date==datefin]

    #di=dg[dg.Date<=datefin]

    #di=dg[dg.Date==datenew].copy()

    #

    #dii=dg.groupby('hot_region')[['index','Date']].max()

    dih=dg.groupby('hot_region')[['index']].max()

    #dii=d.dg['index'].isin(dih.iloc[:,0].values)

    dii=dg['index'].isin(dih['index'].values)

    for i in range(rounds):

     if i==0:

      #australian capital territory shows di.Date.max() might be omitted for some regions due to identicality omission prior

      di=dg[dii].copy()

     else:

      di=dg.loc[dg.Date==datenew].copy()

     #strangely di does not retain its memory of x3 after the second loop unless prior line included within loop

     #also for reference if not already done di needs to be in sorted order by hot_region, Date

     di.iloc[:,-self.tscol*k:]=di.iloc[:,-self.tscol*k:].shift(-1,axis=1)

     #update fc5,l5,m5,k5

     #di['fc{}'.format(k)]=di['diffF_C']

     #di['l{}'.format(k)]=di['logexp1_diffC']

     #di['m{}'.format(k)]=di['cummax{}_ratio'.format(k)]

     #di['k{}'.format(k)]=di['logexp{}_diffC'.format(k)]

     di.loc[:,'fc{}'.format(k)]=di['diffF_C_']

     #di.loc[:,'l{}'.format(k)]=di['logexp1_diffC']

     di.loc[:,'m{}'.format(k)]=di['cummax{}_ratio'.format(k)]

     di.loc[:,'k{}'.format(k)]=di['logexp{}_diffC'.format(k)]

     if model!=None:

       """

       if hot_region!=None:

        for j in hot_region:

         pass

         x00=numpy.zeros((1,self.t0.document_count))

         x00[0,j]=1

         model1=model[j]

       else:

        pass

        #model = load_model('model.h5')

        model1 = load_model(model)

       """

       #x000=d.di.iloc[:,5] #country

       x000=di.iloc[:,5] #country

       #x00=to_categorical(x000,num_classes=304+1)

       x00=to_categorical(x000,num_classes=self.t0.document_count+1)

       x2 = self.sc_X.transform(x2)

       labels = self.km.predict(x2)+1

       x01=to_categorical(labels,num_classes=self.km.n_clusters+1)

       #x01=to_categorical(x001)

       #x00=numpy.concatenate([x00,x01],axis=1)

 

       inputs=[x00, x01,x0]

       #inputs=[x00,x0]

       #x3=cl1.model2.predict(inputs)

       #x3=model.predict(inputs).reshape(-1,)

       x3=model.predict(inputs)[:,0].reshape(-1,)

       #di.iloc[:,14]=x3

       di.loc[:,'logexp{}_diffC'.format(k)]=x3

       #

       #di.iloc[:,13]=numpy.exp(x3)

       di.loc[:,'exp{}_diffC'.format(k)]=numpy.exp(x3)

       #dj2.loc[dji,'exp{}_diffC'.format(k)]=numpy.exp(x3)

       #di.loc[:,'exp{}_diffC'.format(k)]=dj2['exp{}diffC'.format(k)].rolling(window=k,center=False).apply(gmean)[dji]

       #di.cummmax=di[[*[di.columns[13,15]]]].max(axis=1)

       #di.cummmax=di[['cummax{}_ratio'.format(k),'cummmax']].max(axis=1)

       di.cummmax=numpy.maximum(di.cummmax,di['exp{}_diffC'.format(k)])

       di.loc[:,'cummax{}_ratio'.format(k)]=di['exp{}_diffC'.format(k)]/di.cummmax

       #

       #datenew=datefin+numpy.timedelta64(1,'D')

       datenew=datenew+numpy.timedelta64(1,'D')

       self.datenew=datenew

       di.Date=datenew

       #

       #dj2 defined for computation

       #dj=pandas.concat([dj.iloc[:,3:],di.iloc[:,1:15]],axis=0)

       #dj2=pandas.concat([dj,di.iloc[:,1:18]],axis=0)

       dj2=pandas.concat([dj,di.iloc[:,1:-self.tscol*k]],axis=0)

       #dj=dj.sort_values([dj.columns[4],dj.columns[0]])

       dj2=dj2.sort_values(['hot_region','Date'])

       dji=dj2.Date==datenew

       #

       die=numpy.exp(x3)**k

       for j in range(1,k):

         die/=dj2['exp{}diffC'.format(k)].shift(j)[dji]

       di.loc[:,'exp{}diffC'.format(k)]=die

       #dj2.loc[dji,'exp{}diffC'.format(k)]=die

       #

       ##djh=(dj.diffC.shift(k)[dji])*(dj.exp5_diffC[dji])

       #djj=(dj2.Date-dj2.Date.shift(k)).apply(lambda x: x.days)

       ##djh=(dj2.diffC.shift(k)[dji])*(dj2.exp5_diffC[dji])**djj[dji]

       ##djh=(dj2.diffC.shift(k)[dji])*(dj2.loc[dji,'exp{}_diffC'.format(k)])**djj[dji]

       #djh=(dj2.diffC.shift(k)[dji])*(dj2.loc[dji,'exp{}diffC'.format(k)])**djj[dji]

       ##dj.diffC[dji]=djh

       djh=(dj2.diffC.shift(1)[dji])*(dj2.loc[dji,'exp{}_diffC'.format(k)])

       #dj.ConfirmedCases[dji]=dj.ConfirmedCases.shift(1)[dji]+dj.diffC[dji]

       #dj.exp1_diffC[dji]=dj.diffC[dji]/dj.diffC.shift(1)[dji]

       #dj['logexp{}_diffC'.format(1)][dji]=numpy.log(dj.exp1_diffC[dji])

       #dj.loc[dji,'diffC']=djh

       #dj.loc[dji,'ConfirmedCases']=dj.ConfirmedCases.shift(1)[dji]+dj.diffC[dji]

       #dj.loc[dji,'exp1_diffC']=dj.diffC[dji]/dj.diffC.shift(1)[dji]

       #dj.loc[dji,'logexp{}_diffC'.format(1)]=numpy.log(dj.exp1_diffC[dji])

       di.diffC=djh

       di.ConfirmedCases=dj2.ConfirmedCases.shift(1)[dji]+djh

       #di.exp1_diffC=djh/dj2.diffC.shift(1)[dji]

       #di.logexp1_diffC=numpy.log(di.exp1_diffC)

       """

       diffF_C

       diffF

       Fatalities

       """

       dg=pandas.concat([dg,di],axis=0)

       self.dg=dg.sort_values(['hot_region','Date'])

       dj=pandas.concat([dj,di.iloc[:,1:-self.tscol*k]],axis=0)

       self.dj=dj

     #self.di=di

     #self.dj=dj





class postproc:

  def __init__(self, csv='/kaggle/input/covid19-global-forecasting-week-4/train.csv'):

    self.df0 = pandas.read_csv(csv)

    self.df0.Date=self.df0.Date.apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    self.datefin=self.df0['Date'].unique().max()

    t0 = Tokenizer(filters='',split='|')

    self.t0=t0

    dfa= self.df0.iloc[:,2]+';'+self.df0.iloc[:,1].fillna(value='')

    #t0.fit_on_texts(df0.region.unique())

    #self.t0a=t0.texts_to_sequences(df.region)

    t0.fit_on_texts(dfa.unique())

    dfb=self.t0.texts_to_sequences(dfa)

    self.df0['hot_region']=numpy.array(dfb)

    self.df0=self.df0.drop([*self.df0.columns[:3]],axis=1)

    #self.t0a=t0.texts_to_sequences(dfa)

    #check d.t0a equals df.hot_region

    #

  def func3(self,csv='/kaggle/input/covid19-global-forecasting-week-4/test.csv'):

    df1 = pandas.read_csv(csv)

    self.df1=df1

    #self.df1=df1

    df1.Date=df1.Date.apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    dfa= df1.iloc[:,2]+';'+df1.iloc[:,1].fillna(value='')

    #self.datepen=df['Date'].unique()[-2]

    #self.datefin=df['Date'].unique()[-1]

    #label encode via tokenizer

    #from keras.preprocessing.text import Tokenizer

    #from keras.utils import to_categorical

    dfb=self.t0.texts_to_sequences(dfa)

    df1['hot_region']=numpy.array(dfb)

    self.df1=df1.drop([*df1.columns[1:3]],axis=1)

    #datek=sorted(df['Date'].unique())[k]

  def func4(self,csv='/kaggle/input/submissions/sub0.csv'):

    df = pandas.read_csv(csv)

    self.df = df

    self.df.Date=self.df.Date.apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    datefin=self.datefin

    dfi=self.df0.Date<=datefin

    dfj=self.df.Date>datefin

    #self.df1 =df1.merge(self.df[['Date','hot_region','ConfirmedCases','Fatalities']], how='left',  on=['Date','hot_region'])

    #self.df2 =self.df1.merge(self.df.loc[dfi,['Date','hot_region','ConfirmedCases','Fatalities']], how='left',  on=['Date','hot_region'])

    #self.df3 =self.df1.merge(self.df.loc[dfj,['Date','hot_region','ConfirmedCases','Fatalities']], how='left',  on=['Date','hot_region'])

    self.df3 =self.df1.merge(self.df0.loc[dfi,:], how='left',  on=['Date','hot_region'])

    self.df4 =self.df1.merge(self.df.loc[dfj,:].drop(['Unnamed: 0','Fatalities'],axis=1), how='left',  on=['Date','hot_region'])

    self.df3.update(self.df4)

  def func5(self,csv='/kaggle/input/submissions/sub2a.csv'):

    self.df2 = pandas.read_csv(csv)

    dfj=self.df1.Date<=self.datefin

    self.df2.loc[dfj,:]=numpy.nan

    #self.df2 = self.df2.drop('ConfirmedCases',axis=1)

    self.df3.update(self.df2)

    self.df3=self.df3.drop([*self.df3.columns[1:3]],axis=1)







if not __name__ == '__main__':

 #k=3

 k=5

 #d=preproc(csv='train.csv')

 d=preproc(csv='/kaggle/input/covid19-global-forecasting-week-4/train.csv')

 d.func0(k)

 #d.func1(modelcl=cl0,col=2,pat0=12)

 #d.func2(model=cl1.model2,datefin0=None,rounds=30)

 #d.func2(model=cl1,datefin0=None,rounds=30) #this is just cl1.predict method, which is chained model1 predict followed by xgboost

 #d.func2(model=d.cl1.model0,datefin0=None,rounds=2)

 #d.func2(model=d.cl1.model2,datefin0=None)

 #d.func2(model=d.cl1.model0,datefin0=None)

 #url='https://github.com/drcyle/kaggle/blob/master/sub1.csv'

 #url='https://raw.githubusercontent.com/drcyle/kaggle/master/sub1.csv'

 #sub=pandas.read_csv(url)

 #sub.drop(sub.columns[0],axis=1).to_csv('submission.csv',index=False)

 """

 df0.read_csv('sub4.csv')

 df1=df0[['Date','hot_region','ConfirmedCases','Fatalities']]

 df1.to_csv('sub0.csv')

 """

 e=postproc()

 e.func3()

 e.func4()

 e.func5()

 """

 e.df3.ConfirmedCases[140]=0

 e.df3.ConfirmedCases[174]=0

 e.df3.ConfirmedCases[175]=0

 e.df3.ConfirmedCases[181]=0

 e.df3.ConfirmedCases[182]=0

 """

 e.df3.to_csv('submission.csv',index=False)

    

"""

ERROR: Could not parse '' into expected type of Double (Line 140, Column 5)

ERROR: Could not parse '' into expected type of Double (Line 140, Column 6)

ERROR: Could not parse '' into expected type of Double (Line 174, Column 5)

ERROR: Could not parse '' into expected type of Double (Line 174, Column 6)

ERROR: Could not parse '' into expected type of Double (Line 175, Column 5)

ERROR: Could not parse '' into expected type of Double (Line 175, Column 6)

ERROR: Could not parse '' into expected type of Double (Line 181, Column 5)

ERROR: Could not parse '' into expected type of Double (Line 181, Column 6)

ERROR: Could not parse '' into expected type of Double (Line 182, Column 5)

ERROR: Could not parse '' into expected type of Double (Line 182, Column 6)

"""



if __name__ == '__main__':

 #k=3

 subf=pandas.read_csv('/kaggle/input/submissions/sub3.csv')

 subf.to_csv('submission.csv',index=False)

 subf=subf.drop(['Unnamed: 0'],axis=1)

 subf.ForecastId=subf.ForecastId.astype(int)

 subf.to_csv('submission.csv',index=False)




