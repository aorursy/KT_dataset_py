import requests,pickle,os,pandas as pd,numpy as np

from time import time

begins=time()



#target='p77d3_xlmr_300fast_.tgz'#older version more runs ..

target='p77d4_xlmr_300fast_.tgz'

target='P8b_DistillRoberta_192_Fast_.pickle'

target='p77d4_xlmr192Fast_.tgz'

target='p8_p2xlmr192Fast_All8.tgz';#0.8275 -- Left, pas terrible ( moins entrainé pour l'instant .. )

target='p8_p2xlmr192Fast_NotEn_17.tgz'#0.854, puis 0.854 toujours pas terrible -- Right,last one -- supposed be the best -- needs more training

target='p8_p2xlmr192Fast_NotEn7.tgz'#0.9178, puis 0.9187 -- Center excluding en train, has more trainings -- works better

target='ptotal.tgz'#0.9178, puis 0.9187 -- Center excluding en train, has more trainings -- works better





avg=1

latestResults='http://1.x24.fr/a/jupyter/poc7/'#

r=requests.get(latestResults+target,stream=True)



with open(target,'wb') as f:

    f.write(r.raw.read())       



if target.endswith('.tgz'):

    os.system('tar xf '+target+';rm -f '+target)



target=target.replace('.tgz','.pickle')

data = open(target, "rb")

preds = pickle.load(data)

data.close()

os.system('rm '+target)



sep=pd.DataFrame({})#



if'Direct':

    sep['id']=list(preds.keys())

    sep['toxic']=list(preds.values())

sep=sep.sort_values(by='id')

fn='submission.csv'

sep['id,toxic'.split(',')].to_csv(fn,index=False)

#display(sep)

print('exec time:',round(time()-begins),'sec')

assert(False)
%%script False

sep=pd.DataFrame({'id':list(range(0,63812))})



if(type(preds)==list):

  j=0;res={}

  for i in preds:

    p(i)

    res[j]=i

    j+=1

  preds=res



if True & bool('dict'):

  pk=preds.keys();

  for i in pk:

    if type(preds[i][0])==np.ndarray:

      preds[i]=[ji[0] for ji in preds[i]]



print('number of predictions',len(pk))    

    

df=pd.DataFrame(preds)

means=df.mean(axis=1).values

    

pk=list(preds.keys())



if avg:

    sep['toxic']=means

else:

    sep['toxic']=preds[pk[-1]];#last prediction : scores less than the average of submissions



fn='submission.csv'

sep['id,toxic'.split(',')].to_csv(fn,index=False)

print('exec time:',round(time()-begins),'sec')