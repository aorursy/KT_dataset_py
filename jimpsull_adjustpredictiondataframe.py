# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir('../input/blendmodels'))

# Any results you write to the current directory are saved as output.
trmeta=pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')
temeta=pd.read_csv('../input/PLAsTiCC-2018/test_set_metadata.csv')

pdf=pd.read_csv('../input/blendmodels/blend_submission.csv')

print(trmeta.shape)
print(temeta.shape)
print(pdf.shape)
trmeta=trmeta.fillna(0)
temeta=temeta.fillna(0)

trIgFilt=trmeta.loc[:,'distmod']==0
trEgFilt=trmeta.loc[:,'distmod']!=0
teIgFilt=temeta.loc[:,'distmod']==0

igClasses=trmeta.loc[trIgFilt,'target'].unique()
egClasses=trmeta.loc[trEgFilt,'target'].unique()


trIgFrac = trIgFilt.sum()/trmeta.shape[0]
teIgFrac = teIgFilt.sum()/temeta.shape[0]


trEgFrac = 1.0 - trIgFrac
teEgFrac = 1.0 - teIgFrac


print(trIgFrac)
print(teIgFrac)
import copy
def setZeroProbas(opdf, temeta, igClasses, egClasses):
    pdf=copy.deepcopy(opdf)
    pdf=pdf.merge(temeta, on='object_id')
    rdf=pd.DataFrame()
    rdf['object_id']=pdf['object_id']

    igFilter=pdf.loc[:,'distmod']==0
    egFilter=pdf.loc[:,'distmod']!=0
    
    for eg in egClasses:
        pdf.loc[igFilter,'class_' + str(eg)]=0
        rdf.loc[:,'class_' + str(eg)]=pdf.loc[:,'class_' + str(eg)]
        
    for ig in igClasses:
        pdf.loc[egFilter,'class_' + str(ig)]=0
        rdf.loc[:,'class_' + str(ig)]=pdf.loc[:,'class_' + str(ig)]
    
    rdf['class_99']=pdf['class_99']
    return rdf

predictions=setZeroProbas(pdf, temeta, igClasses, egClasses)
predictions.to_csv('justSetZeroProbas.csv', index=False)

#predictions=rdf
predictions.describe()
fil={}
distmod={}
sigma={}

trFracs={}
teFracs={}
teMult={}


for i in trmeta.loc[:,'target'].unique():
    fil[i] = trmeta['target']==i
    distmod[i] = np.average(trmeta.loc[fil[i],'distmod'])
    sigma[i] = np.std(trmeta.loc[fil[i],'distmod'])
    
    print('class ' + str(i) + ': ' + str(distmod[i]) +' +/- ' + str(sigma[i]))
    print(fil[i].sum())
    
for ig in igClasses:
    trFracs[ig]=fil[ig].sum() / trmeta.shape[0]
    teFracs[ig]=trFracs[ig] * teIgFrac / trIgFrac
    print('class ' + str(ig) + 'tr : ' + str(trFracs[ig]) +', te : ' + str(teFracs[ig]))
    #teMult[ig]=teFracs[ig] / np.average(rdf.loc[:,'class_' + str(ig)])
    #print(teMult[ig])
    
for eg in egClasses:
    trFracs[eg]=fil[eg].sum() / trmeta.shape[0]
    teFracs[eg]=trFracs[eg] * teEgFrac / trEgFrac
    print('class ' + str(eg) + 'tr : ' + str(trFracs[eg]) +', te : ' + str(teFracs[eg]))
    #teMult[eg]=teFracs[eg] / np.average(rdf.loc[:,'class_' + str(eg)])
    #print(teMult[eg])
    

def applyRebalance(ordf, teMult):
    
    rdf=copy.deepcopy(ordf)
    for cindex in rdf.columns:
        if cindex != 'object_id':
            theClass = int(cindex[6:])
            print(theClass)
            rdf[cindex]*=teMult[theClass]
            
    return rdf
#fadf=applyRebalance(rdf, teMult)
#fadf.describe()
#from Scirpus discussion:

def GenUnknown(data):
    return ((((((data["mymedian"]) + (((data["mymean"]) / 2.0)))/2.0)) + (((((1.0) - (((data["mymax"]) * (((data["mymax"]) * (data["mymax"]))))))) / 2.0)))/2.0)
#predictions['class_99'] = 1 - predictions.max(axis=1)
#predictions['object_id'] = object_ids

#pdf=predictions
feats = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']

y = pd.DataFrame()
y['mymean'] = predictions[feats].mean(axis=1)
y['mymedian'] = predictions[feats].median(axis=1)
y['mymax'] = predictions[feats].max(axis=1)

predictions['class_99'] = GenUnknown(y)
#meta=pd.read_csv('../input/PLAsTiCC-2018/test_set_metadata.csv')
#import copy
def modUnknown(opdf, meta, ddfMult=0.5, mwMult=0.5, preserveMed=False):
    pdf=copy.deepcopy(opdf)
    mdf=pdf.merge(meta,on='object_id')
    ddfilter=mdf.loc[:,'ddf']==1
    mwfilter=mdf.loc[:,'hostgal_photoz']==0
    print(ddfilter.sum())
    print(mwfilter.sum())
    
    mdf.loc[mwfilter,'class_99']=mwMult*mdf.loc[mwfilter,'class_99']
    mdf.loc[ddfilter,'class_99']=ddfMult*mdf.loc[ddfilter,'class_99']
    pdf.loc[:,'class_99']=mdf.loc[:,'class_99']
    
    return pdf

npdf=modUnknown(predictions, temeta)
npdf.head()
npdf.describe()
npdf.to_csv('probaZeroAndRe99.csv', index=False)
