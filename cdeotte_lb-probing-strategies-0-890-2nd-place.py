import numpy as np, pandas as pd, os

np.random.seed(300)



# GENERATE RANDOM DATA

data = pd.DataFrame(np.zeros((20000,300)))

for i in range(300): data.iloc[:,i] = np.random.normal(0,1,20000)



# SET TARGET AS LINEAR COMBINATION OF 50 A'S PLUS NOISE 

important = 35; noise = 3.5

a = np.random.normal(0,1,300)

x = np.random.choice(np.arange(300),300-important,replace=False); a[x] = 0

data['target'] = data.values.dot(a) + np.random.normal(0,noise,20000)



# MAKE 64% TARGET=1, 36% TARGET=0

data.sort_values('target',inplace=True)

data.iloc[:7200,300] = 0.0

data.iloc[7200:,300] = 1.0



# RANDOMLY SELECT TRAIN, PUBLIC, PRIVATE

train = data.sample(250)

public = data[ ~data.index.isin(train.index) ].sample(1975)

private = data[ ~data.index.isin(train.index) & ~data.index.isin(public.index) ].sample(frac=1) 



# RESET INDICES

train.reset_index(drop=True,inplace=True)

public.reset_index(drop=True,inplace=True)

private.reset_index(drop=True,inplace=True)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import roc_auc_score



oof = np.zeros(len(train))

predsPU= np.zeros(len(public))

predsPR= np.zeros(len(private))

rskf = RepeatedStratifiedKFold(n_splits=25, n_repeats=5)

for train_index, test_index in rskf.split(train.iloc[:,:-1], train['target']):

    clf = LogisticRegression(solver='liblinear',penalty='l2',C=1.0,class_weight='balanced')

    clf.fit(train.loc[train_index].iloc[:,:-1],train.loc[train_index]['target'])

    oof[test_index] += clf.predict_proba(train.loc[test_index].iloc[:,:-1])[:,1]

    predsPU += clf.predict_proba(public.iloc[:,:-1])[:,1]

    predsPR += clf.predict_proba(private.iloc[:,:-1])[:,1]

aucTR = round(roc_auc_score(train['target'],oof),5)

aucPU = round(roc_auc_score(public['target'],predsPU),5)

aucPR = round(roc_auc_score(private['target'],predsPR),5)

print('LR Model with L2-penalty: CV =',aucTR,'LB =',aucPU,'Private score =',aucPR)
oof = np.zeros(len(train))

predsPU= np.zeros(len(public))

predsPR= np.zeros(len(private))

rskf = RepeatedStratifiedKFold(n_splits=25, n_repeats=5)

for train_index, test_index in rskf.split(train.iloc[:,:-1], train['target']):

    clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.1,class_weight='balanced')

    clf.fit(train.loc[train_index].iloc[:,:-1],train.loc[train_index]['target'])

    oof[test_index] += clf.predict_proba(train.loc[test_index].iloc[:,:-1])[:,1]

    predsPU += clf.predict_proba(public.iloc[:,:-1])[:,1]

    predsPR += clf.predict_proba(private.iloc[:,:-1])[:,1]

aucTR = round(roc_auc_score(train['target'],oof),5)

aucPU = round(roc_auc_score(public['target'],predsPU),5)

aucPR = round(roc_auc_score(private['target'],predsPR),5)

print('LR Model with L1-penalty: CV =',aucTR,'LB =',aucPU,'Private score =',aucPR)
# START WITH BEST TRAINING DATA MODEL

clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.1, class_weight='balanced')

clf.fit(train.iloc[:,:-1],train['target']) 

u0 = clf.coef_[0]

u0 = u0 / np.sqrt(u0.dot(u0))



# INITIAL SCORES

aucPU = round(roc_auc_score(public['target'],u0.dot(public.iloc[:,:-1].values.transpose())),5)

aucPR = round(roc_auc_score(private['target'],u0.dot(private.iloc[:,:-1].values.transpose())),5)

bestPU = aucPU; currentPR = aucPR; initial = u0.copy()

print('Our starting model has LB =',aucPU,'and Private score =',aucPR)



# ACCELERATE RANDOM SEARCH BY NEGLECTING 250 LEAST IMPORTANT VARIABLES FROM TRAINING DATA

df = pd.DataFrame({'var':np.arange(300),'CV':np.zeros(300),'diff':np.zeros(300)})

for i in range(300):

    df.loc[i,'CV'] = roc_auc_score(train['target'],train[i])

    df.loc[i,'diff'] = abs(df.loc[i,'CV']-0.5)

df.sort_values('diff',inplace=True,ascending=False)
u0 = initial

angDeg = 5

angRad = 2*np.pi*angDeg/360



np.random.seed(42)

for k in range(150):

    # CHOOSE RANDOM SEARCH DIRECTION

    u1 = np.random.normal(0,1,300)

    # REMOVE 250 UNIMPORTANT DIMENSIONS

    u1[ df.iloc[100:,0] ] = 0.0

    # ROTATE 5 DEGREES IN THIS NEW DIRECTION

    u1 = u1 - u1.dot(u0)*u0

    u1 = u1 / np.sqrt(u1.dot(u1))

    u2 = u0*np.cos(angRad) + u1*np.sin(angRad)

    # CALCULATE LB AND PRIVATE SCORE

    aucPU = round(roc_auc_score(public['target'],u2.dot(public.iloc[:,:-1].values.transpose())),5)

    aucPR = round(roc_auc_score(private['target'],u2.dot(private.iloc[:,:-1].values.transpose())),5)

    # IF SCORE INCREASES PRINT RESULTS

    if (aucPU>bestPU)|(k==0):

        bestPU = aucPU

        currentPR = aucPR

        u0 = u2.copy()

        print('Submission',k+1,': Best LB =',bestPU,'and Private score =',currentPR)
df = pd.DataFrame({'var':np.arange(300),'CV':np.zeros(300),'diff':np.zeros(300)})

for i in range(300):

    df.loc[i,'CV'] = roc_auc_score(train['target'],train[i])

    df.loc[i,'diff'] = abs(df.loc[i,'CV']-0.5)

print('We need to LB probe',len(df.loc[ df['diff']>0.04 ,'CV']),'variables')
# LB PROBE 100 VARIABLES STARTING WITH MOST IMPORTANT CV SCORE

df.sort_values('diff',inplace=True,ascending=False)

LBprobe = list(df.loc[ df['diff']>0.04, 'var'])

df.sort_values('var',inplace=True)



# INITIALIZE VARIABLES

df['LB'] = 0.5; df['A'] = 0; ct=0



# PERFORM LB PROBING TO DETERMINE A_K'S

keep = []

for i in LBprobe:

    ct += 1; found = True

    # CALCUATE LB SCORE FOR VAR_K

    df.loc[i,'LB'] = roc_auc_score(public['target'],public[i])

    if (df.loc[i,'LB']<0.47) | (df.loc[i,'LB']>0.53): keep.append(i) 

    else: found = False

    # UPDATE A_K'S

    df.loc[keep,'A'] = (8/9)*df.loc[keep,'LB']+(1/9)*df.loc[keep,'CV']-0.5

    # PREDICT PUBLIC

    predPU = df['A'].values.dot(public.iloc[:,:300].values.transpose())

    aucPU = round( roc_auc_score(public['target'],predPU) ,3)

    # PREDICT PRIVATE

    predPR = df['A'].values.dot(private.iloc[:,:300].values.transpose())

    aucPR = round( roc_auc_score(private['target'],predPR) ,3)

    # DISPLAY CURRENT LB AND PRIVATE SCORE

    if found: print('Submission',ct,': Best LB =',aucPU,'and Private score ='

            ,aucPR,'with',len(keep),'keep')
train = pd.read_csv('../input/train.csv')

df = pd.DataFrame({'var':np.arange(300),'CV':np.zeros(300),'diff':np.zeros(300),'LB':0.5*np.ones(300)})

for i in range(300):

    df.loc[i,'CV'] = roc_auc_score(train['target'],train[str(i)])

    df.loc[i,'diff'] = abs(df.loc[i,'CV']-0.5)

df.sort_values('diff',inplace=True,ascending=False)

df.head()
df.loc[ df['var']==33, 'LB' ] = 0.671

df.loc[ df['var']==65, 'LB' ] = 0.671

df.loc[ df['var']==217, 'LB' ] = 0.382

df.loc[ df['var']==117, 'LB' ] = 0.405

df.loc[ df['var']==91, 'LB' ] = 0.382
df.loc[ df['var']==295, 'LB' ] = 0.506

df.loc[ df['var']==24, 'LB' ] = 0.501

df.loc[ df['var']==199, 'LB' ] = 0.613

df.loc[ df['var']==80, 'LB' ] = 0.483

df.loc[ df['var']==73, 'LB' ] = 0.394
df.loc[ df['var']==194, 'LB' ] = 0.472

df.loc[ df['var']==189, 'LB' ] = 0.454

df.loc[ df['var']==16, 'LB' ] = 0.437

df.loc[ df['var']==183, 'LB' ] = 0.506

df.loc[ df['var']==82, 'LB' ] = 0.494
df.loc[ df['var']==258, 'LB' ] = 0.474

df.loc[ df['var']==63, 'LB' ] = 0.428

df.loc[ df['var']==298, 'LB' ] = 0.466

df.loc[ df['var']==201, 'LB' ] = 0.489

df.loc[ df['var']==165, 'LB' ] = 0.498
df.loc[ df['var']==133, 'LB' ] = 0.486

df.loc[ df['var']==209, 'LB' ] = 0.431

df.loc[ df['var']==164, 'LB' ] = 0.542

df.loc[ df['var']==129, 'LB' ] = 0.506

df.loc[ df['var']==134, 'LB' ] = 0.471
df.loc[ df['var']==226, 'LB' ] = 0.492

df.loc[ df['var']==237, 'LB' ] = 0.474

df.loc[ df['var']==39, 'LB' ] = 0.488

df.loc[ df['var']==17, 'LB' ] = 0.508

df.loc[ df['var']==30, 'LB' ] = 0.494
df.loc[ df['var']==114, 'LB' ] = 0.479

df.loc[ df['var']==272, 'LB' ] = 0.471

df.loc[ df['var']==108, 'LB' ] = 0.459

df.loc[ df['var']==220, 'LB' ] = 0.511

df.loc[ df['var']==150, 'LB' ] = 0.482
df.loc[ df['var']==230, 'LB' ] = 0.482

df.loc[ df['var']==90, 'LB' ] = 0.486

df.loc[ df['var']==289, 'LB' ] = 0.482

df.loc[ df['var']==241, 'LB' ] = 0.494

df.loc[ df['var']==4, 'LB' ] = 0.516
df.loc[ df['var']==43, 'LB' ] = 0.474

df.loc[ df['var']==239, 'LB' ] = 0.457

df.loc[ df['var']==127, 'LB' ] = 0.501

df.loc[ df['var']==45, 'LB' ] = 0.449

df.loc[ df['var']==151, 'LB' ] = 0.504
df.loc[ df['var']==244, 'LB' ] = 0.509

df.loc[ df['var']==26, 'LB' ] = 0.511

df.loc[ df['var']==105, 'LB' ] = 0.505

df.loc[ df['var']==176, 'LB' ] = 0.525

df.loc[ df['var']==101, 'LB' ] = 0.535
df['A'] = 0

df['A'] = (8/9)*df['LB'] + (1/9)*df['CV'] - 0.500

keep_threshold = 0.04 # YIELDS 15 NON-ZEROS A'S

df.loc[ abs(df['A'])<keep_threshold , 'A' ] = 0

df.sort_values('var',inplace=True)

for i in range(300):

    if df.loc[i,'LB'] != 0.500:           

        print('A_'+str(i)+' = ',round(df.loc[i,'A'],6))
test = pd.read_csv('../input/test.csv')

pred = test.iloc[:,1:].values.dot(df['A'].values)

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = pred

sub.to_csv('submission.csv',index=False)
import seaborn as sns

import matplotlib.pyplot as plt



idx = df.loc[ df['A']!=0, 'var' ].values

idx = np.sort(idx); idx2 = []

for i in idx: idx2.append(str(i))



plt.figure(figsize=(15,15))

sns.heatmap(train[idx2+['target']].corr(), cmap='RdBu_r', annot=True, center=0.0)

plt.title('Correlation Among Useful Variables',fontsize=20)

plt.show()