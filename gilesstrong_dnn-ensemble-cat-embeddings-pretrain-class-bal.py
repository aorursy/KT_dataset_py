!git clone https://github.com/fastai/fastai.git
!mv fastai/fastai/ fastai/Fastai
import sys
sys.path.append('./fastai/')
%matplotlib inline
%reload_ext autoreload
%autoreload 2
from Fastai.structured import *
from Fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from sklearn.model_selection import train_test_split, StratifiedKFold
!cp ../input/*csv .
PATH='./'
train = pd.read_csv(f'{PATH}train.csv')
test = pd.read_csv(f'{PATH}test.csv')
len(train),len(test)
test.loc[test['rez_esc'] == 99.0 , 'rez_esc'] = 5
train.drop(columns=[x for x in train.columns if 'SQB' in x or x == 'agesq'], inplace=True)
test.drop(columns=[x for x in test.columns if 'SQB' in x or x == 'agesq'], inplace=True)
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)
train.columns[train.isna().any()].tolist(), test.columns[test.isna().any()].tolist()
#Fill na (from https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm)
def repalce_v18q1(x):
    if x['v18q'] == 0:
        return x['v18q']
    else:
        return x['v18q1']

train['v18q1'] = train.apply(lambda x : repalce_v18q1(x),axis=1)
test['v18q1'] = test.apply(lambda x : repalce_v18q1(x),axis=1)

train['v2a1'] = train['v2a1'].fillna(value=train['tipovivi3'])
test['v2a1'] = test['v2a1'].fillna(value=test['tipovivi3'])
train['rez_esc'] = train.v18q1.fillna(0).astype(np.int32)
test['rez_esc'] = test.v18q1.fillna(0).astype(np.int32)
train['meaneduc'] = train.v18q1.fillna(0).astype(np.float32)
test['meaneduc'] = test.v18q1.fillna(0).astype(np.float32)
train.columns[train.isna().any()].tolist(), test.columns[test.isna().any()].tolist()
train['roof_waste_material'] = np.nan
test['roof_waste_material'] = np.nan
train['electricity_other'] = np.nan
test['electricity_other'] = np.nan

def fill_roof_exception(x):
    if (x['techozinc'] == 0) and (x['techoentrepiso'] == 0) and (x['techocane'] == 0) and (x['techootro'] == 0):
        return 1
    else:
        return 0
    
def fill_no_electricity(x):
    if (x['public'] == 0) and (x['planpri'] == 0) and (x['noelec'] == 0) and (x['coopele'] == 0):
        return 1
    else:
        return 0

train['roof_waste_material'] = train.apply(lambda x : fill_roof_exception(x),axis=1)
test['roof_waste_material'] = test.apply(lambda x : fill_roof_exception(x),axis=1)
train['electricity_other'] = train.apply(lambda x : fill_no_electricity(x),axis=1)
test['electricity_other'] = test.apply(lambda x : fill_no_electricity(x),axis=1)
train.head().T.head(142)
for c in train.columns:
    print(c, len(set(train[c])))
ignore = [x for x in train.columns if x == 'Target' or x == 'idhogar'] + ['edjefe', 'edjefa', 'Id']
train[train.idhogar == 'fd8a6d014'].T.head(142)
#from https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
train['escolari_age'] = train['escolari']/train['age']
test['escolari_age'] = test['escolari']/test['age']
#from https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
df_train = pd.DataFrame()
df_test = pd.DataFrame()

aggr_mean_list = ['rez_esc', 'dis', 'male', 'female',
                  'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4',
                  'estadocivil5', 'estadocivil6', 'estadocivil7',
                  'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7',
                  'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12',
                  'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
                  'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']

other_list = ['escolari', 'age', 'escolari_age']

for item in aggr_mean_list:
    group_train_mean = train[item].groupby(train['idhogar']).mean()
    group_test_mean = test[item].groupby(test['idhogar']).mean()
    new_col = item + '_aggr_mean'
    df_train[new_col] = group_train_mean
    df_test[new_col] = group_test_mean

for item in other_list:
    for function in ['mean','std','min','max','sum']:
        group_train = train[item].groupby(train['idhogar']).agg(function)
        group_test = test[item].groupby(test['idhogar']).agg(function)
        new_col = item + '_' + function
        df_train[new_col] = group_train
        df_test[new_col] = group_test
test.head()
#from https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
df_test = df_test.reset_index()
df_train = df_train.reset_index()

train = pd.merge(train, df_train, on='idhogar')
test = pd.merge(test, df_test, on='idhogar')

#fill all na as 0
train.fillna(value=0, inplace=True)
test.fillna(value=0, inplace=True)
train.drop(columns=['parentesco' + str(i+2) for i in range(11)], inplace=True)
ignore.append('parentesco1')
test.head()
def toCategorical(inData, columns, name):
    inData[name] = np.zeros_like(len(inData))
    for i, c in enumerate(columns):
        inData.loc[:, name] += (i+1)*inData.loc[:, c]
    inData.drop(columns=columns, inplace=True)
wall_mat = ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother']
toCategorical(train, wall_mat, 'wall_mat')
toCategorical(test, wall_mat, 'wall_mat')
floor_mat = ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']
toCategorical(train, floor_mat, 'floor_mat')
toCategorical(test, floor_mat, 'floor_mat')
roof_mat = ['techozinc', 'techoentrepiso', 'techocane', 'techootro', 'roof_waste_material']
toCategorical(train, roof_mat, 'roof_mat')
toCategorical(test, roof_mat, 'roof_mat')
water_prov = ['abastaguadentro', 'abastaguafuera', 'abastaguano']
toCategorical(train, water_prov, 'water_prov')
toCategorical(test, water_prov, 'water_prov')
elec_prov = ['public', 'planpri', 'noelec', 'coopele', 'electricity_other']
toCategorical(train, elec_prov, 'elec_prov')
toCategorical(test, elec_prov, 'elec_prov')
toilet = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']
toCategorical(train, toilet, 'toilet')
toCategorical(test, toilet, 'toilet')
cooking = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']
toCategorical(train, cooking, 'cooking')
toCategorical(test, cooking, 'cooking')
rubbish = ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']
toCategorical(train, rubbish, 'rubbish')
toCategorical(test, rubbish, 'rubbish')
wall_quality = ['epared1', 'epared2', 'epared3']
toCategorical(train, wall_quality, 'wall_quality')
toCategorical(test, wall_quality, 'wall_quality')
roof_quality = ['etecho1', 'etecho2', 'etecho3']
toCategorical(train, roof_quality, 'roof_quality')
toCategorical(test, roof_quality, 'roof_quality')
floor_quality = ['eviv1', 'eviv2', 'eviv3']
toCategorical(train, floor_quality, 'floor_quality')
toCategorical(test, floor_quality, 'floor_quality')
gender = ['male', 'female']
toCategorical(train, gender, 'gender')
toCategorical(test, gender, 'gender')
civil_status = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']
toCategorical(train, civil_status, 'civil_status')
toCategorical(test, civil_status, 'civil_status')
education = ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
             'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']
toCategorical(train, education, 'education')
toCategorical(test, education, 'education')
house_ownership = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
toCategorical(train, house_ownership, 'house_ownership')
toCategorical(test, house_ownership, 'house_ownership')
region = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']
toCategorical(train, region, 'region')
toCategorical(test, region, 'region')
area = ['area1', 'area2']
toCategorical(train, area, 'area')
toCategorical(test, area, 'area')
for c in [x for x in train.columns if x not in ignore]:
    print(c, len(set(train[c])))
cat_vars = [
'hacdor',
'hacapo',
'v14a',
'refrig',
'v18q',
'cielorazo',
'dis',
'computer',
'television',
'mobilephone',
'wall_mat',
'floor_mat',
'roof_mat',
'water_prov',
'elec_prov',
'toilet',
'cooking',
'rubbish',
'wall_quality',
'roof_quality',
'floor_quality',
'gender',
'civil_status',
'education',
'house_ownership',
'region',
'area']
train.replace({'dependency': {'no': 0, 'yes': 1}}, inplace=True)
test.replace({'dependency': {'no': 0, 'yes': 1}}, inplace=True)
contin_vars = [x for x in train.columns if x not in ignore and x not in cat_vars]
for c in [x for x in contin_vars]:
    print(c, len(set(train[c])))
dep = 'Target'
test.index=test['Id']
test[dep] = 0
len(test)
for v in cat_vars: 
    train[v] = train[v].astype('category').cat.as_ordered()
    
apply_cats(test, train)

for v in contin_vars:
    train[v] = train[v].fillna(0).astype('float32')
    test[v] = test[v].fillna(0).astype('float32')

train.head(2)
pretrain = train[train.parentesco1 == 0].copy()
train = train[train.parentesco1 == 1].copy()
train.reset_index(inplace=True)
pretrain.reset_index(inplace=True)
n = len(train)
print(f'Pretraining on {len(pretrain)}, final training on {n} points')
df, y, nas, mapper = proc_df(train[cat_vars+contin_vars+[dep]], 'Target', do_scale=True)
df_test, _, nas, mapper = proc_df(test[cat_vars+contin_vars+[dep]], 'Target', do_scale=True,
                                  mapper=mapper, na_dict=nas)
df_pre, yp, nas, mapper = proc_df(pretrain[cat_vars+contin_vars+[dep]], 'Target', do_scale=True,
                                  mapper=mapper, na_dict=nas)
cat_sz = [(c, len(train[c].cat.categories)+1) for c in cat_vars]
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
#Just checking we've not missed anything
[x for x in df_test.columns if x not in df.columns] , [x for x in df.columns if x not in df_test.columns]
def inv_y(a): return np.exp(a) #The model will output the log of predictions
def macro(y_pred, y_true): #Metric for comparison
    y_pred = np.argmax(inv_y(y_pred), axis=1) #We take the highest class prediction as the prediction
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    return f1
#Simple model with one layer of 100 neurons, embeddings have dropout of 0.05, fully connected layers use DO=0.5
def getPreModel(md): 
    return md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                          0.05, 5, [100], [0.5]) 
def getModel(md):
    return md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                          0.05, 5, [100, 100], [0.5, 0.5]) 
def preTrainModel(trn_idx, val_idx):
    #Get indeces of pretrain data
    trn_Housholds = train['idhogar'].iloc[trn_idx]
    val_Housholds = train['idhogar'].iloc[val_idx]
    ptrn_idx = pretrain[pretrain['idhogar'].isin(trn_Housholds)].index
    pval_idx = pretrain[pretrain['idhogar'].isin(val_Housholds)].index
    
    #Class balancing
    tmpDF = df_pre.copy()
    tmpY = yp.copy()
    maxN = np.sum(np.equal(4, yp[ptrn_idx]))
    for c in [1,2,3]:
        rows = pd.Series(np.equal(c, yp[ptrn_idx]), name='bools')
        n = np.sum(rows)
        nCopyMult = maxN//n
        for j in range(nCopyMult):
            tmpDF = tmpDF.append(df_pre.iloc[ptrn_idx][rows.values].copy(), ignore_index=True)
            tmpY = np.append(tmpY, yp[ptrn_idx][rows].copy())
    
    #Load data
    pmd = ColumnarModelData.from_data_frame(PATH, pval_idx, tmpDF, tmpY.astype(int), cat_flds=cat_vars, bs=16,
                                            is_reg=False, is_multi=False)
    
    #Create pre model and train with 1-cycle
    print('Pretraining model')
    m = getPreModel(pmd)
    m.fit(2e-3, 1, wds=1e-3, metrics=[macro], cycle_len=15,use_clr=(5,8), best_save_name='pre')
def trainModel(trn_idx, val_idx):
    #Balance training classes
    tmpDF = df.copy()
    tmpY = y.copy()
    maxN = np.sum(np.equal(4, y[trn_idx]))
    for c in [1,2,3]:
        rows = pd.Series(np.equal(c, y[trn_idx]), name='bools')
        n = np.sum(rows)
        nCopyMult = maxN//n
        for j in range(nCopyMult):
            tmpDF = tmpDF.append(df.iloc[trn_idx][rows.values].copy(), ignore_index=True)
            tmpY = np.append(tmpY, y[trn_idx][rows].copy())
            
    #Load data
    md = ColumnarModelData.from_data_frame(PATH, val_idx, tmpDF, tmpY.astype(int), cat_flds=cat_vars, bs=16,
                                           test_df=df_test, is_reg=False, is_multi=False)
    
    #Create new model and initialise with pretrained model
    m = getModel(md)
    m.model.load_state_dict(torch.load(m.get_model_path('pre')), strict=False)
    
    #Freeze all but last layer, to avoid destroying the pretrained weights, train with 1-cycle
    m.freeze_to(2)
    print('Training last layer')
    m.fit(8e-2,1,wds=1e-3,cycle_len=15,use_clr=(5,8), metrics=[macro], best_save_name='tmpbest')
    
    #Load best, unfreeze all layers for final training
    m.load('tmpbest')
    m.unfreeze()
    m.bn_freeze(True)
    
    #Final training, use differential learning rates, and train via 1-cycle
    lr = 8e-3
    print('Final training')
    m.fit(np.array([lr/9,lr/3,lr])/5, 1, wds=1e-3, metrics=[macro], cycle_len=15,use_clr=(5,8), best_save_name='best')
    m.load('best')
    
    return m
%%time
nSplits = 10
skf = StratifiedKFold(nSplits, True, 1234)
folds = skf.split(df, y)

pred_test = []
valScore = 0
for i, (trn_idx, val_idx) in enumerate(folds):
    print('________________________')
    print('Running fold', i)
    
    preTrainModel(trn_idx, val_idx)
    
    m = trainModel(trn_idx, val_idx)
    
    #Test on val
    score = macro(*m.predict_with_targs())
    valScore += score
    print('Fold', i, 'score:', score)
    
    #Predict test and append for averaging
    pred_test.append(m.predict(True))
    print('________________________\n')
print("\nCV finished, mean validation score:", valScore/nSplits)
testClassPred = np.argmax(inv_y(np.mean(pred_test, axis=0)), axis=1)
testClassPred
test['Target']=testClassPred
csv_fn=f'{PATH}sub.csv'
test.head()
test[['Target']].to_csv(csv_fn, index=True)
len(test)
!rm -rf fastai