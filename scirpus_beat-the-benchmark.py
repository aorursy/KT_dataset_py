import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
train = pd.read_csv('../input/train.csv', index_col = "item_id",parse_dates = ["activation_date"])
dealprobs = train.deal_probability.values
train.drop('deal_probability',inplace=True,axis=1)
test = pd.read_csv('../input/test.csv', index_col = "item_id",parse_dates = ["activation_date"])
train.insert(0,'dow',train.activation_date.dt.dayofweek)
test.insert(0,'dow',test.activation_date.dt.dayofweek)
train.insert(0,'month',train.activation_date.dt.month)
test.insert(0,'month',test.activation_date.dt.month)
textfeats = ["description", "title"]
for cols in textfeats:
    print(cols)
    train[cols] = train[cols].astype(str) 
    train[cols] = train[cols].astype(str).fillna('totallyblank') # FILL NA
    train[cols] = train[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    train[cols + '_num_chars'] = train[cols].apply(len) # Count number of Characters
    train[cols + '_num_words'] = train[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    train[cols + '_num_unique_words'] = train[cols].apply(lambda comment: len(set(w for w in comment.split())))
    train[cols + '_words_vs_unique'] = train[cols+'_num_unique_words'] / train[cols+'_num_words'] # Count Unique Words
    
    test[cols] = test[cols].astype(str) 
    test[cols] = test[cols].astype(str).fillna('totallyblank') # FILL NA
    test[cols] = test[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    test[cols + '_num_chars'] = test[cols].apply(len) # Count number of Characters
    test[cols + '_num_words'] = test[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    test[cols + '_num_unique_words'] = test[cols].apply(lambda comment: len(set(w for w in comment.split())))
    test[cols + '_words_vs_unique'] = test[cols+'_num_unique_words'] / test[cols+'_num_words'] # Count Unique Words
train.drop(['activation_date','image'],inplace=True,axis=1)
test.drop(['activation_date','image'],inplace=True,axis=1)
train['deal_probability'] = dealprobs
test['deal_probability'] = 0
numerics = ['price',                                                                  
            'item_seq_number',
            'image_top_1',
            'description_num_chars',
            'description_num_words',
            'description_num_unique_words',
            'description_words_vs_unique',
            'title_num_chars',
            'title_num_words',
            'title_num_unique_words',
            'title_words_vs_unique'
           ]
ntrainrows = train.shape[0]
alldata = pd.concat([train,test])
alldata[numerics] = alldata[numerics].fillna(-1)
for c in numerics:
    print('Binning: ', c)
    print('Min Value', alldata.loc[alldata[c]!=-1,c].min())
    print('Max Value', alldata.loc[alldata[c]!=-1,c].max())
    print('No Of NULLs:', alldata.loc[alldata[c].isnull()].shape[0])
    print('No Of Uniques:', len(alldata.loc[alldata[c]!=-1,c].unique()))
    if(len(alldata.loc[alldata[c]!=-1,c].unique())<=50):
        print('Nothing to Do')
    elif(alldata.loc[alldata[c]!=-1,c].min()==0):
        alldata.loc[alldata[c]!=-1,c] = pd.cut(np.log1p(alldata.loc[alldata[c]!=-1,c]), 50, labels=False)
    else:
        alldata.loc[alldata[c]!=-1,c] = pd.cut(np.log(alldata.loc[alldata[c]!=-1,c]), 50, labels=False)
test = alldata[ntrainrows:]
train = alldata[:ntrainrows]
tecols = ['category_name', 'city', 'description',
          'description_num_chars', 'description_num_unique_words',
          'description_num_words', 'description_words_vs_unique', 'dow',
          'image_top_1', 'item_seq_number', 'month', 'param_1', 'param_2',
          'param_3', 'parent_category_name', 'price', 'region', 'title',
          'title_num_chars', 'title_num_unique_words', 'title_num_words',
          'title_words_vs_unique', 'user_id', 'user_type']

train = train.reset_index(drop=False)
test = test.reset_index(drop=False)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
for col in tecols:
    print(col)
    train['te_'+col] = 0.
    test['te_'+col] = 0.
    SMOOTHING = test[~test[col].isin(train[col])].shape[0]/test.shape[0]
    _, test['te_'+col] = target_encode(train[col], 
                                      test[col], 
                                      target=train['deal_probability'], 
                                      min_samples_leaf=10,
                                      smoothing=SMOOTHING,
                                      noise_level=0.0)
    for f, (vis_index, blind_index) in enumerate(kf.split(train.index)):
        print(f)
        _, train.loc[blind_index, 'te_'+col] = target_encode(train.loc[vis_index, col], 
                                                            train.loc[blind_index, col], 
                                                            target=train.loc[vis_index,'deal_probability'], 
                                                            min_samples_leaf=10,
                                                            smoothing=SMOOTHING,
                                                            noise_level=0.01)
def Output(p):
    return 1./(1.+np.exp(-p))

def GPI(data):
    return Output(  0.100000*np.tanh(((((data["te_title"]) + (np.tanh((np.tanh((((((data["te_param_1"]) * ((14.27754878997802734)))) - ((7.0)))))))))) * ((11.44603633880615234)))) +
                    0.100000*np.tanh(((((((data["te_title"]) * (((data["te_param_1"]) + (((data["te_param_1"]) + ((14.26261806488037109)))))))) - ((7.32064056396484375)))) * ((4.60418081283569336)))) +
                    0.100000*np.tanh((((((((9.0)) * (data["te_user_id"]))) + ((((((((6.12634897232055664)) * (data["te_param_2"]))) * 2.0)) - ((7.04345417022705078)))))) * 2.0)) +
                    0.100000*np.tanh((((((((6.0)) * (((data["te_title"]) + ((((((6.60653972625732422)) * (data["te_param_2"]))) + (-3.0))))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((((((((8.0)) * (data["te_param_1"]))) * 2.0)) * 2.0)) - ((14.12178039550781250)))) +
                    0.100000*np.tanh((((6.0)) * (((((((data["te_title"]) * ((8.71852207183837891)))) - ((7.44896841049194336)))) - (((-3.0) - (1.0))))))) +
                    0.100000*np.tanh(((np.minimum(((((((data["te_user_id"]) * 2.0)) * 2.0))), (((((((((12.76275444030761719)) * 2.0)) * (data["te_param_2"]))) - ((9.38495445251464844))))))) * 2.0)) +
                    0.100000*np.tanh((((6.07989358901977539)) * (((((((13.99245452880859375)) * ((((-1.0) + (((data["te_title"]) * 2.0)))/2.0)))) + (2.0))/2.0)))) +
                    0.100000*np.tanh(((((((((data["te_user_id"]) * 2.0)) + ((((((13.39045619964599609)) * (((data["te_param_1"]) * 2.0)))) - ((9.0)))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((-3.0) - ((12.04623603820800781)))) +
                    0.100000*np.tanh((((10.53458595275878906)) * ((((10.0)) * ((((((((14.56688404083251953)) + ((14.56688404083251953)))) * (data["te_user_id"]))) - ((9.01885128021240234)))))))) +
                    0.100000*np.tanh(((np.where((((((((((0.35881170630455017)) > (data["te_user_id"]))*1.)) / 2.0)) > (((data["te_param_3"]) * 2.0)))*1.)>0, -3.0, (5.61410808563232422) )) * 2.0)) +
                    0.099961*np.tanh((((((((((((data["te_param_2"]) * ((-1.0*(((8.60091304779052734))))))) * 2.0)) * 2.0)) + ((8.60091590881347656)))/2.0)) * (((-2.0) * 2.0)))) +
                    0.100000*np.tanh(((((((-3.0) - (((((-3.0) * (((np.maximum(((data["te_title"])), ((data["te_category_name"])))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((((((((10.62176132202148438)) * ((((((10.0)) * (data["te_category_name"]))) * (data["te_param_2"]))))) + (-3.0))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((((((data["te_title"]) * ((6.93708467483520508)))) - (3.0))) + ((((10.0)) * (data["te_param_1"]))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((6.23275041580200195)) * (((((np.tanh((data["te_title"]))) * 2.0)) - ((((data["te_param_1"]) < (((data["te_parent_category_name"]) * 2.0)))*1.)))))) +
                    0.100000*np.tanh(((((((((np.maximum(((data["te_parent_category_name"])), ((((data["te_title"]) * 2.0))))) * ((10.58407497406005859)))) + (((-2.0) * 2.0)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((-3.0) - (((((data["te_param_2"]) - (data["te_title"]))) * ((14.98566627502441406)))))) * ((14.98566627502441406)))) +
                    0.100000*np.tanh((((((((((10.0)) * ((((((14.12048912048339844)) * (data["te_param_1"]))) + (-1.0))))) - ((9.0)))) * 2.0)) * 2.0)) +
                    0.099941*np.tanh(np.where((((data["te_param_3"]) > (data["te_image_top_1"]))*1.)>0, -3.0, np.where((((data["te_title_num_words"]) > (data["te_param_1"]))*1.)>0, -3.0, 2.0 ) )) +
                    0.099961*np.tanh(((((((((((((data["te_item_seq_number"]) - ((((data["te_param_1"]) < (data["te_title_num_chars"]))*1.)))) * 2.0)) - (data["te_category_name"]))) * 2.0)) * 2.0)) * 2.0)) +
                    0.099902*np.tanh((((((np.tanh((data["te_price"]))) < (data["te_param_3"]))*1.)) - (np.where((((data["te_price"]) < (data["te_param_1"]))*1.)>0, 1.0, (10.0) )))) +
                    0.099980*np.tanh((((((((data["te_item_seq_number"]) / 2.0)) > (np.minimum(((data["te_category_name"])), (((((data["te_item_seq_number"]) > (((data["te_category_name"]) / 2.0)))*1.))))))*1.)) * (-2.0))) +
                    0.099687*np.tanh(((((-1.0) - (((np.minimum((((((data["te_image_top_1"]) > (data["te_title"]))*1.))), ((data["te_param_1"])))) * ((-1.0*(((9.0))))))))) * 2.0)) +
                    0.099980*np.tanh(np.where(((((0.13747933506965637)) > (((data["te_param_1"]) * 2.0)))*1.)>0, ((-2.0) * 2.0), (((np.tanh((data["te_category_name"]))) > (data["te_param_1"]))*1.) )) +
                    0.099980*np.tanh(((((data["te_param_1"]) - (data["te_image_top_1"]))) - ((((((np.maximum(((data["te_category_name"])), ((data["te_price"])))) > (((data["te_image_top_1"]) * 2.0)))*1.)) * 2.0)))) +
                    0.100000*np.tanh(((-3.0) * (((((0.06090761721134186)) > (np.minimum(((np.minimum(((((data["te_title"]) - ((0.06090761721134186))))), ((data["te_param_2"]))))), ((data["te_param_3"])))))*1.)))) +
                    0.100000*np.tanh((((((data["te_item_seq_number"]) > (data["te_param_2"]))*1.)) - ((((((np.minimum(((data["te_item_seq_number"])), ((data["te_param_2"])))) < (((data["te_price"]) / 2.0)))*1.)) * 2.0)))) +
                    0.100000*np.tanh(((data["te_param_1"]) - ((((((((((data["te_param_3"]) + (data["te_parent_category_name"]))/2.0)) < (data["te_param_2"]))*1.)) + ((((data["te_param_1"]) > (data["te_param_2"]))*1.)))/2.0)))) +
                    0.099961*np.tanh(np.where(((data["te_param_3"]) - (((((((data["te_item_seq_number"]) < (((data["te_image_top_1"]) / 2.0)))*1.)) + (data["te_item_seq_number"]))/2.0)))>0, (1.0), -3.0 )) +
                    0.099980*np.tanh((((((-3.0) * ((((((data["te_description_num_chars"]) < (np.tanh((data["te_param_3"]))))*1.)) * 2.0)))) + ((((data["te_description_num_chars"]) > (data["te_price"]))*1.)))/2.0)) +
                    0.099980*np.tanh(((data["te_title"]) - ((((np.minimum(((data["te_price"])), ((data["te_param_1"])))) > (((((data["te_param_3"]) * (data["te_price"]))) + (data["te_param_3"]))))*1.)))) +
                    0.090232*np.tanh(((((((((((data["te_category_name"]) + (((data["te_param_2"]) * 2.0)))/2.0)) < (data["te_param_1"]))*1.)) * 2.0)) + ((((data["te_param_2"]) + (-1.0))/2.0)))) +
                    0.100000*np.tanh((((((data["te_param_1"]) < (data["te_param_2"]))*1.)) - (((((0.13021591305732727)) > (np.minimum(((np.minimum(((data["te_param_3"])), ((data["te_param_2"]))))), ((data["te_parent_category_name"])))))*1.)))) +
                    0.099961*np.tanh(((((((((((data["te_description_num_unique_words"]) * 2.0)) - ((((((data["te_title"]) * 2.0)) < (data["te_description_num_unique_words"]))*1.)))) - (data["te_param_2"]))) * 2.0)) * 2.0)) +
                    0.099922*np.tanh((-1.0*((np.maximum((((((data["te_item_seq_number"]) < (((data["te_param_1"]) - (data["te_category_name"]))))*1.))), (((((data["te_param_1"]) < (data["te_category_name"]))*1.)))))))) +
                    0.100000*np.tanh(((data["te_param_3"]) - (np.where((((((np.minimum(((data["te_title"])), ((data["te_item_seq_number"])))) / 2.0)) > ((0.05008221790194511)))*1.)>0, 0.0, 3.0 )))) +
                    0.100000*np.tanh((((((((np.tanh((((data["te_parent_category_name"]) * 2.0)))) < (data["te_param_2"]))*1.)) - ((((data["te_param_1"]) < (((data["te_param_3"]) / 2.0)))*1.)))) * 2.0)) +
                    0.100000*np.tanh((((((((data["te_title"]) < (((data["te_image_top_1"]) / 2.0)))*1.)) * (-3.0))) - ((((data["te_price"]) < (((data["te_image_top_1"]) / 2.0)))*1.)))) +
                    0.096855*np.tanh(((((((-1.0*(((((-1.0*(((0.61978352069854736))))) * (data["te_title"])))))) > (data["te_param_1"]))*1.)) - (data["te_title"]))) +
                    0.099785*np.tanh(((np.where(((data["te_description_num_chars"]) - (data["te_title"]))>0, data["te_item_seq_number"], (((((data["te_description_num_chars"]) > (data["te_price"]))*1.)) - (data["te_title"])) )) * 2.0)) +
                    0.100000*np.tanh((((((-1.0*(((((data["te_item_seq_number"]) < (np.minimum(((data["te_parent_category_name"])), ((((data["te_image_top_1"]) - (((data["te_title"]) / 2.0))))))))*1.))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((8.0)) * ((((2.0) < ((((8.0)) * (np.maximum(((data["te_price"])), ((((data["te_param_1"]) - (data["te_param_2"])))))))))*1.)))) +
                    0.100000*np.tanh(((data["te_category_name"]) - ((((np.minimum(((np.tanh((np.minimum(((data["te_title"])), ((data["te_description_num_unique_words"]))))))), ((data["te_param_2"])))) < (((data["te_category_name"]) / 2.0)))*1.)))) +
                    0.099980*np.tanh((((np.maximum(((data["te_price"])), ((np.where((((data["te_title"]) < (((data["te_parent_category_name"]) / 2.0)))*1.)>0, data["te_parent_category_name"], data["te_param_3"] ))))) > ((0.33367761969566345)))*1.)) +
                    0.099980*np.tanh((-1.0*(((((np.maximum(((data["te_param_3"])), (((((((data["te_category_name"]) < (((data["te_param_2"]) / 2.0)))*1.)) / 2.0))))) > (((data["te_param_1"]) * 2.0)))*1.))))) +
                    0.099941*np.tanh((((-1.0*(((((np.minimum(((data["te_parent_category_name"])), ((data["te_param_3"])))) < (((np.tanh((np.minimum(((data["te_title"])), ((data["te_param_1"])))))) / 2.0)))*1.))))) * 2.0)) +
                    0.099941*np.tanh((((-1.0*((((((((data["te_param_2"]) > (((data["te_category_name"]) * 2.0)))*1.)) > (((((data["te_param_1"]) * 2.0)) * 2.0)))*1.))))) * ((6.0)))) +
                    0.099961*np.tanh((((((((data["te_param_3"]) + (data["te_parent_category_name"]))) < (data["te_param_2"]))*1.)) - ((((((np.tanh((data["te_image_top_1"]))) / 2.0)) > (data["te_item_seq_number"]))*1.)))) )

def GPII(data):
    return Output(  0.100000*np.tanh((((((((((((5.0)) * (data["te_param_2"]))) + (-3.0))) + (((data["te_image_top_1"]) * 2.0)))) * ((13.97144412994384766)))) * 2.0)) +
                    0.100000*np.tanh((((11.35549640655517578)) * ((((12.05701923370361328)) * ((-1.0*(((((7.0)) - (((data["te_param_1"]) * ((13.80763912200927734))))))))))))) +
                    0.100000*np.tanh(((((((((data["te_parent_category_name"]) * 2.0)) * 2.0)) - ((6.28572225570678711)))) * (((((1.0) - (((data["te_param_2"]) * 2.0)))) * 2.0)))) +
                    0.100000*np.tanh((((((data["te_title"]) + (((((((data["te_title"]) * 2.0)) + (-1.0))) * ((6.45414733886718750)))))/2.0)) * ((12.60820579528808594)))) +
                    0.100000*np.tanh((((((((11.88877964019775391)) * (((-1.0) - ((((-1.0*((data["te_param_2"])))) * 2.0)))))) - (-2.0))) * 2.0)) +
                    0.100000*np.tanh((((((((-1.0*((data["te_param_2"])))) * ((((-1.0*((data["te_item_seq_number"])))) - ((14.97994422912597656)))))) - ((6.0)))) * 2.0)) +
                    0.100000*np.tanh((((-1.0*((((((((((((-2.0) - ((7.0)))) * 2.0)) * (data["te_title"]))) * 2.0)) + ((13.60573673248291016))))))) * 2.0)) +
                    0.100000*np.tanh((((((14.20429992675781250)) * (((np.maximum(((np.maximum(((data["te_param_2"])), ((data["te_param_3"]))))), ((data["te_price"])))) * 2.0)))) - ((10.0)))) +
                    0.100000*np.tanh(((((((np.maximum(((data["te_category_name"])), ((data["te_user_id"])))) * 2.0)) * ((12.44879245758056641)))) - ((10.0)))) +
                    0.100000*np.tanh(((((((((data["te_param_1"]) * ((((13.42785644531250000)) * 2.0)))) - ((8.58090019226074219)))) * ((((13.50540447235107422)) * 2.0)))) - (2.0))) +
                    0.100000*np.tanh((((14.54305934906005859)) * (((((-2.0) + (((((-3.0) + (((data["te_user_id"]) * ((12.95226860046386719)))))) * 2.0)))) * 2.0)))) +
                    0.100000*np.tanh((((((10.0)) * (((-1.0) + (((((data["te_param_2"]) + (data["te_param_2"]))) * 2.0)))))) * 2.0)) +
                    0.100000*np.tanh(((((((((data["te_user_id"]) * ((10.73872756958007812)))) + (((((data["te_param_1"]) * ((13.28394317626953125)))) - ((4.72308158874511719)))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((((((-1.0*(((7.0))))) * ((((-1.0*(((7.0))))) * (data["te_title"]))))) - ((7.56813335418701172)))) - ((4.47601795196533203)))) +
                    0.100000*np.tanh((((4.0)) * ((((((((13.31715965270996094)) * (data["te_param_1"]))) - ((4.50778961181640625)))) + ((((10.89849376678466797)) * (data["te_title"]))))))) +
                    0.099980*np.tanh(((((((((data["te_title"]) * 2.0)) - ((((((data["te_parent_category_name"]) * 2.0)) > (data["te_title"]))*1.)))) * ((10.24160671234130859)))) - ((5.0)))) +
                    0.100000*np.tanh((-1.0*(((((8.89357757568359375)) * (((((data["te_param_1"]) * ((-1.0*((((-3.0) + ((14.58576679229736328))))))))) - (-2.0)))))))) +
                    0.100000*np.tanh((((-1.0*(((14.20679950714111328))))) * ((((data["te_price"]) > (((data["te_parent_category_name"]) * (((data["te_parent_category_name"]) - (data["te_price"]))))))*1.)))) +
                    0.100000*np.tanh(((((data["te_category_name"]) * ((14.73734378814697266)))) + (((((((data["te_param_1"]) * ((13.52457332611083984)))) + (-2.0))) * ((14.73734378814697266)))))) +
                    0.100000*np.tanh(((np.minimum((((((((((((data["te_title"]) * 2.0)) * 2.0)) + (-1.0))/2.0)) * ((8.89320945739746094))))), ((data["te_param_3"])))) - (data["te_param_2"]))) +
                    0.100000*np.tanh(((np.where((((-2.0) < (((data["te_param_1"]) * ((((-1.0*(((9.0))))) * 2.0)))))*1.)>0, -2.0, data["te_image_top_1"] )) * 2.0)) +
                    0.100000*np.tanh(((np.minimum(((((((data["te_param_1"]) * ((14.75029754638671875)))) + (-2.0)))), (((-1.0*((((data["te_param_2"]) * (data["te_param_2"]))))))))) * 2.0)) +
                    0.100000*np.tanh(((((((np.minimum(((((((data["te_param_1"]) * ((12.21925258636474609)))) + (-2.0)))), ((data["te_image_top_1"])))) - (data["te_param_2"]))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((((((data["te_price"]) < (data["te_param_3"]))*1.)) - ((((((data["te_category_name"]) < (np.tanh((data["te_parent_category_name"]))))*1.)) * 2.0)))) - (data["te_param_2"]))) +
                    0.100000*np.tanh(((((data["te_image_top_1"]) - (((((2.0)) > ((((((10.0)) * (np.minimum(((data["te_title"])), ((data["te_param_1"])))))) * 2.0)))*1.)))) * 2.0)) +
                    0.100000*np.tanh((-1.0*((np.maximum((((((((((np.tanh((data["te_price"]))) > (np.maximum(((data["te_param_3"])), ((data["te_parent_category_name"])))))*1.)) * 2.0)) * 2.0))), ((data["te_parent_category_name"]))))))) +
                    0.100000*np.tanh(((((((data["te_category_name"]) - ((((np.maximum(((data["te_description_num_chars"])), ((data["te_param_2"])))) > ((((data["te_description_num_chars"]) + (data["te_image_top_1"]))/2.0)))*1.)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((data["te_title_num_chars"]) - ((((((np.minimum(((data["te_title"])), ((data["te_param_2"])))) * 2.0)) < (data["te_price"]))*1.)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((data["te_price"]) - ((((np.minimum(((data["te_param_2"])), ((data["te_image_top_1"])))) < (np.minimum(((data["te_category_name"])), ((np.minimum(((data["te_price"])), ((data["te_param_1"]))))))))*1.)))) +
                    0.100000*np.tanh((-1.0*(((((((((np.minimum(((data["te_param_2"])), ((data["te_title"])))) * 2.0)) * 2.0)) > ((((data["te_price"]) < (((data["te_param_2"]) * 2.0)))*1.)))*1.))))) +
                    0.100000*np.tanh((((((((((((data["te_param_1"]) < (data["te_param_2"]))*1.)) / 2.0)) - ((((((data["te_param_3"]) / 2.0)) > (data["te_param_1"]))*1.)))) * 2.0)) * 2.0)) +
                    0.099980*np.tanh(((((data["te_title"]) + ((((((((data["te_param_2"]) > (((data["te_parent_category_name"]) * 2.0)))*1.)) * 2.0)) - (((data["te_param_2"]) * 2.0)))))) * 2.0)) +
                    0.093593*np.tanh((((((((data["te_category_name"]) < (np.minimum(((data["te_param_1"])), ((data["te_parent_category_name"])))))*1.)) - ((((((data["te_price"]) / 2.0)) > (data["te_param_1"]))*1.)))) * 2.0)) +
                    0.100000*np.tanh((-1.0*(((((((((np.minimum(((data["te_param_2"])), ((np.minimum(((data["te_title"])), ((data["te_param_3"]))))))) * ((7.38131237030029297)))) < ((0.44618737697601318)))*1.)) * 2.0))))) +
                    0.099961*np.tanh((-1.0*((((((((((((data["te_image_top_1"]) + (((data["te_param_2"]) * (((data["te_parent_category_name"]) / 2.0)))))/2.0)) > (data["te_param_3"]))*1.)) * 2.0)) * 2.0))))) +
                    0.099961*np.tanh(((((((((data["te_title_num_chars"]) + (np.minimum(((data["te_category_name"])), ((data["te_description_num_chars"])))))/2.0)) > (data["te_param_2"]))*1.)) - ((((data["te_title_num_chars"]) > (data["te_description_num_chars"]))*1.)))) +
                    0.099980*np.tanh(((data["te_parent_category_name"]) - ((((((((((data["te_image_top_1"]) > (((np.minimum(((data["te_category_name"])), ((data["te_price"])))) * 2.0)))*1.)) * 2.0)) * 2.0)) * 2.0)))) +
                    0.100000*np.tanh(((((np.minimum(((data["te_param_2"])), (((((data["te_title"]) < (data["te_description_num_unique_words"]))*1.))))) - ((((data["te_title"]) < ((((0.18750433623790741)) / 2.0)))*1.)))) * 2.0)) +
                    0.099980*np.tanh((-1.0*(((((((((np.minimum(((data["te_category_name"])), ((data["te_title"])))) * 2.0)) < (data["te_image_top_1"]))*1.)) + (np.minimum(((data["te_category_name"])), ((data["te_title"]))))))))) +
                    0.099980*np.tanh((((((((np.maximum(((((data["te_category_name"]) / 2.0))), ((data["te_title"])))) * ((((data["te_param_1"]) < (data["te_image_top_1"]))*1.)))) > (data["te_image_top_1"]))*1.)) * 2.0)) +
                    0.100000*np.tanh((((((data["te_parent_category_name"]) < (np.minimum((((((data["te_price"]) + (((data["te_price"]) / 2.0)))/2.0))), (((((data["te_description_num_chars"]) > (data["te_price"]))*1.))))))*1.)) * 2.0)) +
                    0.099980*np.tanh((-1.0*(((((data["te_param_3"]) < ((((((((((np.minimum(((data["te_param_3"])), ((data["te_title"])))) < (data["te_category_name"]))*1.)) / 2.0)) / 2.0)) / 2.0)))*1.))))) +
                    0.099961*np.tanh(((-3.0) * ((((np.minimum(((data["te_description"])), ((((data["te_param_2"]) * 2.0))))) < ((((((data["te_description"]) / 2.0)) + (data["te_title_num_chars"]))/2.0)))*1.)))) +
                    0.100000*np.tanh((((10.0)) * ((-1.0*(((((np.minimum(((data["te_param_1"])), ((((data["te_title"]) * 2.0))))) > (np.tanh((((data["te_image_top_1"]) * 2.0)))))*1.))))))) +
                    0.100000*np.tanh((((np.maximum(((((data["te_param_1"]) * (data["te_category_name"])))), ((((data["te_price"]) * ((((data["te_param_3"]) > (data["te_param_1"]))*1.))))))) > (data["te_param_2"]))*1.)) +
                    0.099980*np.tanh((-1.0*((((((((((((((data["te_param_3"]) < (data["te_title_num_chars"]))*1.)) + (data["te_param_3"]))/2.0)) + (data["te_parent_category_name"]))/2.0)) < (data["te_title_num_words"]))*1.))))) +
                    0.099101*np.tanh(((((data["te_param_2"]) * ((((data["te_image_top_1"]) > (data["te_title"]))*1.)))) + ((((data["te_title"]) > (((data["te_category_name"]) + (data["te_param_2"]))))*1.)))) +
                    0.100000*np.tanh((((((-1.0*(((((np.minimum(((data["te_param_3"])), ((data["te_param_1"])))) < (np.maximum((((0.05219222232699394))), ((((data["te_param_3"]) / 2.0))))))*1.))))) * 2.0)) * 2.0)) +
                    0.099961*np.tanh(((((((((4.0)) * (data["te_price"]))) > ((((data["te_price"]) > (((data["te_title"]) - (data["te_param_1"]))))*1.)))*1.)) * 2.0)) +
                    0.100000*np.tanh(((((((np.where((((((data["te_user_type"]) > (data["te_description"]))*1.)) * (data["te_description"]))>0, data["te_description"], -3.0 )) * 2.0)) * 2.0)) * 2.0)))

def GP(data):
    return np.sqrt(GPI(data)*GPII(data))
np.sqrt(mean_squared_error(train.deal_probability,GP(train)))
gpsub = pd.DataFrame()
gpsub['item_id'] = test.item_id.values
gpsub['deal_probability'] = GP(test).values
gpsub.to_csv("gpsub.csv",index=False)
gpsub.head(10)