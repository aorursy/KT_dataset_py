import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt



import itertools
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
def preprocess(df, train_df=True):

    df_data = []

    for mol_id in df['id'].unique():

        sample_data = df.loc[df['id'] == mol_id]

        sample_seq_length = sample_data.seq_length.values[0]



        # bpp

        bpp = np.load(f'../input/stanford-covid-vaccine/bpps/{mol_id}.npy')



        rng = 68 if train_df else sample_seq_length

        for i in range(rng):

            # mean of bpp for position i

            bpp_i = bpp[:, i]

            aux = [bi for ix, bi in enumerate(bpp_i)]

            sum_bpp = np.sum(aux)



            # top 3 values in bpp

            top_3_ix = bpp_i.argsort()[-3:][::-1]

            

            if train_df:

                sample_dict = {'id' : sample_data['id'].values[0],

                               'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),

                               'sequence' : sample_data['sequence'].values[0][i],

                               'structure' : sample_data['structure'].values[0][i],

                               'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],

                               'sum_bpp': sum_bpp,

                               'sum_bpp_top1': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-1]]) if ix != top_3_ix[-1]]),

                               'sum_bpp_top2': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-2]]) if ix != top_3_ix[-2]]),

                               'sum_bpp_top3': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-3]]) if ix != top_3_ix[-3]]),

                               'sequence_top1': sample_data['sequence'].values[0][top_3_ix[-1]],

                               'sequence_top2': sample_data['sequence'].values[0][top_3_ix[-2]],

                               'sequence_top3': sample_data['sequence'].values[0][top_3_ix[-3]],

                               'reactivity' : sample_data['reactivity'].values[0][i],

        #                        'reactivity_error' : sample_data['reactivity_error'].values[0][i],

                               'deg_Mg_pH10' : sample_data['deg_Mg_pH10'].values[0][i],

        #                        'deg_error_Mg_pH10' : sample_data['deg_error_Mg_pH10'].values[0][i],

                               'deg_pH10' : sample_data['deg_pH10'].values[0][i],

        #                        'deg_error_pH10' : sample_data['deg_error_pH10'].values[0][i],

                               'deg_Mg_50C' : sample_data['deg_Mg_50C'].values[0][i],

        #                        'deg_error_Mg_50C' : sample_data['deg_error_Mg_50C'].values[0][i],

                               'deg_50C' : sample_data['deg_50C'].values[0][i],

        #                        'deg_error_50C' : sample_data['deg_error_50C'].values[0][i]

                }

            else:

                sample_dict = {'id' : sample_data['id'].values[0],

                               'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),

                               'sequence' : sample_data['sequence'].values[0][i],

                               'structure' : sample_data['structure'].values[0][i],

                               'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],

                               'sum_bpp': sum_bpp,

                               'sum_bpp_top1': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-1]]) if ix != top_3_ix[-1]]),

                               'sum_bpp_top2': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-2]]) if ix != top_3_ix[-2]]),

                               'sum_bpp_top3': np.sum([bi for ix, bi in enumerate(bpp[:, top_3_ix[-3]]) if ix != top_3_ix[-3]]),

                               'sequence_top1': sample_data['sequence'].values[0][top_3_ix[-1]],

                               'sequence_top2': sample_data['sequence'].values[0][top_3_ix[-2]],

                               'sequence_top3': sample_data['sequence'].values[0][top_3_ix[-3]],

                }



            shifts = [1, 2, 3]

            shift_cols = ['sequence', 'structure', 'predicted_loop_type']

            for shift,col in itertools.product(shifts, shift_cols):

                if i - shift >= 0:

                    sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]

                else:

                    sample_dict['b'+str(shift)+'_'+col] = -1



                if i + shift <= sample_seq_length - 1:

                    sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]

                else:

                    sample_dict['a'+str(shift)+'_'+col] = -1





            df_data.append(sample_dict)

    df_data = pd.DataFrame(df_data)

    

    return df_data
train_data = preprocess(train)

test_data = preprocess(test, train_df=False)
train_data.head()
targets = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

num_feats = ['sum_bpp', 'sum_bpp_top1', 'sum_bpp_top2', 'sum_bpp_top3']

not_use_cols = ['id', 'id_seqpos']

features = [f for f in train_data.columns if f not in not_use_cols if f not in targets]

cat_feats = [f for f in features if f not in num_feats]
# label_encoding

sequence_encmap = {'A': 0, 'G' : 1, 'C' : 2, 'U' : 3}

structure_encmap = {'.' : 0, '(' : 1, ')' : 1}

looptype_encmap = {'S':0, 'E':1, 'H':2, 'I':3, 'X':4, 'M':5, 'B':6}



enc_targets = ['sequence', 'a1_sequence', 'a2_sequence', 'a3_sequence',

               'b1_sequence', 'b2_sequence', 'b3_sequence',

               'sequence_top1', 'sequence_top2', 'sequence_top3',

               'structure', 'a1_structure', 'a2_structure', 'a3_structure',

               'b1_structure', 'b2_structure', 'b3_structure',

               'predicted_loop_type', 'a1_predicted_loop_type', 'a2_predicted_loop_type',

               'a3_predicted_loop_type', 'b1_predicted_loop_type', 'b2_predicted_loop_type',

               'b3_predicted_loop_type']

enc_maps = [sequence_encmap, sequence_encmap, sequence_encmap, sequence_encmap,

            sequence_encmap, sequence_encmap, sequence_encmap,

            sequence_encmap, sequence_encmap, sequence_encmap,

            structure_encmap, structure_encmap, structure_encmap, structure_encmap,

            structure_encmap, structure_encmap, structure_encmap,

            looptype_encmap, looptype_encmap, looptype_encmap,

            looptype_encmap, looptype_encmap, looptype_encmap,

            looptype_encmap,]



for t, m in zip(enc_targets, enc_maps):

    print(t)

    train_data[t] = train_data[t].apply(lambda x: m[x] if x in m else -1)

    test_data[t] = test_data[t].apply(lambda x: m[x] if x in m else -1)
train_data.head()
import lightgbm as lgb
# params

seed = 2020

params = {

    'objective': 'regression',

    'boosting': 'gbdt',

    'metric': 'rmse',

    'num_leaves': 32,

    'max_bin': 512,

    'reg_lambda': 0.5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'learning_rate': 0.08,

    'min_data_in_leaf': 200,

    'seed' : seed,

    'n_jobs': -1,

}



# cv

cv_results = dict()

models = dict()

preds = dict()

for tgt in targets:

    print('-'*30, tgt, '-'*30,)

    DTrain = lgb.Dataset(train_data[features], train_data[tgt], categorical_feature=cat_feats)

    m = lgb.cv(params, DTrain, num_boost_round=300, 

               nfold=10, stratified=False, verbose_eval=100,

               early_stopping_rounds=30)

    cv_results[tgt] = m['rmse-mean'][-1]

    

    # train

    DTrain = lgb.Dataset(train_data[features], train_data[tgt], categorical_feature=cat_feats)

    model = lgb.train(params, DTrain, num_boost_round=len(m['rmse-mean']))

    # predict test

    test_data[tgt] = model.predict(test_data[features])

    # store model

    models[tgt] = model

    

cv_results
# feature_importances

for tgt in targets:

    tmp = pd.Series(models[tgt].feature_importance('gain'), index=features)



    fig, ax = plt.subplots(figsize=(10, 5))

    tmp.sort_values(ascending=False).plot.barh(ax=ax)

    ax.set_title(tgt)

    fig.tight_layout()
submission = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv', usecols=['id_seqpos'])

submission = submission.merge(test_data[['id_seqpos'] + targets], on='id_seqpos')



submission.head()
submission.to_csv('submission.csv', index=False)