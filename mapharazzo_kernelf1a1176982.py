%matplotlib inline
%reload_ext autoreload
%autoreload 2
ls
from fastai.structured import *
from fastai.column_data import *

PATH='../input/'
df = pd.read_csv('{}train.csv'.format(PATH), engine='python')
df_test_csv = pd.read_csv('{}test.csv'.format(PATH), engine='python')
df_test_csv['winPlacePerc'] = 0
df.head()
df['groupId']
# arr1 = [1,2,3,4,5]
# arr2 = [6,7,8,9,10]
# arr_idx = [1,3,5]
# list(map(lambda x, idx: sum(x) if idx not in arr_idx else x[0], list(zip(arr1, arr2)), list(range(len(arr1)))))
df = df.sort_values(by='groupId')
columns = df.columns.tolist()
ignore_idx = [0, 1, 2, 14, 15, 20, 25]
list(enumerate(columns))
values = df.values
values[0]
dic = dict()
def dict_func(arr1, arr2):
    return list(map(lambda x, idx: sum(x) if idx not in ignore_idx else x[0], list(zip(arr1, arr2)), list(range(len(arr1)))))
for entry in values:
    if entry[1] in dic:
        dic.update({entry[1]: dict_func(dic[entry[1]], entry)})
    else:
        dic[entry[1]] = list(entry)
max(dic.keys())
dict_values = list(dic.values())
dict_values
processed_df = pd.DataFrame(data=dict_values, columns=columns)
processed_df.head()
def mae(y_pred, targ):
    return sum(list(map(lambda x: torch.abs(x[0] - x[1]), zip(y_pred, targ)))) / len(y_pred)
to_drop = ['Id', 'groupId', 'matchId', 'headshotKills', 'longestKill', 'maxPlace', 'swimDistance']
df, y, nas, mapper = proc_df(processed_df, 'winPlacePerc', do_scale=True, skip_flds=to_drop)
df_test, _, nas, mapper = proc_df(df_test_csv, 'winPlacePerc', do_scale=True, skip_flds=to_drop,
                                  mapper=mapper, na_dict=nas)
df_test.head()
train_ratio = 0.7
train_size = int(len(df) * train_ratio)
val_idx = list(range(train_size, len(df)))
md = ColumnarModelData.from_data_frame('.', val_idx, df, y.astype(np.float32), cat_flds=[], bs=128, test_df=df_test)
m0 = md.get_learner([], len(df.columns), 0.04, 1, [1500, 750], [0.001, 0.01], y_range=[0, 1])
m0.summary()
m0.lr_find()
m0.sched.plot()
lr = 1.5 * 1e-5
from sklearn.metrics import mean_squared_error
m0.fit(lr, 3, metrics=[mae])
m0.fit(lr, 3, metrics=[mae], cycle_len=2)
#m0.fit(lr, 3, metrics=[mae], cycle_len=2)
m0.save('mae058')
m0.load('mae058')
x,y=m0.predict_with_targs()
pred_test=m0.predict(True)
pred_test
results_df = pd.DataFrame(columns=['Id', 'winPlacePerc'])
results_df.winPlacePerc = [x[0] for x in pred_test]
results_df.Id = df_test_csv.Id
results_df.to_csv('results.csv', index=False)
df_test.head()
df.head()