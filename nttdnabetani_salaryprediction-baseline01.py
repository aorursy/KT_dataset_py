# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
HOME_DIR = '/kaggle/input/exam-for-students20200923/'
train_data = pd.read_csv(HOME_DIR + 'train.csv', low_memory=False)
# 行数、列数の確認
train_data.shape
train_data.head(n=10)
for col_name in train_data.columns:
    print('{}: {}'.format(col_name, type(train_data[col_name][0])))
test_data = pd.read_csv(HOME_DIR + 'test.csv')
# 行数、列数の確認
test_data.shape
test_data.head(n=10)
survey_dict = pd.read_csv(HOME_DIR + 'survey_dictionary.csv')
survey_dict
submission_sample = pd.read_csv(HOME_DIR + 'sample_submission.csv')
submission_sample
country_info = pd.read_csv(HOME_DIR + 'country_info.csv')
country_info.head(n=10)
feature_list = train_data.columns
feature_list
# int型
train_data.select_dtypes(include=int)
# float型
train_data.select_dtypes(include=float)
train_data.select_dtypes(include=float).shape
# 数値型
numeric_col_name_list = train_data.select_dtypes(include=[float, int]).columns.to_list()
# str(object)型
train_data.select_dtypes(include=object)
train_data.select_dtypes(include=object).shape
# object型のうちunique valueがN種類以下のものはカテゴリ変数として扱う
## N=5: 25個の特徴量
## N=10: 55個の特徴量
## N=15: 62個の特徴量
# N=10を採用
CAT_MAX_UNIQUE_COUNT = 10

category_col_name_list = []
text_col_name_list = []

for col_name in train_data.select_dtypes(include=object).columns:
    if len(train_data[col_name].unique()) <= CAT_MAX_UNIQUE_COUNT:
        category_col_name_list.append(col_name)
    else:
        text_col_name_list.append(col_name)

print('{}: {} 個の特徴量がカテゴリに該当'.format(CAT_MAX_UNIQUE_COUNT, len(category_col_name_list)))
print('{}: {} 個の特徴量がテキストに該当'.format(CAT_MAX_UNIQUE_COUNT, len(text_col_name_list)))
category_col_name_list
text_col_name_list
# 学習データ
nan_flg = train_data.isnull()

nan_df = pd.concat({'count': nan_flg.sum(), 'rate': 100*nan_flg.mean()}, axis=1)
nan_df.query('count > 0').sort_values('rate')
# テストデータ
nan_flg = test_data.isnull()

nan_df = pd.concat({'count': nan_flg.sum(), 'rate': 100*nan_flg.mean()}, axis=1)
nan_df.query('count > 0').sort_values('rate')
for col_name in train_data.columns:
    if len(train_data[col_name].unique()) == train_data.shape[0]:
        print(col_name)
# Respondentは一意なデータなので分析には使わない
numeric_col_name_list.remove('Respondent')

# ConvertedSalaryはターゲットなので特徴量リストから除く
numeric_col_name_list.remove('ConvertedSalary')
# 集計用DataFrameを作成する
train_df = train_data.copy()
train_df['Type'] = 'train'

test_df = test_data.copy()
test_df['ConvertedSalary'] = np.NaN
test_df['Type'] = 'test'

# 列順を合わせる
test_df = test_df[train_df.columns]

summary_df = pd.concat([train_df, test_df], ignore_index=True)

train_flg = summary_df['Type'] == 'train'
test_flg = summary_df['Type'] == 'test'
summary_df.head(n=5)
summary_df.tail(n=5)
summary_df.shape
# 集計用の列を追加
summary_df['Count'] = 1

summary_df['Ratio'] = 0
summary_df.loc[train_flg, 'Ratio'] = 1 / sum(train_flg)
summary_df.loc[test_flg, 'Ratio'] = 1 / sum(test_flg)
def get_binned_data(x, bins=10, label_format='{:02}_{:.0f}-{:.0f}'):
    '''
    指定したbin数にビン分割したデータを生成する
    * ユニーク数が10未満の場合は値をそのままカテゴリ化する
    * binsにリストを指定した場合はその値で分割する
    * 欠損値は'NaN'に変換する
    Parameters
    ----------
    x : list
        値のリスト(データ型はstr, int, floatを想定)
    bins : int or list
        binの数(default: 10)
    label_format : str
        labelの表示形式
    '''
    if len(x) == 0:
        return []
    
    # データ型チェック
    if type(x) not in (pd.Series, pd.DataFrame):
        x = pd.Series(x)
        
    v = x[x.index[0]]

    if not(type(v) in (str, int, float, np.float64, np.int64, np.uint8)):
        print('Unexpected type: {}'.format(type(v)))
        return x

    # カテゴリデータの場合
    if type(v) is str:
        binned_x = x.fillna('NaN')
        return binned_x

    # unique数が10未満の場合は文字列に変換する
    if len(x.unique()) < 10:
        binned_x = pd.Series([str(val) for val in x])
    else:
        if type(bins) is int:
            binned_value, bin_def = pd.qcut(x, bins, retbins=True, duplicates='drop')
        else:
            bin_def = bins
        
        labels = [label_format.format(i, bin_def[i], bin_def[i+1]) for i in range(len(bin_def)-1)]

        if type(bins) is int:
            binned_x = pd.qcut(x, bins, labels=labels, duplicates='drop')
        else:
            binned_x = pd.cut(x, bins, labels=labels)

        binned_x = pd.Series([str(val) for val in binned_x])

    return binned_x
def plot_train_test_histogram(col_name, summary_df, bins=10):
    '''
    学習用データと評価用データのヒストグラムを描画する
    Parameters
    ----------
    col_name : str
        ヒストグラムを描画する列名
    summary_df : pd.DataFrame
        全データ
    bins : int
        ヒストグラムのbinの数
    '''
    
    # ビン分割
    all_values = summary_df[col_name]
    all_binned_values = get_binned_data(all_values, bins)
    
    train_flg = summary_df['Type'] == 'train'
    train_binned_values = all_binned_values[train_flg]
    
    test_flg = summary_df['Type'] == 'test'
    test_binned_values = all_binned_values[test_flg]
    
    # カテゴリごとに件数を集計
    train_plot_data = pd.DataFrame({'train': train_binned_values.value_counts() / sum(train_flg)})
    test_plot_data = pd.DataFrame({'test': test_binned_values.value_counts() / sum(test_flg)})
    all_plot_data = pd.DataFrame({'all': all_binned_values.value_counts()})
    
    if all_values.dtype == np.int64:
        train_plot_data.index = train_plot_data.index.astype(int)
        test_plot_data.index = test_plot_data.index.astype(int)
        all_plot_data.index = all_plot_data.index.astype(int)
        
    train_plot_data = train_plot_data.sort_index() 
    test_plot_data = test_plot_data.sort_index() 
    all_plot_data = all_plot_data.sort_index() 
    
    # 全体カテゴリのindexに合わせる
    train_plot_data = pd.concat([all_plot_data, train_plot_data], axis=1).fillna(0)['train']
    test_plot_data = pd.concat([all_plot_data, test_plot_data], axis=1).fillna(0)['test']
    
    x = np.arange(len(all_plot_data))
    w = 0.4
    
    plt.bar(x, train_plot_data, width=w, label='train', color='blue')
    plt.bar(x+w, test_plot_data, width=w, label='test', color='red')
    plt.xticks(x+w/2, all_plot_data.index, rotation=90)
    plt.legend(loc='best')
feature = numeric_col_name_list[0]

stat_df = pd.DataFrame(
    {
        'train': summary_df[train_flg][feature].describe(),
        'test': summary_df[test_flg][feature].describe()
    }
)

print('feature: {}'.format(feature))
print(stat_df)

if len(summary_df[feature].unique()) <= CAT_MAX_UNIQUE_COUNT:
    print()
    print(summary_df.groupby(feature).sum()[['Count', 'Ratio']])

plot_train_test_histogram(feature, summary_df)
# カテゴリ型
feature = category_col_name_list[0]

stat_df = pd.DataFrame(
    {
        'train': summary_df[train_flg][feature].describe(),
        'test': summary_df[test_flg][feature].describe()
    }
)

print('feature: {}'.format(feature))
print(stat_df)

if len(summary_df[feature].unique()) <= CAT_MAX_UNIQUE_COUNT:
    print()
    print(summary_df.groupby(feature).sum()[['Count', 'Ratio']])

plot_train_test_histogram(feature, summary_df)
plt.hist(train_data['ConvertedSalary'])
plt.hist(np.log(train_data['ConvertedSalary']+1))
summary_df['LogConvertedSalary'] = np.log(summary_df['ConvertedSalary'] + 1)
summary_df
def plot_target_rate(col_name, summary_df, target, bins=10, label_format='{:02}_{:.0f}-{:.0f}'):
    '''
    特徴量の値ごとの特徴量の平均値を描画する
    Parameters
    ----------
    col_name : str
        ヒストグラムを描画する列名
    summary_df : pd.DataFrame
        全データ
    target : str
        ターゲットの列名
    bins : int
        ヒストグラムのbinの数。valuesがstr型の場合は無視される
    '''
    
    # ビン分割
    all_values = summary_df[col_name]
    all_binned_values = get_binned_data(all_values, bins=bins, label_format=label_format)

    train_flg = summary_df['Type'] == 'train'
    train_binned_values = all_binned_values[train_flg]

    # カテゴリごとに集計する
    feature_df = pd.DataFrame({col_name : train_binned_values, target : summary_df[target]})
    target_rate_df = feature_df.groupby(col_name).mean()
    count_df = feature_df.groupby(col_name).count()
    count_df.columns = ['count']
    
    category_target_df = target_rate_df.join(count_df)
    
    if all_values.dtype == np.int64:
        category_target_df.index = category_target_df.index.astype(int)
        category_target_df = category_target_df.sort_index()
        category_target_df.index = category_target_df.index.astype(str)

    # ヒストグラムと生存率をplot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.bar(category_target_df.index, category_target_df['count'], alpha=0.5)
    ax1.set_ylabel('count')
    ax1.set_xticklabels(category_target_df.index, rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(category_target_df.index, category_target_df[target], color='red', label=target)
    ax2.set_ylabel('Average {}'.format(target))
    #ax2.set_ylim([0, 1.2])
    ax2.legend(loc='best')

    ax1.set_title('Average {target} by {col}'.format(target=target, col=col_name))
    ax1.set_xlabel(col_name)

    print(category_target_df.to_string(formatters={target: '{:.1%}'.format}))
TARGET = 'LogConvertedSalary'
feature = numeric_col_name_list[1]
plot_target_rate(feature, summary_df, TARGET)
feature = category_col_name_list[0]
plot_target_rate(feature, summary_df, TARGET)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
feature_num_df = summary_df[numeric_col_name_list].copy()

# 欠損値処理(数値系)
for col_name in numeric_col_name_list:
    min_val = min(feature_num_df[col_name])
    feature_num_df[col_name] = feature_num_df[col_name].fillna(min_val - 1)
    
feature_cat_df = summary_df[category_col_name_list].copy()
feature_cat_df[TARGET] = summary_df[TARGET]

# 欠損値処理(カテゴリ系)
feature_cat_df = feature_cat_df.fillna('NaN')
def count_rank_encoding(df, feature, key):
    feature_rank = df.groupby(feature).count()[key].rank(ascending=False).astype(int)
    feature_dict = feature_rank.to_dict()
    
    return df[feature].map(lambda x: feature_dict[x])
# カテゴリ変数をCount rank encoding
for col_name in category_col_name_list:
    feature_cat_df[col_name] = count_rank_encoding(feature_cat_df, col_name, TARGET)

feature_cat_df.drop(TARGET, axis=1, inplace=True)
# 学習/テストデータ
feature_df = pd.concat([feature_num_df, feature_cat_df], axis=1)

X_train = feature_df[train_flg]
y_train = summary_df[TARGET][train_flg]

X_test = feature_df[test_flg]
# サイズ確認
print(feature_df.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
# 相関を確認
cor_matrix = X_train.corr()

# リスト形式に変換
cor_list = pd.DataFrame()

for i, col1 in enumerate(X_train.columns):
    for j in range(i+1, len(X_train.columns)):
        col2 = X_train.columns[j]
        cor = cor_matrix[col1][col2]
        cor_df = pd.DataFrame([[col1, col2, cor]])
        
        cor_list = pd.concat([cor_list,  cor_df], axis=0)

cor_list.columns = ['Feature1', 'Feature2', 'cor']
cor_list.reset_index(drop=True, inplace=True)
cor_list['abs_cor'] = abs(cor_list['cor'])
cor_list.sort_values(by='abs_cor', ascending=False, inplace=True)

# 相関係数（絶対値）のTop5
cor_list[0:5]
# 相関行列をheatmapで表示
abs_cor_matrix = abs(cor_matrix)

sns.heatmap(abs_cor_matrix,
            xticklabels=X_train.columns,
            yticklabels=X_train.columns,
            cmap='Blues'
           )
# モデルチューニング
model = RandomForestRegressor(random_state=0)

n_estimators_params = [50, 100, 200] # default=100
max_depth_params = [1, 3, 5, 7]
max_features_params = [5, 10, 15]   # default=sqrt(#features)=9.7

parameters = {
    'n_estimators': n_estimators_params,
    'max_features': max_features_params,
    'max_depth': max_depth_params
}

# パラメタチューニング
skf = KFold(n_splits=5, shuffle=True, random_state=0)

model_tuning = GridSearchCV(
    estimator = model,   # 識別器
    param_grid = parameters,    # パラメタ
    scoring='neg_mean_squared_error',    # MSEで評価
    refit = 'neg_mean_squared_error',       # MSE最小パラメタで学習データ全体を再学習
    cv = skf,                      # Stratified Cross validation                      
    n_jobs = -1,               # 並列実行数(-1: コア数で並列実行)
    verbose = 1,                # メッセージ出力レベル
)

model_tuning.fit(X_train, y_train)
print('Finished')
def get_grid_search_result(model_tuning):
    '''
    チューニング結果をまとめたDataFrameを取得する
    
    Parameters
    ----------
    model_tuning : 
        GridSearchCVでのチューニング結果
    '''
    # パラメタとスコアをまとめる
    score_df = pd.DataFrame()

    for i, test_score in enumerate(model_tuning.cv_results_['mean_test_score']):
        param = model_tuning.cv_results_['params'][i]
        param_df = pd.DataFrame(param.values(), index=param.keys()).T

        # Negative MSEの場合はMSEに変換する
        if model_tuning.scoring == 'neg_mean_squared_error':
            test_score *= -1
        
        param_df['score'] = test_score

        score_df = pd.concat([score_df, param_df], axis=0)

    score_df.reset_index(drop=True, inplace=True)
    
    return score_df
def plot_rf_tuning_result(model_tuning, x_param_name):
    '''
    RandomForestのチューニングの結果をplot
    
    Parameters
    ----------
    model_tuning : 
        GridSearchCVでのチューニング結果 
    
    x_param_name : str
        x軸に表示するパラメタ名
    '''
    score_df = get_grid_search_result(model_tuning)
    
    # x軸に使うパラメタ以外のパラメタ
    line_param_name = score_df.columns.to_list()
    line_param_name.remove(x_param_name)
    line_param_name.remove('score')
    
    # 折れ線の凡例: 「パラメタ名=パラメタ値」
    line_name_list = []

    for i, item in score_df.iterrows():
        line_name = ''

        for param_name in line_param_name:
            line_name += ', ' if line_name != '' else ''
            line_name += param_name + '=' + str(item[param_name])

        line_name_list.append(line_name)

    score_df['line_name'] = line_name_list
    
    # x_paramをx軸、line_paramを折れ線グラフで表現
    _, ax = plt.subplots(1,1)
    
    for line_name in np.unique(line_name_list):
        plot_df = score_df.query('line_name == "{}"'.format(line_name))
        plot_df = plot_df.sort_values(x_param_name)
        
        ax.plot(plot_df[x_param_name], plot_df['score'], '-o', label=line_name)
        
    ax.set_title("Grid Search", fontsize=20, fontweight='bold')
    ax.set_xlabel(x_param_name, fontsize=16)
    ax.set_ylabel('CV Average MSE', fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 0.95, 0.5, .100), fontsize=10)
    ax.grid('on')
def plot_rf_param_tuning_result(model_tuning, param_name):
    '''
    パラメタごとの結果(平均Score)をplot
    
    Parameters
    ----------
    model_tuning : 
        GridSearchCVでのチューニング結果 
    
    param_name : str
        集計軸にとるパラメタ名
    '''
    score_df = get_grid_search_result(model_tuning)
    
    # 指定したパラメタ軸で平均scoreを集計する
    plot_df = score_df.groupby(param_name).mean()
    plot_df = plot_df.sort_values(param_name)
    
    # x_paramをx軸、line_paramを折れ線グラフで表現
    _, ax = plt.subplots(1,1)
    
    ax.plot(plot_df.index, plot_df['score'], '-o', label='average score')
        
    ax.set_title("Grid Search: " + param_name, fontsize=20, fontweight='bold')
    ax.set_xlabel(param_name, fontsize=16)
    ax.set_ylabel('CV Average MSE', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid('on')    
plot_rf_tuning_result(model_tuning, 'n_estimators')
plot_rf_param_tuning_result(model_tuning, 'n_estimators')
plot_rf_param_tuning_result(model_tuning, 'max_features')
plot_rf_param_tuning_result(model_tuning, 'max_depth')
# ベストパラメタ
print('* Best Score: {:.3f}'.format(-model_tuning.best_score_))
print('* Best parameter: {}'.format(model_tuning.best_params_))
# モデルチューニング
model = RandomForestRegressor(random_state=0)

n_estimators_params = [200] # default=100
max_depth_params = [7, 9, 11]
max_features_params = [10, 15, 20]   # default=sqrt(#features)=9.7

parameters = {
    'n_estimators': n_estimators_params,
    'max_features': max_features_params,
    'max_depth': max_depth_params
}

# パラメタチューニング
skf = KFold(n_splits=5, shuffle=True, random_state=0)

model_tuning = GridSearchCV(
    estimator = model,   # 識別器
    param_grid = parameters,    # パラメタ
    scoring='neg_mean_squared_error',    # MSEで評価
    refit = 'neg_mean_squared_error',       # MSE最小パラメタで学習データ全体を再学習
    cv = skf,                      # Cross validation                      
    n_jobs = -1,               # 並列実行数(-1: コア数で並列実行)
    verbose = 1,                # メッセージ出力レベル
)

model_tuning.fit(X_train, y_train)
print('Finished')
plot_rf_tuning_result(model_tuning, 'n_estimators')
plot_rf_param_tuning_result(model_tuning, 'n_estimators')
plot_rf_param_tuning_result(model_tuning, 'max_features')
plot_rf_param_tuning_result(model_tuning, 'max_depth')
# ベストパラメタ
print('* Best Score: {:.3f}'.format(-model_tuning.best_score_))
print('* Best parameter: {}'.format(model_tuning.best_params_))
best_model = model_tuning.best_estimator_
z_pred = best_model.predict(X_test)

# z = log(y + 1)と変換しているのでy = exp(z) - 1に変換
y_pred = np.exp(z_pred) - 1
plt.hist(y_pred)
pd.DataFrame(y_pred).describe()
summary_df['ConvertedSalary'].describe()
submission_sample['ConvertedSalary'] = y_pred
submission_sample
submission_sample.to_csv('submission_baseline01.csv', index=False)
country_info
country_info.columns
# 小数点が","になっているので変換
col_list = ['Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)', 'Net migration', 'Infant mortality (per 1000 births)', 
            'Literacy (%)', 'Phones (per 1000)', 'Arable (%)', 'Crops (%)', 'Other (%)', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']

for col_name in col_list:
    country_info[col_name] = country_info[col_name].map(lambda x: str(x).replace(',', '.')).astype(float)
    
country_info.describe()
country_df = pd.DataFrame({'Country': summary_df['Country'].unique()})
pd.merge(left=country_df, right=country_info, on='Country', how='left')
summary_country_df = pd.merge(left=summary_df, right=country_info, on='Country', how='left')
summary_country_df.shape
summary_country_df
# 特徴量リストの再定義
numeric_col_name_list = summary_country_df.select_dtypes(include=[float, int]).columns.to_list()

# Respondentは一意なデータなので分析には使わない
numeric_col_name_list.remove('Respondent')

# ConvertedSalaryはターゲットなので特徴量リストから除く
numeric_col_name_list.remove('ConvertedSalary')
numeric_col_name_list.remove('LogConvertedSalary')

# 補助データも除く
numeric_col_name_list.remove('Count')
numeric_col_name_list.remove('Ratio')

# object型のうちunique valueがN種類以下のものはカテゴリ変数として扱う
## N=5: 25個の特徴量
## N=10: 55個の特徴量
## N=15: 62個の特徴量
# N=10を採用
category_col_name_list = []
text_col_name_list = []

for col_name in summary_country_df.select_dtypes(include=object).columns:
    if len(summary_country_df[col_name].unique()) <= CAT_MAX_UNIQUE_COUNT:
        category_col_name_list.append(col_name)
    else:
        text_col_name_list.append(col_name)

category_col_name_list.remove('Type')        
        
print('{}: {} 個の特徴量がカテゴリに該当'.format(CAT_MAX_UNIQUE_COUNT, len(category_col_name_list)))
print('{}: {} 個の特徴量がテキストに該当'.format(CAT_MAX_UNIQUE_COUNT, len(text_col_name_list)))
numeric_col_name_list
len(numeric_col_name_list)
feature_num_df = summary_country_df[numeric_col_name_list].copy()

# 欠損値処理(数値系)
for col_name in numeric_col_name_list:
    min_val = min(feature_num_df[col_name])
    feature_num_df[col_name] = feature_num_df[col_name].fillna(min_val - 1)
    
feature_cat_df = summary_country_df[category_col_name_list].copy()
feature_cat_df[TARGET] = summary_df[TARGET]

# 欠損値処理(カテゴリ系)
feature_cat_df = feature_cat_df.fillna('NaN')
# カテゴリ変数をCount rank encoding
for col_name in category_col_name_list:
    feature_cat_df[col_name] = count_rank_encoding(feature_cat_df, col_name, TARGET)

feature_cat_df.drop(TARGET, axis=1, inplace=True)
# 学習/テストデータ
feature_df = pd.concat([feature_num_df, feature_cat_df], axis=1)

X_train = feature_df[train_flg]
y_train = summary_df[TARGET][train_flg]

X_test = feature_df[test_flg]
# サイズ確認
print(feature_df.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
# 相関を確認
cor_matrix = X_train.corr()

# リスト形式に変換
cor_list = pd.DataFrame()

for i, col1 in enumerate(X_train.columns):
    for j in range(i+1, len(X_train.columns)):
        col2 = X_train.columns[j]
        cor = cor_matrix[col1][col2]
        cor_df = pd.DataFrame([[col1, col2, cor]])
        
        cor_list = pd.concat([cor_list,  cor_df], axis=0)

cor_list.columns = ['Feature1', 'Feature2', 'cor']
cor_list.reset_index(drop=True, inplace=True)
cor_list['abs_cor'] = abs(cor_list['cor'])
cor_list.sort_values(by='abs_cor', ascending=False, inplace=True)

# 相関係数（絶対値）のTop5
cor_list[0:5]
# 相関行列をheatmapで表示
abs_cor_matrix = abs(cor_matrix)

sns.heatmap(abs_cor_matrix,
            xticklabels=X_train.columns,
            yticklabels=X_train.columns,
            cmap='Blues'
           )
# モデルチューニング
model = RandomForestRegressor(random_state=0)

n_estimators_params = [200] # default=100
max_depth_params = [13]
max_features_params = [20]   # default=sqrt(#features)=9.7

parameters = {
    'n_estimators': n_estimators_params,
    'max_features': max_features_params,
    'max_depth': max_depth_params
}

# パラメタチューニング
skf = KFold(n_splits=5, shuffle=True, random_state=0)

model_tuning = GridSearchCV(
    estimator = model,   # 識別器
    param_grid = parameters,    # パラメタ
    scoring='neg_mean_squared_error',    # MSEで評価
    refit = 'neg_mean_squared_error',       # MSE最小パラメタで学習データ全体を再学習
    cv = skf,                      # Cross validation                      
    n_jobs = -1,               # 並列実行数(-1: コア数で並列実行)
    verbose = 1,                # メッセージ出力レベル
)

model_tuning.fit(X_train, y_train)
print('Finished')
plot_rf_tuning_result(model_tuning, 'n_estimators')
plot_rf_param_tuning_result(model_tuning, 'n_estimators')
plot_rf_param_tuning_result(model_tuning, 'max_features')
plot_rf_param_tuning_result(model_tuning, 'max_depth')
# ベストパラメタ
print('* Best Score: {:.3f}'.format(-model_tuning.best_score_))
print('* Best parameter: {}'.format(model_tuning.best_params_))
# Feature importance
model = model_tuning.best_estimator_

plot_df = pd.DataFrame([X_train.columns, model.feature_importances_]).T
plot_df.columns = ['Feature', 'Importance']
plot_df = plot_df.sort_values('Importance', ascending=False)

plt.bar(x=range(plot_df.shape[0]), height=plot_df['Importance'], tick_label=plot_df['Feature'])
plt.xticks(rotation=90)
plt.title('Feature Importance')
plot_df.tail(n=10)
best_model = model_tuning.best_estimator_
z_pred = best_model.predict(X_test)

# z = log(y + 1)と変換しているのでy = exp(z) - 1に変換
y_pred = np.exp(z_pred) - 1
submission_sample['ConvertedSalary'] = y_pred
submission_sample.to_csv('submission_country_info.csv', index=False)
submission_sample.describe()
drop_feature_list = plot_df.query('Importance < 0.005')['Feature']
numeric_col_name_list_country_info = numeric_col_name_list.copy()
category_col_name_list_coutry_info = category_col_name_list.copy()

for col in drop_feature_list:
    if col in numeric_col_name_list:
        numeric_col_name_list.remove(col)
        
    if col in category_col_name_list:
        category_col_name_list.remove(col)
len(numeric_col_name_list)
len(category_col_name_list)
feature_num_df = summary_country_df[numeric_col_name_list].copy()

# 欠損値処理(数値系)
for col_name in numeric_col_name_list:
    min_val = min(feature_num_df[col_name])
    feature_num_df[col_name] = feature_num_df[col_name].fillna(min_val - 1)
    
feature_cat_df = summary_country_df[category_col_name_list].copy()
feature_cat_df[TARGET] = summary_df[TARGET]

# 欠損値処理(カテゴリ系)
feature_cat_df = feature_cat_df.fillna('NaN')
# カテゴリ変数をCount rank encoding
for col_name in category_col_name_list:
    feature_cat_df[col_name] = count_rank_encoding(feature_cat_df, col_name, TARGET)

feature_cat_df.drop(TARGET, axis=1, inplace=True)
# 学習/テストデータ
feature_df = pd.concat([feature_num_df, feature_cat_df], axis=1)

X_train = feature_df[train_flg]
y_train = summary_df[TARGET][train_flg]

X_test = feature_df[test_flg]
# サイズ確認
print(feature_df.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
# 相関を確認
cor_matrix = X_train.corr()

# リスト形式に変換
cor_list = pd.DataFrame()

for i, col1 in enumerate(X_train.columns):
    for j in range(i+1, len(X_train.columns)):
        col2 = X_train.columns[j]
        cor = cor_matrix[col1][col2]
        cor_df = pd.DataFrame([[col1, col2, cor]])
        
        cor_list = pd.concat([cor_list,  cor_df], axis=0)

cor_list.columns = ['Feature1', 'Feature2', 'cor']
cor_list.reset_index(drop=True, inplace=True)
cor_list['abs_cor'] = abs(cor_list['cor'])
cor_list.sort_values(by='abs_cor', ascending=False, inplace=True)

# 相関係数（絶対値）のTop5
cor_list[0:5]
# モデルチューニング
model = RandomForestRegressor(random_state=0)

n_estimators_params = [500] # default=100
max_depth_params = [15]
max_features_params = [20]   # default=sqrt(#features)=9.7

parameters = {
    'n_estimators': n_estimators_params,
    'max_features': max_features_params,
    'max_depth': max_depth_params
}

# パラメタチューニング
skf = KFold(n_splits=5, shuffle=True, random_state=0)

model_tuning = GridSearchCV(
    estimator = model,   # 識別器
    param_grid = parameters,    # パラメタ
    scoring='neg_mean_squared_error',    # MSEで評価
    refit = 'neg_mean_squared_error',       # MSE最小パラメタで学習データ全体を再学習
    cv = skf,                      # Cross validation                      
    n_jobs = -1,               # 並列実行数(-1: コア数で並列実行)
    verbose = 1,                # メッセージ出力レベル
)

model_tuning.fit(X_train, y_train)
print('Finished')
plot_rf_tuning_result(model_tuning, 'n_estimators')
plot_rf_param_tuning_result(model_tuning, 'n_estimators')
plot_rf_param_tuning_result(model_tuning, 'max_features')
plot_rf_param_tuning_result(model_tuning, 'max_depth')
# ベストパラメタ
print('* Best Score: {:.3f}'.format(-model_tuning.best_score_))
print('* Best parameter: {}'.format(model_tuning.best_params_))
# Feature importance
model = model_tuning.best_estimator_

plot_df = pd.DataFrame([X_train.columns, model.feature_importances_]).T
plot_df.columns = ['Feature', 'Importance']
plot_df = plot_df.sort_values('Importance', ascending=False)

plt.bar(x=range(plot_df.shape[0]), height=plot_df['Importance'], tick_label=plot_df['Feature'])
plt.xticks(rotation=90)
plt.title('Feature Importance')
plot_df.tail(n=10)
best_model = model_tuning.best_estimator_
z_pred = best_model.predict(X_test)

# z = log(y + 1)と変換しているのでy = exp(z) - 1に変換
y_pred = np.exp(z_pred) - 1
submission_sample['ConvertedSalary'] = y_pred
submission_sample.to_csv('submission_country_info_tuned.csv', index=False)
