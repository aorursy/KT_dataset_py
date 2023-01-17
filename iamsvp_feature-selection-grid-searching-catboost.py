import numpy  as np
import pandas  as pd
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
import catboost
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import f1_score # f1_score(y_true, y_pred, average='macro')
import shap
train = pd.read_csv('../input/mf-accelerator/contest_train.csv')
test = pd.read_csv('../input/mf-accelerator/contest_test.csv')
train['TARGET'].plot.hist()
plt.xticks(train['TARGET'].unique());
train.info(verbose=True, null_counts=True)
def empty_features(test=test, train=train, threshhold=0, verbose=False):
    if verbose:
        print(' column \t test \t\t train\n', '*'*40)
    i=1
    empty_list=[]
    for col in test.columns:
        percentage_train = (train[col].isnull().sum()/len(train))*100
        percentage_test = (test[col].isnull().sum()/len(test))*100
        if percentage_train and percentage_test:
            if ((percentage_test>=threshhold)|(percentage_train>=threshhold)):
                empty_list.append(col)
                if verbose:
                    print(i,'{}{} % \t{} %'.format(col.ljust(15,' '), round(percentage_test, 3), round(percentage_train,3 )))
                    i+=1
    return empty_list
extra_features=empty_features(verbose=True, threshhold=45) # features with more then 45% (threshold) empty values in both datasets
extra_features.append('ID') # have no usefull info either

extra_features
train1 = train.copy()
test1 = test.copy()
train1.drop(extra_features, axis=1, inplace=True)
test1.drop(extra_features, axis=1, inplace=True)
only_1_value = [feature for feature in train1.nunique().index if train1.nunique()[feature]==1]
only_1_value
train1.drop(only_1_value, axis=1, inplace=True)
test1.drop(only_1_value, axis=1, inplace=True)
len(train1)
searching_for_cat = train1.nunique().sort_values(ascending=True)
cat_features = [feature for feature in searching_for_cat.index if ((searching_for_cat[feature]<=(len(train)/1000))&(feature!='TARGET'))]
searching_for_cat = train1[cat_features].nunique().sort_values(ascending=True)

len(searching_for_cat)
fig,ax = plt.subplots(figsize=(20,3))

ax.bar(searching_for_cat[::-1].index, searching_for_cat[::-1].values);
for label in ax.get_xmajorticklabels():
    label.set_rotation(45)
    label.set_horizontalalignment("right")
def pipelining_preprocessor_for_catboost(df1, dropcolls = None, filler = 'pop', target = None): # return processed  dataFrame
    df = df1.copy()
    if target:
        y = df[target]
        df.drop(target, axis=1, inplace=True)
    if dropcolls:
        df.drop(dropcolls, axis=1, inplace=True)
    def fill_empty_by_pop(df, filler=filler): # filler for empty values in cols by most popular values (order columns by original)
        df_nums = df.select_dtypes(exclude='object')
        if filler == 'pop':
            digits = df_nums.median()
        elif filler == 'zero':
            digits = 0
        elif filler == 'out_of_range':
            digits = -9999
        else:
            raise ValueError('filler vallues is not allowed ["zero", "pop", "out_of_range"]')
        return df_nums.fillna(digits)[df.columns]
    output = fill_empty_by_pop(df, filler=filler)
    if target:
        return output,y
    else:
        return output
X,y = pipelining_preprocessor_for_catboost(train, filler='out_of_range', target='TARGET', dropcolls=extra_features+only_1_value)
X[cat_features] = X[cat_features].astype(np.int)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2, shuffle=True)

X_test = pipelining_preprocessor_for_catboost(test, filler='out_of_range', dropcolls=extra_features+only_1_value)
X_test[cat_features] = X_test[cat_features].astype(np.int)
len(X.columns), len(train.columns) # выбросили ненужные признаки
# создаем обучающий пул
train_pool = catboost.Pool(X_train, y_train, cat_features=cat_features)
val_pool = catboost.Pool(X_val, y_val, cat_features=cat_features)
test_pool = catboost.Pool(X_test, cat_features=cat_features)
full_pool = catboost.Pool(X, y, cat_features=cat_features)
model = CatBoostClassifier(
    eval_metric='TotalF1:average=Macro', # метрика, заявленная организаторами
    loss_function='MultiClass',
    iterations=600,
    random_seed=2,
    auto_class_weights='Balanced', 
    random_strength=1, 
    l2_leaf_reg=18, learning_rate=0.005, 
    max_ctr_complexity=1, 
    max_depth = 12, grow_policy='SymmetricTree',
    task_type='GPU', # для ускорения, можно использовать вычисления на граф. процессоре
)
'''
cat_params = {
    'learning_rate':[0.005, 0.001, 0.0001], # 0.005
    'l2_leaf_reg' : [12, 18, 30], # 18
}

grid_search_results = model.grid_search(cat_params, full_pool, plot=True,
                                          cv = 5, stratified=True, shuffle=True,
                                          search_by_train_test_split=False,
                                          partition_random_seed=2)
grid_search_results['params']
'''
model.fit(train_pool, eval_set=val_pool, plot=False)
# model.fit(full_pool, plot=False)
shap_values = model.get_feature_importance(train_pool, type='ShapValues')

original_shape = shap_values.shape
shap_values_transposed = shap_values.transpose(1, 0, 2)

shap.summary_plot(
    list(shap_values_transposed[:,:,:-1]),
    features=X_train,
    class_names=y_train.unique(),
    plot_type='bar',
    max_display=15,
)
vals= np.abs(shap_values).mean(0)

feature_importance_shap = pd.DataFrame(list(zip(X_train.columns, sum(vals))), columns=['feature','feature_importance_shap'])
feature_importance_shap.sort_values(by=['feature_importance_shap'], ascending=False, inplace=True)
feature_importance_shap.reset_index(inplace=True, drop=True)
feature_importance_shap
n_features = 125
new_cat_features = [feature for feature in feature_importance_shap['feature'][0:n_features] if feature in cat_features]

X_train, X_val, y_train, y_val = train_test_split(X[feature_importance_shap['feature'][0:n_features]], y, test_size=0.2, random_state=2, shuffle=True)

X_test = (pipelining_preprocessor_for_catboost(test, filler='out_of_range', dropcolls=extra_features+only_1_value))[feature_importance_shap['feature'][0:n_features]]
X_test[new_cat_features] = X_test[new_cat_features].astype(np.int)

# создаем обучающий пул
train_pool_2 = catboost.Pool(X_train, y_train, cat_features=new_cat_features)
val_pool_2 = catboost.Pool(X_val, y_val, cat_features=new_cat_features)
test_pool_2 = catboost.Pool(X_test, cat_features=new_cat_features)
full_pool_2 = catboost.Pool(X[new_cat_features], y, cat_features=new_cat_features)
model.fit(train_pool_2, eval_set=val_pool_2, plot=False)
gb_pred = list(map(int, model.predict(test_pool)))
gb_output = pd.DataFrame({'ID': test['ID'], 'Predicted': gb_pred})
gb_output.to_csv('submission_catboost.csv', index=False)
gb_output