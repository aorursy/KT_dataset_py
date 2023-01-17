# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv', decimal=',')

df['date'] = pd.to_datetime(df['date'])

df.head(2)
print('Shape: ', df.shape)

print('Columns: ')

print(df.columns)

print('Datatypes:')

print(df.dtypes)
counts = df.groupby('date').count()

counts[counts['% Iron Feed'] < 180]
# get a series of unique hourly timestamps

hours = pd.Series(df['date'].unique())

hours.index = hours

len(hours)
# create a date time index from the first to the last hour included in the date column

date_range = pd.date_range(start=df.iloc[0,0], end='2017-09-09 23:59:40', freq='20S')

# remove first couple observations consistent with the counts exploration above

date_range = date_range[6:]

date_range[-5:]
# create lists from both the hours series and the new datetime index

hours_list = hours.index.format()

print(hours_list[:5])

seconds_list = date_range.format()

print(seconds_list[:5])
# match the new datetime index to the hours series and only append the timestamps if the datea and hour match the hours list

new_index = []

for idx in seconds_list:

    if (idx[:13] + ':00:00') in hours_list:

        new_index.append(idx)



#remove the one missing interval within the hour which we found earlier using the counts

new_index.remove('2017-04-10 00:00:00')

new_index[-20:]
print(len(new_index))

print(len(df))
df['index'] = new_index

df['index'] = pd.to_datetime(df['index'])

df.index = df['index']

df = df.loc[:, df.columns[:-1]]

df.rename(columns={'date': 'datetime hours'}, inplace=True)

df.head()
unique_avg = []

for col in df.columns:

    unique_avg.append(df.groupby('datetime hours').apply(lambda x: len(x[col].unique())).mean())

plt.plot(np.arange(len(unique_avg)), unique_avg)

plt.title('Average Count of Unique Values per Hour for every Variable')

plt.ylabel('Count')

plt.xticks(list(range(len(unique_avg))), list(df.columns), rotation='vertical')

plt.show()
# some values for Silica Concentration seem interpolated so we're removing the values for all those hours

#some imports

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



#get list of % Silica Concentrate values, see the first 5 that have more than one hourly value

silica_unique = df.groupby('datetime hours').apply(lambda x: len(x['% Silica Concentrate'].unique()))

print(silica_unique[silica_unique > 1][:5])



# plot before the interpolations are taken out of the dataframe

plt.plot(df['% Silica Concentrate'][df['datetime hours'] == silica_unique[silica_unique > 1].index[0]])



# take out interpolated hours for % Silica Concentrate

interpolated_hours = silica_unique[silica_unique >1].index.format()

clean_df=df[~df['datetime hours'].isin(interpolated_hours)]



#finish the graph

plt.title('Interpolated hour of % Silica Concentrate')

plt.xlabel('Datetime Index [Day Hour:Minute]')

plt.ylabel('Silica Concentrate [%]')

plt.legend(loc='best')

plt.show()
plt.plot(clean_df.index, clean_df['% Iron Feed'])

plt.plot(clean_df.index, clean_df['% Silica Feed'])

plt.title('Iron and Silica Percentage of Input Feed')

plt.legend(loc='best')

plt.ylabel('Iron/Silica Feed [%]')

plt.xlabel('Date [Year-Month]')

plt.show()
#if_unique = clean_df.groupby('datetime hours').apply(lambda x: len(x['% Iron Feed'].unique()))

#sf_unique = clean_df.groupby('datetime hours').apply(lambda x: len(x['% Silica Feed'].unique()))

#print(if_unique[if_unique > 1][:5])

print('Count of unique hours in cleaned df: ', len(clean_df.groupby('datetime hours').mean()))

print('Count of unique % Iron Feed values: ',len(clean_df['% Iron Feed'].unique()))

print('Count of unique % Silica Feed values: ',len(clean_df['% Silica Feed'].unique()))

print('Reference: Count of unique % Silica Concentrate values: ',len(clean_df['% Silica Concentrate'].unique()))
# function to get unique values of a df column and their counts 

def get_unique_counts(column):

    df = pd.DataFrame()

    

    uv_list, count_list = list(column.unique()), []

    

    for uv in uv_list:

        count_list.append(len(column[column == uv]))

        

    df['unique_values'] = uv_list

    df['count'] = count_list

    return df
if_unique = get_unique_counts(clean_df['% Iron Feed']).sort_values('count',ascending=False)

sf_unique = get_unique_counts(clean_df['% Silica Feed']).sort_values('count',ascending=False)

print(if_unique.head(10))

print(sf_unique.head(10))
for i in range(6):

    clean_df['% Silica Feed'][clean_df['% Silica Feed'] == sf_unique.iloc[i,0]].plot()

    clean_df['% Iron Feed'][clean_df['% Iron Feed'] == if_unique.iloc[i,0]].plot() 

    plt.show()
clean_df.groupby([clean_df.index.date, clean_df.index.hour]).mean()
dirty_idx = []

for i in range(4):

    dirty_idx.extend(clean_df['% Silica Feed'][clean_df['% Silica Feed'] == sf_unique.iloc[i,0]].index.format())

dirty_idx

print(len(dirty_idx), len(clean_df))

clean_df=clean_df[~clean_df.index.isin(dirty_idx)]

print(clean_df.shape)

clean_df['% Silica Feed'].plot()

clean_df['% Iron Feed'].plot() 

plt.show()
pair_cols = list(df.columns[1:8])

pair_cols.extend(df.columns[-2:])

print(pair_cols)

#smol_df = clean_df.loc[:,pair_cols]

sns.pairplot(clean_df.loc[:,pair_cols])

plt.show()
# minute correlations 

corr_df = pd.DataFrame(index=clean_df.columns[1:])

for minute in range(60):

    min_df = clean_df[clean_df.index.minute == minute]

    corr_df[str(minute)] = min_df.groupby([min_df.index.date, min_df.index.hour, min_df.index.minute]).mean().corr().iloc[:,-1]

corr_df = corr_df.transpose()



corr_df.iloc[:,:-2].plot(legend=False)

plt.title("Correlations of Variables vs Silica Concentrate Grouped by Minute")

plt.ylabel("Correlation")

plt.xlabel("Minute of the Hour")

plt.show()
def rmse(actual, preds):

    return np.sqrt(np.sum((np.array(actual)-np.array(preds))**2) / len(actual))

def mape(actual, preds):

    return np.sum(np.abs((np.array(actual)-np.array(preds))/(np.array(actual)))) / len(actual)

def mae(actual, preds):

    return np.sum(np.abs((np.array(actual)-np.array(preds)))) / len(actual)
import xgboost as xgb
start=0

end=100



X = clean_df.iloc[start*24*180:end*24*180,1:-2]

y = clean_df.iloc[start*24*180:end*24*180,-1]

X_test = clean_df.iloc[(end)*24*180:,1:-2]

y_test = clean_df.iloc[(end)*24*180:,-1]



xgbr= xgb.XGBRegressor(max_depth=8, n_estimators=50, min_sample_split = 500, subsample=0.5, silent=True, colsample_bytree=0.8, gamma=100)



xgbr.fit(X,y)

print(xgbr.feature_importances_)

print(xgbr.score(X,y))
preds=xgbr.predict(X_test)

pred_df = pd.DataFrame(preds, columns=['predictions'])

print('Train RMSE: ' + str(rmse(y, xgbr.predict(X))))

print('RMSE: ' + str(rmse(y_test, preds)))

print('MAE: ' + str(mae(y_test, preds)))

print('MAPE: ' + str(mape(y_test, preds)))

plt.plot(y_test, label = 'Actual')

pred_df.index = y_test.index

plt.plot(pred_df, label = 'Prediction')

plt.legend()

plt.ylim(0,6)

plt.title('XGBoost Regressor Model Forecast')

plt.xticks(rotation='vertical')

plt.ylabel('Silica Concentrate [%]')

plt.xlabel('Time [Days]')

plt.show()
from sklearn.linear_model import Ridge



start=0

end=100

X = clean_df.iloc[start*24*180:end*24*180,1:-2]

y = clean_df.iloc[start*24*180:end*24*180,-1]

X_test = clean_df.iloc[(end)*24*180:,1:-2]

y_test = clean_df.iloc[(end)*24*180:,-1]



rr= Ridge(alpha= 1, fit_intercept=False, normalize=True)



rr.fit(X,y)

print(rr.get_params)

print(rr.score(X,y))

print(rr.score(X_test,y_test))
preds=rr.predict(X_test)

pred_df = pd.DataFrame(preds, columns=['predictions'])

y_pred= df.iloc[140*24*180:147*24*180,-1]

print('Train RMSE: ', rmse(y, rr.predict(X)))

print('RMSE: ', rmse(y_test, preds))

print('MAPE: ', mape(y_test, preds))

print('MAE: ', mae(y_test, preds))

plt.plot(y_test, label = 'Actual')

pred_df.index = y_test.index

plt.plot(pred_df, label = 'Predictions')

plt.title('Ridge Regression Model Forecast')

plt.xticks(rotation='vertical')

plt.ylabel('Silica Concentrate [%]')

plt.xlabel('Time [Days]')

plt.ylim(0,6)

plt.legend()

plt.show()
sns.distplot(clean_df['% Silica Concentrate'])

plt.title('Distribution Plot for % Silica Concentrate')

plt.ylabel('Relative Frequency')

plt.xlabel('Silica Concentrate [%]')

plt.show()
#create hour column

cdf = clean_df.copy(deep=True)

cdf['hour'] = cdf.index.hour



# get labels 

cdf['label'] = 0

cdf['label'][cdf['% Silica Concentrate'] > 3] = 1

print(cdf['label'][cdf['label'] == 1].count())

print(cdf['label'][cdf['label'] == 0].count())

print(cdf['label'][cdf['label'] == 0].count() / cdf['label'][cdf['label'] == 1].count())
import random 



random.seed(69)

#start=0

#end=138



mdf = cdf.drop(columns = ['datetime hours', '% Iron Concentrate', '% Silica Concentrate'])



#create and sample train set for equal class distribution

#train = mdf.iloc[start*24*180:end*24*180]

train = mdf.iloc[:-14*24*180]

zero_idx = train[train['label'] == 0].index

sample_idx = random.sample(list(zero_idx), train[train['label'] == 1].shape[0])

sample_idx.extend(list(train[train['label'] == 1].index))

sample_idx = pd.DatetimeIndex(sample_idx).sort_values()

train = train.reindex(sample_idx)



X = train.iloc[:,:-1]

y = train.iloc[:,-1]



#X_eval = mdf.iloc[(end)*24*180:(end+7)*24*180,:-1]

#y_eval = mdf.iloc[(end)*24*180:(end+7)*24*180,-1]

#X_test = mdf.iloc[(end+7)*24*180:(end+14)*24*180,:-1]

#y_test = mdf.iloc[(end+7)*24*180:(end+14)*24*180,-1]

#X_eval = mdf.iloc[-14*24*180:-7*24*180,:-1]

#y_eval = mdf.iloc[-14*24*180:-7*24*180,-1]

X_test = mdf.iloc[-14*24*180:,:-1]

y_test = mdf.iloc[-14*24*180:,-1]



print(y[y == 0].count() / y[y==1].count())
xgbc= xgb.XGBClassifier(max_depth=4, n_estimators=5, subsample=0.5, eval_metric='logloss', colsample_bytree=0.8, 

                        min_child_weight=100, gamma=50)



xgbc.fit(X,y)

print(xgbc.feature_importances_)

print(xgbc.score(X,y))
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

preds=xgbc.predict(X_test)

results = confusion_matrix(y_test, preds) 

print('Test Set Results')

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_test, preds) )

print('Report : ')

print(classification_report(y_test, preds))
eval_preds=xgbc.predict(X_eval)

eval_results = confusion_matrix(y_eval, eval_preds) 

print('Evaluation Set Results')

print('Confusion Matrix :')

print(eval_results) 

print('Accuracy Score :',accuracy_score(y_eval, eval_preds) )

print('Report : ')

print(classification_report(y_eval, eval_preds))
train_preds=xgbc.predict(X)

train_results = confusion_matrix(y, train_preds) 

print('Training Set Results')

print('Confusion Matrix :')

print(train_results) 

print('Accuracy Score :',accuracy_score(y, train_preds) )

print('Report : ')

print(classification_report(y, train_preds))
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        print()

        #print('Confusion matrix, without normalization')



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
plot_confusion_matrix(np.array(y_test), np.array(preds), classes=np.array(['pure (below cutoff)', 'impure (above cutoff)']), normalize=False)

plt.show()
import random 



random.seed(69)

#start=0

end=-14



mdf = cdf.drop(columns = ['datetime hours', '% Iron Concentrate', '% Silica Concentrate'])



#create and sample train set for equal class distribution

#train = mdf.iloc[start*24*180:end*24*180]

train = mdf.iloc[:-14*24*180]

zero_idx = train[train['label'] == 0].index

sample_idx = random.sample(list(zero_idx), train[train['label'] == 1].shape[0])

sample_idx.extend(list(train[train['label'] == 1].index))

sample_idx = pd.DatetimeIndex(sample_idx).sort_values()

train = train.reindex(sample_idx)



X = train.iloc[:,:-1]

y = train.iloc[:,-1]



X_eval = mdf.iloc[(end)*24*180:(end+7)*24*180,:-1]

y_eval = mdf.iloc[(end)*24*180:(end+7)*24*180,-1]

X_test = mdf.iloc[(end+7)*24*180:,:-1]

y_test = mdf.iloc[(end+7)*24*180:,-1]

#X_test = mdf.iloc[-14*24*180:,:-1]

#y_test = mdf.iloc[-14*24*180:,-1]



print(y[y == 0].count() / y[y==1].count())
#max_depth_list = [5,7,10,15]

#n_trees_list  = [50, 75, 100, 150, 200]



#feature_1 = []

#feature_2 = []

#train_acc = []

#test_acc = []

#test_precision = []

#test_recall = []

#test_trueones = []



#for max_depth in max_depth_list:

#    for n_tree in n_trees_list:

#        xgbc= xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_tree, subsample=0.5, eval_metric='logloss', colsample_bytree=0.8, 

#                        min_child_weight=100, gamma=50)

#        

#        xgbc.fit(X,y)

#        pred = xgbc.predict(X_test)

#        

#        feature_1.append(max_depth)

#        feature_2.append(n_tree)

#        train_acc.append(xgbc.score(X,y))

#        test_acc.append(xgbc.score(X_test, y_test))

#        cm = confusion_matrix(y_test, pred) 

#        test_trueones.append(cm[1,1])

#        test_precision.append((cm[1,1]) / (cm[0,1] + cm[1,1]))

#        test_recall.append((cm[1,1]) / (cm[1,0] + cm[1,1]))

#        print(max_depth,n_tree)
#result_df = pd.DataFrame()

#result_df['feature_1'] = feature_1

#result_df['feature_2'] = feature_2

#result_df['train_acc'] = train_acc

#result_df['test_acc'] = test_acc

#result_df['test_precision'] = test_precision

#result_df['test_recall'] = test_recall

#result_df['test_trueones'] = test_trueones

#result_df
#max_depth_list2 = [2,3,4,5,6,8]

#n_trees_list2  = [3,5,6,7,8,9,10]#



#feature_12 = []

#feature_22 = []

#train_acc2 = []

#test_acc2 = []

#test_precision2 = []

#test_recall2 = []

#test_trueones2 = []



#for max_depth in max_depth_list2:

#    for n_tree in n_trees_list2:

#        xgbc= xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_tree, subsample=0.5, eval_metric='logloss', colsample_bytree=0.8, 

#                        min_child_weight=100, gamma=50)

#       

#        xgbc.fit(X,y)

#        pred = xgbc.predict(X_eval)

#        

#        feature_12.append(max_depth)

#        feature_22.append(n_tree)

#        train_acc2.append(xgbc.score(X,y))

 #       test_acc2.append(xgbc.score(X_eval, y_eval))

#        cm = confusion_matrix(y_eval, pred) 

#        test_trueones2.append(cm[1,1])

#        test_precision2.append((cm[1,1]) / (cm[0,1] + cm[1,1]))

#        test_recall2.append((cm[1,1]) / (cm[1,0] + cm[1,1]))

#        #print(max_depth,n_tree)
#result_df2 = pd.DataFrame()

#result_df2['max_depth'] = feature_12

#result_df2['n_trees'] = feature_22

#result_df2['train_acc'] = train_acc2

#result_df2['test_acc'] = test_acc2

#result_df2['test_precision'] = test_precision2

#result_df2['test_recall'] = test_recall2

#result_df2['test_trueones'] = test_trueones2

#result_df2
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

max_depth_list = [2,3,4,5]

n_trees_list  = [2,3,4,5,6,7,8]

min_child_weight_list = [10,50,100,200,300]

gamma_list = [1,10,30,50,100]



feature_1 = []

feature_2 = []

feature_3 = []

feature_4 = []

train_acc = []

test_acc = []

test_precision = []

test_recall = []

test_trueones = []



for max_depth in max_depth_list:

    for n_tree in n_trees_list:

        for min_child_weight in min_child_weight_list:

            for gamma in gamma_list:

                xgbc= xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_tree, subsample=0.5, eval_metric='logloss', #colsample_bytree=0.8, 

                        min_child_weight=min_child_weight, gamma=gamma)

       

                xgbc.fit(X,y)

                pred = xgbc.predict(X_eval)



                feature_1.append(max_depth)

                feature_2.append(n_tree)

                feature_3.append(min_child_weight)

                feature_4.append(gamma)

                train_acc.append(xgbc.score(X,y))

                test_acc.append(xgbc.score(X_eval, y_eval))

                cm = confusion_matrix(y_eval, pred) 

                test_trueones.append(cm[1,1])

                test_precision.append((cm[1,1]) / (cm[0,1] + cm[1,1]))

                test_recall.append((cm[1,1]) / (cm[1,0] + cm[1,1]))

                #print(max_depth,n_tree)
result_df3 = pd.DataFrame()

result_df3['max_depth'] = feature_1

result_df3['n_trees'] = feature_2

result_df3['min_child_weight'] = feature_3

result_df3['gamma'] = feature_4

result_df3['train_acc'] = train_acc

result_df3['test_acc'] = test_acc

result_df3['test_precision'] = test_precision

result_df3['test_recall'] = test_recall

result_df3['test_trueones'] = test_trueones

result_df3.to_csv('/kaggle/working/xgb_grid_search_results.csv')

result_df3
result_df3.sort_values(['test_trueones'], ascending=False).head(50)
result_df3.groupby([result_df3['max_depth'], result_df3['n_trees']])['test_trueones'].max()
result_df3[(result_df3['max_depth'] == 4) & (result_df3['n_trees'] == 8)]
xgbc= xgb.XGBClassifier(max_depth=5, n_estimators=2, subsample=0.5, eval_metric='logloss', min_child_weight = 300)



xgbc.fit(X,y)

print(xgbc.feature_importances_)

print(xgbc.score(X,y))
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

preds=xgbc.predict(X_eval)

results = confusion_matrix(y_eval, preds) 

print('Test Set Results')

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_eval, preds) )

print('Report : ')

print(classification_report(y_eval, preds))
plot_confusion_matrix(np.array(y_eval), np.array(preds), classes=np.array(['pure (below cutoff)', 'impure (above cutoff)']), normalize=False)

plt.show()
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(np.array(y_eval), xgbc.predict_proba(X_eval)[:,1])

roc_auc = roc_auc_score(np.array(y_eval), xgbc.predict_proba(X_eval)[:,1])



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
xgbc= xgb.XGBClassifier(max_depth=3, n_estimators=7, subsample=0.5, eval_metric='logloss', min_child_weight=300)



xgbc.fit(X,y)

print(xgbc.feature_importances_)

print(xgbc.score(X,y))
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

preds=xgbc.predict(X_eval)

results = confusion_matrix(y_eval, preds) 

print('Test Set Results')

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_eval, preds) )

print('Report : ')

print(classification_report(y_eval, preds))

plot_confusion_matrix(np.array(y_eval), np.array(preds), classes=np.array(['pure (below cutoff)', 'impure (above cutoff)']), normalize=False)

plt.show()
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(np.array(y_eval), xgbc.predict_proba(X_eval)[:,1])

roc_auc = roc_auc_score(np.array(y_eval), xgbc.predict_proba(X_eval)[:,1])



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(fit_intercept=False, C=0.1)



lr.fit(X,y)

print(lr.decision_function(X))

print(lr.score(X,y))
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

preds=lr.predict(X_eval)

results = confusion_matrix(y_eval, preds) 

print('Test Set Results')

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_eval, preds) )

print('Report : ')

print(classification_report(y_eval, preds))
intercept_list = [True, False]

C_list  = [10,1,0.1,0.01,0.001,0.0001]



feature_1 = []

feature_2 = []

train_acc = []

test_acc = []

test_precision = []

test_recall = []

test_trueones = []



for C in C_list:

    for intercept in intercept_list:

        lr = LogisticRegression(fit_intercept=intercept, C=C, solver='liblinear')



       

        lr.fit(X,y)

        pred = lr.predict(X_eval)

        

        feature_1.append(C)

        feature_2.append(intercept)

        train_acc.append(lr.score(X,y))

        test_acc.append(lr.score(X_eval, y_eval))

        cm = confusion_matrix(y_eval, pred) 

        test_trueones.append(cm[1,1])

        test_precision.append((cm[1,1]) / (cm[0,1] + cm[1,1]))

        test_recall.append((cm[1,1]) / (cm[1,0] + cm[1,1]))

        #print(C,intercept)
result_df = pd.DataFrame()

result_df['C'] = feature_1

result_df['intercept'] = feature_2

#result_df['min_child_weight'] = feature_3

#result_df['gamma'] = feature_4

result_df['train_acc'] = train_acc

result_df['test_acc'] = test_acc

result_df['test_precision'] = test_precision

result_df['test_recall'] = test_recall

result_df['test_trueones'] = test_trueones

result_df
C_list  = [10, 7.5, 5, 4, 3, 2,1,0.75, 0.5, 0.3, 0.1,0.05, 0.01]



feature_1 = []

feature_2 = []

train_acc = []

test_acc = []

test_precision = []

test_recall = []

test_trueones = []



for C in C_list:

    lr = LogisticRegression(fit_intercept=True, C=C, solver='liblinear')





    lr.fit(X,y)

    pred = lr.predict(X_eval)



    feature_1.append(C)

    #feature_2.append(intercept)

    train_acc.append(lr.score(X,y))

    test_acc.append(lr.score(X_eval, y_eval))

    cm = confusion_matrix(y_eval, pred) 

    test_trueones.append(cm[1,1])

    test_precision.append((cm[1,1]) / (cm[0,1] + cm[1,1]))

    test_recall.append((cm[1,1]) / (cm[1,0] + cm[1,1]))

    #print(C,intercept)
result_df2 = pd.DataFrame()

result_df2['C'] = feature_1

#result_df2['intercept'] = feature_2

result_df2['train_acc'] = train_acc

result_df2['test_acc'] = test_acc

result_df2['test_precision'] = test_precision

result_df2['test_recall'] = test_recall

result_df2['test_trueones'] = test_trueones

result_df2
import xgboost as xgb

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(fit_intercept=True, C=7.50, solver='liblinear')



lr.fit(X,y)



lr_preds= lr.predict(X_eval)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

results = confusion_matrix(y_eval, lr_preds) 

print('Logistic Regression Results')

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_eval, lr_preds) )

print('Report: ')

print(classification_report(y_eval, lr_preds))

plot_confusion_matrix(np.array(y_eval), np.array(lr_preds), classes=np.array(['pure (below cutoff)', 'impure (above cutoff)']), normalize=False)

plt.show()
import random 



random.seed(69)



mdf = df.drop(columns = ['datetime', '% Iron Concentrate', '% Silica Concentrate'])



train = mdf.iloc[:-14*24*180,:]

zero_idx = train[train['label'] == 0].index

sample_idx = random.sample(list(zero_idx), train[train['label'] == 1].shape[0])

sample_idx.extend(list(train[train['label'] == 1].index))

sample_idx = pd.DatetimeIndex(sample_idx).sort_values()

train = train.reindex(sample_idx)



X = train.iloc[:,:-1]

y = train.iloc[:,-1]



X_eval = mdf.iloc[-14*24*180:-7*24*180,:-1]

y_eval = mdf.iloc[-14*24*180:-7*24*180,-1]

X_test = mdf.iloc[-7*24*180:,:-1]

y_test = mdf.iloc[-7*24*180:,-1]



print(y[y == 0].count() / y[y==1].count())
import xgboost as xgb

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(fit_intercept=True, C=3.0, solver='liblinear')

xgbc= xgb.XGBClassifier(max_depth=2, n_estimators=2, eval_metric='logloss', subsample=0.5)

xgbc2 = xgb.XGBClassifier(max_depth=4, n_estimators=8, eval_metric='logloss', subsample=0.5, min_child_weight=300)



lr.fit(X,y)

xgbc.fit(X,y)

xgbc2.fit(X,y)



lr_preds= lr.predict(X_eval)

xgb_pred = xgbc.predict(X_eval)

xgb2_pred = xgbc2.predict(X_eval)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

results = confusion_matrix(y_eval, xgb_pred) 

print('XGBoost Results')

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_eval, xgb_pred) )

print('Report: ')

print(classification_report(y_eval, xgb_pred))
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

results = confusion_matrix(y_eval, xgb2_pred) 

print('XGBoost Results')

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_eval, xgb2_pred) )

print('Report: ')

print(classification_report(y_eval, xgb2_pred))
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

results = confusion_matrix(y_eval, lr_preds) 

print('Logistic Regression Results')

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_eval, lr_preds) )

print('Report: ')

print(classification_report(y_eval, lr_preds))
eval_df = pd.DataFrame()

eval_df['LogReg'] = lr_preds

eval_df['XGBoost'] = xgb_pred

eval_df['XGBoost2'] = xgb2_pred

eval_df.index = y_eval.index

eval_df.head()
hour_counts = eval_df.groupby([eval_df.index.date, eval_df.index.hour]).sum() / 180

hour_counts['Actual'] = y_eval.groupby([y_eval.index.date, y_eval.index.hour]).mean()

hour_counts.plot()
sns.boxplot(x="Actual", y="LogReg", data=hour_counts)
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(np.array(hour_counts["Actual"]), np.array(hour_counts["LogReg"]))

roc_auc = roc_auc_score(np.array(hour_counts["Actual"]), np.array(hour_counts["LogReg"]))



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
sns.boxplot(x="Actual", y="XGBoost", data=hour_counts)
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(np.array(hour_counts["Actual"]), np.array(hour_counts["XGBoost"]))

roc_auc = roc_auc_score(np.array(hour_counts["Actual"]), np.array(hour_counts["XGBoost"]))



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
sns.boxplot(x="Actual", y="XGBoost2", data=hour_counts)
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(np.array(hour_counts["Actual"]), np.array(hour_counts["XGBoost2"]))

roc_auc = roc_auc_score(np.array(hour_counts["Actual"]), np.array(hour_counts["XGBoost2"]))



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
avg_eval_preds = (hour_counts['LogReg'] + hour_counts['XGBoost'] + hour_counts['XGBoost2']) / 3

sns.boxplot(x=hour_counts["Actual"], y=avg_eval_preds)
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(np.array(hour_counts["Actual"]), np.array(avg_eval_preds))

roc_auc = roc_auc_score(np.array(hour_counts["Actual"]), np.array(avg_eval_preds))



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
cutoff_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

acc = []

truezeros = []

falsezeros = []

trueones = []

falseones = []

imp_precision = []

imp_recall = []



for cutoff in cutoff_list:

    ddf = pd.DataFrame(index=avg_eval_preds.index)

    ddf['class'] = 0

    ddf['class'][avg_eval_preds > cutoff] = 1

    

    cm = confusion_matrix(hour_counts["Actual"], ddf['class']) 

    acc.append((cm[1,1]+cm[0,0]) / len(ddf))

    truezeros.append(cm[0,0])

    falsezeros.append(cm[1,0])

    trueones.append(cm[1,1])

    falseones.append(cm[0,1])

    imp_precision.append((cm[1,1]) / (cm[0,1] + cm[1,1]))

    imp_recall.append((cm[1,1]) / (cm[1,0] + cm[1,1]))

    #print(C,intercept)
cutoffs = pd.DataFrame()

cutoffs['Cutoff'] = cutoff_list

cutoffs['Accuracy'] = acc

cutoffs['True Zeros'] = truezeros

cutoffs['False Zeros'] = falsezeros

cutoffs['True Ones'] = trueones

cutoffs['False Ones'] = falseones

cutoffs['Impure Precision'] = imp_precision

cutoffs['Impure Recall'] = imp_recall

cutoffs
#for i in range(len(cutoff_list)):

#    ddf = pd.DataFrame(index=avg_eval_preds.index)

#    ddf['class'] = 0

#    ddf['class'][avg_eval_preds > cutoff_list[i]] = 1

#    

#    print(cutoff_list[i])

#    plot_confusion_matrix(np.array(hour_counts["Actual"]), np.array(ddf['class']), classes=np.array(['pure (below cutoff)', 'impure (above cutoff)']), normalize=False)
ddf = pd.DataFrame(index=hour_counts['LogReg'].index)

ddf['class'] = 0

ddf['class'][hour_counts['LogReg'] > 0.45] = 1



plot_confusion_matrix(np.array(hour_counts["Actual"]), np.array(ddf['class']), classes=np.array(['pure (below cutoff)', 'impure (above cutoff)']), normalize=False)

plt.show()
#import random #



#random.seed(69)



#mdf = df.drop(columns = ['datetime', '% Iron Concentrate', '% Silica Concentrate'])



#train = mdf.iloc[:-14*24*180]

#zero_idx = train[train['label'] == 0].index

#sample_idx = random.sample(list(zero_idx), train[train['label'] == 1].shape[0])

#sample_idx.extend(list(train[train['label'] == 1].index))

#sample_idx = pd.DatetimeIndex(sample_idx).sort_values()

#train = train.reindex(sample_idx)



#X = train.iloc[:,:-1]

#y = train.iloc[:,-1]



#X_eval = mdf.iloc[-42*24*180:-14*24*180,:-1]

#y_eval = mdf.iloc[-42*24*180:-14*24*180,-1]

#X_test = mdf.iloc[-14*24*180:,:-1]

#y_test = mdf.iloc[-14*24*180:,-1]



#print(y[y == 0].count() / y[y==1].count())
#lr = LogisticRegression(fit_intercept=True, C=3.0, solver='liblinear')

#xgbc= xgb.XGBClassifier(max_depth=2, n_estimators=2, eval_metric='logloss', subsample=0.5)

#xgbc2 = xgb.XGBClassifier(max_depth=4, n_estimators=8, eval_metric='logloss', subsample=0.5, min_child_weight=300)



#lr.fit(X,y)

#xgbc.fit(X,y)

#xgbc2.fit(X,y)



#lr_preds= lr.predict(X_eval)

#xgb_pred = xgbc.predict(X_eval)

#xgb2_pred = xgbc2.predict(X_eval)
#eval_df = pd.DataFrame()

#eval_df['LogReg'] = lr_preds

#eval_df['XGBoost'] = xgb_pred

#eval_df['XGBoost2'] = xgb2_pred

#eval_df.index = X_test.index

#hour_counts = eval_df.groupby([eval_df.index.date, eval_df.index.hour]).sum() / 180

#hour_counts['Actual'] = y_test.groupby([y_test.index.date, y_test.index.hour]).mean()

#hour_counts['Average'] = (hour_counts['LogReg'] + hour_counts['XGBoost'] + hour_counts['XGBoost2']) / 3

#hour_counts['Prediction'] = 0

#hour_counts['Prediction'][hour_counts['Average'] > 0.45] = 1

#plot_confusion_matrix(np.array(hour_counts["Actual"]), np.array(hour_counts["Prediction"]), classes=np.array(['pure (below cutoff)', 'impure (above cutoff)']), normalize=False)

#plt.show()
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#results = confusion_matrix(hour_counts["Actual"], hour_counts["Prediction"]) 

#print('Logistic Regression Results')

#print('Confusion Matrix :')

#print(results) 

#print('Accuracy Score :',accuracy_score(hour_counts["Actual"], hour_counts["Prediction"]) )

#print('Report: ')

#print(classification_report(hour_counts["Actual"], hour_counts["Prediction"]))