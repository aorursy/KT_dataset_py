import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import metrics
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import datetime
class Classifier:
    
    def __init__(self):
        pass
    
    def file_input(self,filename):
        data = pd.read_csv(filename,engine='python',header=None)
        return data
    
    def my_function(self,list1,list2,list3):
        if list3 == 1:
            if list1 in list2:
                return 1
            else:
                return 0
            
    def convertCategory(self,x):
        if x == 'S':
            return -1
        elif x in ['1','2','3','4','5','6','7','8','9','10','11','12']:
            return x
        else:
            return 13
    
    def data_exploration_clicks(self,clicks):
        
        category_analaysis = clicks[(clicks['category'].isin(['S','0','1','2','3','4','5','6','7','8','9','10','11','12']))]
      
        chart_4 = sns.barplot(x=category_analaysis['category'].value_counts().index, y=category_analaysis['category'].value_counts())
        chart_4.set_xticklabels(chart_4.get_xticklabels(),rotation=45)
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.title('Count of clicks against each Category')
        plt.show()
   
        chart_5 = sns.barplot(x=clicks['item_id'].value_counts().nlargest(10).index, y=clicks['item_id'].value_counts().nlargest(10))
        chart_5.set_xticklabels(chart_5.get_xticklabels(),rotation=45)
        plt.xlabel('Item_id')
        plt.ylabel('Count')
        plt.title('Top 10 Items having maximum clicks:')
        plt.show()
        
    def data_exploration_buys(self,data):
        
        print("Top 10 Items which have been bought the maximum.")
        #print(buys['item_id'].value_counts().nlargest(10))
        chart_1 = sns.barplot(x=buys['item_id'].value_counts().nlargest(10).index, y=buys['item_id'].value_counts().nlargest(10))
        chart_1.set_xticklabels(chart_1.get_xticklabels(),rotation=45)
        plt.xlabel('Item_id')
        plt.ylabel('Count')
        plt.title('Top 10 Items which have been bought the maximum.')
        plt.show()
        
        print("Top 10 items which are purchased in larger quantities.")
        quantity_analysis = buys[['item_id','qty']].groupby('item_id').agg(total_quantity=pd.NamedAgg(column='qty',aggfunc=sum))
        quant_analysis = quantity_analysis.sort_values('total_quantity',ascending=False).nlargest(10,columns='total_quantity')
        chart_2 = sns.barplot(x = quant_analysis.index, y = quant_analysis['total_quantity'] ,data = quant_analysis)
        chart_2.set_xticklabels(chart_2.get_xticklabels(),rotation=45)
        plt.xlabel('Item_id')
        plt.ylabel('Quantity')
        plt.title('Top 10 items which are purchased in larger quantities.')
        plt.show()
        
        print("Top 10 items Identifying the items having the maximum price.")
        buys_plot = buys[['item_id','price']].drop_duplicates().sort_values('price',ascending=False).nlargest(10,columns='price')
        chart_3 = sns.barplot(x = buys_plot['item_id'] , y = buys_plot['price'] ,data = buys_plot)
        chart_3.set_xticklabels(chart_3.get_xticklabels(),rotation=45)
        plt.xlabel('Item_id')
        plt.ylabel('Price')
        plt.title('Top 10 items having the maximum cost price.')
        plt.show()
        
    def transforming_buys(self,buys):
        print("Transforming the buys file ...!!!")
        grouped = buys.groupby("session")
        buys_g = pd.DataFrame(index=grouped.groups.keys())        
        buys_g["Number_items_bought"] = grouped.item_id.count()
        buys_g["unique_items_bought"] = grouped.item_id.unique()
        buys_g["is_buy"] = 1
        buys_g.index.name = "session"
        print("Transformation of the buys file completed...!!!")
        return buys_g
    
    def chunk_load_data(self,chunk):
        return chunk
        
        
    def transforming_clicks(self,clicks):
        
        clicks_new = clicks.groupby('session')['timestamp'].agg([min,max])

        clicks_new['dwell_time'] = clicks_new['max'] - clicks_new['min'] #cal the dwell time of the session.
        clicks_new['dwell_time_seconds'] = clicks_new['dwell_time'].dt.total_seconds() #converting dwell time into seconds
        
        clicks.loc[clicks['category'] == 'S',['category']] = -1

        grouped = clicks.groupby('session')
            
        #print("Calculating the total clicks")
        clicks_new['total_clicks'] = grouped.item_id.count()
        
        #print("Calculating the day of week")
        clicks_new['dayofweek'] = clicks_new['min'].dt.dayofweek
        
        #print("Calculating the day of month")
        clicks_new['dayofmonth'] = clicks_new['min'].dt.day
        
        #print("Calculating hour of click")
        clicks_new['hourofclick'] = clicks_new['min'].dt.hour
        
        #print("Calculating time of click")
        b = [0,4,8,12,16,20,24]
        l = ['Late Night', 'Early Morning','Morning','Noon','Evening','Night']
        clicks_new['timeofday'] = pd.cut(clicks_new['hourofclick'], bins=b, labels=l, include_lowest=True)
        
        #print("Calculating clickrate")
        clicks_new["click_rate"] = clicks_new["total_clicks"] / clicks_new["dwell_time_seconds"]
        clicks_new.click_rate = clicks_new.click_rate.replace(np.inf, np.nan)
        clicks_new.click_rate = clicks_new.click_rate.fillna(0)
        
        #print("** Transformed**")
        return clicks_new

    def transforming_clicks2(self,clicks):

        grouped = clicks.groupby('session').agg({'item_id':['first','last','nunique'],'category':['nunique']})        
        return grouped

    def transforming_clicks3(self,clicks):
       
        keys, values = clicks.sort_values('session').values.T
        ukeys, index = np.unique(keys, True)
        arrays = np.split(values, index[1:])
        df2 = pd.DataFrame({'a':ukeys, 'b':(a for a in arrays)})
        return df2
    
    def transforming_clicks3_cat(self,clicks):
       
        keys, values = clicks.sort_values('session').values.T
        ukeys, index = np.unique(keys, True)
        arrays = np.split(values, index[1:])
        df2 = pd.DataFrame({'a':ukeys, 'b':(a for a in arrays)})
        return df2

    def data_preparation(self,X,y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
        
    def undersampling(self,train_data):   
        
        count_class_0, count_class_1 = train_data['is_buy'].value_counts()
        df_class_0 = train_data[train_data['is_buy'] == 0]
        df_class_1 = train_data[train_data['is_buy'] == 1]
        df_class_0_under = df_class_0.sample(count_class_1)
        df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        df_test_under['is_buy'].value_counts()
        return df_test_under
        
    def one_hot_encode(self,column_name,training_data):
        temp = pd.get_dummies(training_data[column_name])
        training_data = pd.concat([training_data, temp], axis=1)
        return training_data
        
    def get_preds(self,threshold, probabilities):
        return [1 if prob > threshold else 0 for prob in probabilities]
        
    def logit_model(self,train_x,train_y,test_x,test_y,thres=0.5):
        
        model = LogisticRegression(solver='sag')
        model.fit(train_x,train_y.values.ravel())
        probas = model.predict_proba(test_x)[:, 1]
        print("Threshold Value : ",thres)
        y_pred = self.get_preds(thres,probas)
        return y_pred,probas

    def calc_special_offer(self,x):
        if -1 in x:
            return 1
        else:
            return 0
    
    def error_metrics(self,prediction,test_y,probas):
        
        accuracy = accuracy_score(prediction,test_y)        
        print('Accuracy =',accuracy)
        print("")
        print(pd.DataFrame(confusion_matrix(test_y, prediction), columns=['Predicted 0', "Predicted 1"], index=['Actual 0', 'Actual 1']))
        print("classification_Report:")
        print(classification_report(test_y,prediction))
        fig, ax = plt.subplots(figsize=(10,7))
        fpr, tpr, threshold = metrics.roc_curve(test_y,probas,pos_label=1)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        print("Optimum Threshold Value:",list(roc_t['threshold']))
        
        auc = metrics.roc_auc_score(test_y, probas)
        plt.plot(fpr,tpr,label="auc="+str(auc))
        ax.plot(np.linspace(0, 1, 100),np.linspace(0, 1, 100),label='baseline',linestyle='--')
        plt.title('Receiver Operating Characteristic Curve', fontsize=18)
        plt.ylabel('TPR', fontsize=16)
        plt.xlabel('FPR', fontsize=16)
        plt.legend(fontsize=12);
        plt.show()
clicks_file = "/input/recsys-challenge-2015/yoochoose-data/yoochoose-clicks.dat"
buys_file = "/input/recsys-challenge-2015/yoochoose-data/yoochoose-buys.dat"
data = Classifier()
result=None
count=1
names=["session", "timestamp", "item_id", "category"]
for chunk in pd.read_csv(clicks_file,names=names,usecols = ['session','timestamp','item_id','category'],parse_dates=["timestamp"],chunksize=500000):
    print("Executing Chunk ",count,"/67")
    click_df = data.transforming_clicks(chunk)
    if result is None:
      result=click_df
      count=count+1
    else:
      result = result.append(click_df)  
      count=count+1
print("Done 1st Transformation of Clicks file to calculate \n1.Session Start Time \n2.Session End Time  \n3.Session Dwell Time \n4.Dwell time seconds \n5.Total clicks \n6.Dayofweek \n7.Dayofmonth \n8.Hourofclick \n9.Timeofday \n10.Click_Rate")
result_2=None
count=1
names=["session", "timestamp", "item_id", "category"]
for chunk in pd.read_csv(clicks_file,usecols = ['session','item_id','category'],names=names,chunksize=500000):
    print("Executing Chunk ",count,"/67")
    click_df = data.transforming_clicks2(chunk)
    if result_2 is None:
      result_2=click_df
      count=count + 1
    else:
      result_2 = result_2.append(click_df) 
      count= count + 1
print("Done Transforming Clicks input file to calculate \n 1.First Clicked item \n 2.Last Clicked Item \n 3.Total Unique Items  \n 4.Total Unique Categories ")
colnames=['first_clicked_item','last_clicked_item','total_unique_items','total_unique_categories']
result_2.columns = colnames
#result_2.set_index('session')
result_3=None
count=1
names=["session", "timestamp", "item_id", "category"]
for chunk in pd.read_csv(clicks_file,names=names,usecols = ['session','item_id'],chunksize=500000):
    print("Executing Chunk ",count,"/67")
    click_df = data.transforming_clicks3(chunk)
    if result_3 is None:
      result_3=click_df
      count = count + 1
    else:
      result_3 = result_3.append(click_df) 
      count = count + 1
    
print("Done Transforming Clicks")
colnames=['session','visited_items']
result_3.columns = colnames
result_3 = result_3.set_index('session')
print(datetime.datetime.now())
result_4=None
count=1
names=["session", "timestamp", "item_id", "category"]
for chunk in pd.read_csv(clicks_file,names=names,usecols = ['session','category'],converters={"category": data.convertCategory},chunksize=500000):
    print("Executing Chunk ",count,"/67")
    click_df = data.transforming_clicks3_cat(chunk)
    if result_4 is None:
      result_4 = click_df
      count = count + 1
    else:
      result_4 = result_4.append(click_df)  
      count = count + 1
colnames=['session','visited_categories']
result_4.columns = colnames
result_4 = result_4.set_index('session')
result_4['Number_clicked_visited_categories'] = result_4['visited_categories'].apply(lambda x : len(x))
result_4['Special_offer_click']=result_4['visited_categories'].apply(data.calc_special_offer)
print("Done Transforming Clicks to calculate \n 1.Unique Visited categories \n 2. Total Visited Categories \n 3.Special offer click " )
buys = data.file_input(buys_file)
names=["session","timestamp","item_id","price","qty"]
buys.columns = names
data.data_exploration_buys(buys)
buys_g = data.transforming_buys(buys)
clicks_tranformed_updated = pd.concat([result,result_2,result_3,result_4], axis=1)
training_data = pd.merge(clicks_tranformed_updated,buys_g['is_buy'],how='left',left_index=True,right_index=True)
training_data['is_buy'] = training_data['is_buy'].fillna(0)
training_data.head()
print("calculating the popularity index for first and last item clicked")
names=["session", "timestamp", "item_id", "category"]
result_items = pd.concat([ chunk.apply(pd.Series.value_counts) for chunk in pd.read_csv(clicks_file,names=names,usecols = ['item_id'],index_col=0,chunksize=500000)])
df = pd.DataFrame(result_items.index.value_counts())
df.index.name = "item_id"
df.columns = ['count']
val = df['count'].sum()
df['popularity'] = df['count'].apply(lambda x : x / val )
df['popularity'] = df['popularity'].round(5)
print("Done..!!")
updated_training_df = pd.merge(training_data, df, left_on='first_clicked_item',right_on=df.index,how='inner')
updated_training_df.rename(columns={'popularity': 'first_clicked_item_popularity'},inplace=True)
updated_training_df = pd.merge(updated_training_df, df, left_on='last_clicked_item',right_on=df.index,how='inner')
updated_training_df.rename(columns={'popularity': 'last_clicked_item_popularity'},inplace=True)
temp = training_data[training_data['is_buy'] == 1]

#df['col_3'] = df.apply(lambda x: f(x.col_1, x.col_2), axis=1)
temp['first_item_probab_check'] = temp.apply(lambda x : data.my_function(x.first_clicked_item,x.unique_items_bought,x.is_buy),axis=1)
temp['last_item_probab_check'] = temp.apply(lambda x : data.my_function(x.last_clicked_item,x.unique_items_bought,x.is_buy),axis=1)

import seaborn as sns
from matplotlib import pyplot as plt

sns.barplot(x=temp.first_item_probab_check.value_counts().index, y=temp.first_item_probab_check.value_counts()/temp.first_item_probab_check.value_counts().sum())
plt.title('Probability of First clicked item being Purchased')
plt.xlabel('First Clicked Item')
plt.legend()


sns.barplot(x=temp.last_item_probab_check.value_counts().index, y=temp.last_item_probab_check.value_counts()/temp.last_item_probab_check.value_counts().sum())
plt.title('Probability of Last clicked item being Purchased')
plt.xlabel('Last Clicked Item')
plt.legend()
training_data[training_data['is_buy'] == 1]['dwell_time_seconds'].mean()
training_data[training_data['is_buy'] == 0]['dwell_time_seconds'].mean()
ax = sns.barplot(x=training_data['dayofweek'].value_counts().index,y=training_data['dayofweek'].value_counts(),data=training_data)
plt.xlabel('DaysofWeek')
plt.ylabel('Countofsessions')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Most Popular Days based on Number of sessions.[0 - 6] -> [Monday - Sunday]')
training_data_temp = training_data[training_data['is_buy'] == 1]
ax = sns.barplot(x=training_data_temp['dayofweek'].value_counts().index,y=training_data_temp['dayofweek'].value_counts(),data=training_data_temp)
plt.xlabel('DaysofWeek')
plt.ylabel('Countofsessions')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Popular Days for Buying Events based on Number of Sessions.')
training_data_temp['timeofday'].value_counts()
ax = sns.barplot(x=training_data_temp['timeofday'].value_counts().index,y=training_data_temp['timeofday'].value_counts(),data=training_data_temp)
plt.xlabel('timeofday')
plt.ylabel('Countofsessions')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Best Time of the Day for a buying Event.')
print(updated_training_df['is_buy'].value_counts())
new_balanced_data = data.undersampling(updated_training_df)
print(new_balanced_data['is_buy'].value_counts())
updated_training_data = data.one_hot_encode("timeofday",new_balanced_data)
preprocessed_training_data = updated_training_data.loc[:,~updated_training_data.columns.isin([
    'min', 'max', 'dwell_time',
       'first_clicked_item', 'last_clicked_item','timeofday',
       'visited_items', 'visited_categories','hourofclick',
       'Number_items_bought', 'unique_items_bought', 'count_x',
       'count_y'
])]
import seaborn as sn
import matplotlib.pyplot as plt


corrMatrix = preprocessed_training_data.corr()
plt.figure(figsize=(20,10))
sn.heatmap(corrMatrix, annot=True)
plt.show()
X = preprocessed_training_data.loc[:,~preprocessed_training_data.columns.isin(['is_buy'])]
y = preprocessed_training_data.loc[:,preprocessed_training_data.columns.isin(['is_buy'])]
X_train, X_test, y_train, y_test = data.data_preparation(X,y)
y_test['is_buy'] = pd.to_numeric(y_test['is_buy']).round(0).astype(int)
y_train['is_buy'] = pd.to_numeric(y_train['is_buy']).round(0).astype(int)
pred,prob = data.logit_model(X_train,y_train,X_test,y_test,0.49)
data.error_metrics(pred,y_test['is_buy'],prob)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
updated_training_data = data.one_hot_encode("timeofday",new_balanced_data)

preprocessed_training_data = new_balanced_data.loc[:,~new_balanced_data.columns.isin([
    'min', 'max', 'dwell_time',
       'first_clicked_item', 'last_clicked_item','timeofday',
       'visited_items', 'visited_categories','hourofclick',
       'Number_items_bought', 'unique_items_bought', 'count_x',
       'count_y'
])]
X = preprocessed_training_data.loc[:,~preprocessed_training_data.columns.isin(['is_buy'])]
y = preprocessed_training_data.loc[:,preprocessed_training_data.columns.isin(['is_buy'])]
X_train, X_test, y_train, y_test = data.data_preparation(X,y)
estimator = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate = 0.125, metric = 'l1', n_estimators = 20, num_leaves = 38)


#param_grid = {
#    'n_estimators': [x for x in [150,200,250]],
#    'learning_rate': [0.30,0.40,0.50],
#    'num_leaves': [30,35,40],
#    'boosting_type' : ['gbdt'],
#    'objective' : ['binary'],
#    'random_state' : [501]}
     
param_grid = {
    'n_estimators': [x for x in [150]],
    'learning_rate': [0.25],
    'num_leaves': [32],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501]}

gridsearch = GridSearchCV(estimator, param_grid)

gridsearch.fit(X_train, y_train.values.ravel(),eval_set = [(X_test, y_test)],eval_metric = ['auc', 'binary_logloss'],early_stopping_rounds = 10)
print('Best parameters found by grid search are:', gridsearch.best_params_)
gbm = lgb.LGBMClassifier(learning_rate = 0.35, metric = 'l1', n_estimators = 150,num_leaves= 35)


gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=20)
ax = lgb.plot_importance(gbm, height = 0.4, 
                         max_num_features = 15, 
                         xlim = (0,1000), ylim = (0,10), 
                         figsize = (10,6))
plt.show()
y_pred_prob = gbm.predict_proba(X_test)[:, 1]
auc_roc_0 = str(roc_auc_score(y_test, y_pred_prob))
print('AUC: \n' + auc_roc_0)
pred = []
for i in y_pred_prob:
    if i > 0.5:
        pred.append(1)
    else:
        pred.append(0)
data.error_metrics(pred,y_test['is_buy'],y_pred_prob)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
preprocessed_training_data = new_balanced_data.loc[:,~new_balanced_data.columns.isin([
    'min', 'max', 'dwell_time',
       'first_clicked_item', 'last_clicked_item','timeofday',
       'visited_items', 'visited_categories','hourofclick',
       'Number_items_bought', 'unique_items_bought', 'count_x',
       'count_y'
])]
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [150,200,250],
    'max_features': ['log2'],
    'max_depth' : [4,6],
    'criterion' :['gini']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 2)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=4, criterion='gini')
rfc1.fit(X_train, y_train)
probab_pred = rfc1.predict_proba(X_test)
pred = rfc1.predict(X_test)
auc_roc_0 = str(roc_auc_score(y_test, pred))
print('AUC: \n' + auc_roc_0)
print(list(zip(X_train,rfc1.feature_importances_)))
data.error_metrics(pred,y_test['is_buy'],probab_pred[:, 1])
from keras import models
from keras import layers
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras import backend as K
from matplotlib import pyplot as plt
updated_training_data = data.one_hot_encode("timeofday",new_balanced_data)

preprocessed_training_data = updated_training_data.loc[:,~updated_training_data.columns.isin([
    'min', 'max', 'dwell_time',
       'first_clicked_item', 'last_clicked_item','timeofday',
       'visited_items', 'visited_categories',#'hourofclick',
       'Number_items_bought', 'unique_items_bought', 'count_x',
       'count_y'
])]
preprocessed_training_data = preprocessed_training_data.reindex(columns=['dwell_time_seconds','total_clicks','dayofweek','dayofmonth','hourofclick','click_rate','total_unique_items','total_unique_categories','Number_clicked_visited_categories','Special_offer_click','first_clicked_item_popularity','last_clicked_item_popularity','Late Night','Early Morning','Morning','Noon','Evening','Night','is_buy'])
dataset = preprocessed_training_data.values
print(dataset.shape)
X = dataset[:,0:18]
y = dataset[:,18]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80,test_size=0.20, random_state=101)
class Neural_Net_CLassifier:
    
    def predicted(self,prediction):
        list1=[]
        for i in prediction:
            print("")
            for j in i:
                if j > 0.5:
                    list1.append(1)
                else:
                    list1.append(0)
        return list1

    def NN_arch4(self,lrate=0.0001):
        #2 hidden layer with a relu activation
        model = models.Sequential()
        model.add(layers.Dense(18,input_dim = 18, activation='sigmoid'))
        model.add(layers.Dense(6, activation='relu'))
        model.add(layers.Dense(5, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(lr=lrate)
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
        return model

    def model_fit(self,model,epoch_val=50):
        model.fit(X_train, y_train, epochs=epoch_val,batch_size=100)
        val_loss, val_acc = model.evaluate(X_test,y_test)
        print(val_loss, val_acc)
        return val_loss,val_acc

    def history_plot(self,history):
        training_loss = history.history['loss']
        test_loss = history.history['val_loss']

        training_acc = history.history['accuracy']
        test_acc = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history
        plt.figure(figsize=(5,3))

        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        #Visualize accuracy history
        plt.plot(epoch_count, training_acc, 'r--')
        plt.plot(epoch_count, test_acc, 'b-')
        plt.legend(['Training acc', 'Test acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.show();
NN_classifier = Neural_Net_CLassifier()
model = NN_classifier.NN_arch4(0.0006)
#print("Learning rate before second fit:", model.optimizer.learning_rate.numpy())
val_loss,val_acc = NN_classifier.model_fit(model,50)
#history = model.fit(X_train,y_train,epochs=10,verbose=0,validation_data=(X_test, y_test)) 
#NN_classifier.history_plot(history)
y_pred_keras = model.predict(X_test).ravel()
yhat_classes = model.predict_classes([X_test], verbose=0)
from sklearn.metrics import roc_curve
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
auc_roc_0 = str(roc_auc_score(y_test, prediction))
yhat_classes = yhat_classes[:, 0]
print('AUC: \n' + auc_roc_0)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)
class Item_Predictor:

    def p_root(self,value, root): 
        root_value = 1 / float(root) 
        return round (Decimal(value) **Decimal(root_value), 3) 

    def minkowski_distance(self,x, y, p_value): 
        return (self.p_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)), p_value))
    
    def data_transformation(self,result_4,df_clicks,df_buys):
        #clicks = result_4[result_4['session'] == 11628]
        clicks_new = result_4.groupby(['session','item_id','category']).size().reset_index()
        clicks_updated = pd.merge(clicks_new,df_clicks['popularity'],left_on=clicks_new.item_id,right_on=df_clicks.index)
        clicks_updated.columns = ['item_id_0','session','item_id','category','special_offer_click','click_popularity']
        clicks_updated_2 = clicks_updated.merge(df_buys['popularity'],how='left',left_on=clicks_updated.item_id,right_on=df_buys.index)
        click_updated_3 = clicks_updated_2.fillna(0)
        return click_updated_3
        
    def purchace_item_validation_func(self,data):
        if (data.Predicted_1 in data.Unique_Purchased_Items) or (data.Predicted_2 in data.Unique_Purchased_Items):
            print('True')
            list_1.append('True')
        else:
            list_1.append('False')
            print('False')
            
    def predict(self,data,df,temp):
        scores=[]        
        data = data[data['session'] == temp]
        if data.count()[0] > 1:
            for i in range(0,len(list(zip(*[data[col] for col in data])))):
                for j in range(i+1,len(list(zip(*[data[col] for col in data])))):    
                    out_1 = [t for t in list(zip(*[data[col] for col in data]))[i]]
                    out_2 = [t for t in list(zip(*[data[col] for col in data]))[j]]
                    score = item_pred.minkowski_distance(out_1,out_2,2)
                    scores.append([out_1[1],out_2[1],score])  
            list_1 = sorted(scores,key=lambda l:l[2])
            df.loc[temp] = [list_1[0][0],list_1[0][1]]            
        else:
            scores.append(data['item_id'])  
            #print("Items to be Purchased(ItemID):")
            df.loc[temp] = [data.iloc[0]['item_id'],None]
item_pred = Item_Predictor()
def fun(chunk):
    return chunk
  
result_4=None
count=1
names=["session", "timestamp", "item_id", "category"]
for chunk in pd.read_csv(clicks_file,names=names,usecols=["session","item_id","category"],converters={"category": data.convertCategory},chunksize=500000):
    print("Executing Chunk ",count,"/67")
    click_df = fun(chunk)
    if result_4 is None:
        result_4 = click_df
        count = count + 1
    else:
        result_4 = result_4.append(click_df)  
        count = count + 1
print("Done Transforming Clicks" )

result_4 = result_4.set_index('session')

colnames=['item_id','category']
result_4.columns = colnames
buys_data = data.file_input(buys_file)
names=["session", "timestamp", "item_id", "price", "qty"]
buys_data.columns = names
buys_transformed_data = data.transforming_buys(buys_data)
buys_transformed_data.head(10)
result_3=None
names=["session", "timestamp", "item_id", "category"]
count=1
for chunk in pd.read_csv(clicks_file,names=names,usecols = ['session','item_id'],chunksize=500000):
    print("Executing Chunk ",count,"/67")
    click_df = data.transforming_clicks3(chunk)
    if result_3 is None:
      result_3=click_df
      count = count+1
    else:
      result_3 = result_3.append(click_df)  
      count = count+1
print("Done Transforming Clicks to calculate \n 1.Visited Items per Session")
colnames=['session','visited_items']
result_3.columns = colnames
result_3 = result_3.set_index('session')
names=["session", "timestamp", "item_id", "category"]
result_items = pd.concat([chunk.apply(pd.Series.value_counts) for chunk in pd.read_csv(clicks_file,names=names,usecols = ['item_id'],index_col=0,chunksize=500000)])
df_clicks = pd.DataFrame(result_items.index.value_counts())
df_clicks.index.name = "item_id"
df_clicks.columns = ['count']
val = df_clicks['count'].sum()
df_clicks['popularity'] = df_clicks['count'].apply(lambda x : x / val )
df_clicks['popularity'] = df_clicks['popularity'].round(5)
names=["session", "timestamp", "item_id", "price", "qty"]
result_items = pd.concat([ chunk.apply(pd.Series.value_counts) for chunk in pd.read_csv(buys_file,names=names,usecols = ['item_id'],index_col=0,chunksize=500000)])
df_buys = pd.DataFrame(result_items.index.value_counts())
df_buys.index.name = "item_id"
df_buys.columns = ['count']
val = df_buys['count'].sum()
df_buys['popularity'] = df_buys['count'].apply(lambda x : x / val )
df_buys['popularity'] = df_buys['popularity'].round(5)
transformed_data = item_pred.data_transformation(result_4,df_clicks,df_buys)
transformed_data.drop(['key_0','item_id_0'], axis=1, inplace=True)
transformed_data_new = transformed_data.sort_values('session')
buys_sessions = np.unique(buys_transformed_data.index)
training_data = transformed_data_new[transformed_data_new['session'].isin(buys_sessions)]
from math import *
from decimal import Decimal 

df = pd.DataFrame(columns=['Predicted_1','Predicted_2'])

for i in buys_sessions[0:5000]:
    training_data_input = training_data[training_data['session'] == i]
    item_pred.predict(training_data,df,i)
df = df.fillna(0)
df_merged_predicted = df.merge(buys_transformed_data['unique_items_bought'],left_on=df.index,right_on=buys_transformed_data.index)
df_merged_predicted.columns=['session','Predicted_1','Predicted_2','Unique_Purchased_Items']
df_merged_predicted['Predicted_1'] = df_merged_predicted['Predicted_1'].astype(np.int64)
list_1=[]
df_merged_predicted.apply(item_pred.purchace_item_validation_func,axis=1)
count_T=0
count_F=0
for i in list_1:
    if i == 'True':
        count_T=count_T+1
    else:
        count_F=count_F+1
print('Our model could predict :',count_T/(count_T + count_F),'sample data correctly')
print("Prediction Items with session id are as follows:")
print(df)
