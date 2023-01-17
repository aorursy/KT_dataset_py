import numpy as np

import pandas as pd

from bokeh.io import show , output_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource ,HoverTool

from bokeh.layouts import row , column , widgetbox

from bokeh.models.widgets import Tabs , Panel

from bokeh.application.handlers import FunctionHandler

from bokeh.application import Application

from ipywidgets import interact



import category_encoders as ce

from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier

from sklearn import metrics





output_notebook()
pd.options.display.max_columns = None

pd.options.display.max_rows = None
import os

os.listdir('../input/')
train_df = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test_df = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

samp_submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")

train_df.shape
train_df.head(10)
train_df.isna().sum()
train_df.columns
train_df.iloc[:,:].nunique()
cols = ['bin_0' , 'bin_1','bin_2','bin_3' ,'bin_4','ord_0','ord_1','ord_2','ord_3','ord_4',

        'nom_0','nom_1','nom_2','nom_3','nom_4','target' , 'day','month']

train_dfS = train_df.loc[:,cols].copy()

def Uniques(col):

    

    x = [str(i) for i in train_dfS.loc[:,col].unique() if not pd.isnull(i)]

    return x

import bokeh

from bokeh.models import Select

def modify_doc(doc):

    

    

    def create_figure():

        

        current_feature_name = feature_name.value

        targets = sorted(Uniques(current_feature_name))

        source = ColumnDataSource(data = {

            

            'x' : targets,

            'y' : train_dfS.loc[:,current_feature_name].value_counts().to_list(),

            'color' : bokeh.palettes.plasma(len(targets))

        })

        #print(source.data)

        plot = figure(x_range = targets,title = "Categorical Embedding -II" , plot_height = 500 , plot_width = 500)

        plot.vbar(x = 'x' , top = 'y' , color = 'color' , width = 0.5 , source = source,legend_field = 'x')

        plot.xaxis.axis_label = current_feature_name

        plot.yaxis.axis_label = "Counts"

        plot.legend.orientation = 'horizontal'

        plot.legend.location = 'top_right'

        plot.left[0].formatter.use_scientific = False

        plot.add_tools(HoverTool(tooltips = [('Counts' , '@y')]))

        #show(plot)

        return plot

        

    def update_plot(attr , old , new):

        

        layout.children[1] = create_figure()

        

    

    #Controls

    feature_name = Select(title = "Categorical Columns" , options = cols , value = cols[0])

    feature_name.on_change('value' , update_plot)

    p = create_figure()

    layout = row(widgetbox(feature_name) , p)

    doc.add_root(layout)



handler = FunctionHandler(modify_doc)

app = Application(handler)

        

        

        

        

        
doc = app.create_document()
show(app)

#I dont think kaggle kernels support bokeh interactive plots , but do download this notebook and see the cool interactive plot
#Let's fill in the missing data using mode



for col in train_df.columns:

    

    train_df[col].fillna(train_df[col].mode()[0] , inplace = True)

    



for col in test_df.columns:

    

    test_df[col].fillna(test_df[col].mode()[0] , inplace = True)
test_df.head()
#importing catgeory encoders library



def EncodeMapings(df):

    

    #Encoding for training set

    df_encoded = df.copy()

    

    df_encoded['bin_3'] = df_encoded['bin_3'].apply(lambda x : 0 if x == 'F' else 1)

    df_encoded['bin_4'] = df_encoded['bin_4'].apply(lambda x : 0 if x == 'N' else 1)

    

    df_encoded.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],

                         value = [0, 1, 2, 3, 4], inplace = True)



    df_encoded.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],

                         value = [0, 1, 2, 3, 4, 5], inplace = True)



    df_encoded.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],

                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)



    df_encoded.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 

                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],

                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 

                                  22, 23, 24, 25], inplace = True)

    

    return df_encoded

    

   

    





    

    

    
train_dfP = EncodeMapings(train_df)

test_dfP = EncodeMapings(test_df)
nom_col = ['nom_0','nom_1' ,'nom_2','nom_3','nom_4']

    

        

for col in nom_col:

            woe_enc = ce.WOEEncoder()

        

        

            train_dfP[f'{col}_woe'] = woe_enc.fit_transform(train_dfP[col] , train_dfP.target)

            test_dfP[f'{col}_woe'] = woe_enc.transform(test_dfP[col])

    

 #Using Leave one outencoder for high cardinality data

    

    

high_card = ['nom_5' , 'nom_6' , 'nom_7','nom_8','nom_9','ord_5'] 

        

        

for col in high_card:

            loo_enc = ce.LeaveOneOutEncoder()

        

        

            train_dfP[f'{col}_loo'] = loo_enc.fit_transform(train_dfP[col] , train_dfP.target)

            test_dfP[f'{col}_loo'] = loo_enc.transform(test_dfP[col])

    

train_dfP.drop(['nom_0','nom_1' ,'nom_2','nom_3','nom_4','nom_5' , 'nom_6' , 'nom_7','nom_8','nom_9','ord_5'],

                   inplace = True , axis = 1)

test_dfP.drop(['nom_0','nom_1' ,'nom_2','nom_3','nom_4','nom_5' , 'nom_6' , 'nom_7','nom_8','nom_9','ord_5'],

                   inplace = True , axis = 1)
print("Training Set Shape: {}".format(train_dfP.shape))

print("Test set Shape: {}".format(test_dfP.shape))
train_dfP.to_csv("train_dfP" , index = False)

test_dfP.to_csv("test_dfP", index = False)
import gc

del train_df , train_dfS ,test_df

gc.collect()
y = train_dfP.target.values

X = train_dfP.drop(['target','id'] , axis = 1).values

test_dfP.drop(['id'] , inplace = True , axis = 1)


skf = StratifiedKFold(n_splits = 5 ,random_state = 42 ,shuffle = True)



model = CatBoostClassifier(iterations=600,

                              learning_rate=0.01,

                              depth=5,

                              bootstrap_type='Bernoulli',

                              loss_function='Logloss',

                              subsample=0.9,

                              eval_metric='AUC',

                              metric_period=50,

                              allow_writing_files=False)





oof_y = []

oof_pred = []



scores = []



for train_idx, test_idx in skf.split(X,y):

    

    X_train , X_val = X[train_idx] , X[test_idx]

    y_train , y_val = y[train_idx] , y[test_idx]

    

    model.fit(X_train , y_train , eval_set = (X_val , y_val))

    

    pred = model.predict_proba(X_val)[:,1]

    

    oof_y.append(y_val)

    oof_pred.append(pred)

    score = metrics.roc_auc_score(y_val , pred)

    print("Fold Score :{}".format(score))

    scores.append(score)

    



    

    

    

    

    

    
print("Mean Auc_roc Score : {}".format(sum(scores) / skf.n_splits))
#plot model feature importances



feature_names = [col for col in train_dfP.columns if col not in ['target' , 'id']]

source = ColumnDataSource(data = {'x' : feature_names,

                                  'y' : model.feature_importances_,

                                  'color' : bokeh.palettes.turbo(len(feature_names))

                                 })



plot = figure(x_range = feature_names , title= "Feature Importance" , plot_height = 1500 , plot_width = 1500)

plot.vbar(x = 'x' , top = 'y' ,color  =  'color' , source = source  , width = 0.5)

#plot.legend.orientation = 'horizontal'

#plot.legend.location = 'top_right'

plot.left[0].formatter.use_scientific = False

plot.add_tools(HoverTool(tooltips = [('Value' , '@y')]))

plot.xaxis.axis_label = 'Features'

plot.yaxis.axis_label = 'Values'



show(plot)

df_test = test_dfP.values
test_preds = model.predict_proba(df_test)[:,1]

test_preds

#samp_submission.drop('targets' ,inplace = True , axis = 1)

samp_submission['target'] = test_preds
samp_submission.to_csv("Submission_baseline.csv" , index  =False)

