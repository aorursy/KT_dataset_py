# import all libraries



import pandas as pd #basic

import numpy as np #basic

import pandas_profiling as pp #EDA



from scipy.stats import shapiro #Stats

from scipy.stats import pearsonr #Stats

import scipy.stats as stats #Stats



import plotly.express as px #visualization

import plotly.graph_objs as go#visualization

import plotly.offline as py#visualization

import plotly.figure_factory as ff#visualization



from sklearn.model_selection import train_test_split #Split data

from sklearn import preprocessing #manipulate data

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_score,recall_score,f1_score,roc_auc_score



#load the dataset 

loans = pd.read_csv('../input/Bank_Personal_Loan_Modelling-1.csv')
print ("Rows     : " ,loans.shape[0])

print ("Columns  : " ,loans.shape[1])

print ("\nFeatures : \n" ,loans.columns.tolist())

print ("\nMissing values :  ", loans.isnull().sum().values.sum())

print ("\nUnique values :  \n",loans.nunique())
loans.describe().transpose()
# Get the target column distribution. Your comments

loans['Personal Loan'].value_counts()
df_new=loans.copy()

pp.ProfileReport(df_new)
# Experience column has -ve values as seen from decribe output

# total rows with this issue



print (loans[loans['Experience'] < 0]['Experience'].value_counts().sum())



# we have 52 record thats less than one percent, so we will take abstract values



loans['Experience'] = loans['Experience'].apply(abs)



# New dataframe to get Vizulizations from data



loans_viz = loans.copy()



# replace values



loans_viz['Securities Account'] = loans_viz['Securities Account'].replace({1: 'Yes', 0: 'No'})

loans_viz['CD Account'] = loans_viz['CD Account'].replace({1: 'Yes',0: 'No'})

loans_viz['Online'] = loans_viz['Online'].replace({1: 'Yes', 0: 'No'})

loans_viz['CreditCard'] = loans_viz['CreditCard'].replace({1: 'Yes',0: 'No'})



# Make Mortgage a Yes or No field just for vizulization purposes



def mort_lab(loans_viz):

    if loans_viz['Mortgage'] > 0:

        return 'Yes'

    elif loans_viz['Mortgage'] == 0:

        return 'No'



loans_viz['Mortgage'] = loans_viz.apply(lambda loans_viz:mort_lab(loans_viz), axis=1)



# Separating customers who took loans vs who didnt



loan = loans_viz[loans_viz['Personal Loan'] == 1]

not_loan = loans_viz[loans_viz['Personal Loan'] == 0]



# Separating catagorical and numerical columns



Id_col = ['ID']

target_col = ['Personal Loan']

exclude = ['ZIP Code']# as zip code is not having any value in personal loan predictions

cat_cols = loans_viz.nunique()[loans_viz.nunique() < 6].keys().tolist()

cat_cols = [x for x in cat_cols if x not in target_col + exclude]

num_cols = [x for x in loans_viz.columns if x not in cat_cols

            + target_col + Id_col + exclude]

#check distirbution

def check_dist(col_name):

    stat, p = shapiro(loans[col_name])

#     print('stat=%.3f, p=%.3f' % (stat, p))

    if p > 0.05:

        print('Distribution of {} column is Probably Normal'.format(col_name))

    else:

        print('Distribution of {} column is Probably Not Normal'.format(col_name))

for col in num_cols:

    check_dist(col)        
#check correlation

def check_dependancy(col1,col2):

    stat, p = pearsonr(loans[col1], loans[col2])

    if p > 0.05:

        print('{} and {} are Probably independent<----'.format(col1,col2))

    else:

        print('{} and {} are Probably dependent'.format(col1,col2))

check_dependancy('Experience','Income')

check_dependancy('Age','Income')

check_dependancy('Education','Income')

check_dependancy('Family','Income')

check_dependancy('Family','Mortgage')

check_dependancy('Family','Personal Loan')

check_dependancy('CCAvg','Personal Loan')

check_dependancy('ZIP Code','Personal Loan')

check_dependancy('Income','Personal Loan')

check_dependancy('Age','Personal Loan')
# T-test to check dependency of smoking on charges

Ho = "Income of Loan Customer and non-Loan Customer are same"   # Stating the Null Hypothesis

Ha = "Income of Loan Customer and non-Loan Customer are not same"   # Stating the Alternate Hypothesis



x = np.array(loans[loans['Personal Loan']==1].Income)  # Selecting income corresponding to  Loan Customer as an array

y = np.array(loans[loans['Personal Loan']==0].Income) # Selecting income corresponding to non- Loan Customer as an array



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
Ho = "No. of family has no effect on loan"   # Stating the Null Hypothesis

Ha = "No. of family has an effect on loan"   # Stating the Alternate Hypothesis





one = loans[loans.Family == 1]['Personal Loan']

two = loans[loans.Family == 2]['Personal Loan']

three = loans[loans.Family == 3]['Personal Loan']

four = loans[loans.Family == 4]['Personal Loan']





f_stat, p_value = stats.f_oneway(one,two,three,four)





if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')
names=loans_viz['Personal Loan'].value_counts().keys()

values=loans_viz['Personal Loan'].value_counts().values

fig = go.Figure(data=[go.Pie(labels=names, values=values, pull=[0,  0.2])])

fig.show()
px.scatter(data_frame=loans_viz,x='Age',y='Income',color='Personal Loan')
px.scatter(data_frame=loans_viz,x='Age',y='Experience',color='Personal Loan')
px.strip(data_frame=loans_viz,x='Family',y='Income',color='Personal Loan')
px.strip(data_frame=loans,x='Family',y='Mortgage',color='Personal Loan')
px.strip(data_frame=loans,x='Income',y='Mortgage',color='Personal Loan')
px.scatter(data_frame=loans_viz,x='CCAvg',y='Family',color='Personal Loan')
def plot_pie(column):



    trace1 = go.Pie(

        values=loans_viz[column].value_counts().values.tolist(),

        labels=loans_viz[column].value_counts().keys().tolist(),

        hoverinfo='label+percent+name',

        domain=dict(x=[0, .48]),

        name='Personal Loan Customers',

        marker=dict(line=dict(width=2, color='rgb(243,243,243)')),

        hole=.6,

        )

    trace2 = go.Pie(

        values=not_loan[column].value_counts().values.tolist(),

        labels=not_loan[column].value_counts().keys().tolist(),

        hoverinfo='label+percent+name',

        marker=dict(line=dict(width=2, color='rgb(243,243,243)')),

        domain=dict(x=[.52, 1]),

        hole=.6,

        name='Non Loan Customers',

        )



    layout = go.Layout(dict(title=column

                       + ' Distribution in Personal Loans ',

                       plot_bgcolor='rgb(243,243,243)',

                       paper_bgcolor='rgb(243,243,243)',

                       annotations=[dict(text='Personal Loan Customers'

                       , font=dict(size=13), showarrow=False, x=.15,

                       y=.5), dict(text='Non Loan Customers',

                       font=dict(size=13), showarrow=False, x=.88,

                       y=.5)]))



    data = [trace1, trace2]

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)





    # for all categorical columns plot pie



for i in cat_cols:

    plot_pie(i)
def histogram(column):

    trace1 = go.Histogram(x=loans[column], histnorm='percent',

                          name='Loan Customers',

                          marker=dict(line=dict(width=.5, color='black'

                          )), opacity=.9)



    trace2 = go.Histogram(x=not_loan[column], histnorm='percent',

                          name='Non Loan Customers',

                          marker=dict(line=dict(width=.5, color='black'

                          )), opacity=.9)

    data = [trace1, trace2]

    layout = go.Layout(dict(title=column

                       + ' distribution in Personal Loans ',

                       plot_bgcolor='rgb(243,243,243)',

                       paper_bgcolor='rgb(243,243,243)',

                       xaxis=dict(gridcolor='rgb(255, 255, 255)',

                       title=column, zerolinewidth=1, ticklen=5,

                       gridwidth=2),

                       yaxis=dict(gridcolor='rgb(255, 255, 255)',

                       title='percent', zerolinewidth=1, ticklen=5,

                       gridwidth=2)))

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)

for i in num_cols:

    histogram(i)
##Split the train and test with 30%ratio



(train, test) = train_test_split(loans, test_size=.3, random_state=111)



##seperating dependent and independent variables



cols = [i for i in loans.columns if i not in Id_col + target_col + exclude]

X_train = train[cols]

y_train = train[target_col].values.ravel()

X_test = test[cols]

y_test = test[target_col].values.ravel()
y_train
X_train=preprocessing.scale(X_train)

X_test=preprocessing.scale(X_test)
def print_conf_matrix(conf_mat):

    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(

        z=conf_mat,

        x=['Actual: Not Loan', ' Actual: Loan'],

        y=['Predict: Not Loan', 'Predict: Loan'],

        showscale=False,

        colorscale='Rainbow',

        name='matrix',

        xaxis='x2',

        yaxis='y2',

        ))

    fig.show()
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(k):

    knn = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = k, p = 2, weights = 'uniform')

    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    probabilities = knn.predict_proba(X_test)

    precision = precision_score(y_test, predictions)

    accuracy = accuracy_score(y_test, predictions)

    rs = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    print('f1_score  for k ={}  is  {}'.format(k, f1))

#     print('Accuracy score for k ={}  is  {}'.format(k, accuracy))

for k in range(1, 20, 2):

    knn_classifier(k)
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = 5, p = 2, weights = 'uniform')

    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    print(classification_report(y_test, predictions))
## Logistic regression

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=1,solver='liblinear')

log_reg.fit(X_train, y_train)

predictions = log_reg.predict(X_test)

print(classification_report(y_test, predictions))
from sklearn.naive_bayes import GaussianNB

gb = GaussianNB()

gb.fit(X_train, y_train)

predictions = gb.predict(X_test)

print(classification_report(y_test, predictions))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=11)

dtc.fit(X_train, y_train)

predictions = dtc.predict(X_test)

print(classification_report(y_test, predictions))
from sklearn.neighbors import RadiusNeighborsClassifier

rnc = RadiusNeighborsClassifier(radius=1.5,outlier_label =1)

rnc.fit(X_train, y_train)

predictions = rnc.predict(X_test)

print(classification_report(y_test, predictions))
from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score



#gives model report in dataframe

def model_report(model,name) :

    model.fit(X_train,y_train)

    predictions  = model.predict(X_test)

    accuracy     = accuracy_score(y_test,predictions)

    recallscore  = recall_score(y_test,predictions)

    precision    = precision_score(y_test,predictions)

    roc_auc      = roc_auc_score(y_test,predictions)

    f1score      = f1_score(y_test,predictions) 

    kappa_metric = cohen_kappa_score(y_test,predictions)

    

    df = pd.DataFrame({"Model"           : [name],

                       "Accuracy_score"  : [accuracy],

                       "Recall_score"    : [recallscore],

                       "Precision"       : [precision],

                       "f1_score"        : [f1score],

                       "Area_under_curve": [roc_auc],

                       "Kappa_metric"    : [kappa_metric],

                      })

    return df



#outputs for every model

model1 = model_report(knn,"KNN Classifier")

model2 = model_report(log_reg,"LogisticRegr")

model3 = model_report(gb,"GaussianNB")

model4 = model_report(dtc,"Decision Tree")

model5 = model_report(rnc,"RNC")

#concat all models

model_performances = pd.concat([model1,model2,model3,

                                model4,model5],axis = 0).reset_index()



model_performances = model_performances.drop(columns = "index",axis =1)



table  = ff.create_table(np.round(model_performances,4))



py.iplot(table)
model_performances

def output_tracer(metric,color) :

    tracer = go.Bar(y = model_performances["Model"] ,

                    x = model_performances[metric],

                    orientation = "h",name = metric ,

                    marker = dict(line = dict(width =.7),

                                  color = color)

                   )

    return tracer



layout = go.Layout(dict(title = "Model performances",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "metric",

                                     zerolinewidth=1,

                                     ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        margin = dict(l = 250),

                        height = 780

                       )

                  )





trace1  = output_tracer("Accuracy_score","#6699FF")

trace2  = output_tracer('Recall_score',"red")

trace3  = output_tracer('Precision',"#33CC99")

trace4  = output_tracer('f1_score',"lightgrey")

trace5  = output_tracer('Kappa_metric',"#FFCC99")



data = [trace1,trace2,trace3,trace4,trace5]

fig = go.Figure(data=data,layout=layout)

py.iplot(fig)

from sklearn.ensemble import BaggingClassifier

k_bgc = BaggingClassifier(base_estimator=KNeighborsClassifier(),n_estimators=25,max_samples=.7,oob_score=True,random_state=1)

k_bgc.fit(X_train,y_train)

predictions=k_bgc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
lr_bgc = BaggingClassifier(base_estimator=LogisticRegression(random_state=1,solver='liblinear'),n_estimators=25,max_samples=.7,oob_score=True,random_state=1)

lr_bgc.fit(X_train,y_train)

predictions=lr_bgc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
gb_bgc = BaggingClassifier(base_estimator=GaussianNB(),n_estimators=25,max_samples=.7,oob_score=True,random_state=1)

gb_bgc.fit(X_train,y_train)

predictions=gb_bgc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
dt_bgc = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=25,max_samples=.7,oob_score=True,random_state=1)

dt_bgc.fit(X_train,y_train)

predictions=dt_bgc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
from sklearn.ensemble import AdaBoostClassifier
lr_abc = AdaBoostClassifier(base_estimator=LogisticRegression(random_state=1,solver='liblinear'),random_state=1,learning_rate=.5,n_estimators=75)

lr_abc.fit(X_train,y_train)

predictions = lr_abc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
gb_abc = AdaBoostClassifier(base_estimator=GaussianNB(),random_state=1,learning_rate=.5,n_estimators=75)

gb_abc.fit(X_train,y_train)

predictions = gb_abc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
dtc_abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),random_state=1,learning_rate=.5,n_estimators=75)

dtc_abc.fit(X_train,y_train)

predictions = dtc_abc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
abc = AdaBoostClassifier(random_state=1,learning_rate=.5,n_estimators=75)

abc.fit(X_train,y_train)

predictions = abc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=50,random_state=50)

gbc.fit(X_train,y_train)

predictions = gbc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
from sklearn.ensemble import VotingClassifier
clf1= GradientBoostingClassifier(n_estimators=50,random_state=50)

clf2=AdaBoostClassifier(base_estimator=dtc,random_state=1,learning_rate=.5,n_estimators=75)

clf3=LogisticRegression(random_state=1,solver='liblinear')

clf4=DecisionTreeClassifier(max_depth=11)

clf5=RadiusNeighborsClassifier(radius=1.5,outlier_label =1)



vc = VotingClassifier(estimators=[  ('log_reg', clf3), ('dtc', clf4), ('rnvc', clf5),('gb',clf1),('adab',clf2)],voting='hard')

vc.fit(X_train,y_train)

predictions= vc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
from sklearn.ensemble import RandomForestClassifier

rc = RandomForestClassifier(n_estimators =75,random_state=1)

rc.fit(X_train,y_train)

predictions = rc.predict(X_test)

print(f1_score(y_test,predictions))

print(classification_report(y_test, predictions))
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
params = {

 'task': 'train'

 , 'boosting_type': 'gbdt'

 , 'objective':  'multiclass'

 , 'num_class': 2

 , 'metric':  'multi_logloss'

 , 'min_data': 1

 , 'verbose': -1

}

 

gbm = lgb.train(params, lgb_train, num_boost_round=50)
predictions = gbm.predict(X_test)

predictions_classes = []

for i in predictions:

  predictions_classes.append(np.argmax(i))

predictions_classes = np.array(predictions_classes)

print(classification_report(predictions_classes, y_test))
import xgboost  as xgb
D_train = xgb.DMatrix(X_train, label=y_train)

D_test = xgb.DMatrix(X_test, label=y_test)





param = {

    'eta': 0.3, 

    'max_depth': 3,  

    'objective': 'multi:softprob',  

    'num_class': 3} 



steps = 20  # The number of training iterations







model = xgb.train(param, D_train, steps)







preds = model.predict(D_test)

best_preds = np.asarray([np.argmax(line) for line in preds])



# print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))

# print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))

# print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))

# print("f1 = {}".format(f1_score(y_test, best_preds)))

print(classification_report(y_test,best_preds))
from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score



#gives model report in dataframe

def model_report(model,name) :

    model.fit(X_train,y_train)

    predictions  = model.predict(X_test)

    accuracy     = accuracy_score(y_test,predictions)

    recallscore  = recall_score(y_test,predictions)

    precision    = precision_score(y_test,predictions)

    roc_auc      = roc_auc_score(y_test,predictions)

    f1score      = f1_score(y_test,predictions) 

    kappa_metric = cohen_kappa_score(y_test,predictions)

    

    df = pd.DataFrame({"Model"           : [name],

                       "Accuracy_score"  : [accuracy],

                       "Recall_score"    : [recallscore],

                       "Precision"       : [precision],

                       "f1_score"        : [f1score],

                       "Area_under_curve": [roc_auc],

                       "Kappa_metric"    : [kappa_metric],

                      })

    return df



#outputs for every model

model1 = model_report(k_bgc,"bag_KNN")

model2 = model_report(lr_bgc,"bag_LR")

model3 = model_report(gb_bgc,"bag_GB")

model4 = model_report(dt_bgc,"bag_DT")

model5 = model_report(lr_abc,"ab_LR")

model6 = model_report(gb_abc,"ab_GB")

model7 = model_report(dtc_abc,"ab_DT")

model8 = model_report(abc,"ab_default")

model9 = model_report(gbc,"GBC")

model10 = model_report(vc,"VC")

model11 = model_report(rc,"RFC")

# model12 = model_report(gbm,"LGB")

# model13 = model_report(model,"XGB")

#concat all models

ens_performances = pd.concat([model1,model2,model3,

                                model4,model5,model6,

								model7,model8,model9,

								model10,model11],axis = 0).reset_index()



ens_performances = ens_performances.drop(columns = "index",axis =1)



table  = ff.create_table(np.round(ens_performances,4))



py.iplot(table)
model_performances

def output_tracer(metric,color) :

    tracer = go.Bar(y = ens_performances["Model"] ,

                    x = ens_performances[metric],

                    orientation = "h",name = metric ,

                    marker = dict(line = dict(width =.7),

                                  color = color)

                   )

    return tracer



layout = go.Layout(dict(title = "Model performances",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "metric",

                                     zerolinewidth=1,

                                     ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        margin = dict(l = 250),

                        height = 780

                       )

                  )





trace1  = output_tracer("Accuracy_score","#6699FF")

trace2  = output_tracer('Recall_score',"red")

trace3  = output_tracer('Precision',"#33CC99")

trace4  = output_tracer('f1_score',"lightgrey")

trace5  = output_tracer('Kappa_metric',"#FFCC99")



data = [trace1,trace2,trace3,trace4,trace5]

fig = go.Figure(data=data,layout=layout)

py.iplot(fig)
