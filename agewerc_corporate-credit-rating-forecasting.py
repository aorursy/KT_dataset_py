import pandas as pd

import numpy as np

from numpy import loadtxt

from numpy import sort

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.ticker import PercentFormatter

import matplotlib.ticker as mtick

from wordcloud import WordCloud, STOPWORDS 

from random import sample

import seaborn as sns

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn import svm

from sklearn.metrics import classification_report

from sklearn import metrics

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.datasets import make_classification

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.utils import resample

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel
# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
df_rating = pd.read_csv('../input/corporate-credit-rating/corporate_rating.csv')
# Display the dimensions

print("The credit rating dataset has", df_rating.shape[0], "records, each with", df_rating.shape[1],

    "attributes")
# Display the structure

df_rating.info()
df_rating.head()
df_rating.Rating.value_counts()
rating_dict = {'AAA':'Lowest Risk', 

               'AA':'Low Risk',

               'A':'Low Risk',

               'BBB':'Medium Risk', 

               'BB':'High Risk',

               'B':'High Risk',

               'CCC':'Highest Risk', 

               'CC':'Highest Risk',

               'C':'Highest Risk',

               'D':'In Default'}



df_rating.Rating = df_rating.Rating.map(rating_dict)
ax = df_rating['Rating'].value_counts().plot(kind='bar',

                                             figsize=(8,4),

                                             title="Count of Rating by Type",

                                             grid=True)
df_rating = df_rating[df_rating['Rating']!='Lowest Risk'] # filter Lowest Risk

df_rating = df_rating[df_rating['Rating']!='In Default']  # filter In Default

df_rating.reset_index(inplace = True, drop=True) # reset index
# Statistical summary 

df_rating.describe()
column_list = list(df_rating.columns[6:31])

column_list = sample(column_list,4) 

print(column_list)
figure, axes = plt.subplots(nrows=2, ncols=4, figsize=(9,5))



axes[0, 0].hist(df_rating[column_list[0]])

axes[0, 1].hist(df_rating[column_list[1]])

axes[1, 0].hist(df_rating[column_list[2]])

axes[1, 1].hist(df_rating[column_list[3]])



axes[0, 2].boxplot(df_rating[column_list[0]])

axes[1, 2].boxplot(df_rating[column_list[1]])

axes[0, 3].boxplot(df_rating[column_list[2]])

axes[1, 3].boxplot(df_rating[column_list[3]])



figure.tight_layout()
df_rating.skew(axis=0)
for c in df_rating.columns[6:31]:



    q1 = df_rating[c].quantile(0.25)

    q3 = df_rating[c].quantile(0.75)

    iqr = q3 - q1 #Interquartile range

    fence_low  = q3-1.5*iqr

    fence_high = q1+1.5*iqr

    lower_out = len(df_rating.loc[(df_rating[c] < fence_low)  ,c])

    upper_out = len(df_rating.loc[(df_rating[c] > fence_high)  ,c])

    outlier_count = upper_out+lower_out

    prop_out = outlier_count/len(df_rating)

    print(c, ": "+"{:.2%}".format(prop_out))

df_rating_outlier = df_rating.copy()



for c in df_rating_outlier.columns[6:31]:

    

    q1 = df_rating_outlier[c].quantile(0.25)

    q3 = df_rating_outlier[c].quantile(0.75)

    iqr = q3 - q1 #Interquartile range

    fence_low  = q3-1.5*iqr

    fence_high = q1+1.5*iqr

    

    for i in range(len(df_rating_outlier)):

        

        if df_rating.loc[i,c] < fence_low or df_rating.loc[i,c] > fence_high: # if Outlier

            

            df_rating_outlier.loc[i,c] = 1

        

        else: # Not Outlier

            df_rating_outlier.loc[i,c] = 0
df_rating_outlier.head()
df_rating_outlier["total"] = df_rating_outlier.sum(axis=1)

df_rating_outlier.total.hist(bins = 20)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()



for c in df_rating.columns[6:31]:



    df_rating[[c]] = min_max_scaler.fit_transform(df_rating[[c]].to_numpy())*1000

    df_rating[[c]] = df_rating[c].apply(lambda x: np.log10(x+0.01))
figure, axes = plt.subplots(nrows=2, ncols=4, figsize=(9,5))



axes[0, 0].hist(df_rating[column_list[0]])

axes[0, 1].hist(df_rating[column_list[1]])

axes[1, 0].hist(df_rating[column_list[2]])

axes[1, 1].hist(df_rating[column_list[3]])



axes[0, 2].boxplot(df_rating[column_list[0]])

axes[1, 2].boxplot(df_rating[column_list[1]])

axes[0, 3].boxplot(df_rating[column_list[2]])

axes[1, 3].boxplot(df_rating[column_list[3]])



figure.tight_layout()
df_rating_no_out = df_rating.copy()



for c in df_rating_no_out.columns[6:31]:



    q05 = df_rating_no_out[c].quantile(0.10)

    q95 = df_rating_no_out[c].quantile(0.90)

    iqr = q95 - q05 #Interquartile range

    fence_low  = q05-1.5*iqr

    fence_high = q95+1.5*iqr

    df_rating_no_out.loc[df_rating_no_out[c] > fence_high,c] = df_rating_no_out[c].quantile(0.25)

    df_rating_no_out.loc[df_rating_no_out[c] < fence_low,c] = df_rating_no_out[c].quantile(0.75)

    
figure, axes = plt.subplots(nrows=8, ncols=3, figsize=(20,44))



i = 0 

j = 0



for c in df_rating_no_out.columns[6:30]:

    

    sns.boxplot(x=df_rating_no_out.Rating, y=df_rating_no_out[c], palette="Set3", ax=axes[i, j])

    

    if j == 2:

        j=0

        i+=1

    else:

        j+=1    

df_rating.colors = 'a'

df_rating_no_out.loc[df_rating_no_out['Rating'] == 'Lowest Risk', 'color'] = 'r'

df_rating_no_out.loc[df_rating_no_out['Rating'] == 'Low Risk', 'color'] = 'g'

df_rating_no_out.loc[df_rating_no_out['Rating'] == 'Medium Risk', 'color'] = 'b'

df_rating_no_out.loc[df_rating_no_out['Rating'] == 'High Risk','color'] = 'y'

df_rating_no_out.loc[df_rating_no_out['Rating'] == 'Highest Risk', 'color'] = 'm'
column_list = list(df_rating.columns[6:31])

column_list = sample(column_list,12) 
figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(14,14))



i = 0 

j = 0



for c in range(0,12, 2):



    sns.scatterplot(x = column_list[c], y=column_list[c+1], hue="color", data=df_rating_no_out, ax=axes[j,i])

    

    if i == 1:

        i = 0

        j +=1

    

    else:

        i+=1
le = preprocessing.LabelEncoder()

le.fit(df_rating.Sector)

df_rating.Sector = le.transform(df_rating.Sector) # encode sector

le.fit(df_rating.Rating)

df_rating.Rating = le.transform(df_rating.Rating) # encode rating
df_train, df_test = train_test_split(df_rating, test_size=0.2, random_state = 1234)
X_train, y_train = df_train.iloc[:,5:31], df_train.iloc[:,0]

X_test, y_test = df_test.iloc[:,5:31], df_test.iloc[:,0]
XGB_model = xgb.XGBRegressor(objective ='multi:softmax', num_class =4)

XGB_model.fit(X_train, y_train)

y_pred_XGB = XGB_model.predict(X_test)

Accuracy_XGB = metrics.accuracy_score(y_test, y_pred_XGB)

print("XGB Accuracy:",Accuracy_XGB)
GBT_model = GradientBoostingClassifier(random_state=123)

GBT_model.fit(X_train, y_train)

y_pred_GBT = GBT_model.predict(X_test)

Accuracy_GBT = metrics.accuracy_score(y_test, y_pred_GBT)

print("GBT Accuracy:",Accuracy_GBT)
RF_model = RandomForestClassifier(random_state=1234)

RF_model.fit(X_train,y_train)

y_pred_RF = RF_model.predict(X_test)

Accuracy_RF = metrics.accuracy_score(y_test, y_pred_RF)

print("RF Accuracy:",Accuracy_RF)
SVC_model = svm.SVC(kernel='rbf', gamma= 2, C = 5, random_state=1234)

SVC_model.fit(X_train, y_train)

y_pred_SVM = SVC_model.predict(X_test)

Accuracy_SVM = metrics.accuracy_score(y_test, y_pred_SVM)

print("SVM Accuracy:",Accuracy_SVM)
MLP_model = MLPClassifier(hidden_layer_sizes=(5,5,5), activation='logistic', solver='adam', max_iter=1500)

MLP_model.fit(X_train, y_train)

y_pred_MLP = MLP_model.predict(X_test)

Accuracy_MLP = metrics.accuracy_score(y_test, y_pred_MLP)

print("MLP Accuracy:",Accuracy_MLP)
GNB_model = GaussianNB()

GNB_model.fit(X_train, y_train)

y_pred_GNB = GNB_model.predict(X_test)

Accuracy_GNB = metrics.accuracy_score(y_test, y_pred_GNB)

print("GNB Accuracy:",Accuracy_GNB)
LDA_model = LinearDiscriminantAnalysis()

LDA_model.fit(X_train,y_train)

y_pred_LDA = LDA_model.predict(X_test)

Accuracy_LDA = metrics.accuracy_score(y_test, y_pred_LDA)

print("LDA Accuracy:",Accuracy_LDA)
QDA_model = QuadraticDiscriminantAnalysis()

QDA_model.fit(X_train,y_train)

y_pred_QDA = QDA_model.predict(X_test)

Accuracy_QDA = metrics.accuracy_score(y_test, y_pred_QDA)

print("QDA Accuracy:",Accuracy_QDA)
KNN_model = KNeighborsClassifier(n_neighbors = 3)

KNN_model.fit(X_train,y_train)

y_pred_KNN = KNN_model.predict(X_test)

Accuracy_KNN = metrics.accuracy_score(y_test, y_pred_KNN)

print("KNN Accuracy:",Accuracy_KNN)
LR_model = LogisticRegression(random_state=1234 , multi_class='multinomial', solver='newton-cg')

LR_model = LR_model.fit(X_train, y_train)

y_pred_LR = LR_model.predict(X_test)

Accuracy_LR = metrics.accuracy_score(y_test, y_pred_LR)

print("LR Accuracy:",Accuracy_LR)
accuracy_list = [Accuracy_XGB, Accuracy_GBT, Accuracy_RF, Accuracy_SVM, Accuracy_MLP, Accuracy_GNB, 

                 Accuracy_LDA, Accuracy_QDA, Accuracy_KNN, Accuracy_LR]



model_list = ['XGBboost', 'Gradient Boosting', 'Random Forest', 'Support Vector Machine', 

              "Neural Network", 'Naive Bayes', 'Linear Discriminat', 'Quadratic Discriminat', 

              'KNN', 'Logistic Regression']



df_accuracy = pd.DataFrame({'Model': model_list, 'Accuracy': accuracy_list})
order = list(df_accuracy.sort_values('Accuracy', ascending=False).Model)

df_accuracy = df_accuracy.sort_values('Accuracy', ascending=False).reset_index().drop(['index'], axis=1)



plt.figure(figsize=(12,8))

# make barplot and sort bars

x = sns.barplot(x='Model', y="Accuracy", data=df_accuracy, order = order, palette="rocket")

plt.xlabel("Model", fontsize=20)

plt.ylabel("Accuracy", fontsize=20)

plt.title("Accuracy by Model", fontsize=20)

plt.grid(linestyle='-', linewidth='0.5', color='grey')

plt.xticks(rotation=70, fontsize=12)

plt.ylim(0,1)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))



for i in range(len(model_list)):

    plt.text(x = i, y = df_accuracy.loc[i, 'Accuracy'] + 0.05, s = str(round((df_accuracy.loc[i, 'Accuracy'])*100, 2))+'%', 

             fontsize = 14, color='black',horizontalalignment='center')



y_value=['{:,.2f}'.format(x) + '%' for x in ax.get_yticks()]

ax.set_yticklabels(y_value)



plt.tight_layout()

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
params = XGB_model.get_xgb_params()
params
params['eval_metric'] = "merror"
num_boost_round = 1000
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=50,

    verbose_eval=30)



print("Best merror: {:.2f} with {} rounds".format(

                 model.best_score,

                 model.best_iteration+1))
cv_results = xgb.cv(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    seed=42,

    nfold=5,

    metrics={'merror'},

    early_stopping_rounds=50,

    verbose_eval=30

)

cv_results.tail()
cv_results['test-merror-mean'].min()
gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range(5,12)

    for min_child_weight in range(5,8)

]
# Define initial best params and MAE

min_merror = float("Inf")

best_params = None

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(

                             max_depth,

                             min_child_weight))

    

    # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'merror'},

        early_stopping_rounds=50,

        verbose_eval=False



    )

    # Update best merror

    mean_merror = cv_results['test-merror-mean'].min()

    boost_rounds = cv_results['test-merror-mean'].argmin()

    print("\tMAE {} for {} rounds".format(mean_merror, boost_rounds))

    if mean_merror < min_merror:

        min_merror = mean_merror

        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, merror: {}".format(best_params[0], best_params[1], min_merror))
params['max_depth'] = 7

params['min_child_weight'] = 5
gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/10. for i in range(7,11)]

    for colsample in [i/10. for i in range(7,11)]

]
# Define initial best params and MAE

min_merror = float("Inf")

best_params = None



for subsample, colsample in reversed(gridsearch_params):

    print("CV with subsample={}, colsample={}".format(

                             subsample,

                             colsample))

    # We update our parameters

    params['subsample'] = subsample

    params['colsample_bytree'] = colsample

    

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'merror'},

        early_stopping_rounds=10,

        verbose_eval=False

    )

    

    # Update best MAE

    mean_merror = cv_results['test-merror-mean'].min()

    boost_rounds = cv_results['test-merror-mean'].argmin()

    print("\tMAE {} for {} rounds".format(mean_merror, boost_rounds))

    if mean_merror < min_merror:

        min_merror = mean_merror

        best_params = (subsample,colsample)

        

print("Best params: {}, {}, merror: {}".format(best_params[0], best_params[1], min_merror))
params['subsample'] =0.9

params['colsample_bytree'] = 0.7
%time

# This can take some time…

min_merror = float("Inf")

best_params = None



for eta in [.3, .2, .1, .05, .01, .005]:

    print("CV with eta={}".format(eta))

    # We update our parameters

    params['eta'] = eta

    # Run and time CV

    cv_results = xgb.cv(

            params,

            dtrain,

            num_boost_round=num_boost_round,

            seed=42,

            nfold=5,

            metrics=['merror'],

            early_stopping_rounds=10

)

    # Update best score

    mean_mae = cv_results['test-merror-mean'].min()

    boost_rounds = cv_results['test-merror-mean'].argmin()

    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))

    if mean_merror < min_merror:

        min_merror = mean_merror

        best_params = eta

print("Best params: {}, merror: {}".format(best_params, min_merror))
params['eta'] = .3
params
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=1000,

    verbose_eval=100

)
num_boost_round = model.best_iteration + 1

best_model = xgb.train(

    params,

    dtrain,

    verbose_eval=100,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")]

)
metrics.accuracy_score(best_model.predict(dtest), y_test)
cm = confusion_matrix(y_test, y_pred_XGB)
fig, ax = plt.subplots(figsize=(8,8))



sns.heatmap(cm, annot = True, ax = ax, vmin=0, vmax=150, fmt="d", linewidths=.5, linecolor = 'white', cmap="Reds") # annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['Medium Risk','Highest Risk', 'Low Risk', 'High Risk'])

ax.yaxis.set_ticklabels(['Medium Risk','Highest Risk', 'Low Risk', 'High Risk']);



# This part is to correct a bug from the heatmap funciton from pyplot

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values



plt.show()
print(classification_report(y_test, y_pred_XGB, target_names = ['Medium Risk','Highest Risk', 'Low Risk', 'High Risk']))
thresholds = sort(XGB_model.feature_importances_)

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(XGB_model, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    # train model

    selection_model = XGBClassifier()

    selection_model.fit(select_X_train, y_train)

    # eval model

    select_X_test = selection.transform(X_test)

    y_pred = selection_model.predict(select_X_test)

    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)

    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
from xgboost import plot_importance



fig, ax = plt.subplots(figsize=(8, 8))

# xgboost.plot_importance(..., ax=ax)



plot_importance(model, ax=ax)

plt.show()
def WCloud(dataframe, column, rating):

    

    words = ''

    

    # iterate through the csv file 

    for val in dataframe.loc[dataframe['Rating'] == rating, column]:

      

        # typecaste each val to string 

        val = str(val)

        val = val.replace(".", "")

        val = val.replace(",", "")



        # split the value 

        tokens = val.split()



        #Converts each token into lowercase 

        for i in range(len(tokens)): 

            tokens[i] = tokens[i].lower() 



        words += " ".join(tokens) + " "

        

    return words
stop_words = ['global', 'incorporated', 'corporation', ' corp', 'industries', 'technologies', 'co', 'inc', 'limited', 'ltd', 'technology', 'resources', 'corp', 'group', 'communications',

             'holdings',' holding', 'plc', 'group', 'oil', 'resource', 'company','international', 'states', 'ag', ' sa', 'pty', 'international', 'united', 'states', 'partners', 'group', 

             'spa', 'se', 'lp', '(the)', 'the', 'LLC', 'n.v', 'service', 'products', 'companies', 'company', 'energy','corporation', 'holdings', 'company', 'limited',

             'holding', 'partners', 'industries', 'nv', 'semiconductor', 'rr', 'usa', 'homes', 'eletric', 'petroleum']
comment_wordsHR = WCloud(df_rating, 'Name', 0)

clean_text = [word for word in comment_wordsHR.split() if word not in stop_words]

comment_wordsHR = ' '.join([str(elem) for elem in clean_text])



comment_wordsHRest = WCloud(df_rating, 'Name', 1)

clean_text = [word for word in comment_wordsHRest.split() if word not in stop_words]

comment_wordsHRest = ' '.join([str(elem) for elem in clean_text])



comment_wordsLR = WCloud(df_rating, 'Name', 2)

clean_text = [word for word in comment_wordsLR.split() if word not in stop_words]

comment_wordsLR = ' '.join([str(elem) for elem in clean_text])



comment_wordsMR = WCloud(df_rating, 'Name', 3)

clean_text = [word for word in comment_wordsMR.split() if word not in stop_words]

comment_wordsMR = ' '.join([str(elem) for elem in clean_text])
wordcloudMR = WordCloud(background_color ='white', colormap="twilight", max_font_size = 25,

                min_font_size = 10).generate(comment_wordsHR) 



wordcloudLR = WordCloud(background_color ='white', colormap="twilight", max_font_size = 25,

                min_font_size = 10).generate(comment_wordsLR) 



wordcloudHR = WordCloud(background_color ='white',  colormap="ocean",max_font_size = 25,

                min_font_size = 10).generate(comment_wordsHR) 



wordcloudHRest = WordCloud(background_color ='white', colormap="gnuplot2",max_font_size = 25,

                min_font_size = 10).generate(comment_wordsHRest) 

fig = plt.figure(figsize = (17,10))

axes = fig.subplots(nrows=2, ncols=2)



plt.subplot(2, 2, 1)

plt.imshow(wordcloudMR, interpolation="bilinear") 

plt.axis("off") 

plt.margins(x=0, y=0)

plt.title('Medium Risk Companies', fontsize = 27)



plt.subplot(2, 2, 2)

plt.imshow(wordcloudLR, interpolation="bilinear") 

plt.axis("off") 

plt.margins(x=0, y=0)

plt.title('Low Risk Companies', fontsize = 27)



plt.subplot(2, 2, 3)

plt.imshow(wordcloudHR, interpolation="bilinear") 

plt.axis("off") 

plt.margins(x=0, y=0)

plt.title('High Risk Companies', fontsize = 27)



plt.subplot(2, 2, 4)

plt.imshow(wordcloudHRest, interpolation="bilinear") 

plt.axis("off") 

plt.margins(x=0, y=0)

plt.title('Highest Risk Companies', fontsize = 27, fontweight = 2)