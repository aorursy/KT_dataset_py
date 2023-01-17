import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import string
df = pd.read_csv('dataset_zip.csv')
df = df[['user_id','prod_id','prod_name','date','rating','review_text','label']]
df.head()
# Number of unique customers
df.user_id.nunique()
# Number of unique restaurants
df.prod_id.nunique()
import plotly.express as px
# Plot 1 : Distribution of genuine and fake reviews——Unbalanced dataset
df = df.reset_index()
df_review_distribution = df.groupby('label', as_index = False ).agg({"index":"count"}).rename(columns = {"index":"count"})

colors = ['FireBrick', 'Salmon'] 
fig1 = px.pie(df_review_distribution, values='count', color = 'label', names=['Genuine', 'Fake'], 
                title='Review distribution',  hole=.3)
fig1.update_traces(textposition='inside', textinfo='value+ percent+label', marker=dict(colors=colors))
fig1.update_layout(width = 900, height = 430, showlegend=True, title ={"x": 0.51,"y": 0.9},
                   legend={"x": 0.39,"y": -0.1, "orientation": "h","yanchor": "top"})
fig1.show()
# Plot 2: Number of reviews vs. day of week (genuine/fake)
df['date'] = pd.to_datetime(df['date'])
df['day_of_week_name'] = df['date'].dt.day_name()
df['day_of_week'] = df['date'].apply(lambda x: x.weekday())

df_weekday_distribution = df[['index','day_of_week_name','day_of_week','label']].groupby(['label','day_of_week_name','day_of_week'], 
                           as_index = False).agg({'index':'count'}).rename(columns = {'index':'number'})\
                           .sort_values(by=['label','day_of_week'])
df_weekday_genuine = df_weekday_distribution[df_weekday_distribution.label == 0]
df_weekday_fake = df_weekday_distribution[df_weekday_distribution.label == 1]

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig2 = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], specs=[[{"type": "bar"}, {"type": "bar"}]])

fig2.add_trace(go.Bar(x=df_weekday_genuine.day_of_week_name, y=df_weekday_genuine.number, text=df_weekday_genuine.number, 
                       name='Genuine', marker_color='FireBrick'), row=1, col=1)
fig2.add_trace(go.Bar(x=df_weekday_fake.day_of_week_name, y=df_weekday_fake.number, text=df_weekday_fake.number, 
                      name='Fake', marker_color='lightsalmon'), row=1, col=2)
fig2.update_layout(title = 'Number of Reviews vs. Day of Week', autosize = False, width = 900, height = 400, hovermode = False,
                  showlegend=True, legend={"x": 0.38,"y": -0.3, "orientation": "h","yanchor": "top"})
fig2.update_traces(texttemplate='%{text:.2s}', textposition='inside')
fig2.show()
# Plot 3: Rating distribution
import plotly.figure_factory as ff

df_rating_distribution = df[['index','rating','label']].groupby(['label','rating'], as_index = False)\
                         .agg({'index':'count'}).rename(columns = {'index':'number'}).sort_values(by=['label','rating'])
df_rating_genuine = df_rating_distribution[df_rating_distribution.label == 0]
df_rating_fake = df_rating_distribution[df_rating_distribution.label == 1]

# Plot 3: Rating distribition
fig3 = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], specs=[[{"type": "bar"}, {"type": "bar"}]])
fig3.add_trace(go.Bar(x=df_rating_genuine.rating, y=df_rating_genuine.number, text=df_rating_genuine.number, 
                       name='Genuine', marker_color='FireBrick'), row=1, col=1)
fig3.add_trace(go.Bar(x=df_rating_fake.rating, y=df_rating_fake.number, text=df_rating_fake.number, 
                      name='Fake', marker_color='lightsalmon'), row=1, col=2)
fig3.update_layout(title = 'Rating Distribution', autosize = False, width = 900, height = 400, hovermode = False,
                  showlegend=True, legend={"x": 0.38,"y": -0.1, "orientation": "h","yanchor": "top"})
fig3.update_traces(texttemplate='%{text:.2s}', textposition='inside')
fig3.show()
# Day of week
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].apply(lambda x: x.weekday())

# Number of chars in the review;
df['char_count'] = df['review_text'].apply(len)

# Number of words in the review;
df['word_count'] = df['review_text'].apply(lambda x: len(x.split()))

# Word Density
df['word_density'] = df['char_count'] / (df['word_count']+1)

# Percentage of exclamation
df['exclamation_percent'] = df['review_text'].apply(lambda x: len([i for i in x.split() if i == '!'])/len(x.split())) 

# Percentage of punctuation
df['punctuation_percent'] = df['review_text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))/len(x.split())) 

# Percentage of title words
df['title_word_percent'] = df['review_text'].apply(lambda x: len([i for i in x.split() if i.istitle()])/len(x.split()))
                                                   
# Percentage of upper case words                                                   
df['upper_case_word_percent'] = df['review_text'].apply(lambda x: len([i for i in x.split() if i.isupper()])/len(x.split()))
                                                   
# Percentage of first person                                                   
df['first_person_percent'] = df['review_text'].apply(lambda x: len([i for i in x.split() if (i == 'I' or i == 'i')])/len(x.split()))             
# Calculate polarity score of each review text
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def text_clean(text):
    # Convert to lower case
    lower_text = text.lower()
    
    # Remove punctuation
    lower_text = re.sub("[()\"\'\?\.\,\%\/\!\:\;\-\=\#\&_]", " ", lower_text)
    
    # Tokenize text
    token_text = word_tokenize(lower_text)
    
    # Remove stopwords and words with length <=2
    stop_words = set(stopwords.words('english'))
    filtered_text = [w for w in token_text if (not w in stop_words) and (len(w)>2)]
    
    text = " ".join(filtered_text).strip()
    
    return text

df['review_text_token'] = df['review_text'].apply(text_clean)
from textblob import TextBlob
def text_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['text_polarity'] = df['review_text_token'].apply(text_polarity)
def intragroup(cands, taggings, features):
    dupers = []
    for thisgrp in cands:
        grpmask = taggings == thisgrp
        revcount = grpmask.sum()
        if revcount > 1:
            simgrp = features[grpmask.values]
            if simgrp.nunique() != len(simgrp):
                dupers.append(thisgrp)
    return dupers
# Total number of reviews        
df['user_no_of_review'] = df.groupby('user_id')['user_id'].transform('size')

# Max number of reviews per day
user_max_no_reviews = df.groupby(['user_id','date'], as_index = False).agg({'review_text':'count'}).drop('date',axis = 1)\
                 .groupby(['user_id'],as_index = False).agg({'review_text':'max'}).rename(columns = {'review_text':'user_max_no_review_per_day'})

df = pd.merge(df, user_max_no_reviews, how = 'inner', on = 'user_id')

# Average rating
df['user_avg_rating'] = df.groupby('user_id')['rating'].transform('mean')

# Rating deviation
df['user_rating_std'] = df.groupby('user_id')['rating'].transform('std')

# Average number of words
df['user_avg_no_words'] = df.groupby('user_id')['word_count'].transform('mean')

# Whether user has submitted exact same review text twice
duplicating_users = intragroup(df['user_id'].unique(), df['user_id'], df['review_text']) # slow
df['user_has_dup_text'] = df['user_id'].apply(lambda x: x in duplicating_users).astype(int)
# Total number of reviews        
df['prod_no_of_review'] = df.groupby('prod_id')['prod_id'].transform('size')

# Max number of reviews received per day
prod_max_no_reviews = df.groupby(['prod_id','date'], as_index = False).agg({'review_text':'count'}).drop('date',axis = 1)\
                 .groupby(['prod_id'],as_index = False).agg({'review_text':'max'}).rename(columns = {'review_text':'prod_max_no_review_per_day'})

df = pd.merge(df,prod_max_no_reviews, how = 'inner', on = 'prod_id')

# Average rating
df['prod_avg_rating'] = df.groupby('prod_id')['rating'].transform('mean')

# Rating deviation
df['prod_rating_std'] = df.groupby('prod_id')['rating'].transform('std')

# Average number of words
df['prod_avg_no_words'] = df.groupby('prod_id')['word_count'].transform('mean')

# Whether restaurant has received exact same review text twice
duplicating_prods = intragroup(df['prod_id'].unique(), df['prod_id'], df['review_text'])
df['prod_has_dup_text'] = df['prod_id'].apply(lambda x: x in duplicating_prods).astype(int)
# Plot 4: User Based Features ——Have fake reviews/ no fake reviews
user_fake = df[df.label == 1]['user_id'].drop_duplicates().tolist()
df_user = df[['user_id','user_no_of_review','user_max_no_review_per_day','user_avg_rating','user_avg_no_words']].drop_duplicates()
df_user_fake = df_user[df_user.user_id.isin(user_fake)].drop(['user_id'] ,axis = 1)
df_user_genuine = df_user.drop(df_user_fake.index).drop(['user_id'], axis = 1)
import matplotlib.pyplot as plt
import seaborn as sns

def displot_user(df,color,title):
    plt.style.use('ggplot')
    #sns.set_style('darkgrid', {'figure.facecolor': 'black'})
    fig,axes=plt.subplots(1,4, figsize=(24, 3), dpi = 130)
    fig.suptitle(title,fontsize=20)
    sns.distplot(df.user_no_of_review, kde=True, bins=10,ax=axes[0],color=color)
    #axes[0].tick_params(colors='white') 
    axes[0].set_xlabel('Total number of reviews',fontsize=16, color = 'white')
    sns.distplot(df.user_max_no_review_per_day,kde= True, bins=10,ax=axes[1],color=color)
    #axes[1].tick_params(colors='white') 
    axes[1].set_xlabel('Max number of reviews per day',fontsize=16, color = 'white')
    sns.distplot(df.user_avg_rating,kde=True,bins=10,ax=axes[2], color=color)
    #axes[2].tick_params(colors='white') 
    axes[2].set_xlabel('Average rating',fontsize = 16, color = 'white')
    sns.distplot(df.user_avg_no_words,kde=True,bins=10,ax=axes[3], color=color)
    #axes[3].tick_params(colors='white') 
    axes[3].set_xlabel('Avearage number of words', fontsize = 16, color = 'white')
    plt.show()

displot_user(df_user_genuine, "FireBrick", 'Users whose reviews all detected as genuine') 
displot_user(df_user_fake, "lightsalmon", 'Users having reviews detected as fake')       
# Plot 5: Restaurant Based Features ——Have fake reviews/ no fake reviews
prod_fake = df[df.label == 1]['prod_id'].drop_duplicates().tolist()
df_prod = df[['prod_id','prod_no_of_review','prod_max_no_review_per_day','prod_avg_rating','prod_avg_no_words']].drop_duplicates()
df_prod_fake = df_prod[df_prod.prod_id.isin(prod_fake)].drop_duplicates().drop(['prod_id'], axis = 1)
df_prod_genuine = df_prod.drop(df_prod_fake.index).drop_duplicates().drop(['prod_id'], axis = 1)
def displot_prod(df,color,title):
    plt.style.use('ggplot')
    #sns.set_style('darkgrid', {'figure.facecolor': 'black'})
    fig,axes=plt.subplots(1,4, figsize=(24, 3), dpi = 130)
    fig.suptitle(title, fontsize = 20)
    sns.distplot(df.prod_no_of_review, kde=True, bins=10,ax=axes[0],color=color)
    #axes[0].tick_params(colors='white') 
    axes[0].set_xlabel('Total number of reviews',fontsize = 16, color = 'white')
    sns.distplot(df.prod_max_no_review_per_day,kde= True, bins=10,ax=axes[1],color=color)
    #axes[1].tick_params(colors='white') 
    axes[1].set_xlabel('Max number of reviews received per day',fontsize = 16, color = 'white')
    sns.distplot(df.prod_avg_rating,kde=True,bins=10,ax=axes[2], color=color)
    #axes[2].tick_params(colors='white') 
    axes[2].set_xlabel('Average rating',fontsize = 16, color = 'white')
    sns.distplot(df.prod_avg_no_words,kde=True,bins=10,ax=axes[3],  color=color)
    #axes[3].tick_params(colors='white') 
    axes[3].set_xlabel('Avearage number of words',fontsize = 16, color = 'white')
    plt.show()

displot_prod(df_prod_genuine, "FireBrick", 'Restaurants whose reviews all detected as genuine')  
displot_prod(df_prod_fake, "lightsalmon", 'Restaurants having reviews detected as fake')     
# Drop useless columns
df_model = df.drop(['index','user_id','prod_id','prod_name','review_text','review_text_token','date', 'day_of_week_name'], axis=1)
df_model.fillna(value=0, inplace=True)
df_model.info()
df_model[df_model.label == 1].shape
# Train-test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_model, test_size = 0.2, random_state = 42)
train[train.label == 1].shape
test.tail()
train_0 = train[train.label == 0]
train_1 = train[train.label == 1]
train_0_sample = train_0.sample(n = train_1.shape[0], random_state=42) 
train_1_sample = train_1.sample(n = train_1.shape[0], random_state=42) 
train_sample = pd.concat([train_0_sample, train_1_sample])
print(train_sample[train_sample.label == 0].shape)
print(train_sample[train_sample.label == 1].shape)
y_train_RUS = train_sample['label']
X_train_RUS = train_sample.drop(['label'], axis = 1)
print(y_train_RUS.shape)
print(X_train_RUS.shape)
y_test = test['label']
X_test = test.drop(['label'], axis = 1)
# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_RUS) # train set: fit_transform
X_test_scaled = scaler.transform(X_test) # test set: transform
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

tuned_parameters_svm = [{'C': [0.01,0.1,1,10,100]}] 

# Find optimal parameters by GridSearchCV 
svm_clf = SVC(kernel = 'rbf', probability=True, max_iter=2000, random_state = 42)
svm_gs = GridSearchCV(svm_clf, tuned_parameters_svm,scoring = 'f1', cv=5)

svm_gs.fit(X_train_scaled, y_train_RUS)
print(svm_gs.best_estimator_)
best_svm = svm_gs.best_estimator_
best_svm.fit(X_train_scaled, y_train_RUS)
pred_svm = best_svm.predict(X_test_scaled)
# Accuracy score
from sklearn.metrics import accuracy_score, f1_score
accuracy_svm = accuracy_score(pred_svm, y_test)
print("Accuracy of SVM is ", accuracy_svm)
# F1-score
f1_svm = f1_score(pred_svm, y_test)
print("F1-score of SVM is ", f1_svm)
# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_svm))
# AUC-ROC plot
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
# calculate the fpr and tpr for all thresholds of the classification

def AUC_ROC_plot(model,title):
    probs = model.predict_proba(X_test_scaled) # need to change depending on using pca or scaled
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.style.use('seaborn')
    plt.figure(figsize = (6,5), dpi = 70)
    plt.title('Receiver Operating Characteristic—'+title, y = 1.06)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

AUC_ROC_plot(best_svm, 'SVM')
# Unnormalized confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

cm_svm = confusion_matrix(y_test, pred_svm)
print("Absolute confusion matrix is\n",cm_svm)
# Normalized confusion matrix
def plot_confusion_matrix(y_test, pred, classes, model_name, normalize=False,title=None,cmap=plt.cm.OrRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix —' + str(model_name)
        else:
            title = 'Confusion matrix, without normalization —' + str(model_name)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize = (6,6), dpi = 70)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.title(title, y = 1.06)
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_test, pred_svm, classes=["0","1"], model_name = 'SVM', normalize=True)
from sklearn.linear_model import LogisticRegression

tuned_parameters_lr = [{'C': [0.001,0.01,0.1,1,10,100]}]

# Find optimal parameters by GridSearchCV 
lr_clf = LogisticRegression(random_state = 42)
lr_gs = GridSearchCV(lr_clf, tuned_parameters_lr, scoring = 'f1', cv=5)

lr_gs.fit(X_train_scaled, y_train_RUS)
print(lr_gs.best_estimator_)
best_lr = lr_gs.best_estimator_
best_lr.fit(X_train_scaled, y_train_RUS)
pred_lr = best_lr.predict(X_test_scaled)
# Accuracy score
accuracy_lr = accuracy_score(pred_lr, y_test)
print("Accuracy of Logistic Regression is ", accuracy_lr)
# F1-score
f1_lr = f1_score(pred_lr, y_test)
print("F1-score of Logistic Regression is ", f1_lr)
# Classification report
print(classification_report(y_test, pred_lr))
# AUC-ROC plot
AUC_ROC_plot(best_lr, 'Logistics Regression')
# Unnormalized confusion matrix
cm_lr = confusion_matrix(y_test, pred_lr)
print("Absolute confusion matrix is\n",cm_lr)
# Normalized confusion matrix
plot_confusion_matrix(y_test, pred_lr, classes=["0","1"], model_name = 'Logistics Regression', normalize=True)
from sklearn.ensemble import RandomForestClassifier

tuned_parameters_rf = [{'n_estimators': range(94,97), 
                         'max_depth':range(2,8),
                         'min_samples_leaf': range(3,6)}]


# Find optimal parameters by GridSearchCV 
rf_clf = RandomForestClassifier(oob_score = True, random_state = 42)
rf_gs = GridSearchCV(rf_clf, tuned_parameters_rf, scoring = 'f1', cv=5)

rf_gs.fit(X_train_scaled, y_train_RUS)
print(rf_gs.best_estimator_)
best_rf = rf_gs.best_estimator_
best_rf.fit(X_train_scaled, y_train_RUS)
pred_rf = best_rf.predict(X_test_scaled)
# Accuracy score
accuracy_rf = accuracy_score(pred_rf, y_test)
print("Accuracy of Random Forest is ", accuracy_rf)
# F1 score
f1_rf = f1_score(pred_rf, y_test)
print("F1-score of Random Forest is ", f1_rf)
# Classification report
print(classification_report(y_test, pred_rf))
# AUC-ROC plot
AUC_ROC_plot(best_rf, 'Random Forest')
# Unnormalized confusion matrix
cm_rf = confusion_matrix(y_test, pred_rf)
print("Absolute confusion matrix is\n",cm_rf)
# Normalized confusion matrix
plot_confusion_matrix(y_test, pred_rf, classes=["0","1"], model_name = 'Random Forest', normalize=True)
# Save predicted results
results_rf = pd.DataFrame({'index':X_test.index.values,'label':pred_rf})
results_rf.to_csv('part_1_result_rf.csv', index = False)
import lightgbm as lgb

tuned_parameters_lgb = [{'learning_rate': [0.001, 0.01, 0.1],'n_estimators': range(100,121)}]

# Find optimal parameters by GridSearch 
lgb_clf = lgb.LGBMClassifier(boosting_type= 'gbdt',objective = 'binary',silent = True, random_state = 42)
lgb_gs = GridSearchCV(lgb_clf, tuned_parameters_lgb, scoring = 'f1', cv=5)

lgb_gs.fit(X_train_scaled, y_train_RUS)
print(lgb_gs.best_estimator_)
best_lgb = lgb_gs.best_estimator_
best_lgb.fit(X_train_scaled, y_train_RUS)
pred_lgb = best_lgb.predict(X_test_scaled)
# Accuracy score
accuracy_lgb = accuracy_score(pred_lgb, y_test)
print("Accuracy of LightGBM is ", accuracy_lgb)
# F1 score
f1_lgb = f1_score(pred_lgb, y_test)
print("F1-score of LightGBM is ", f1_lgb)
# Classification report
print(classification_report(y_test, pred_lgb))
# AUC-ROC plot
AUC_ROC_plot(best_lgb, 'LightGBM')
# Unnormalized confusion matrix
cm_lgb = confusion_matrix(y_test, pred_lgb)
print("Absolute confusion matrix is\n",cm_lgb)
# Normalized confusion matrix
plot_confusion_matrix(y_test, pred_lgb, classes=["0","1"], model_name = 'LightGBM', normalize=True)
# Feature importance of the best model
importances_lgb = best_lgb.feature_importances_
feature_importances_lgb = pd.DataFrame({'feature':X_df.columns,'importance':importances_lgb},index=None)
importances_lgb =feature_importances_lgb.sort_values(by='importance', ascending=True).head(10)

plt.style.use('ggplot')
plt.figure(figsize = (6,4),dpi = 80)
plt.barh(importances_lgb['feature'], importances_lgb['importance'], label='Importance',facecolor = 'FireBrick', edgecolor = 'white')
plt.title('LightGBM: Feature Importance')
plt.show()
# Save predicted results
results_lgb = pd.DataFrame({'index':X_test.index.values,'label':pred_lgb})
results_lgb.to_csv('part_1_result_lgb.csv', index = False)
import xgboost as xgb
tuned_parameters_xgb = [{'n_estimators':range(96,101), 
                         'learning_rate': [0.01,0.001,0.1],
                         'max_depth':range(2,5)}]

# Find optimal parameters by GridSearchCV 
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', booster = 'gbtree', 
             colsample_bytree = 0.8, silent = True, random_state = 42)
xgb_gs = GridSearchCV(xgb_clf, tuned_parameters_xgb, scoring = 'f1', cv=5)

xgb_gs.fit(X_train_scaled, y_train_RUS)
print(xgb_gs.best_estimator_)
best_xgb = xgb_gs.best_estimator_
best_xgb.fit(X_train_scaled, y_train_RUS)
pred_xgb = best_xgb.predict(X_test_scaled)
# Accuracy score
accuracy_xgb = accuracy_score(pred_xgb, y_test)
print("Accuracy of XGBoost is ", accuracy_xgb)
# F1 score
f1_xgb = f1_score(pred_xgb, y_test)
print("F1-score of XGBoost is ", f1_xgb)
# Classification report
print(classification_report(y_test, pred_xgb))
# AUC-ROC plot
AUC_ROC_plot(best_xgb, 'XGBoost')
# Unnormalized confusion matrix
cm_xgb = confusion_matrix(y_test, pred_xgb)
print("Absolute confusion matrix is\n",cm_xgb)
# Normalized confusion matrix
plot_confusion_matrix(y_test, pred_xgb, classes=["0","1"], model_name = 'XGBoost', normalize=True)
# Save predicted results
results_xgb = pd.DataFrame({'index':X_test.index.values,'label':pred_xgb})
results_xgb.to_csv('part_1_result_xgb.csv', index = False)
Y_df = df_model['label']
X_df = df_model.drop(['label'], axis=1).astype(float)
X_df.fillna(value=0, inplace=True) 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordTokenizer

vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, ngram_range=(1,1), tokenizer=TreebankWordTokenizer().tokenize) # 1 word best
# vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=0.005, ngram_range=(6,6), analyzer='char_wb') # char not as good
textvec = vec.fit_transform(df['review_text']).asfptype()
# Combine text based features and non-text based features
from scipy import sparse
nummatrix = sparse.coo_matrix(X_df)
combinedmatrix = sparse.hstack([nummatrix, textvec]).tocsr()

sparse.save_npz('combinedmatrix.npz', combinedmatrix)
# combinedmatrix = sparse.load_npz('combinedmatrix.npz')        # load previously combined matrix
# lgbst = lgb.Booster(model_file='dart_auc_001_80_07_08_4890')  # load previously saved model

X_train, X_test, y_train, y_test = train_test_split(combinedmatrix, Y_df, test_size = 0.2, random_state = 42)
# Undersampling for sparse matrix: sample by indices instead, subscript matrix later
from random import sample

positivelist = np.flatnonzero(y_train)
negativelist = list(set([x for x in range(len(y_train))])-set(positivelist))
negativesample = sample(negativelist, len(positivelist))
combinedlist = list(set(positivelist).union(set(negativesample)))

X_train = X_train[combinedlist]
y_train = y_train.iloc[combinedlist]

xmain, xval, ymain, yval = train_test_split(X_train, y_train, test_size=0.2)

lgbmain = lgb.Dataset(xmain, label=ymain)
lgbval = lgb.Dataset(xval, label=yval, reference=lgbmain)
# Parameter tuning
lgparam = {'objective': 'binary',
           'boosting': 'dart',
           'num_threads': 2,
           'metric': 'auc',
           'learning_rate' : 0.01,
           'num_leaves' : 80,
           'feature_fraction': 0.7,
           'bagging_fraction': 0.8
           }

lgbst = lgb.train(lgparam, lgbmain, 4880, valid_sets=[lgbval], early_stopping_rounds=50, verbose_eval=10)

lgpred = lgbst.predict(X_test)
# After maxing AUC, observe how F1 and accuracy change with classification boundary
for thres in [0.6, 0.65, 0.7]:
    catpred = (lgpred > thres).astype(int)
    print(thres)
    print(f1_score(y_test, catpred))
    print(accuracy_score(y_test, catpred))

print(classification_report(y_test, catpred))