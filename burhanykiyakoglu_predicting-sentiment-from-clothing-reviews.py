import pandas as pd

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_curve, auc

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

import sklearn.metrics as mt

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

 

df1 = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')

df = df1[['Review Text','Rating','Class Name','Age']]

#df1.info()

#df1.describe()

df1.head()
# fill NA values by space

df['Review Text'] = df['Review Text'].fillna('')



# CountVectorizer() converts a collection 

# of text documents to a matrix of token counts

vectorizer = CountVectorizer()

# assign a shorter name for the analyze

# which tokenizes the string

analyzer = vectorizer.build_analyzer()



def wordcounts(s):

    c = {}

    # tokenize the string and continue, if it is not empty

    if analyzer(s):

        d = {}

        # find counts of the vocabularies and transform to array 

        w = vectorizer.fit_transform([s]).toarray()

        # vocabulary and index (index of w)

        vc = vectorizer.vocabulary_

        # items() transforms the dictionary's (word, index) tuple pairs

        for k,v in vc.items():

            d[v]=k # d -> index:word 

        for index,i in enumerate(w[0]):

            c[d[index]] = i # c -> word:count

    return  c



# add new column to the dataframe

df['Word Counts'] = df['Review Text'].apply(wordcounts)

df.head()
# selecting some words to examine detailed 

selectedwords = ['awesome','great','fantastic','extraordinary','amazing','super',

                 'magnificent','stunning','impressive','wonderful','breathtaking',

                 'love','content','pleased','happy','glad','satisfied','lucky',

                 'shocking','cheerful','wow','sad','unhappy','horrible','regret',

                 'bad','terrible','annoyed','disappointed','upset','awful','hate']



def selectedcount(dic,word):

    if word in dic:

        return dic[word]

    else:

        return 0

    

dfwc = df.copy()  

for word in selectedwords:

    dfwc[word] = dfwc['Word Counts'].apply(selectedcount,args=(word,))

    

word_sum = dfwc[selectedwords].sum()

print('Selected Words')

print(word_sum.sort_values(ascending=False).iloc[:5])



print('\nClass Names')

print(df['Class Name'].fillna("Empty").value_counts().iloc[:5])



fig, ax = plt.subplots(1,2,figsize=(20,10))

wc0 = WordCloud(background_color='white',

                      width=450,

                      height=400 ).generate_from_frequencies(word_sum)



cn = df['Class Name'].fillna(" ").value_counts()

wc1 = WordCloud(background_color='white',

                      width=450,

                      height=400 

                     ).generate_from_frequencies(cn)



ax[0].imshow(wc0)

ax[0].set_title('Selected Words\n',size=25)

ax[0].axis('off')



ax[1].imshow(wc1)

ax[1].set_title('Class Names\n',size=25)

ax[1].axis('off')



rt = df['Review Text']

plt.subplots(figsize=(18,6))

wordcloud = WordCloud(background_color='white',

                      width=900,

                      height=300

                     ).generate(" ".join(rt))

plt.imshow(wordcloud)

plt.title('All Words in the Reviews\n',size=25)

plt.axis('off')

plt.show()
df1=df['Rating'].value_counts().to_frame()

avgdf1 = df.groupby('Class Name').agg({'Rating': np.average})

avgdf2 = df.groupby('Class Name').agg({'Age': np.average})

avgdf3 = df.groupby('Rating').agg({'Age': np.average})



trace1 = go.Bar(

    x=avgdf1.index,

    y=round(avgdf1['Rating'],2),

    marker=dict(

        color=avgdf1['Rating'],

        colorscale = 'RdBu')

)



trace2 = go.Bar(

    x=df1.index,

    y=df1.Rating,

    marker=dict(

        color=df1['Rating'],

        colorscale = 'RdBu')

)



trace3 = go.Bar(

    x=avgdf2.index,

    y=round(avgdf2['Age'],2),

    marker=dict(

        color=avgdf2['Age'],

        colorscale = 'RdBu')

)



trace4 = go.Bar(

    x=avgdf3.index,

    y=round(avgdf3['Age'],2),

    marker=dict(

        color=avgdf3['Age'],

        colorscale = 'Reds')

)



fig = tools.make_subplots(rows=2, cols=2, print_grid=False)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 2, 2)



fig['layout']['xaxis1'].update(title='Class')

fig['layout']['yaxis1'].update(title='Average Rating')

fig['layout']['xaxis2'].update(title='Rating')

fig['layout']['yaxis2'].update(title='Count')

fig['layout']['xaxis3'].update(title='Class')

fig['layout']['yaxis3'].update(title='Average Age of the Reviewers')

fig['layout']['xaxis4'].update(title='Rating')

fig['layout']['yaxis4'].update(title='Average Age of the Reviewers')



fig['layout'].update(height=800, width=900,showlegend=False)

fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)',

                   'paper_bgcolor':'rgba(0,0,0,0)'})

#fig['layout'].update(plot_bgcolor='rgba(0,0,0,0)')

#fig['layout'].update(paper_bgcolor='rgba(0,0,0,0)')

py.iplot(fig)
cv = df['Class Name'].value_counts()



trace = go.Scatter3d( x = avgdf1.index,

                      y = avgdf1['Rating'],

                      z = cv[avgdf1.index],

                      mode = 'markers',

                      marker = dict(size=10,color=avgdf1['Rating']),

                      hoverinfo ="text",

                      text="Class: "+avgdf1.index+" \ Average Rating: "+avgdf1['Rating'].map(' {:,.2f}'.format).apply(str)+" \ Number of Reviewers: "+cv[avgdf1.index].apply(str)

                      )



data = [trace]

layout = go.Layout(title="Average Rating & Class & Number of Reviewers",

                   scene = dict(

                    xaxis = dict(title='Class'),

                    yaxis = dict(title='Average Rating'),

                    zaxis = dict(title='Number of Sales'),),

                   margin = dict(l=30, r=30, b=30, t=30))

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)

plt.savefig('3D_Scatter.png')
# Rating of 4 or higher -> positive, while the ones with 

# Rating of 2 or lower -> negative 

# Rating of 3 -> neutral

df = df[df['Rating'] != 3]

df['Sentiment'] = df['Rating'] >=4

df.head()



# split data

train_data,test_data = train_test_split(df,train_size=0.8,random_state=0)

# select the columns and 

# prepare data for the models 

X_train = vectorizer.fit_transform(train_data['Review Text'])

y_train = train_data['Sentiment']

X_test = vectorizer.transform(test_data['Review Text'])

y_test = test_data['Sentiment']
start=dt.datetime.now()

lr = LogisticRegression()

lr.fit(X_train,y_train)

print('Elapsed time: ',str(dt.datetime.now()-start))
start=dt.datetime.now()

nb = MultinomialNB()

nb.fit(X_train,y_train)

print('Elapsed time: ',str(dt.datetime.now()-start))
start=dt.datetime.now()

svm = SVC()

svm.fit(X_train,y_train)

print('Elapsed time: ',str(dt.datetime.now()-start))
start=dt.datetime.now()

nn = MLPClassifier()

nn.fit(X_train,y_train)

print('Elapsed time: ',str(dt.datetime.now()-start))
# define a dataframe for the prediction probablities of the models

#df1 = train_data.copy()

#df1['Logistic Regression'] = lr.predict_proba(X_train)[:,1]

#df1['Naive Bayes'] = nb.predict_proba(X_train)[:,1]

#df1['SVM'] = svm.decision_function(X_train)

#df1['Neural Network'] = nn.predict_proba(X_train)[:,1]

#df1=df1.round(2)

#df1.head()



# define a dataframe for the predictions

df2 = train_data.copy()

df2['Logistic Regression'] = lr.predict(X_train)

df2['Naive Bayes'] = nb.predict(X_train)

df2['SVM'] = svm.predict(X_train)

df2['Neural Network'] = nn.predict(X_train)

df2.head()
pred_lr = lr.predict_proba(X_test)[:,1]

fpr_lr,tpr_lr,_ = roc_curve(y_test,pred_lr)

roc_auc_lr = auc(fpr_lr,tpr_lr)



pred_nb = nb.predict_proba(X_test)[:,1]

fpr_nb,tpr_nb,_ = roc_curve(y_test.values,pred_nb)

roc_auc_nb = auc(fpr_nb,tpr_nb)



pred_svm = svm.decision_function(X_test)

fpr_svm,tpr_svm,_ = roc_curve(y_test.values,pred_svm)

roc_auc_svm = auc(fpr_svm,tpr_svm)



pred_nn = nn.predict_proba(X_test)[:,1]

fpr_nn,tpr_nn,_ = roc_curve(y_test.values,pred_nn)

roc_auc_nn = auc(fpr_nn,tpr_nn)



f, axes = plt.subplots(2, 2,figsize=(15,10))

axes[0,0].plot(fpr_lr, tpr_lr, color='darkred', lw=2, label='ROC curve (area = {:0.2f})'.format(roc_auc_lr))

axes[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

axes[0,0].set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05])

axes[0,0].set(xlabel ='False Positive Rate', ylabel = 'True Positive Rate', title = 'Logistic Regression')

axes[0,0].legend(loc='lower right', fontsize=13)



axes[0,1].plot(fpr_nb, tpr_nb, color='darkred', lw=2, label='ROC curve (area = {:0.2f})'.format(roc_auc_nb))

axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

axes[0,1].set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05])

axes[0,1].set(xlabel ='False Positive Rate', ylabel = 'True Positive Rate', title = 'Naive Bayes')

axes[0,1].legend(loc='lower right', fontsize=13)



axes[1,0].plot(fpr_svm, tpr_svm, color='darkred', lw=2, label='ROC curve (area = {:0.2f})'.format(roc_auc_svm))

axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

axes[1,0].set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05])

axes[1,0].set(xlabel ='False Positive Rate', ylabel = 'True Positive Rate', title = 'Support Vector Machine')

axes[1,0].legend(loc='lower right', fontsize=13)



axes[1,1].plot(fpr_nn, tpr_nn, color='darkred', lw=2, label='ROC curve (area = {:0.2f})'.format(roc_auc_nn))

axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

axes[1,1].set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05])

axes[1,1].set(xlabel ='False Positive Rate', ylabel = 'True Positive Rate', title = 'Neural Network')

axes[1,1].legend(loc='lower right', fontsize=13);
# preparation for the confusion matrix

lr_cm=confusion_matrix(y_test.values, lr.predict(X_test))

nb_cm=confusion_matrix(y_test.values, nb.predict(X_test))

svm_cm=confusion_matrix(y_test.values, svm.predict(X_test))

nn_cm=confusion_matrix(y_test.values, nn.predict(X_test))



plt.figure(figsize=(15,12))

plt.suptitle("Confusion Matrices",fontsize=24)



plt.subplot(2,2,1)

plt.title("Logistic Regression")

sns.heatmap(lr_cm, annot = True, cmap="Greens",cbar=False);



plt.subplot(2,2,2)

plt.title("Naive Bayes")

sns.heatmap(nb_cm, annot = True, cmap="Greens",cbar=False);



plt.subplot(2,2,3)

plt.title("Support Vector Machine (SVM)")

sns.heatmap(svm_cm, annot = True, cmap="Greens",cbar=False);



plt.subplot(2,2,4)

plt.title("Neural Network")

sns.heatmap(nn_cm, annot = True, cmap="Greens",cbar=False);
print("Logistic Regression")

print(mt.classification_report(y_test, lr.predict(X_test)))

print("\n Naive Bayes")

print(mt.classification_report(y_test, nb.predict(X_test)))

print("\n Support Vector Machine (SVM)")

print(mt.classification_report(y_test, svm.predict(X_test)))

print("\n Neural Network")

print(mt.classification_report(y_test, nn.predict(X_test)))