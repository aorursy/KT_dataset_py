import pandas as pd
from IPython.core.display import HTML

path = "../input/"

versions = pd.read_csv(path+"KernelVersions.csv")
kernels = pd.read_csv(path+"Kernels.csv")
users = pd.read_csv(path+"Users.csv")

language_map = {'1' : 'R','5' : 'R', '12' : 'R', '13' : 'R', '15' : 'R', '16' : 'R',
                '2' : 'Python','8' : 'Python', '9' : 'Python', '14' : 'Python'}

def pressence_check(title, tokens, ignore = []):
    present = False
    for token in tokens:
        words = token.split()
        if all(wrd.lower().strip() in title.lower() for wrd in words):
            present = True
    for token in ignore:
        if token in title.lower():
            present = False
    return present 

## check if the latest version of the kernel is about the same topic 
def get_latest(idd):
    latest = versions[versions['KernelId'] == idd].sort_values('VersionNumber', ascending = False).iloc(0)[0]
    return latest['VersionNumber']

def get_kernels(tokens, n, ignore = []):
    versions['isRel'] = versions['Title'].apply(lambda x : pressence_check(x, tokens, ignore))
    relevant = versions[versions['isRel'] == 1]
    results = relevant.groupby('KernelId').agg({'TotalVotes' : 'sum', 
                                                'KernelLanguageId' : 'max', 
                                                'Title' : lambda x : "#".join(x).split("#")[-1],
                                                'VersionNumber' : 'max'})
    results = results.reset_index().sort_values('TotalVotes', ascending = False).head(n)
    results = results.rename(columns={'KernelId' : 'Id', 'TotalVotes': 'Votes'})


    results['latest_version']  = results['Id'].apply(lambda x : get_latest(x))
    results['isLatest'] = results.apply(lambda r : 1 if r['VersionNumber'] == r['latest_version'] else 0, axis=1)
    results = results[results['isLatest'] == 1]

    results = results.merge(kernels, on="Id").sort_values('TotalVotes', ascending = False)
    results = results.merge(users.rename(columns={'Id':"AuthorUserId"}), on='AuthorUserId')
    results['Language'] = results['KernelLanguageId'].apply(lambda x : language_map[str(x)] if str(x) in language_map else "")
    results = results.sort_values("TotalVotes", ascending = False)
    return results[['Title', 'CurrentUrlSlug','Language' ,'TotalViews', 'TotalComments', 'TotalVotes', "DisplayName","UserName"]]


def best_kernels(tokens, n = 10, ignore = []):
    response = get_kernels(tokens, n, ignore)     
    hs = """<style>
                .rendered_html tr {font-size: 12px; text-align: left}
            </style>
            <h3><font color="#1768ea">"""+tokens[0].title()+"""</font></h3>
            <table>
            <th>
                <td><b>Kernel</b></td>
                <td><b>Author</b></td>
                <td><b>Language</b></td>
                <td><b>Views</b></td>
                <td><b>Comments</b></td>
                <td><b>Votes</b></td>
            </th>"""
    for i, row in response.iterrows():
        url = "https://www.kaggle.com/"+row['UserName']+"/"+row['CurrentUrlSlug']
        aurl= "https://www.kaggle.com/"+row['UserName']
        hs += """<tr>
                    <td>"""+str(i+1)+"""</td>
                    <td><a href="""+url+""" target="_blank"><b>"""  + row['Title'] + """</b></a></td>
                    <td><a href="""+aurl+""" target="_blank">"""  + row['DisplayName'] + """</a></td>
                    <td>"""+str(row['Language'])+"""</td>
                    <td>"""+str(row['TotalViews'])+"""</td>
                    <td>"""+str(row['TotalComments'])+"""</td>
                    <td>"""+str(row['TotalVotes'])+"""</td>
                    </tr>"""
    hs += "</table>"
    display(HTML(hs))
tokens = ["linear regression"]
best_kernels(tokens, 10)
tokens = ['logistic regression', "logistic"]
best_kernels(tokens, 10)
tokens = ['Ridge']
best_kernels(tokens, 10)
tokens = ['Lasso']
best_kernels(tokens, 10)
tokens = ['ElasticNet']
best_kernels(tokens, 4)
tokens = ['Decision Tree']
best_kernels(tokens, 10)
tokens = ['random forest']
best_kernels(tokens, 10)
tokens = ['lightgbm', 'light gbm', 'lgb']
best_kernels(tokens, 10)
tokens = ['xgboost', 'xgb']
best_kernels(tokens, 10)
tokens = ['catboost']
best_kernels(tokens, 10)
tokens = ['neural network']
best_kernels(tokens, 10)
tokens = ['autoencoder']
best_kernels(tokens, 10)
tokens = ['deep learning']
best_kernels(tokens, 10)
tokens = ['convolutional neural networks', 'cnn']
best_kernels(tokens, 10)
tokens = ['lstm']
best_kernels(tokens, 10)
tokens = ['gru']
ignore = ['grupo']
best_kernels(tokens, 10, ignore)
tokens = ['mxnet']
best_kernels(tokens, 10)
tokens = ['resnet']
best_kernels(tokens, 10)
tokens = ['Capsule network', 'capsulenet']
best_kernels(tokens, 5)
tokens = ['vgg']
best_kernels(tokens, 5)
tokens = ['inception']
best_kernels(tokens, 5)
tokens = ['computer vision']
best_kernels(tokens, 5)
tokens = ['transfer learning']
best_kernels(tokens, 5)
tokens = ['kmeans', 'k means']
best_kernels(tokens, 10)
tokens = ['hierarchical clustering']
best_kernels(tokens, 3)
tokens = ['dbscan']
best_kernels(tokens, 10)
tokens = ['unsupervised']
best_kernels(tokens, 10)
tokens = ['naive bayes']
best_kernels(tokens, 10)
tokens = ['svm']
best_kernels(tokens, 10)
tokens = ['knn']
best_kernels(tokens, 10)
tokens = ['recommendation engine']
best_kernels(tokens, 5)
tokens = ['EDA', 'exploration']
best_kernels(tokens, 10)
tokens = ['feature engineering']
best_kernels(tokens, 10)
tokens = ['feature selection']
best_kernels(tokens, 10)
tokens = ['outlier treatment', 'outlier']
best_kernels(tokens, 10)
tokens = ['anomaly detection', 'anomaly']
best_kernels(tokens, 8)
tokens = ['smote']
best_kernels(tokens, 5)
tokens = ['pipeline']
best_kernels(tokens, 10)
tokens = ['dataset decomposition', 'dimentionality reduction']
best_kernels(tokens, 2)
tokens = ['PCA']
best_kernels(tokens, 10)
tokens = ['Tsne', 't-sne']
best_kernels(tokens, 10)
tokens = ['cross validation']
best_kernels(tokens, 10)
tokens = ['model selection']
best_kernels(tokens, 10)
tokens = ['model tuning', 'tuning']
best_kernels(tokens, 10)
tokens = ['gridsearch', 'grid search']
best_kernels(tokens, 10)
tokens = ['ensemble']
best_kernels(tokens, 10)
tokens = ['stacking', 'stack']
best_kernels(tokens, 10)
tokens = ['bagging']
best_kernels(tokens, 10)
tokens = ['NLP', 'Natural Language Processing', 'text mining']
best_kernels(tokens, 10)
tokens = ['topic modelling']
best_kernels(tokens, 8)
tokens = ['word embedding','fasttext', 'glove', 'word2vec']
best_kernels(tokens, 8)
tokens = ['scikit']
best_kernels(tokens, 10)
tokens = ['tensorflow', 'tensor flow']
best_kernels(tokens, 10)
tokens = ['theano']
best_kernels(tokens, 10)
tokens = ['keras']
best_kernels(tokens, 10)
tokens = ['pytorch']
best_kernels(tokens, 10)
tokens = ['vowpal wabbit','vowpalwabbit']
best_kernels(tokens, 10)
tokens = ['eli5']
best_kernels(tokens, 10)
tokens = ['hyperopt']
best_kernels(tokens, 5)
tokens = ['pandas']
best_kernels(tokens, 10)
tokens = ['SQL']
best_kernels(tokens, 10)
tokens = ['bigquery', 'big query']
best_kernels(tokens, 10)
tokens = ['visualization', 'visualisation']
best_kernels(tokens, 10)
tokens = ['plotly', 'plot.ly']
best_kernels(tokens, 10)
tokens = ['seaborn']
best_kernels(tokens, 10)
tokens = ['d3.js']
best_kernels(tokens, 4)
tokens = ['bokeh']
best_kernels(tokens, 10)

