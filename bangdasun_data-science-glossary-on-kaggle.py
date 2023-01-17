import pandas as pd
from IPython.core.display import HTML

path = "../input/"

versions = pd.read_csv(path+"KernelVersions.csv")
kernels = pd.read_csv(path+"Kernels.csv")
users = pd.read_csv(path+"Users.csv")

def pressence_check(title, tokens):
    present = False
    for token in tokens:
        words = token.split()
        if all(wrd.lower().strip() in title.lower() for wrd in words):
            present = True
    return present 
    
def get_kernels(tokens, n):
    versions['isRel'] = versions['Title'].apply(lambda x : pressence_check(x, tokens))
    relevant = versions[versions['isRel'] == 1]
    relevant = relevant.groupby('KernelId').agg({'TotalVotes' : 'sum', 'Title' : lambda x : "#".join(x).split("#")[0]})
    results = relevant.reset_index().sort_values('TotalVotes', ascending = False).head(n)
    results = results.rename(columns={'KernelId' : 'Id', 'TotalVotes': 'Votes'})
    results = results.merge(kernels, on="Id").sort_values('TotalVotes', ascending = False)
    results = results.merge(users.rename(columns={'Id':"AuthorUserId"}), on='AuthorUserId')
    return results[['Title', 'CurrentUrlSlug', 'TotalViews', 'TotalComments', 'TotalVotes', "DisplayName","UserName"]]


def best_kernels(tokens, n = 10):
    response = get_kernels(tokens, n)     
    hs = """<style>
                .rendered_html tr {font-size: 12px; text-align: left}
            </style>
            <h3><font color="#1768ea">"""+tokens[0].title()+"""</font></h3>
            <table>
            <th>
                <td><b>Kernel Title</b></td>
                <td><b>Author</b></td>
                <td><b>Total Views</b></td>
                <td><b>Total Comments</b></td>
                <td><b>Total Votes</b></td>
            </th>"""
    for i, row in response.iterrows():
        url = "https://www.kaggle.com/"+row['UserName']+"/"+row['CurrentUrlSlug']
        aurl= "https://www.kaggle.com/"+row['UserName']
        hs += """<tr>
                    <td>"""+str(i+1)+"""</td>
                    <td><a href="""+url+""" target="_blank"><b>"""  + row['Title'] + """</b></a></td>
                    <td><a href="""+aurl+""" target="_blank">"""  + row['DisplayName'] + """</a></td>
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
tokens = ['Stepwise regression']
best_kernels(tokens, 10)
tokens = ['polynomial regression']
best_kernels(tokens, 5)
tokens = ['multivariate regression']
best_kernels(tokens, 5)
tokens = ['Ridge']
best_kernels(tokens, 10)
tokens = ['Lasso']
best_kernels(tokens, 10)
tokens = ['ElasticNet']
best_kernels(tokens, 10)
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
tokens = ['adaboost']
best_kernels(tokens, 5)
tokens = ['neural network']
best_kernels(tokens, 10)
tokens = ['backpropagation']
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
best_kernels(tokens, 10)
tokens = ['mxnet']
best_kernels(tokens, 10)
tokens = ['resnet']
best_kernels(tokens, 10)
tokens = ['Capsule network', 'capsulenet']
best_kernels(tokens, 10)
tokens = ['kmeans', 'k means']
best_kernels(tokens, 10)
tokens = ['hierarchical clustering']
best_kernels(tokens, 10)
tokens = ['dbscan']
best_kernels(tokens, 10)
tokens = ['naive bayes']
best_kernels(tokens, 10)
tokens = ['svm']
best_kernels(tokens, 10)
tokens = ['ensemble']
best_kernels(tokens, 10)
tokens = ['stacking', 'stack']
best_kernels(tokens, 10)
tokens = ['feature engineering']
best_kernels(tokens, 10)
tokens = ['feature selection']
best_kernels(tokens, 10)
tokens = ['cross validation']
best_kernels(tokens, 10)
tokens = ['model selection']
best_kernels(tokens, 10)
tokens = ['smote']
best_kernels(tokens, 10)
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
tokens = ['tensorflow', 'tensor flow']
best_kernels(tokens, 10)
tokens = ['eli5']
best_kernels(tokens, 10)
tokens = ['visualization', 'visualisation']
best_kernels(tokens, 10)
tokens = ['plotly', 'plot.ly']
best_kernels(tokens, 10)
tokens = ['seaborn']
best_kernels(tokens, 10)
tokens = ['bokeh']
best_kernels(tokens, 10)
tokens = ['PCA']
best_kernels(tokens, 10)
tokens = ['Tsne', 't-sne']
best_kernels(tokens, 10)